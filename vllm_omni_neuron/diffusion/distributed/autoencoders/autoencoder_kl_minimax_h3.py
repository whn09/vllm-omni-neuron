# SPDX-License-Identifier: Apache-2.0
"""H3-VisualVAE for Neuron — causal 3D CNN encoder plus non-causal ViT decoder.

Raw ``nn.Parameter`` with weight loaders, matching the Wan 2.2 port and the MiniMax-H3 DiT.

Where the parallelism goes
-------------------------
H3 decodes in 256-pixel tiles with a 64-pixel overlap that is linearly cross-faded on the host
(``use_tiling`` is on by default and the released frames are the blended-tile ones, so this is
part of the output contract). The tiles are independent, so the decoder — 36 ViT blocks at
``dim = 2048``, ~2.4B parameters, 4.8 GB in float16 — is **replicated** on every rank and the
tiles of the whole clip (every temporal chunk at once) are split over the world: each rank
decodes its share, the pixels go to the host, and the shares are gathered there. The decode
issues no device collective at all. Unlike Wan2.2's decoder, which convolves across the whole
latent plane and has to exchange halos when it is split, H3's own tiling already supplies the
overlap.

The encoder (~110M parameters of causal 3D convolution) is only needed for keyframe or image
conditioning, so a text-to-video pipeline builds the VAE with ``with_encoder=False``.

Neuron-specific departures from ``diffusers.models.autoencoders.autoencoder_kl_minimax_h3``
-------------------------------------------------------------------------------------------
1. **Reflect padding is expressed as `cat` of edge slices.** A reflect pad of 1
   is ``cat([x[..., 1:2], x, x[..., w-2:w-1]])``; the two-sided spatial pad is that
   applied twice. Negative slice indices are avoided throughout (the Neuron lowering
   does not accept them), so every bound is computed from the shape.
2. **Per-frame GroupNorm without the transposes.** The reference folds the temporal
   axis into the batch axis with two ``permute(0, 2, 1, 3, 4).contiguous()`` calls
   around ``nn.GroupNorm``. Group statistics are over ``(channels_per_group, H, W)``
   at fixed frame, which is a reduction this port takes directly on a
   ``(B, G, C // G, F, H, W)`` view — same numbers, no materialized transpose.
3. **Rotary cos/sin are host-computed.** The decoder's position grid is a pure
   function of the tile's latent shape, so it is built once per shape on the host
   and passed in as a tensor, keeping ``meshgrid`` out of the trace.
4. **Attention is explicit matmul + float32 softmax, not the flash kernel.** The
   token count is ``num_patches + num_register_tokens + 1`` — 1797 for a 256-pixel
   tile — which is not a multiple of the flash kernel's sequence tile. At that
   length the scores fit comfortably (4 heads/rank x 1797^2 in float16 is ~25 MB),
   so the dense form is both correct and cheap here.

Precision
---------
The released checkpoint is float32 and the verified decode recipe is float16
*autocast over float32 weights*. This port holds the matmul weights in
``compute_dtype`` (default ``float16``, to stay on that recipe rather than round the
weights to bfloat16's 7 mantissa bits) and keeps every normalization, the softmax
and the two rotary tables in float32, which is what the autocast did.
"""

import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)


def _col_bias_loader(shard_size, num_shards):
    """Column-parallel bias loader: shard along dim 0."""
    return sharding_weight_loader(shard_dim=0, shard_size=shard_size, num_shards=num_shards)


def _row_weight_loader(shard_size, num_shards):
    """Row-parallel weight loader: shard the input dim (transposed storage)."""
    return sharding_weight_loader(
        shard_dim=0, shard_size=shard_size, num_shards=num_shards, is_storage_transposed=True
    )


# The decoder is replicated: every rank holds all of it and decodes its own share of the tiles,
# so the decode issues no device collective at all. H3's tiles are already independent (256-px
# tiles, cross-faded on the host), and its 2.4B-parameter decoder fits next to the DiT's TP shard.
def get_num_tile_groups() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def get_tile_group_index() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


_GATHER_CALLS = 0
_SAME_HOST: bool | None = None


def _cpu_group():
    from vllm_omni.diffusion.distributed.parallel_state import get_world_group

    return get_world_group().cpu_group


def _all_on_one_host() -> bool:
    """Whether every rank runs on this host (asked once, over the default group)."""
    global _SAME_HOST
    if _SAME_HOST is None:
        import socket

        hosts = [None] * dist.get_world_size()
        dist.all_gather_object(hosts, socket.gethostname(), group=_cpu_group())
        _SAME_HOST = len(set(hosts)) == 1
    return _SAME_HOST


def _gather_to_rank0(mine: list[torch.Tensor], num_groups: int, rank: int) -> list | None:
    """Every rank's host tensors on rank 0, in rank order; ``None`` elsewhere.

    On one host the tensors go through shared memory: each rank writes its own to ``/dev/shm``
    and rank 0 maps them back after a barrier, at memory speed. A 1344x768 clip is ~1 GB of
    decoded tiles, which an object gather pickles and pushes through the CPU group's sockets —
    6 s, most of the decode. Across hosts that is the fallback.
    """
    if not _all_on_one_host():
        shares = [None] * num_groups if rank == 0 else None
        dist.gather_object(mine, shares, dst=0)
        return shares
    global _GATHER_CALLS
    _GATHER_CALLS += 1
    import numpy as np

    # Every worker is a child of the same executor process, and they call this in lockstep.
    # Raw .npy rather than `torch.save`, which consults the (unimplemented) Lite device hooks
    # even for host tensors; bfloat16 travels as its int16 bits.
    def path(index: int) -> str:
        return f"/dev/shm/minimax_h3_{os.getppid()}_{_GATHER_CALLS}_{index}.npy"

    dtype = mine[0].dtype if mine else None
    if mine:
        stacked = torch.stack([tile.contiguous() for tile in mine])
        array = (stacked.view(torch.int16) if dtype == torch.bfloat16 else stacked).numpy()
    else:
        array = np.zeros((0,), dtype=np.float16)
    np.save(path(rank), array)
    dtypes = [None] * num_groups if rank == 0 else None
    dist.gather_object(dtype, dtypes, dst=0, group=_cpu_group())
    if rank != 0:
        return None
    shares = []
    for index in range(num_groups):
        array = np.load(path(index), mmap_mode="r")
        os.unlink(path(index))
        if dtypes[index] is None:
            shares.append([])
            continue
        stacked = torch.from_numpy(array)
        if dtypes[index] == torch.bfloat16:
            stacked = stacked.view(torch.bfloat16)
        shares.append(list(stacked.unbind(0)))
    return shares


def tiles_for(num_tiles: int, num_groups: int, group_index: int) -> tuple[int, int]:
    """The ``[start, end)`` tiles ``group_index`` decodes, the leading groups taking the extra."""
    per_group = -(-num_tiles // num_groups)
    start = min(group_index * per_group, num_tiles)
    return start, min(start + per_group, num_tiles)


def _slice(hidden_states: torch.Tensor, dim: int, start: int, length: int) -> torch.Tensor:
    """Take ``length`` elements from ``dim``, as a fresh contiguous tensor.

    The Neuron backend rejects **every** operation on a non-contiguous tensor,
    including the ones that would make it contiguous: `contiguous`, `clone`, `+ 0` and
    even `.cpu()` all raise ``Expected self.is_contiguous() to be true, but got false``.
    So a strided view cannot be repaired after the fact — it must never be created. A
    slice on any axis but the first produces one (a leading-dim slice stays contiguous).

    `index_select` is the way out: it *gathers* rather than restrides, so it works on a
    contiguous input along any axis and returns a contiguous result.

    On CPU that gather is the wrong tool, and it is not a small difference: at the stitch's
    shape — a `(1, 3, 25, 288, 288)` float32 tile, trimming 64 off the last axis —
    `index_select` takes 17.2 ms where `narrow().contiguous()` takes 1.33 ms, a 13x gap that
    grows to 34x on the second-to-last axis. Both return the same contiguous tensor; the
    gather just reads an index vector and scatters element-wise where the copy moves runs.
    The constraint above is a *device* constraint, so it is applied where it holds. This is
    dispatched on the tensor rather than on a flag because the callers are shared: the same
    `_slice` trims latents on their way to the device and pixels on their way back.

    Only for eager code either way. Inside a traced graph plain slicing is correct and
    cheaper — the compiler handles layout itself.
    """
    # `length` is clamped rather than checked because every caller here replaces a
    # `[start : start + length]` expression, and that is what Python slicing does when
    # the stop runs past the end.
    length = min(length, hidden_states.shape[dim] - start)
    if start == 0 and length == hidden_states.shape[dim]:
        return hidden_states
    if hidden_states.device.type == "cpu":
        return hidden_states.narrow(dim, start, length).contiguous()
    index = torch.arange(start, start + length, device=hidden_states.device)
    return torch.index_select(hidden_states, dim, index)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Host-time accounting, off unless `MINIMAX_H3_DECODE_PROFILE` is set
# ---------------------------------------------------------------------
#
# Why this exists at all. The device profile says the decoder's 28 calls span 14.3 s but
# only 7.6 s of that is on-core, so 6.7 s is host time in the gaps -- and it is 2.5% of a
# 163 s request, more than anything else outside the DiT. What the device profile cannot say
# is *which* host phase: there are four candidates per chunk (tile slicing, `_to_host`, the
# `all_gather_object`, and the stitch), and they are all pure PyTorch on 1.2 GB of pixels.
#
# `time.perf_counter()` around host code is honest here in a way it is not around device
# code: nothing in these phases is asynchronous. The one exception is the device `run`, whose
# timing includes however much of the previous call was still in flight -- so it is recorded
# for completeness but the profiler's `nc_exec_running` is the number to trust for it.
#
# What it found, so the next reader does not repeat it. `to_host` and `stitch` held all of it;
# the other six phases together are under 3%. `stitch` was real host cost and is now 168 ms
# from 441 (`_slice` was gathering with `index_select` on CPU, see there). `to_host` is *not*
# host cost at all: its first tile's copy alone is 287 ms against 28 ms for the other three, so
# it is the sync point for the decode enqueued just above it. The lesson generalizes past this
# decoder -- on a device whose queue is implicit, the phase that first reads a result absorbs
# everything still in flight, and a host timer around it cannot tell you that.
_PHASES: dict[str, list[float]] = defaultdict(list)


def _profiling_decode() -> bool:
    return os.environ.get("MINIMAX_H3_DECODE_PROFILE", "").lower() in {"1", "true", "yes"}


class _phase:
    """Accumulate wall time under `name`. A no-op unless profiling is on."""

    def __init__(self, name: str):
        self.name = name
        self.started = 0.0

    def __enter__(self):
        if _profiling_decode():
            self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.started:
            _PHASES[self.name].append(time.perf_counter() - self.started)
        return False


def log_decode_phases() -> None:
    """Log the accumulated per-phase host time, then reset.

    Called from `decode` so the numbers cover one whole video and not one chunk, since the
    thing being explained -- 6.7 s -- is a total over 28 calls.
    """
    if not _PHASES:
        return
    total = sum(sum(times) for times in _PHASES.values())
    logger.info("video VAE host time: %.2f s total over one decode", total)
    for name, times in sorted(_PHASES.items(), key=lambda kv: -sum(kv[1])):
        seconds = sum(times)
        logger.info(
            "  %-14s %6.2f s  %5.1f%%  %4d call(s)  %6.1f ms each",
            name,
            seconds,
            100 * seconds / total,
            len(times),
            1e3 * seconds / len(times),
        )
    _PHASES.clear()


def _as_float32(hidden_states: torch.Tensor, what: str) -> torch.Tensor:
    """Return ``hidden_states`` as float32, casting only while it is still on the host.

    A dtype change is one more thing the Neuron backend will not do: `.to(fp16)` on
    a device tensor raises ``Expected self.dtype() == dst.dtype() to be true``. So a
    caller that hands over a device tensor has to have cast it before the move, and all
    this can do is say so — hence the message rather than a silent bad cast.

    Both traced graphs take float32 in: the encoder's convolutions and `post_quant_conv`
    are held in float32 by `_param_dtype`, and the decoder's transformer casts down to
    `compute_dtype` at its own `proj_in`, inside the graph where the compiler owns it.
    """
    if hidden_states.dtype == torch.float32:
        return hidden_states
    if hidden_states.device.type == "cpu":
        return hidden_states.to(torch.float32)
    raise ValueError(
        f"{what} must be float32, got {hidden_states.dtype}. It is already on "
        f"{hidden_states.device}, where the dtype can no longer be changed — cast it "
        f"before moving it off the host."
    )


def _cat(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
    """`torch.cat` for eager code, asserting what the Neuron backend silently requires.

    Every input has to be contiguous already — see `_slice` for why one cannot be made
    contiguous here. The assertion turns a confusing backend error deep in a `cat` into a
    pointer at whichever caller built a view.
    """
    for i, tensor in enumerate(tensors):
        assert tensor.is_contiguous(), (
            f"cat input {i} of {len(tensors)} is a strided view; build it with `_slice`."
        )
    return torch.cat(tensors, dim=dim)


def _concat_frames(chunks: list[torch.Tensor]) -> torch.Tensor:
    """Join decoded chunks along the frame axis, by copying into one buffer.

    Same result as ``_cat(chunks, dim=2)``, measurably faster for the size this runs at. On
    1.59 GB of 768p chunks `torch.cat` takes ~200 ms where `copy_` into a preallocated tensor
    takes 17-70 ms; the destination being freshly allocated accounts for at most 5 ms of the
    difference, so this is not first-touch page faults — `cat` is simply several times slower
    than the memcpy it ought to reduce to at this scale. On the box the phase measured 2.4 s
    against 200 ms in isolation, because all 32 ranks do it at once and it is bandwidth the
    whole time.

    Host only. `_cat` explains why an eager device path must not build views, and this builds
    one per chunk to copy through.
    """
    if len(chunks) == 1:
        return chunks[0]
    if chunks[0].device.type != "cpu":
        return _cat(chunks, dim=2)

    frames = sum(chunk.shape[2] for chunk in chunks)
    shape = list(chunks[0].shape)
    shape[2] = frames
    joined = chunks[0].new_empty(shape)
    offset = 0
    for chunk in chunks:
        joined[:, :, offset : offset + chunk.shape[2]].copy_(chunk)
        offset += chunk.shape[2]
    return joined


@dataclass
class MiniMaxH3VideoVAEConfig:
    """Configuration of `NeuronAutoencoderKLMiniMaxH3`.

    Field-for-field the diffusers config of the released ``vae/``. The derived
    temporal geometry is computed in `__post_init__` exactly as the reference's
    ``__init__`` does.
    """

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 24
    block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024)
    layers_per_block: int = 2
    spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1)
    temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1)
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    spatial_padding_mode: str = "reflect"
    decoder_num_layers: int = 36
    decoder_num_attention_heads: int = 32
    decoder_attention_head_dim: int = 64
    decoder_num_register_tokens: int = 4
    decoder_ffn_mult: int = 4
    decoder_rope_theta: float = 100.0
    decoder_rope_dim_ratio: float = 0.75
    decoder_norm_eps: float = 1e-5
    clip_length: int = 17
    token_drop: int = 3
    latents_mean: tuple[float, ...] = (0.0,) * 24
    latents_std: tuple[float, ...] = (1.0,) * 24

    spatial_compression_ratio: int = field(init=False)
    temporal_compression_ratio: int = field(init=False)
    frame_pre_padding: int = field(init=False)
    tokens_chunk_size: int = field(init=False)
    token_overlap: int = field(init=False)
    frame_overlap: int = field(init=False)

    def __post_init__(self):
        for name in ("block_out_channels", "spatial_downsample_factors", "temporal_downsample_factors"):
            setattr(self, name, tuple(getattr(self, name)))
        self.latents_mean = tuple(self.latents_mean)
        self.latents_std = tuple(self.latents_std)

        self.spatial_compression_ratio = math.prod(self.spatial_downsample_factors)
        self.temporal_compression_ratio = math.prod(self.temporal_downsample_factors)

        # `clip_length` (17) is not a multiple of the temporal ratio (4), so the decoder
        # has to re-derive the implicit leading pad and the overlap `token_drop` leaves.
        self.frame_pre_padding = (-self.clip_length) % self.temporal_compression_ratio
        self.tokens_chunk_size = math.ceil(self.clip_length / self.temporal_compression_ratio)
        self.token_overlap = (-self.token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(
            self.token_overlap * self.temporal_compression_ratio - self.frame_pre_padding, 0
        )


# ===================================================================
# Padding and normalization primitives
# ===================================================================


def _reflect_pad_1(hidden_states: torch.Tensor, dim: int, before: int, after: int):
    """Reflect-pad ``dim`` by at most one element on each side, via edge slices.

    ``F.pad(..., mode="reflect")`` on a 5-D tensor asks the backend for a 3-D reflect
    lowering; a `cat` of two explicit slices is the same tensor and traces on Neuron.
    Bounds are computed from the shape rather than written as negative indices, which
    the backend does not accept.
    """
    if before == 0 and after == 0:
        return hidden_states
    size = hidden_states.shape[dim]
    parts = []
    if before:
        parts.append(hidden_states.narrow(dim, 1, 1))
    parts.append(hidden_states)
    if after:
        parts.append(hidden_states.narrow(dim, size - 2, 1))
    return torch.cat(parts, dim=dim)


class MiniMaxH3VideoCausalConv3d(nn.Conv3d):
    """Encoder convolution: symmetric reflect spatial pad, causal zero temporal pad.

    Same geometry as the reference — ``kernel_size_t - 1`` zero frames prepended and
    nothing appended — with the reflect pad routed through `_reflect_pad_1`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        padding = self.spatial_padding
        if padding > 0:
            if padding != 1:
                raise ValueError(
                    f"`spatial_padding` is 1 everywhere in the released checkpoint, got {padding}; "
                    "`_reflect_pad_1` only reflects a single element."
                )
            hidden_states = _reflect_pad_1(hidden_states, dim=3, before=1, after=1)
            hidden_states = _reflect_pad_1(hidden_states, dim=4, before=1, after=1)
        if self.temporal_padding > 0:
            hidden_states = F.pad(
                hidden_states, (0, 0, 0, 0, self.temporal_padding, 0), mode="constant"
            )
        return F.conv3d(
            hidden_states, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation
        )


class MiniMaxH3VideoGroupNorm(nn.Module):
    """Group normalization with statistics taken per latent frame.

    The reference (``use_t_isolated_gn``) folds the temporal axis into the batch axis
    with a permute either side of `nn.GroupNorm`. Group statistics are over
    ``(channels_per_group, H, W)`` at a fixed frame, so this takes the reduction on a
    ``(B, G, C // G, F, H, W)`` view instead — identical numbers, no transpose, and
    the reduction is in float32 regardless of the compute dtype.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        if num_channels % num_groups:
            raise ValueError(f"{num_channels} channels do not split into {num_groups} groups.")
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        grouped = hidden_states.float().view(
            batch_size, self.num_groups, num_channels // self.num_groups, num_frames, height, width
        )
        # Reduce over the channels within a group and over space, but NOT over frames.
        mean = grouped.mean(dim=(2, 4, 5), keepdim=True)
        variance = grouped.var(dim=(2, 4, 5), unbiased=False, keepdim=True)
        grouped = (grouped - mean) * torch.rsqrt(variance + self.eps)
        normalized = grouped.view(batch_size, num_channels, num_frames, height, width)
        weight = self.weight.float().view(1, -1, 1, 1, 1)
        bias = self.bias.float().view(1, -1, 1, 1, 1)
        return (normalized * weight + bias).to(input_dtype)


class RMSNorm(nn.Module):
    """RMSNorm over the last dimension, reduced in float32.

    ``elementwise_affine`` mirrors `nn.RMSNorm`: the decoder's block norms are affine,
    the per-head query/key norms are not.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight.float()
        return hidden_states.to(input_dtype)


class LayerNorm(nn.Module):
    """LayerNorm over the last dimension, reduced in float32."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.eps)
        return (hidden_states * self.weight.float() + self.bias.float()).to(input_dtype)


# ===================================================================
# Encoder — replicated causal 3D CNN
# ===================================================================


class MiniMaxH3VideoResnetBlock3d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, norm_num_groups: int = 32, norm_eps: float = 1e-6
    ):
        super().__init__()
        self.norm1 = MiniMaxH3VideoGroupNorm(norm_num_groups, in_channels, eps=norm_eps)
        self.conv1 = MiniMaxH3VideoCausalConv3d(
            in_channels, out_channels, kernel_size=3, spatial_padding=1, temporal_padding=2
        )
        self.norm2 = MiniMaxH3VideoGroupNorm(norm_num_groups, out_channels, eps=norm_eps)
        self.conv2 = MiniMaxH3VideoCausalConv3d(
            out_channels, out_channels, kernel_size=3, spatial_padding=1, temporal_padding=2
        )
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = MiniMaxH3VideoCausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(F.silu(self.norm1(hidden_states)))
        hidden_states = self.conv2(F.silu(self.norm2(hidden_states)))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class MiniMaxH3VideoDownsample3d(nn.Module):
    """Strided 3x3x3 convolution.

    A spatial stride of 2 is preceded by an asymmetric *bottom/right* reflect pad of 1
    and the convolution itself carries no spatial padding, so the output is exactly
    ``ceil(size / 2)``.
    """

    def __init__(
        self, in_channels: int, out_channels: int, temporal_stride: int = 1, spatial_stride: int = 2
    ):
        super().__init__()
        self.spatial_stride = spatial_stride
        self.conv = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            spatial_padding=0,
            temporal_padding=2,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_stride == 2:
            hidden_states = _reflect_pad_1(hidden_states, dim=3, before=0, after=1)
            hidden_states = _reflect_pad_1(hidden_states, dim=4, before=0, after=1)
        return self.conv(hidden_states)


class MiniMaxH3VideoDownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        temporal_downsample_factor: int,
        spatial_downsample_factor: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                MiniMaxH3VideoResnetBlock3d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
                for i in range(num_layers)
            ]
        )
        self.downsamplers = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsamplers = nn.ModuleList(
                [
                    MiniMaxH3VideoDownsample3d(
                        out_channels,
                        out_channels,
                        temporal_stride=temporal_downsample_factor,
                        spatial_stride=spatial_downsample_factor,
                    )
                ]
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class MiniMaxH3VideoEncoder3d(nn.Module):
    """Causal 3D CNN encoder, left replicated across TP ranks.

    ~110M parameters against the ViT decoder's 2.4B, and it runs only for keyframe /
    reference conditioning rather than once per denoising step, so sharding it would
    add a collective per convolution for no footprint that matters.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 48,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.conv_in = MiniMaxH3VideoCausalConv3d(
            in_channels, block_out_channels[0], kernel_size=3, spatial_padding=1, temporal_padding=2
        )
        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoDownBlock3d(
                    in_channels=block_in_channels[i],
                    out_channels=block_out_channels[i],
                    num_layers=layers_per_block,
                    temporal_downsample_factor=temporal_downsample_factors[i],
                    spatial_downsample_factor=spatial_downsample_factors[i],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
                for i in range(len(block_out_channels))
            ]
        )
        self.norm_out = MiniMaxH3VideoGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps)
        self.conv_out = MiniMaxH3VideoCausalConv3d(
            block_out_channels[-1], out_channels, kernel_size=3, spatial_padding=1, temporal_padding=2
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        return self.conv_out(F.silu(self.norm_out(hidden_states)))


# ===================================================================
# Decoder — head-parallel ViT
# ===================================================================


def build_decoder_rotary_tables(
    num_frames: int,
    height: int,
    width: int,
    num_suffix_tokens: int,
    rotary_dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Host-side ``(cos, sin)`` for one latent tile shape.

    Reproduces `MiniMaxH3VideoRotaryPosEmbed` over the reference's position grid:
    each axis is length-normalized to ``[-1, 1)`` as ``2 * (arange(0.5, size) / size) - 1``,
    the three angle sets are concatenated and then duplicated, and the suffix tokens
    (the register tokens and the zero cls token) all sit at position ``0``.

    Returns two ``(1, num_tokens, 1, rotary_dim)`` float32 tensors — the head axis is
    kept as a broadcast dimension so a TP rank's slice of heads needs no reshaping.
    """
    num_axes = 3
    if rotary_dim % (2 * num_axes):
        raise ValueError(f"`rotary_dim` {rotary_dim} must be divisible by {2 * num_axes}.")
    inv_freq = 1.0 / theta ** torch.arange(0, 1, 2 * num_axes / rotary_dim, dtype=torch.float32)

    grids = [
        2.0 * (torch.arange(0.5, size, dtype=torch.float32) / size) - 1.0
        for size in (num_frames, height, width)
    ]
    position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
    suffix_ids = position_ids.new_zeros((num_suffix_tokens, num_axes))
    position_ids = torch.cat([position_ids, suffix_ids], dim=0).unsqueeze(0)

    angles = 2.0 * math.pi * position_ids[:, :, :, None] * inv_freq[None, None, None, :]
    angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
    return angles.cos(), angles.sin()


def _apply_rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate the leading ``rotary_dim`` channels of every head, pass the rest through.

    ``hidden_states`` is ``(B, S, num_heads, head_dim)``; ``cos`` / ``sin`` are
    ``(1, S, 1, rotary_dim)`` — 48 of the 64 head channels for the released config.
    """
    rotary_dim = cos.shape[-1]
    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]

    cos = cos.to(hidden_states.dtype)
    sin = sin.to(hidden_states.dtype)
    # `tensor_split` rather than `chunk`: XLA mislowers `split` on some axes
    # (pytorch/xla#8640) and `tensor_split` takes indices, which lowers correctly.
    first, second = torch.tensor_split(hidden_states_rotary, [rotary_dim // 2], dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    hidden_states_rotary = hidden_states_rotary * cos + rotated * sin
    return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1)


def _dense_attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float):
    """Non-causal attention as an explicit matmul with a float32 softmax.

    Inputs are ``(B, num_heads, S, head_dim)``. The softmax is taken in float32 as the
    reference's autocast did.

    Two claims that used to be in this docstring were both wrong, and are corrected here
    because they pointed optimization work at the wrong place:

    1. "The flash kernel is not used here: the token count (1797) is not a multiple of its
       sequence tile." **Not true.** ``NF.flash_attention`` (the `attention_cte` NKI kernel,
       already used by the DiT via `_nf_attend`) passes its constraint check at these shapes
       and lowers on device at S=1797, agreeing with this function to cosine 0.99993.
    2. "The fp32 score matrix is the decoder's likely bottleneck, 5.1 s per traced call."
       **Not true, and the 5.1 s was an arithmetic error** — the decoder runs 7 temporal
       chunks x ceil(tiles/4) traced batches per rank (28 unsplit, not 4), so the real cost
       is 0.64 s/call. Measured per block at the shipped shape, this function is 0.88 ms of a
       7.72 ms block — ~11%, not the bottleneck.

    So `attention_cte` is available and slightly faster here (0.73 vs 0.88 ms/block, ~2% of a
    call). It is left unused only because the win is small and this form is the one the
    parity gates were run against; switching is a safe, low-value cleanup. The decoder's real
    cost is spread across the weight-heavy projections plus collectives and the host
    round-trip.
    """
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    weights = torch.softmax(scores, dim=-1).to(value.dtype)
    return torch.matmul(weights, value)


def _fused_qkv_bias_loader(shard_size: int, num_shards: int):
    """Fuse and shard the Q, K, V bias tensors — the 1-D form of `fused_qkv_weight_loader`."""

    def transform(slices, rank):
        assert len(slices) == 3
        start = (rank % num_shards) * shard_size
        return torch.cat([sl[start : start + shard_size] for sl in slices], dim=0)

    return SafetensorsWeightLoader(transform=transform)


def _gated_half_weight_loader(half: int, inner_dim: int, shard_size: int, num_shards: int):
    """Column-parallel loader for one half of the fused SwiGLU projection.

    The converted checkpoint stores ``ff.net.0.proj.weight`` as one
    ``(2 * inner_dim, dim)`` tensor holding ``[value; gate]`` — diffusers' `SwiGLU`
    order, which the conversion script produces by swapping the reference's
    ``[gate; value]``. Splitting it here keeps the fused checkpoint key while giving
    the traced graph two plain matmuls instead of a `chunk`.
    """

    def transform(slices, rank):
        assert len(slices) == 1
        start = half * inner_dim + (rank % num_shards) * shard_size
        # Checkpoint is `(2 * inner_dim, dim)`; the parameter is `(dim, shard_size)`.
        return slices[0][start : start + shard_size, :].T

    return SafetensorsWeightLoader(transform=transform)


def _gated_half_bias_loader(half: int, inner_dim: int, shard_size: int, num_shards: int):
    """Column-parallel loader for one half of the fused SwiGLU bias."""

    def transform(slices, rank):
        assert len(slices) == 1
        start = half * inner_dim + (rank % num_shards) * shard_size
        return slices[0][start : start + shard_size]

    return SafetensorsWeightLoader(transform=transform)


# The decoder is replicated (see `get_num_tile_groups`), so its weight loaders take the whole
# checkpoint tensor; they still fuse Q/K/V and split the gated feed-forward.
_DECODER_SHARDS = 1


def _decoder_shard_count() -> int:
    return _DECODER_SHARDS


def _decoder_shard_rank() -> int:
    return 0


class MiniMaxH3VideoAttention(nn.Module):
    """Full self-attention over one tile's tokens.

    The per-head query/key norms run in float32 regardless of the compute dtype, as in the
    reference.
    """

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-5):
        super().__init__()
        tp_size = _decoder_shard_count()
        if heads % tp_size:
            raise ValueError(
                f"The H3 video decoder shards attention over its {heads} heads, which a shard "
                f"count of {tp_size} does not divide."
            )
        self.tp_size = tp_size
        self.num_heads = heads // tp_size
        self.head_dim = dim_head
        tp_inner_dim = self.num_heads * dim_head
        self.scale = 1.0 / math.sqrt(dim_head)

        self.qkv_split = [tp_inner_dim, 2 * tp_inner_dim]
        self.qkv_proj_weight = nn.Parameter(torch.empty(dim, 3 * tp_inner_dim))
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=tp_inner_dim,
                kv_size=tp_inner_dim,
                shard_dim=1,
                num_shards=tp_size,
                is_storage_transposed=True,
            ),
        )
        self.qkv_proj_bias = nn.Parameter(torch.empty(3 * tp_inner_dim))
        set_weight_loader(self.qkv_proj_bias, _fused_qkv_bias_loader(tp_inner_dim, tp_size))

        self.norm_q = RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = RMSNorm(dim_head, eps=eps, elementwise_affine=False)

        self.o_proj_weight = nn.Parameter(torch.empty(tp_inner_dim, dim))
        set_weight_loader(self.o_proj_weight, _row_weight_loader(tp_inner_dim, tp_size))
        self.o_proj_bias = nn.Parameter(torch.empty(dim))

    def forward(self, hidden_states: torch.Tensor, rotary_emb) -> torch.Tensor:
        # Projections run at the weights' compute dtype, as the reference's float16 autocast
        # does; the residual stream may be float32 (the branch gates are).
        qkv = torch.matmul(hidden_states.to(self.qkv_proj_weight.dtype), self.qkv_proj_weight) + self.qkv_proj_bias
        query, key, value = torch.tensor_split(qkv, self.qkv_split, dim=-1)

        query = query.unflatten(-1, (self.num_heads, self.head_dim))
        key = key.unflatten(-1, (self.num_heads, self.head_dim))
        value = value.unflatten(-1, (self.num_heads, self.head_dim))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        attn_output = _dense_attend(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), self.scale
        )
        output = attn_output.transpose(1, 2).flatten(2)
        return torch.matmul(output.to(self.o_proj_weight.dtype), self.o_proj_weight) + self.o_proj_bias


class MiniMaxH3VideoFeedForward(nn.Module):
    """SwiGLU feed-forward, column-parallel on both halves and row-parallel down.

    ``value * silu(gate)`` then the down projection, matching diffusers' `SwiGLU`
    inside `FeedForward` — the order the converted checkpoint stores (see
    `_gated_half_weight_loader`). All three projections carry a bias here, unlike the
    DiT's.
    """

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        tp_size = _decoder_shard_count()
        self.tp_size = tp_size
        inner_dim = dim * mult
        if inner_dim % tp_size:
            raise ValueError(f"ffn dim {inner_dim} is not divisible by shard count {tp_size}.")
        inner_per_rank = inner_dim // tp_size

        self.value_proj_weight = nn.Parameter(torch.empty(dim, inner_per_rank))
        set_weight_loader(
            self.value_proj_weight, _gated_half_weight_loader(0, inner_dim, inner_per_rank, tp_size)
        )
        self.value_proj_bias = nn.Parameter(torch.empty(inner_per_rank))
        set_weight_loader(
            self.value_proj_bias, _gated_half_bias_loader(0, inner_dim, inner_per_rank, tp_size)
        )
        self.gate_proj_weight = nn.Parameter(torch.empty(dim, inner_per_rank))
        set_weight_loader(
            self.gate_proj_weight, _gated_half_weight_loader(1, inner_dim, inner_per_rank, tp_size)
        )
        self.gate_proj_bias = nn.Parameter(torch.empty(inner_per_rank))
        set_weight_loader(
            self.gate_proj_bias, _gated_half_bias_loader(1, inner_dim, inner_per_rank, tp_size)
        )

        self.down_proj_weight = nn.Parameter(torch.empty(inner_per_rank, dim))
        set_weight_loader(self.down_proj_weight, _row_weight_loader(inner_per_rank, tp_size))
        self.down_proj_bias = nn.Parameter(torch.empty(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.value_proj_weight.dtype)
        value = torch.matmul(hidden_states, self.value_proj_weight) + self.value_proj_bias
        gate = torch.matmul(hidden_states, self.gate_proj_weight) + self.gate_proj_bias
        hidden = (value * F.silu(gate)).to(self.down_proj_weight.dtype)
        return torch.matmul(hidden, self.down_proj_weight) + self.down_proj_bias


class MiniMaxH3VideoTransformerBlock(nn.Module):
    """Pre-norm ViT block with learned per-branch output gates.

    ``scale1`` / ``scale2`` are zero-initialized in training, so the residual stream is
    the identity at initialization; in the released checkpoint they are the learned
    per-channel weight each branch contributes.
    """

    def __init__(self, dim: int, heads: int, dim_head: int, ffn_mult: int = 4, eps: float = 1e-5):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.attn = MiniMaxH3VideoAttention(dim=dim, heads=heads, dim_head=dim_head, eps=eps)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = RMSNorm(dim, eps=eps)
        self.ff = MiniMaxH3VideoFeedForward(dim, mult=ffn_mult)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: torch.Tensor, rotary_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), rotary_emb) * self.scale1
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states)) * self.scale2
        return hidden_states


class MiniMaxH3VideoViTDecoder3d(nn.Module):
    """Non-causal ViT decoder over one batch of latent tiles.

    Every latent voxel becomes one token; ``num_register_tokens`` learned register
    tokens plus a single all-zero token are appended (all at position ``0``), attended
    over with full self-attention, and dropped again before the patch projection
    expands each token into a ``patch_size_t x patch_size x patch_size`` pixel block.

    ``forward`` takes the rotary tables as arguments rather than building them, so the
    position grid — a pure function of the tile's latent shape — stays on the host.
    """

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        patch_size_t: int = 4,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.dim = dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoTransformerBlock(
                    dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    ffn_mult=ffn_mult,
                    eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = LayerNorm(dim, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def forward(
        self, hidden_states: torch.Tensor, rotary_cos: torch.Tensor, rotary_sin: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
            batch_size, num_frames * height * width, num_channels
        )
        hidden_states = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        num_patches = hidden_states.shape[1]

        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        rotary_emb = (rotary_cos, rotary_sin)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)

        hidden_states = self.proj_out(self.norm_out(hidden_states).to(self.proj_out.weight.dtype))
        hidden_states = hidden_states[:, :num_patches, :]

        patch_size, patch_size_t = self.patch_size, self.patch_size_t
        hidden_states = hidden_states.view(
            batch_size,
            num_frames,
            height,
            width,
            self.out_channels,
            patch_size_t,
            patch_size,
            patch_size,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return hidden_states.reshape(
            batch_size,
            self.out_channels,
            num_frames * patch_size_t,
            height * patch_size,
            width * patch_size,
        )


class NeuronMiniMaxH3VideoDecoder(nn.Module):
    """`post_quant_conv` + the ViT decoder, the unit that gets traced as one NEFF."""

    def __init__(self, post_quant_conv: nn.Module, decoder: MiniMaxH3VideoViTDecoder3d):
        super().__init__()
        self.post_quant_conv = post_quant_conv
        self.decoder = decoder

    def forward(self, z: torch.Tensor, rotary_cos: torch.Tensor, rotary_sin: torch.Tensor):
        return self.decoder(self.post_quant_conv(z), rotary_cos, rotary_sin)


class NeuronMiniMaxH3VideoEncoder(nn.Module):
    """The CNN encoder + `quant_conv`, as one module (the eager path)."""

    def __init__(self, encoder: MiniMaxH3VideoEncoder3d, quant_conv: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv

    def forward(self, x: torch.Tensor):
        return self.quant_conv(self.encoder(x))


class _EncoderStage(nn.Module):
    """A run of consecutive encoder modules, traced as one NEFF."""

    def __init__(self, *parts: nn.Module):
        super().__init__()
        self.parts = nn.ModuleList(parts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for part in self.parts:
            x = part(x)
        return x


class _EncoderHead(nn.Module):
    def __init__(self, encoder: MiniMaxH3VideoEncoder3d, quant_conv: nn.Module):
        super().__init__()
        self.norm_out = encoder.norm_out
        self.conv_out = encoder.conv_out
        self.quant_conv = quant_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.conv_out(F.silu(self.norm_out(x))))


class _Drained(nn.Module):
    """A stage that also returns one element of its output, for the host to read.

    The Lite runtime retires an execution only once one of its outputs is read back to the
    host; neither ``synchronize()`` nor freeing the outputs does. Chained stages whose
    outputs stay in HBM therefore fill its execution queue after ~8 launches ("Execution
    Queue Full"). Reading a one-element probe retires the launch for free.
    """

    def __init__(self, stage: nn.Module):
        super().__init__()
        self.stage = stage

    def forward(self, x: torch.Tensor):
        y = self.stage(x)
        return y, y.reshape(-1)[:1] * 1


def encoder_stages(encoder: MiniMaxH3VideoEncoder3d, quant_conv: nn.Module) -> list[nn.Module]:
    """The encoder + `quant_conv` cut into stages that each compile as their own graph.

    As one graph, the float32 encoder over a 17-frame 256x256 tile takes neuronx-cc over half an
    hour and ~100 GB of host memory per rank — and every rank compiles its own. The cost sits in
    the full-resolution blocks, so the first two down blocks are cut per resnet and the rest per
    block. The activations between stages go through HBM: a few hundred MB per tile.
    """

    def block_parts(block):
        return list(block.resnets) + list(block.downsamplers or [])

    first, second = (block_parts(encoder.down_blocks[i]) for i in range(2))
    stages = [
        _EncoderStage(encoder.conv_in, first[0]),
        _EncoderStage(*first[1:]),
        _EncoderStage(second[0]),
        _EncoderStage(*second[1:]),
    ]
    stages += [_EncoderStage(block) for block in encoder.down_blocks[2:]]
    stages.append(_EncoderHead(encoder, quant_conv))
    return stages


# ===================================================================
# Top-level autoencoder
# ===================================================================


class NeuronAutoencoderKLMiniMaxH3(nn.Module):
    r"""H3-VisualVAE for Neuron.

    Latents are normalized with per-channel ``latents_mean`` / ``latents_std`` rather
    than a ``scaling_factor``: a pipeline encodes with ``(latent - mean) / std`` and
    decodes with ``latent * std + mean``.

    The pixel convention is ImageNet-normalized RGB over a ``[0, 1]`` base range, not
    the usual ``[-1, 1]``: `encode` expects ``(pixel - imagenet_mean) / imagenet_std``
    and `decode` returns values in that same space, so a pipeline applies
    ``sample * imagenet_std + imagenet_mean`` and clamps to ``[0, 1]`` before
    postprocessing.

    Spatial tiling is **on**: MiniMax-H3 was released with tiling enabled and the
    released frames are the blended-tile ones, so turning it off changes the output.
    """

    def __init__(self, compute_dtype: torch.dtype = torch.float16, with_encoder: bool = True, **kwargs):
        super().__init__()
        self.config = MiniMaxH3VideoVAEConfig(
            **{k: v for k, v in kwargs.items() if k in MiniMaxH3VideoVAEConfig.__dataclass_fields__}
        )
        config = self.config
        self.compute_dtype = compute_dtype

        # The encoder runs only for keyframe / image-reference conditioning, so a t2va pipeline
        # leaves it unbuilt: replicated and float32 it is 688 MB of HBM on every rank. `None`
        # rather than absent, so `_encoder_module` can tell "not built" from "not loaded" and
        # `load_weights` never asks the checkpoint for its parameters.
        self._encoder_skipped = not with_encoder
        if self._encoder_skipped:
            self.encoder = None
            self.quant_conv = None
        else:
            self.encoder = MiniMaxH3VideoEncoder3d(
                in_channels=config.in_channels,
                out_channels=2 * config.latent_channels,
                block_out_channels=config.block_out_channels,
                layers_per_block=config.layers_per_block,
                spatial_downsample_factors=config.spatial_downsample_factors,
                temporal_downsample_factors=config.temporal_downsample_factors,
                norm_num_groups=config.norm_num_groups,
                norm_eps=config.norm_eps,
            )
            self.quant_conv = nn.Conv3d(
                2 * config.latent_channels, 2 * config.latent_channels, kernel_size=1
            )
        self.post_quant_conv = nn.Conv3d(
            config.latent_channels, config.latent_channels, kernel_size=1
        )
        self.decoder = MiniMaxH3VideoViTDecoder3d(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            patch_size=config.spatial_compression_ratio,
            patch_size_t=config.temporal_compression_ratio,
            num_layers=config.decoder_num_layers,
            num_attention_heads=config.decoder_num_attention_heads,
            attention_head_dim=config.decoder_attention_head_dim,
            num_register_tokens=config.decoder_num_register_tokens,
            ffn_mult=config.decoder_ffn_mult,
            norm_eps=config.decoder_norm_eps,
        )

        self.use_tiling = True
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64

        # How many spatial tiles go through the decoder in one traced call. Batching the
        # whole grid at once is what one wants for utilization, but the ViT decoder's
        # sequence is `frames * height * width` patches *per tile*, and stacking 15 tiles
        # of a 544x960 frame overruns on-chip SBUF at compile time — neuronx-cc gives up
        # with `[NCC_INLA001] Allocated memory out of bound ... (128x498240)`. Four keeps
        # the graph inside SBUF while still amortizing most of the per-call overhead. It
        # only affects batching, never the arithmetic: each tile is independent, and the
        # cross-fade that joins them happens afterwards on the host.
        self.tile_batch_size = 4

        self._compiled_decoder = None
        self._compiled_encoder = None
        # Rotary tables, keyed by latent tile shape. There is one shape per compiled
        # decoder graph, so this holds a single entry in steady state.
        self._rotary_cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    # ---------------------------------------------------------------
    # Tiling geometry (host-side; identical arithmetic to the reference)
    # ---------------------------------------------------------------

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_overlap_height: int | None = None,
        tile_sample_min_overlap_width: int | None = None,
    ) -> None:
        """Change the tile geometry. Changing it after `compile` forces a retrace."""
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_overlap_height = (
            tile_sample_min_overlap_height or self.tile_sample_min_overlap_height
        )
        self.tile_sample_min_overlap_width = (
            tile_sample_min_overlap_width or self.tile_sample_min_overlap_width
        )

    def disable_tiling(self) -> None:
        """Turn tiling off. This changes the output relative to the release."""
        self.use_tiling = False

    @property
    def device(self) -> torch.device:
        """Where the weights live — read off a parameter rather than tracked separately.

        The clip helpers hand their results back on the host, so they need this to put
        the next clip's input back where the graph expects it.
        """
        return self.post_quant_conv.weight.device

    def _split_tiles(self, length: int, tile_size: int, min_overlap: int):
        """Lay ``tile_size``-wide tiles over ``length`` pixels.

        The tile count is the smallest one whose union covers ``length`` with every
        overlap at least ``min_overlap``; the slack is then distributed round-robin over
        the overlaps in whole ``spatial_compression_ratio`` steps so that every tile
        boundary stays latent-aligned.
        """
        ratio = self.config.spatial_compression_ratio
        if tile_size >= length:
            return [0], [length], []

        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1

        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // ratio):
            overlaps[i % (num_tiles - 1)] += ratio

        tile_start_indices = [0]
        for i in range(num_tiles - 1):
            tile_start_indices.append(tile_start_indices[-1] + tile_size - overlaps[i])
        return tile_start_indices, [tile_size] * num_tiles, overlaps

    def _blend(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        """Linear cross-fade of the last ``blend_extent`` of ``a`` into the first of ``b``."""
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).view(shape)
        weight_b = (positions / blend_extent).view(shape)

        size_a = a.shape[dim]
        blended = (
            _slice(a, dim, size_a - blend_extent, blend_extent) * weight_a
            + _slice(b, dim, 0, blend_extent) * weight_b
        )
        if blend_extent == b.shape[dim]:
            return blended
        rest = _slice(b, dim, blend_extent, b.shape[dim] - blend_extent)
        return _cat([blended, rest], dim=dim)

    def _blend_temporal(
        self, previous: torch.Tensor, chunk: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        """Cross-fade the tail of ``previous`` into the head of ``chunk``, along frames.

        Same result as ``_blend(previous, chunk, blend_extent, dim=-3)``, but it does not
        rebuild the chunk to do it. `_blend` ends by concatenating the faded strip back onto
        the untouched remainder, and these are *full-resolution pixel* chunks — at 768p that
        `cat` copies about 90 MB to alter the handful of frames in the seam. Writing the fade
        into ``chunk`` in place touches only those frames.

        Overwriting the caller's tensor is safe because the caller has just built it with
        `_slice`, whose result is a fresh copy — except in the one case where `_slice` passes
        its input straight through, which the check below covers rather than assumes away.
        In-place on a host tensor only, for the reason `_slice` gives.
        """
        blend_extent = min(previous.shape[-3], chunk.shape[-3], blend_extent)
        if chunk.device.type != "cpu" or blend_extent == chunk.shape[-3]:
            return self._blend(previous, chunk, blend_extent, dim=-3)

        weight = (torch.arange(blend_extent, dtype=chunk.dtype) / blend_extent).view(
            blend_extent, 1, 1
        )
        tail = previous[..., previous.shape[-3] - blend_extent :, :, :]
        chunk[..., :blend_extent, :, :].mul_(weight).add_(tail * (1 - weight))
        return chunk

    def _run_tiles(self, tiles: list[torch.Tensor], run, batch_size: int | None = None) -> list[torch.Tensor]:
        """Push ``tiles`` through ``run`` in fixed-size batches, results on the host.

        Every tile has the same shape, so a batch is just a `cat` along the leading axis. The
        last batch is padded up to `tile_batch_size` by repeating its final tile rather than
        left short: a shorter batch is a different graph shape, i.e. a second compile. The
        padding decodes to garbage that is dropped here.
        """
        batch_size = batch_size or self.tile_batch_size
        outputs: list[torch.Tensor] = []
        logger.info(
            "video VAE: %d tiles of %s in batches of %d",
            len(tiles),
            tuple(tiles[0].shape),
            batch_size,
        )
        for start in range(0, len(tiles), batch_size):
            batch = tiles[start : start + batch_size]
            num_real = len(batch)
            batch = batch + [batch[-1]] * (batch_size - num_real)
            with _phase("batch_cat"):
                batched = _cat(batch, dim=0)
            with _phase("device_run"):
                decoded = run(batched)
            # `split` over the leading axis, so each piece stays contiguous — `_slice` says
            # why that matters before anything moves to the host. The first copy to the host
            # is also where the decode enqueued above is waited on.
            per_tile = decoded.split(tiles[0].shape[0], dim=0)
            with _phase("to_host"):
                outputs.extend(self._to_host(per_tile[:num_real]))
        return outputs

    def _run_tiles_shared(self, tiles: list[torch.Tensor], run, batch_size: int) -> list[torch.Tensor]:
        """Run this rank's share of ``tiles`` and all-gather every share onto every rank's host.

        The encoder's counterpart of `_run_tiles_split`: every rank needs the encoded
        conditioning, and its tiles are latents (16x smaller per axis than the pixels), so an
        all-gather over the CPU group is cheap.
        """
        num_groups = get_num_tile_groups()
        if num_groups <= 1:
            return self._run_tiles(tiles, run, batch_size)
        start, end = tiles_for(len(tiles), num_groups, get_tile_group_index())
        mine = self._run_tiles(tiles[start:end], run, batch_size) if end > start else []
        if end <= start:
            # Reach the one-time compile together with the other ranks (see `_run_tiles_split`).
            self._run_tiles(tiles[:1], run, batch_size)
        shares: list[list[torch.Tensor] | None] = [None] * num_groups
        dist.all_gather_object(shares, mine)
        return [tile for share in shares for tile in share]

    def _run_tiles_split(self, tiles: list[torch.Tensor], run) -> list[torch.Tensor] | None:
        """Decode this rank's share of ``tiles``, then gather every share on rank 0's host.

        The gather is object-based over the CPU group, not a device collective: the pixels are
        already on the host (`_to_host`), so it costs no HBM and is not subject to the device
        mesh's replica-group constraints. Only rank 0 receives them (a 1344x768 clip is ~4 GB of
        decoded tiles, which an all-gather would copy to every rank), so the stitch and the
        temporal blend run there alone and every other rank gets ``None``.
        """
        num_groups = get_num_tile_groups()
        if num_groups <= 1:
            return self._run_tiles(tiles, run)

        start, end = tiles_for(len(tiles), num_groups, get_tile_group_index())
        # One graph for every rank's share: the share size, not a fixed batch, sets the shape.
        self.tile_batch_size = max(1, min(self.tile_batch_size, -(-len(tiles) // num_groups)))
        logger.info(
            "video VAE: %d tiles over %d rank(s); this rank decodes [%d, %d).",
            len(tiles),
            num_groups,
            start,
            end,
        )
        # A rank with no tiles still decodes one (discarded) so that every rank reaches the
        # one-time compile together rather than waiting in the gather while the others compile.
        mine = self._run_tiles(tiles[start:end], run) if end > start else []
        if end <= start:
            self._run_tiles(tiles[:1], run)
        rank = get_tile_group_index()
        with _phase("tile_gather"):
            shares = _gather_to_rank0(mine, num_groups, rank)
        if rank != 0:
            return None
        gathered = [tile for share in shares for tile in share]
        if len(gathered) != len(tiles):
            raise RuntimeError(f"Gathered {len(gathered)} decoded tiles for {len(tiles)} inputs.")
        return gathered

    @staticmethod
    def _to_host(tiles: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        """Move decoded tiles off the device before they are stitched.

        Stitching is a cross-fade and a `cat` over *full-resolution* tiles, and doing it
        on device runs the 24 GiB HBM out: the tiles, the blended overlaps and each
        partially assembled row all have to be resident at once, on top of the decoder's
        weights and its NEFF's scratchpad, for a tensor whose destination is the host
        anyway (`nrt_tensor_allocate status=4`). Nothing in the stitch is arithmetic the
        accelerator is needed for.

        `split` along the leading axis is used to produce these, so they are contiguous
        and `.cpu()` will accept them — see `_slice` for why that matters.
        """
        return [tile.to("cpu") for tile in tiles]

    def _stitch_tiles(self, tiles, height_overlaps, width_overlaps) -> torch.Tensor:
        """Reassemble a tile grid into one canvas, cross-fading the seams.

        Host-side; `_to_host` explains why. Two implementations, agreeing bit for bit:
        `_stitch_into_buffer` when the tiles are on the host, which they are for every
        shipped path, and `_stitch_by_concatenation` otherwise. The split exists because the
        fast version writes into slices of a preallocated tensor in place, and that is
        precisely what the Neuron backend refuses (`_slice`).

        An overlap as wide as a whole tile also falls back, because there the two stop being
        equivalent: `_blend` returns just the faded strip in that case, so the reference's
        output stops being the grid the buffer is sized from. No grid `_split_tiles` produces
        does that; the check is here so that claim is tested rather than assumed.
        """
        tile_height, tile_width = tiles[0][0].shape[-2], tiles[0][0].shape[-1]
        if any(o >= tile_height for o in height_overlaps) or any(
            o >= tile_width for o in width_overlaps
        ):
            return self._stitch_by_concatenation(tiles, height_overlaps, width_overlaps)
        if all(tile.device.type == "cpu" for row in tiles for tile in row):
            return self._stitch_into_buffer(tiles, height_overlaps, width_overlaps)
        return self._stitch_by_concatenation(tiles, height_overlaps, width_overlaps)

    def _stitch_by_concatenation(self, tiles, height_overlaps, width_overlaps) -> torch.Tensor:
        """Blend up (dim -2) then left (dim -1), trim the non-last tiles, concatenate.

        The reference for `_stitch_into_buffer`, and the fallback for tiles that are not on
        the host. Every step allocates: each `_blend` returns a whole fresh tile, each trim
        another, and the two `_cat`s copy every tile again — five or six passes over the
        pixels to emit one. At 768p that is 353 ms per temporal chunk against 63 ms for the
        buffer version, on 617 MB of tiles.
        """
        result_rows = []
        for i, row in enumerate(tiles):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1:
                    tile = _slice(tile, -2, 0, tile.shape[-2] - height_overlaps[i])
                if j < len(row) - 1:
                    tile = _slice(tile, -1, 0, tile.shape[-1] - width_overlaps[j])
                result_row.append(tile)
            result_rows.append(_cat(result_row, dim=-1))
        return _cat(result_rows, dim=-2)

    def _stitch_into_buffer(self, tiles, height_overlaps, width_overlaps) -> torch.Tensor:
        """Write each tile into one output buffer, fading only the seams, in place.

        The concatenating version's `cat`s are there to put tiles next to each other; with
        the canvas allocated up front, placement is an assignment instead. What is left is
        the only arithmetic that differs from a copy — the cross-fade across each seam — and
        that runs on strips 64 to 96 pixels wide rather than riding along on whole tiles.

        This is bit-for-bit identical to `_stitch_by_concatenation`, not merely equivalent,
        which rests on two details:

        * **The fades read the original neighbouring tiles, never the buffer.** In the
          reference, ``row[j - 1]`` is the *untouched* tile, not the height-blended one that
          was emitted, so where both fades meet the weights come out
          ``(1 - wx)*left + wx*(1 - wy)*above + wx*wy*tile`` and the diagonal neighbour
          carries no weight at all. Reading the buffer would give the bilinear form instead:
          also a partition of unity, also plausible-looking, a different number.
        * **Height first, then width, both in place.** Once the tile is placed and its top
          strip faded, the buffer holds exactly the reference's height-blended tile, which is
          what its width blend consumes — so the width fade may read the buffer, because what
          it is reading is that tensor.

        Tiles are written at full extent, overrunning the region the trims used to cut away.
        In row-major order that is safe: the overrun of ``(i, j)`` lands only in territory
        owned by ``(i, j+1)``, ``(i+1, j)`` and ``(i+1, j+1)``, all written afterwards, and
        none of which read the buffer for a fade operand.
        """
        rows, cols = len(tiles), len(tiles[0])
        tile_height, tile_width = tiles[0][0].shape[-2], tiles[0][0].shape[-1]

        # The output extent is the tiles minus every overlap, which is what the reference's
        # trims and concatenations add up to.
        height = tile_height * rows - sum(height_overlaps)
        width = tile_width * cols - sum(width_overlaps)
        canvas = tiles[0][0].new_empty((*tiles[0][0].shape[:-2], height, width))

        y_starts = [0]
        for i in range(rows - 1):
            y_starts.append(y_starts[-1] + tile_height - height_overlaps[i])
        x_starts = [0]
        for j in range(cols - 1):
            x_starts.append(x_starts[-1] + tile_width - width_overlaps[j])

        for i, row in enumerate(tiles):
            for j, tile in enumerate(row):
                y_start, x_start = y_starts[i], x_starts[j]
                canvas[
                    ..., y_start : y_start + tile_height, x_start : x_start + tile_width
                ] = tile

                if i > 0:
                    above = tiles[i - 1][j]
                    extent = min(above.shape[-2], tile_height, height_overlaps[i - 1])
                    weight = torch.arange(extent, dtype=tile.dtype).view(extent, 1) / extent
                    canvas[
                        ..., y_start : y_start + extent, x_start : x_start + tile_width
                    ].mul_(weight).add_(above[..., above.shape[-2] - extent :, :] * (1 - weight))
                if j > 0:
                    left = row[j - 1]
                    extent = min(left.shape[-1], tile_width, width_overlaps[j - 1])
                    weight = torch.arange(extent, dtype=tile.dtype) / extent
                    canvas[
                        ..., y_start : y_start + tile_height, x_start : x_start + extent
                    ].mul_(weight).add_(left[..., left.shape[-1] - extent :] * (1 - weight))
        return canvas

    # ---------------------------------------------------------------
    # Compilation
    # ---------------------------------------------------------------

    def compile(self, *args, **compiler_kwargs):
        """Trace the decoder as one graph and the encoder as `encoder_stages`.

        Every tile of a clip has the same shape, so both graphs are shape-static: the
        decoder sees ``(num_tiles, latent_channels, tokens_chunk_size + token_overlap,
        tile_h, tile_w)`` and the encoder ``(num_tiles, in_channels, clip_length,
        tile_h, tile_w)``. The tile count varies with the canvas size, which is one
        bucket per resolution.
        """
        self._compiled_decoder = torch.compile(
            NeuronMiniMaxH3VideoDecoder(self.post_quant_conv, self.decoder),
            *args,
            **compiler_kwargs,
        )
        if not self._encoder_skipped:
            # The stages share `_Drained.forward`, and dynamo caches graphs per code object:
            # nine stages at two tile shapes (a keyframe, a video clip) exceed the default 8.
            import torch._dynamo.config as dynamo_config

            dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 64)
            self._compiled_encoder = [
                torch.compile(_Drained(stage), *args, **compiler_kwargs)
                for stage in encoder_stages(self.encoder, self.quant_conv)
            ]
        return self

    # `is not None`, not `or`: an `nn.Module` defines no `__bool__`, so truth-testing one
    # falls back to `__len__`, and `torch.compile`'s `OptimizedModule` raises `TypeError`
    # from `__len__` rather than reporting a length. Eager modules answer 0 and would
    # silently take the uncompiled branch.
    def _decoder_module(self):
        if self._compiled_decoder is not None:
            return self._compiled_decoder
        return NeuronMiniMaxH3VideoDecoder(self.post_quant_conv, self.decoder)

    def _encoder_module(self):
        if self._encoder_skipped:
            raise RuntimeError(
                "The video VAE encoder was not built (`with_encoder=False`). It is only "
                "needed to condition on a keyframe or a reference image."
            )
        if self._compiled_encoder is None:
            return NeuronMiniMaxH3VideoEncoder(self.encoder, self.quant_conv)
        stages = self._compiled_encoder

        def run(x: torch.Tensor) -> torch.Tensor:
            for stage in stages:
                x, probe = stage(x)
                probe.to("cpu")  # retires the launch; see `_Drained`
            return x

        return run

    def _rotary_tables(self, num_frames: int, height: int, width: int, device: torch.device):
        """Fetch (or build) the host-side rotary tables for one latent tile shape."""
        key = (num_frames, height, width)
        if key not in self._rotary_cache:
            config = self.config
            rotary_dim = int(config.decoder_attention_head_dim * config.decoder_rope_dim_ratio)
            cos, sin = build_decoder_rotary_tables(
                num_frames=num_frames,
                height=height,
                width=width,
                num_suffix_tokens=config.decoder_num_register_tokens + 1,
                rotary_dim=rotary_dim,
                theta=config.decoder_rope_theta,
            )
            self._rotary_cache[key] = (cos.to(device), sin.to(device))
        return self._rotary_cache[key]

    # ---------------------------------------------------------------
    # Tiled clip encode / decode
    # ---------------------------------------------------------------

    def _tile_grid(self, height: int, width: int):
        """Tile starts, lengths and overlaps for both spatial axes, in pixel space."""
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )
        return (y_indices, y_lengths, y_overlaps), (x_indices, x_lengths, x_overlaps)

    def _encode_clip(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one temporal clip, spatially tiled, all tiles in one traced call.

        MiniMax-H3 encodes a keyframe or an image reference through here rather than
        through `encode`, because a single frame must not go through temporal chunking.

        Returns a **host** tensor either way, so that the two branches agree and the
        chunk-level code above can stay in one place; see `_to_host`.
        """
        encoder = self._encoder_module()
        x = _as_float32(x, "`encode` input").to(self.device)
        if not self.use_tiling:
            return encoder(x).to("cpu")

        (y_indices, y_lengths, y_overlaps), (x_indices, x_lengths, x_overlaps) = self._tile_grid(
            x.shape[-2], x.shape[-1]
        )
        tiles = [
            _slice(_slice(x, -2, i_pos, i_len), -1, j_pos, j_len)
            for i_pos, i_len in zip(y_indices, y_lengths)
            for j_pos, j_len in zip(x_indices, x_lengths)
        ]
        # The tiles are spread over the ranks. A single frame runs several tiles per call; a
        # multi-frame clip (a video reference) runs one, because the batched float32 encoder
        # graph over 17-frame tiles takes neuronx-cc the better part of an hour to compile.
        batch_size = self.tile_batch_size if x.shape[2] == 1 else 1
        host_tiles = self._run_tiles_shared(tiles, encoder, batch_size)

        num_x = len(x_indices)
        rows = [host_tiles[i * num_x : (i + 1) * num_x] for i in range(len(y_indices))]
        ratio = self.config.spatial_compression_ratio
        return self._stitch_tiles(
            rows, [o // ratio for o in y_overlaps], [o // ratio for o in x_overlaps]
        )

    def _decode_clip(self, z: torch.Tensor):
        """One temporal clip's latent tiles and tile grid, or its pixels when tiling is off.

        Untiled, the clip is decoded here and comes back as **pixels on the host**, because a
        full-resolution pixel clip is far too big to keep in HBM alongside the decoder — see
        `_to_host`. Tiled, `_decode_clips` decodes and stitches the tiles.
        """
        ratio = self.config.spatial_compression_ratio
        decoder = self._decoder_module()
        z = _as_float32(z, "`decode` input").to(self.device)

        if not self.use_tiling:
            cos, sin = self._rotary_tables(z.shape[2], z.shape[3], z.shape[4], z.device)
            return decoder(z, cos, sin).to("cpu")

        # Tiles are laid out in pixel space and then mapped back onto the latent grid.
        height = z.shape[-2] * ratio
        width = z.shape[-1] * ratio
        (y_indices, y_lengths, y_overlaps), (x_indices, x_lengths, x_overlaps) = self._tile_grid(
            height, width
        )

        with _phase("tile_slice"):
            tiles = [
                _slice(
                    _slice(z, -2, i_pos // ratio, i_len // ratio),
                    -1,
                    j_pos // ratio,
                    j_len // ratio,
                )
                for i_pos, i_len in zip(y_indices, y_lengths)
                for j_pos, j_len in zip(x_indices, x_lengths)
            ]
        return tiles, (len(y_indices), len(x_indices), y_overlaps, x_overlaps)

    def _decode_clips(self, clips: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode several temporal clips with all their tiles in one pass over the world.

        Every clip has the same tile grid, so the tiles of all clips are decoded together and
        the world's ranks share the whole video rather than one temporal chunk at a time (a
        1344x768 chunk is only 28 tiles, which would leave most of a 64-rank world idle).
        """
        if not self.use_tiling:
            return [self._decode_clip(clip) for clip in clips]
        decoder = self._decoder_module()
        tiles, grid = [], None
        for clip in clips:
            clip_tiles, grid = self._decode_clip(clip)
            tiles.extend(clip_tiles)
        # The rotary tables depend on a tile's latent shape, which every tile shares, so
        # they are built once and reused across batches.
        tile = tiles[0]
        cos, sin = self._rotary_tables(tile.shape[2], tile.shape[3], tile.shape[4], tile.device)
        host_tiles = self._run_tiles_split(tiles, lambda batch: decoder(batch, cos, sin))
        if host_tiles is None:
            return None
        num_y, num_x, y_overlaps, x_overlaps = grid
        per_clip = num_y * num_x
        decoded = []
        for index in range(len(clips)):
            clip_tiles = host_tiles[index * per_clip : (index + 1) * per_clip]
            rows = [clip_tiles[i * num_x : (i + 1) * num_x] for i in range(num_y)]
            with _phase("stitch"):
                decoded.append(self._stitch_tiles(rows, y_overlaps, x_overlaps))
        return decoded

    # ---------------------------------------------------------------
    # Temporal chunking
    # ---------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a video in ``clip_length``-frame chunks, dropping ``token_drop`` tail
        latent frames.

        Returns the ``2 * latent_channels`` moments; the caller takes the mode or samples
        the posterior. MiniMax-H3 encodes a video reference through here because the
        posterior is sampled under a fixed generator rather than through a distribution
        object.
        """
        clip_length = self.config.clip_length
        num_frames = x.shape[2]
        if num_frames % clip_length != 0:
            pad = (-num_frames) % clip_length
            last = _slice(x, 2, num_frames - 1, 1)
            x = _cat([x, last.repeat(1, 1, pad, 1, 1)], dim=2)

        moments = _cat(
            [
                self._encode_clip(_slice(x, 2, i * clip_length, clip_length))
                for i in range(x.shape[2] // clip_length)
            ],
            dim=2,
        )
        token_drop = self.config.token_drop
        if token_drop > 0:
            moments = _slice(moments, 2, 0, moments.shape[2] - token_drop)
        return moments

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent video, mirroring the chunking `_encode` applied.

        ``token_drop`` removed the tail of every encoded chunk, so consecutive decoded
        chunks overlap by ``frame_overlap`` pixel frames and are linearly cross-faded.
        Latent frames are repeated at the end when the length is not a whole number of
        chunks, and the extra pixel frames are cut off again at the end.
        """
        config = self.config
        tokens_chunk_size = config.tokens_chunk_size
        token_drop = config.token_drop
        temporal_ratio = config.temporal_compression_ratio
        chunk_num_frames = tokens_chunk_size * temporal_ratio

        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            last = _slice(z, 2, z.shape[2] - 1, 1)
            z = _cat([z, last.repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        decoded_chunks = []
        overlap = None
        clips = self._decode_clips(
            [
                _slice(z, 2, i * tokens_chunk_size, tokens_chunk_size + config.token_overlap)
                for i in range(num_chunks)
            ]
        )
        if clips is None:
            # Not rank 0: the decoded tiles were gathered there.
            return None
        for clip in clips:
            with _phase("chunk_blend"):
                for j in range(int(token_drop > 0) + 1):
                    # One slice, not two. These are full-resolution pixel tensors and every
                    # `_slice` on them is a whole copy, so taking the frames and then dropping
                    # the pre-padding separately paid for the canvas twice to keep one window.
                    frame_start = j * chunk_num_frames + config.frame_pre_padding
                    chunk = _slice(
                        clip, 2, frame_start, chunk_num_frames - config.frame_pre_padding
                    )
                    if chunk is clip:
                        # `_slice` hands back its input when the window is the whole tensor,
                        # and `_blend_temporal` writes into what it is given. Only reachable
                        # for a single unpadded chunk, where the copy costs nothing that the
                        # old two-slice version was not already paying.
                        chunk = chunk.clone()
                    if j == 0:
                        if overlap is not None:
                            chunk = self._blend_temporal(overlap, chunk, config.frame_overlap)
                        decoded_chunks.append(chunk)
                    else:
                        overlap = chunk
        if overlap is not None:
            decoded_chunks.append(overlap)

        with _phase("chunk_cat"):
            dec = _concat_frames(decoded_chunks)

        # The repeated latent frames produced trailing pixel frames nobody asked for. A
        # chunk's last latent frame only covers `clip_length % temporal_ratio` pixel
        # frames; the others cover `temporal_ratio`.
        if pad_tokens > 0:
            intra_tail = config.clip_length % temporal_ratio
            num_tokens_before_pad = z.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail
                if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0
                else temporal_ratio
                for k in range(pad_tokens)
            )
            dec = _slice(dec, 2, 0, dec.shape[2] - pad_frames)
        return dec

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a ``(B, 3, F, H, W)`` video into ``(B, 48, F', H', W')`` moments."""
        return self._encode(x)

    def encode_clip(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single ``(B, 3, F, H, W)`` clip without temporal chunking.

        The keyframe / reference-image path: one frame must not be split into
        ``clip_length`` chunks.
        """
        return self._encode_clip(x)

    def decode(self, z: torch.Tensor, return_dict: bool = False):
        """Decode ``(B, 24, F', H', W')`` latents into ``(B, 3, F, H, W)`` pixels.

        The returned pixels are in the checkpoint's ImageNet-normalized space; the
        caller denormalizes and clamps.
        """
        decoded = self._decode(z)
        log_decode_phases()
        if not return_dict:
            return (decoded,)
        from diffusers.models.autoencoders.vae import DecoderOutput

        return DecoderOutput(sample=decoded)

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """``(latent - latents_mean) / latents_std``, per channel."""
        mean, std = self._latent_stats(latents)
        return (latents - mean) / std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """``latent * latents_std + latents_mean``, per channel — the inverse."""
        mean, std = self._latent_stats(latents)
        return latents * std + mean

    def _latent_stats(self, latents: torch.Tensor):
        shape = (1, self.config.latent_channels, 1, 1, 1)
        mean = torch.tensor(
            self.config.latents_mean, device=latents.device, dtype=latents.dtype
        ).view(shape)
        std = torch.tensor(self.config.latents_std, device=latents.device, dtype=latents.dtype).view(
            shape
        )
        return mean, std

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # The cached rotary tables live outside the parameter tree.
        device = None
        for arg in args:
            if isinstance(arg, torch.device | str):
                device = arg
        device = kwargs.get("device", device)
        if device is not None:
            self._rotary_cache = {
                key: (cos.to(device), sin.to(device))
                for key, (cos, sin) in self._rotary_cache.items()
            }
        return self

    # ---------------------------------------------------------------
    # Weight loading
    # ---------------------------------------------------------------

    def _weight_mappings(self) -> dict[str, str | list[str]]:
        """Model parameter name -> converted-diffusers checkpoint key(s).

        Only the renamed keys appear; the loader maps everything else by identity. The
        encoder, `quant_conv`, `post_quant_conv` and the decoder's `proj_in` /
        `register_tokens` / `norm_out` / `proj_out` / `norm{1,2}` / `scale{1,2}` all
        keep the reference's names.
        """
        mappings: dict[str, str | list[str]] = {}
        for i in range(self.config.decoder_num_layers):
            prefix = f"decoder.transformer_blocks.{i}"
            mappings[f"{prefix}.attn.qkv_proj_weight"] = [
                f"{prefix}.attn.to_q.weight",
                f"{prefix}.attn.to_k.weight",
                f"{prefix}.attn.to_v.weight",
            ]
            mappings[f"{prefix}.attn.qkv_proj_bias"] = [
                f"{prefix}.attn.to_q.bias",
                f"{prefix}.attn.to_k.bias",
                f"{prefix}.attn.to_v.bias",
            ]
            mappings[f"{prefix}.attn.o_proj_weight"] = f"{prefix}.attn.to_out.0.weight"
            mappings[f"{prefix}.attn.o_proj_bias"] = f"{prefix}.attn.to_out.0.bias"
            # Both SwiGLU halves read the one fused `[value; gate]` tensor; the loaders
            # pick their half (see `_gated_half_weight_loader`).
            mappings[f"{prefix}.ff.value_proj_weight"] = f"{prefix}.ff.net.0.proj.weight"
            mappings[f"{prefix}.ff.value_proj_bias"] = f"{prefix}.ff.net.0.proj.bias"
            mappings[f"{prefix}.ff.gate_proj_weight"] = f"{prefix}.ff.net.0.proj.weight"
            mappings[f"{prefix}.ff.gate_proj_bias"] = f"{prefix}.ff.net.0.proj.bias"
            mappings[f"{prefix}.ff.down_proj_weight"] = f"{prefix}.ff.net.2.weight"
            mappings[f"{prefix}.ff.down_proj_bias"] = f"{prefix}.ff.net.2.bias"
        return mappings

    def _param_dtype(self, name: str) -> torch.dtype:
        """The dtype a parameter is held in.

        The matmul weights go to ``compute_dtype``; everything a normalization, a gate
        or a convolution reads stays float32, which is what the released recipe's
        float16 autocast over float32 weights amounted to.
        """
        if name.startswith("encoder.") or name.startswith("quant_conv."):
            return torch.float32
        if any(
            part in name
            for part in (".norm1.", ".norm2.", "norm_out.", ".scale1", ".scale2", "post_quant_conv.")
        ):
            return torch.float32
        return self.compute_dtype

    def load_weights(
        self,
        model_name_or_path: str,
        device: torch.device = torch.device("cpu"),
        cache_dir: str | None = None,
    ) -> None:
        """Load a rank-sharded checkpoint to ``device`` with pipelined data movement.

        ``dtype_override`` carries `_param_dtype`'s mixed precision into the loader, which
        otherwise casts to the dtype each parameter was *built* with — bfloat16, under
        `DiffusersPipelineLoader`'s default-dtype context — and would downcast the float32
        normalizations and convolutions before this cast them back.

        The decoder is replicated, so every rank loads the whole checkpoint.
        """
        tp_rank = _decoder_shard_rank()
        tp_size = _decoder_shard_count()

        checkpoint = SafetensorsCheckpoint(model_name_or_path, cache_dir)
        dtype_override = {
            name: self._param_dtype(name)
            for name, _ in list(self.named_parameters()) + list(self.named_buffers())
        }
        load_result = checkpoint.load_sharded_pipelined(
            tp_rank,
            tp_size,
            self,
            self._weight_mappings(),
            device,
            dtype_override=dtype_override,
        )
        state_dict = load_result.state_dict

        for name, tensor in state_dict.items():
            target_dtype = self._param_dtype(name)
            if tensor.dtype != target_dtype:
                state_dict[name] = tensor.to(target_dtype)

        self.load_state_dict(state_dict, strict=False, assign=True)


__all__ = [
    "MiniMaxH3VideoVAEConfig",
    "NeuronAutoencoderKLMiniMaxH3",
    "build_decoder_rotary_tables",
]
