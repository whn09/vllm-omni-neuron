# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 joint video+audio transformer for Neuron, following the plugin's DiT pattern.

Raw ``nn.Parameter`` plus vLLM-Neuron weight loaders (no CPL/RPL modules), as in the Wan2.2 port.

MiniMax-H3 runs one stack of 50 blocks over a **single packed 1-D sequence** holding text,
keyframe conditioning, target audio and target video rows at once. Attention is full
self-attention over that sequence; there is no cross-attention and no per-modality block weights.
Modality-specific behaviour comes only from the two input patch projections, the per-row AdaLN
modality tag and the two output heads.

Three things differ from a straight transcription of
``diffusers.models.transformers.transformer_minimax_h3``, all because a traced NEFF wants static
shapes and hates gathers:

1. **The packed sequence is built by `cat`, not `index_copy`.** The released row order is
   ``[text | keyframe conditions | target audio | target video]`` — contiguous blocks whose
   sizes are known before the trace.

2. **AdaLN modulation is applied per *run*, not per row.** The reference gathers six
   ``(seq_len, hidden_size)`` modulation tensors out of the ``(timestep, modality)`` table in
   every block, though the table only ever holds a handful of distinct rows. The sequence is
   decomposed host-side into ``(start, end)`` runs over which the row is constant, the
   ``(num_runs, hidden_size)`` slice is gathered once, and each run is modulated by a broadcast.
   See `packing.DiTRowOrder.runs`.

3. **Only the rows that reach an output head run through the output stack.** Both heads are
   row-wise linear maps, so slicing first is identical to the reference's "project every row,
   then `index_select`", and the text rows skip `norm_out` entirely.

Parallelism: heads are sharded over the TP group (56 heads, so TP is 1, 2, 4, 7 or 8) and the
packed sequence over the CP group, whose ranks all-gather keys and values for attention; see
`parallel` and `attention`.

The mixed-precision contract of the checkpoint is preserved: the two patch projections, the
timestep MLP and the two output heads stay float32 while the block stack (including the AdaLN
projections) runs bfloat16.
"""

import logging
import os
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

from vllm_omni_neuron.diffusion.distributed.parallel_state import (
    get_cp_group,
    get_cp_replica_groups,
    register_replica_groups,
)
from vllm_omni_neuron.diffusion.models.minimax_h3 import parallel
from vllm_omni_neuron.diffusion.models.minimax_h3.attention import (
    can_run_ring_attention,
    context_parallel_attention,
    heads_first_attention_dmajor,
    local_attention,
    ring_attention,
    ring_bound_holds,
)

logger = logging.getLogger(__name__)

#: ``0`` keeps every block on the gathered-K/V path (an A/B switch for the ring kernel).
_RING_ATTENTION_ENV = "MINIMAX_H3_RING_ATTENTION"
#: ``0`` keeps the block stack's rows replicated over the TP group (an A/B switch for SP).
_SEQUENCE_PARALLEL_ENV = "MINIMAX_H3_SEQUENCE_PARALLEL"

# `MINIMAX_H3_ADALN_ON_HOST=1` keeps the 50 AdaLN projections (13B parameters, ~40% of the DiT)
# off the NeuronCores. They read only the timestep embedding, so each rank evaluates its
# row-parallel partial on the host once per distinct timestep value and the graph keeps only the
# TP all-reduce. Frees 26 GB of HBM (6.5 GB/core at TP=4) -- what lets TP=4 fit one trn2.3xlarge.
ADALN_ON_HOST = os.environ.get("MINIMAX_H3_ADALN_ON_HOST", "0") == "1"

# MiniMax-H3 tags every row of the packed sequence with the modality it belongs to and
# keeps one set of AdaLN modulation parameters per (timestep, modality) pair:
# 0 = video, 1 = text, 2 = audio.
MINIMAX_H3_MODALITY_NUM = 3

# The parameters the checkpoint ships in float32 while the block stack is bfloat16.
# Matched as substrings of the parameter name, exactly as diffusers'
# `_keep_in_fp32_modules` does, so `proj_in` / `proj_out` also cover the audio heads.
_KEEP_IN_FP32 = ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out")

# `MINIMAX_H3_HEADS_FIRST` (batch-1 requests): keep q/k/v heads-first `[N, S, D]` from the
# projection through attention (`tp_out` d-major output into the O projection), so attention_cte
# needs no relayout of its operands. `bmm` stores QKV as `[3N, H, D]` (one batched matmul);
# `view` keeps the fused token-major matmul and transposes into the following elementwise ops.
HEADS_FIRST = os.environ.get("MINIMAX_H3_HEADS_FIRST", "")
if HEADS_FIRST not in ("", "bmm", "view"):
    raise ValueError(f"MINIMAX_H3_HEADS_FIRST must be '', 'bmm' or 'view', got {HEADS_FIRST!r}")

# `MINIMAX_H3_FUSED_RMSNORM=1`: RMSNorm via `torch.rms_norm` (fused lowering) instead of the
# hand-written reduction.
FUSED_RMSNORM = os.environ.get("MINIMAX_H3_FUSED_RMSNORM", "0") == "1"

# `MINIMAX_H3_FUSED_ADALN=1`: the block's RMSNorm + modulation (and the attention branch's gated
# residual add before norm2) run as the plugin's fused `adaln_modulate` NKI kernel, one launch per
# row run, instead of separate norm / multiply / add passes over the replicated residual stream.
FUSED_ADALN = os.environ.get("MINIMAX_H3_FUSED_ADALN", "0") == "1"

# `MINIMAX_H3_LOCAL_CONSTMAX=1` (CP=1): blocks whose QK-norm gains bound their scores
# (`ring_bound_holds`, 47 of 50) run the const-max kernel as a one-rank ring instead of
# attention_cte: no online max, mask or P-transpose (the row sum rides MM2). Measured 134 vs 158 ms
# per call at 14 heads x 37888 rows. Per-row Cauchy-Schwarz slack on those blocks is <= 43 nats, far
# from the ~85-nat underflow floor. Token-major path only (not with `MINIMAX_H3_HEADS_FIRST`).
LOCAL_CONSTMAX = os.environ.get("MINIMAX_H3_LOCAL_CONSTMAX", "0") == "1"

# `MINIMAX_H3_SPLIT_BLOCK=1` (per-block compile): compile each block as two graphs, the attention
# half and the feed-forward half.
SPLIT_BLOCK = os.environ.get("MINIMAX_H3_SPLIT_BLOCK", "0") == "1"

# `MINIMAX_H3_SP_FUSED=1` (CP=1, with `MINIMAX_H3_SEQUENCE_PARALLEL=1` and `MINIMAX_H3_SPLIT_BLOCK=1`):
# sequence parallelism that keeps the CP=1 kernels. Each half all-gathers the *raw* residual rows,
# runs the fused RMSNorm + modulate (`adaln_modulate`, per run span, identical on every rank) and the
# heads-first attention / FFN on the full sequence, reduce-scatters the projection output, and does
# the gated residual on this rank's 1/TP of the rows. Same collective bytes as the all-reduce path.
SP_FUSED = os.environ.get("MINIMAX_H3_SP_FUSED", "0") == "1"


def _fused_norm_modulate_runs(hidden, norm_weight, eps, shift, scale, runs, residual=None, gate=None):
    """Per run: ``Norm(residual + gate * hidden) * (w * (1 + scale)) + shift`` with the fused kernel.

    Returns ``modulated`` or, with ``residual``, ``(residual_out, modulated)``.
    """
    from vllm_omni_neuron.diffusion.quantization.adaln_kernels import adaln_modulate

    res_parts, mod_parts = [], []
    for index, (start, end) in enumerate(runs):
        mul = (norm_weight.float() * (1.0 + scale[index : index + 1].float())).to(hidden.dtype)
        add = shift[index : index + 1]
        h = hidden[:, start:end]
        if residual is None:
            mod_parts.append(adaln_modulate(h, mul, add, eps=eps, norm_type="rms_norm"))
        else:
            r, m = adaln_modulate(
                h, mul, add, eps=eps, norm_type="rms_norm",
                residual=residual[:, start:end], gate=gate[index : index + 1],
            )
            res_parts.append(r)
            mod_parts.append(m)
    cat = lambda parts: torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    return cat(mod_parts) if residual is None else (cat(res_parts), cat(mod_parts))



@dataclass
class MiniMaxH3Config:
    """Configuration of `NeuronMiniMaxH3Transformer3DModel`.

    Mirrors the diffusers config of the released checkpoints. Note that
    ``num_attention_heads * attention_head_dim`` (7168) is *larger* than ``hidden_size``
    (5376) in MiniMax-H3 — the attention inner dimension is not the residual stream.
    """

    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5


@dataclass
class MiniMaxH3RowRuns:
    """The static row geometry one compiled graph serves.

    The DiT holds the packed sequence as ``[padding | text | conditions | audio | video]`` (see
    `packing.DiTRowOrder`): the prompt is left-padded to its length bucket, so the rows that act
    as keys are the one interval ``[padding, sequence_length)`` and the rows that even out the
    CP shards trail it.

    Attributes:
        num_text_rows: Length of the leading text block, i.e. the prompt bucket.
        num_condition_rows: Length of the visual conditioning block (keyframes, references).
        num_condition_audio_rows: Leading rows of the audio block that are reference soundtracks.
        num_audio_rows: Length of the audio block (both stereo channels, references included).
        num_video_rows: Length of the target video block.
        runs: ``(start, end)`` spans of the full sequence over which the
            ``(timestep, modality)`` pair — and hence the AdaLN table row — is constant.
        num_timesteps: Length of the distinct-timestep table every step is padded to.
    """

    num_text_rows: int
    num_condition_rows: int
    num_audio_rows: int
    num_video_rows: int
    runs: tuple[tuple[int, int], ...]
    num_timesteps: int
    num_condition_audio_rows: int = 0
    media_runs: tuple[tuple[int, int], ...] = field(init=False)

    def __post_init__(self):
        # The media suffix — everything the two output heads read — is `[conditions |
        # reference audio | audio | video]`, four spans each uniform in its timestep.
        condition_end = self.num_condition_rows
        reference_audio_end = condition_end + self.num_condition_audio_rows
        audio_end = condition_end + self.num_audio_rows
        video_end = audio_end + self.num_video_rows
        self.media_runs = (
            (0, condition_end),
            (condition_end, reference_audio_end),
            (reference_audio_end, audio_end),
            (audio_end, video_end),
        )

    @property
    def media_length(self) -> int:
        return self.num_condition_rows + self.num_audio_rows + self.num_video_rows

    @property
    def sequence_length(self) -> int:
        return self.media_length + self.num_text_rows


# ===================================================================
# Weight loader helpers
# ===================================================================


def _col_weight_loader(shard_size, num_shards):
    """Column-parallel weight loader: shard the output dim (transposed storage)."""
    return sharding_weight_loader(
        shard_dim=1,
        shard_size=shard_size,
        num_shards=num_shards,
        is_storage_transposed=True,
    )


def _col_bias_loader(shard_size, num_shards):
    """Column-parallel bias loader: shard along dim 0."""
    return sharding_weight_loader(shard_dim=0, shard_size=shard_size, num_shards=num_shards)


def _row_weight_loader(shard_size, num_shards):
    """Row-parallel weight loader: shard the input dim (transposed storage)."""
    return sharding_weight_loader(
        shard_dim=0,
        shard_size=shard_size,
        num_shards=num_shards,
        is_storage_transposed=True,
    )


def _row_bias_loader(tp_size):
    """Row-parallel bias loader: pre-divide by tp_size.

    Each rank adds ``bias / tp_size`` before the all-reduce, so the sum over ranks
    restores exactly the original bias. The GPT-OSS `scaled_bias_loader` pattern from
    vllm-neuron.
    """

    def transform(slices, rank):
        assert len(slices) == 1
        return slices[0][:] / tp_size

    return SafetensorsWeightLoader(transform=transform)


def _gated_half_weight_loader(half: int, inner_dim: int, shard_size: int, num_shards: int):
    """Column-parallel loader for one half of a fused SwiGLU projection.

    The converted checkpoint stores ``ff.net.0.proj.weight`` as one
    ``(2 * ffn_dim, hidden_size)`` tensor holding ``[value; gate]`` — diffusers' `SwiGLU`
    order, which the conversion script produces by swapping the reference's
    ``[gate; value]``. Splitting it into two parameters here keeps the fused checkpoint
    key while giving the traced graph two plain matmuls instead of a `chunk`, and lets
    each half be sharded independently over the TP ranks.

    Args:
        half: 0 for the value half, 1 for the gate half.
        inner_dim: ``ffn_dim``, the size of one half.
        shard_size: ``ffn_dim // tp_size``.
        num_shards: TP world size.
    """

    def transform(slices, rank):
        assert len(slices) == 1
        start = half * inner_dim + (rank % num_shards) * shard_size
        # The checkpoint is `(2 * ffn_dim, hidden_size)`; the parameter is
        # `(hidden_size, shard_size)`, hence the transpose.
        return slices[0][start : start + shard_size, :].T

    return SafetensorsWeightLoader(transform=transform)


# ===================================================================
# Primitives
# ===================================================================


class RMSNorm(nn.Module):
    """RMSNorm over the last dimension, reduced in float32.

    The residual stream and the per-head query/key norms are both unsharded along the
    normalized axis (the residual stream is replicated across TP ranks and the head
    dimension is never split), so no cross-rank reduction is needed here — unlike the
    Wan port's `DistributedRMSNorm`.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if FUSED_RMSNORM:
            # `torch.rms_norm` with explicit eps and weight lowers to one fused custom call on
            # Neuron; the hand-written form decomposes into ~17 ops.
            return torch.rms_norm(hidden_states.float(), (hidden_states.shape[-1],), self.weight.float(), self.eps).to(
                hidden_states.dtype
            )
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (hidden_states * self.weight.float()).to(input_dtype)


# `MINIMAX_H3_ROPE_MATMUL=1`: partial rotate-half RoPE as one full-head-width expression,
# ``x * cos' + (x @ R) * sin'`` with ``R`` the signed rotate-half permutation of the first
# ``rotary_dim`` channels (zero on the pass-through ones) and ``cos'``/``sin'`` padded with 1/0.
# Bit-exact vs the slice/cat form (each output is one +-x term times sin, the identity elsewhere), but
# the compiler lowers it as a [*,128]x[128,128] TensorE matmul plus two full-width elementwise ops
# instead of strided slices and concatenations: at 14 heads x 37888 rows (TP4) the QKV+norm+RoPE
# window went 87.6 -> 22.9 ms.
ROPE_MATMUL = os.environ.get("MINIMAX_H3_ROPE_MATMUL", "0") == "1"
_ROPE_MATRICES: dict = {}
#: The 128x128 rotate matrix as a device constant, keyed by dtype. Filled by
#: `NeuronMiniMaxH3Transformer3DModel.prepare_rope_matrix` from a registered buffer, so every block
#: graph takes it as an input rather than building it from zeros + index writes.
_ROPE_ROTATE: dict = {}
_ROPE_ROTARY_DIM = 96
_ROPE_CHUNK_ROWS = 32768


def _rope_matrix(head_dim: int, rotary_dim: int, dtype, device) -> torch.Tensor:
    key = (head_dim, rotary_dim, dtype, str(device))
    mat = _ROPE_MATRICES.get(key)
    if mat is None:
        half = rotary_dim // 2
        mat = torch.zeros(head_dim, head_dim)
        index = torch.arange(half)
        mat[index + half, index] = -1.0  # out[i]        = -x[i + half]
        mat[index, index + half] = 1.0  # out[i + half] =  x[i]
        mat = mat.to(dtype=dtype, device=device)
        _ROPE_MATRICES[key] = mat
    return mat


def rope_full_width_tables(cos: torch.Tensor, sin: torch.Tensor, head_dim: int):
    """``(cos', sin')``: the rotary tables padded to ``head_dim`` with 1 / 0 (pass-through channels).

    Built once per request by the pipeline, so no block graph rebuilds them.
    """
    pad = head_dim - cos.shape[-1]
    if pad == 0:
        return cos, sin
    return (
        torch.cat([cos, cos.new_ones(*cos.shape[:-1], pad)], dim=-1),
        torch.cat([sin, sin.new_zeros(*sin.shape[:-1], pad)], dim=-1),
    )


def _rope_by_matmul(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cos_sin_view):
    head_dim = hidden_states.shape[-1]
    rotate = _ROPE_ROTATE.get(hidden_states.dtype)
    if rotate is None:  # not prepared (e.g. CPU unit tests): build it inline
        rotate = _rope_matrix(head_dim, _ROPE_ROTARY_DIM, hidden_states.dtype, hidden_states.device)
    cos, sin = rope_full_width_tables(cos, sin, head_dim)  # no-op when the pipeline pre-padded them
    cos, sin = cos_sin_view(cos.to(hidden_states.dtype)), cos_sin_view(sin.to(hidden_states.dtype))
    # Rotate in row chunks of at most `_ROPE_CHUNK_ROWS`: one flat `(rows, 128) @ (128, 128)` over all
    # heads (530K rows at TP4) overflows a uint16 access-pattern count in the backend (NCC_IBTN021).
    flat = hidden_states.reshape(-1, head_dim)
    rotated = torch.cat(
        [torch.matmul(piece, rotate) for piece in torch.split(flat, _ROPE_CHUNK_ROWS, dim=0)], dim=0
    ).reshape(hidden_states.shape)
    return hidden_states * cos + rotated * sin


def _apply_rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate the leading ``rotary_dim`` channels of every head, pass the rest through.

    ``hidden_states`` is ``(batch_size, seq_len, num_heads, head_dim)`` and ``cos`` /
    ``sin`` are ``(seq_len, rotary_dim)`` with ``rotary_dim = 2 * 3 * rope_freq_dim`` —
    96 of the 128 head channels for the released config.
    """
    if ROPE_MATMUL:
        return _rope_by_matmul(hidden_states, cos, sin, lambda t: t[None, :, None, :])
    rotary_dim = cos.shape[-1]
    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]

    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    # `tensor_split` rather than `chunk`: XLA mislowers `split` on some axes
    # (pytorch/xla#8640) and `tensor_split` takes indices, which lowers correctly.
    x1, x2 = torch.tensor_split(hidden_states_rotary, [rotary_dim // 2], dim=-1)
    hidden_states_rotated = torch.cat((-x2, x1), dim=-1)
    hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
    return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1)


def _apply_rotary_emb_heads_first(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """`_apply_rotary_emb` for heads-first ``(num_heads, seq_len, head_dim)``."""
    if ROPE_MATMUL:
        return _rope_by_matmul(hidden_states, cos, sin, lambda t: t[None])
    rotary_dim = cos.shape[-1]
    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]
    cos = cos.to(hidden_states.dtype)[None]
    sin = sin.to(hidden_states.dtype)[None]
    x1, x2 = torch.tensor_split(hidden_states_rotary, [rotary_dim // 2], dim=-1)
    hidden_states_rotated = torch.cat((-x2, x1), dim=-1)
    hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
    return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1)


#: Rows a row-wise op takes per call. The patch projections, `norm_out` and the output heads see
#: every media row (before the CP split and after the gather); in one piece a ref2va-sized
#: sequence (~52K media rows) overflows the on-chip state buffer at compile time (NCC_IBIR229).
_ROW_CHUNK = 16384


#: Staged launches allowed in flight before the oldest probe is read back (see `_Probed`).
_MAX_IN_FLIGHT = 4


class _Probed(nn.Module):
    """``fn``'s outputs plus one element of its first tensor output, for the host to read.

    The Lite runtime retires an execution only when one of its outputs is read back to the
    host — neither ``synchronize()`` nor freeing the outputs does — so a chain of graphs whose
    outputs stay in HBM fills its execution queue after ~8 launches.
    """

    def __init__(self, fn, probe_input: int | None = None):
        super().__init__()
        self.fn = fn
        # Probe from this positional input instead of the first output (see below).
        self.probe_input = probe_input

    def forward(self, *args):
        outputs = self.fn(*args)
        if self.probe_input is not None:
            # A probe that reads the graph's (large) output forces the compiler to keep that
            # output's producer chain addressable as one flat buffer: on the sequence-parallel
            # FFN graph (all_gather -> SwiGLU -> reduce_scatter -> gated residual, 37,888 rows at
            # TP4) that alone is NCC_EXSP001 (27.75 GB). Reading one element of a small input
            # still retires the execution and compiles. Same on SDK 2.32 and 2.33.
            source = args[self.probe_input]
            if isinstance(source, (tuple, list)):
                source = source[0]
            return outputs, source.reshape(-1)[:1] * 1
        first = outputs[0] if isinstance(outputs, tuple) else outputs
        return outputs, first.reshape(-1)[:1] * 1


def _rowwise(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``layer(x)`` for a row-wise ``layer``, over ``(B, rows, C)`` in chunks of `_ROW_CHUNK` rows."""
    if x.shape[1] <= _ROW_CHUNK:
        return layer(x)
    return torch.cat([layer(piece) for piece in torch.split(x, _ROW_CHUNK, dim=1)], dim=1)


def _tp_gather_rows(hidden_states: torch.Tensor) -> torch.Tensor:
    """Sequence parallelism: every TP rank's rows, in rank order (dim 1)."""
    return parallel.tp_group().all_gather(hidden_states.contiguous(), dim=1)


def _tp_scatter_rows(hidden_states: torch.Tensor) -> torch.Tensor:
    """Sequence parallelism: sum over the TP group, each rank keeping its own rows (dim 1).

    The functional collective, which Lite lowers directly; the omni `GroupCoordinator` has no
    ``reduce_scatter``.
    """
    import torch.distributed._functional_collectives as funcol

    batch, rows, hidden = hidden_states.shape
    flat = hidden_states.reshape(batch * rows, hidden) if batch == 1 else hidden_states.transpose(0, 1).contiguous()
    out = funcol.reduce_scatter_tensor(flat, "sum", 0, parallel.tp_device_group())
    if batch == 1:
        return out.reshape(1, rows // parallel.tp_size(), hidden)
    return out.transpose(0, 1)


def _modulate_runs(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    runs: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Apply ``h * (1 + scale) + shift`` with one ``(1, hidden_size)`` row per run.

    ``shift`` / ``scale`` are ``(num_runs, hidden_size)`` — already reduced to one row per
    run — so each span is modulated by a broadcast instead of a per-row gather out of
    the full ``(timestep, modality)`` table.
    """
    parts = []
    for index, (start, end) in enumerate(runs):
        segment = hidden_states[:, start:end]
        parts.append(segment * (1.0 + scale[index : index + 1]) + shift[index : index + 1])
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


# ===================================================================
# Model classes
# ===================================================================


class MiniMaxH3FeedForward(nn.Module):
    """TP-enabled SwiGLU feed-forward with raw ``nn.Parameter``.

    ``value * silu(gate)`` then the down projection, matching diffusers' `SwiGLU` inside
    `FeedForward` — which is the order the converted checkpoint stores (see
    `_gated_half_weight_loader`). All three projections are bias-free in MiniMax-H3.
    """

    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        tp_size = parallel.tp_size()
        self.tp_size = tp_size
        inner_per_rank = ffn_dim // tp_size

        self.value_proj_weight = nn.Parameter(torch.empty(hidden_size, inner_per_rank))
        set_weight_loader(
            self.value_proj_weight,
            _gated_half_weight_loader(0, ffn_dim, inner_per_rank, tp_size),
        )
        self.gate_proj_weight = nn.Parameter(torch.empty(hidden_size, inner_per_rank))
        set_weight_loader(
            self.gate_proj_weight,
            _gated_half_weight_loader(1, ffn_dim, inner_per_rank, tp_size),
        )
        self.down_proj_weight = nn.Parameter(torch.empty(inner_per_rank, hidden_size))
        set_weight_loader(self.down_proj_weight, _row_weight_loader(inner_per_rank, tp_size))

    def partial(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """`SP_FUSED`: the un-reduced output on already-gathered rows."""
        value = hidden_states @ self.value_proj_weight
        gate = hidden_states @ self.gate_proj_weight
        return (value * F.silu(gate)) @ self.down_proj_weight

    def forward(self, hidden_states: torch.Tensor, sequence_parallel: bool = False) -> torch.Tensor:
        if sequence_parallel:
            hidden_states = _tp_gather_rows(hidden_states)
        value = hidden_states @ self.value_proj_weight
        gate = hidden_states @ self.gate_proj_weight
        output = (value * F.silu(gate)) @ self.down_proj_weight
        if sequence_parallel:
            return _tp_scatter_rows(output)
        if self.tp_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=parallel.tp_device_group())
        return output


class MiniMaxH3Attention(nn.Module):
    """Full self-attention over the packed sequence, with a fused QKV projection.

    Sharded over heads, so ``num_attention_heads`` (56) must be divisible by the TP world size.
    The per-head query/key norms normalize ``attention_head_dim``, which is never split, so they
    stay local. Under context parallelism the rank holds one shard of the rows and attends to the
    whole sequence: over a K/V ring when the block's QK-norm gains bound its scores (``use_ring``,
    set after loading), else over all-gathered keys and values; see `attention`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm_eps: float = 1e-5,
        sequence_is_sharded: bool = True,
    ):
        super().__init__()
        self.head_dim = attention_head_dim
        # The token refiner runs on the replicated text stream *before* the sequence split, so
        # its attention is local even under CP.
        self.sequence_is_sharded = sequence_is_sharded
        tp_size = parallel.tp_size()
        if num_attention_heads % tp_size:
            raise ValueError(
                f"MiniMax-H3 shards attention over its {num_attention_heads} heads, which "
                f"a tensor-parallel size of {tp_size} does not divide."
            )
        self.tp_size = tp_size
        self.num_heads = num_attention_heads // tp_size
        tp_inner_dim = self.num_heads * attention_head_dim

        # Fused QKV: `(hidden_size, 3 * tp_inner_dim)`. MHA, so q == k == v per rank.
        self.qkv_split = [tp_inner_dim, 2 * tp_inner_dim]
        self.qkv_proj_weight = nn.Parameter(torch.empty(hidden_size, 3 * tp_inner_dim))
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

        self.norm_q = RMSNorm(attention_head_dim, eps=qk_norm_eps)
        self.norm_k = RMSNorm(attention_head_dim, eps=qk_norm_eps)

        self.o_proj_weight = nn.Parameter(torch.empty(tp_inner_dim, hidden_size))
        set_weight_loader(self.o_proj_weight, _row_weight_loader(tp_inner_dim, tp_size))

        self.scale = 1.0 / (attention_head_dim**0.5)
        self.use_ring = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb=None,
        context: "_AttentionContext | None" = None,
    ) -> torch.Tensor:
        if getattr(self, "_heads_first", False) and hidden_states.shape[0] == 1:
            return self._forward_heads_first(hidden_states, rotary_emb, context)
        sequence_parallel = context.sequence_parallel and self.sequence_is_sharded
        if sequence_parallel:
            hidden_states = _tp_gather_rows(hidden_states)
        qkv = torch.matmul(hidden_states, self.qkv_proj_weight)
        query, key, value = torch.tensor_split(qkv, self.qkv_split, dim=-1)

        # Token-major `[B, S, N, D]` throughout: the layout both attention kernels take.
        query = query.unflatten(-1, (self.num_heads, self.head_dim))
        key = key.unflatten(-1, (self.num_heads, self.head_dim))
        value = value.unflatten(-1, (self.num_heads, self.head_dim))

        # The reference norms after unflattening, i.e. per head — not over the fused
        # inner dimension. Keep that order.
        query = self.norm_q(query)
        key = self.norm_k(key)

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        sharded = context.cp_group is not None and self.sequence_is_sharded
        if sharded and self.use_ring and context.ring_groups and can_run_ring_attention(query):
            attn_output = ring_attention(
                query, key, value, self.scale, context.ring_groups, context.cp_size,
                context.key_start, context.key_end, context.shard_start,
            )
        elif sharded:
            attn_output = context_parallel_attention(
                query, key, value, self.scale, context.cp_group, context.key_start, context.key_end
            )
        elif LOCAL_CONSTMAX and self.use_ring and self.sequence_is_sharded and can_run_ring_attention(query):
            # CP=1: the const-max kernel with a one-rank ring (no collective). Same per-block gate as
            # under CP (`ring_bound_holds`); the others keep attention_cte.
            attn_output = ring_attention(
                query, key, value, self.scale, (), 1, context.key_start, context.key_end, 0,
            )
        else:
            attn_output = local_attention(
                query, key, value, self.scale, context.key_start, context.key_end
            )

        output = torch.matmul(attn_output.flatten(2), self.o_proj_weight)
        if sequence_parallel:
            return _tp_scatter_rows(output)
        if self.tp_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=parallel.tp_device_group())
        return output

    def relayout_for_heads_first(self) -> None:
        """`HEADS_FIRST=bmm`: store the fused QKV weight as ``[3N, H, D]`` so one batched matmul
        emits q/k/v already heads-first, and fold ``1/sqrt(D)`` into ``norm_q``'s weight."""
        if HEADS_FIRST == "bmm" and hasattr(self, "qkv_proj_weight"):
            weight = self.qkv_proj_weight.data
            hidden = weight.shape[0]
            heads_weight = weight.view(hidden, 3 * self.num_heads, self.head_dim).permute(1, 0, 2).contiguous()
            del self.qkv_proj_weight
            self.qkv_heads_weight = nn.Parameter(heads_weight, requires_grad=False)
        with torch.no_grad():
            self.norm_q.weight.mul_(self.scale)
        self._scale_folded = True
        self._heads_first = True

    def _forward_heads_first(self, hidden_states, rotary_emb, context) -> torch.Tensor:
        """Batch-1 path that keeps q/k/v heads-first ``[N, S, D]`` end to end: the attention kernel
        reads its operands without a relayout, and writes ``[N, D, S]`` (``tp_out``), which the
        output projection contracts over ``N * D`` directly."""
        num_heads, head_dim = self.num_heads, self.head_dim
        x = hidden_states[0]
        seq = x.shape[0]
        if HEADS_FIRST == "bmm":
            qkv = torch.matmul(x.unsqueeze(0), self.qkv_heads_weight)  # [3N, S, D]
        else:  # "view": token-major matmul, transposed into the elementwise ops that follow
            qkv = torch.matmul(x, self.qkv_proj_weight).view(seq, 3 * num_heads, head_dim).transpose(0, 1)
        query, key, value = qkv[:num_heads], qkv[num_heads : 2 * num_heads], qkv[2 * num_heads :]
        query = self.norm_q(query)
        key = self.norm_k(key)
        if not getattr(self, "_scale_folded", False):
            query = query * self.scale
        if rotary_emb is not None:
            query = _apply_rotary_emb_heads_first(query, *rotary_emb)
            key = _apply_rotary_emb_heads_first(key, *rotary_emb)
        cp_group = context.cp_group if self.sequence_is_sharded else None
        attn = heads_first_attention_dmajor(
            query.contiguous(), key.contiguous(), value.contiguous(), context.key_start, context.key_end, cp_group
        )  # [N, D, S]
        output = torch.matmul(attn.reshape(num_heads * head_dim, seq).transpose(0, 1), self.o_proj_weight)
        output = output.unsqueeze(0)
        if getattr(self, "_return_partial", False):
            return output  # `SP_FUSED`: the caller reduce-scatters
        if self.tp_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=parallel.tp_device_group())
        return output

    def heads_first_partial(self, hidden_states, rotary_emb, context) -> torch.Tensor:
        """`SP_FUSED`: heads-first attention on already-gathered rows, un-reduced output."""
        self._return_partial = True
        try:
            return self._forward_heads_first(hidden_states, rotary_emb, context)
        finally:
            self._return_partial = False


@dataclass
class _AttentionContext:
    """What an attention layer needs beyond its inputs. Built once per forward.

    Attributes:
        key_start: ``(1,)`` int32 first row that acts as a key (the end of the prompt padding).
        key_end: One past the last row that acts as a key.
        cp_group: The CP group when the rows are sharded over it, else None.
        ring_groups: Every CP ring's global ranks, for the ring kernel; None keeps every block on
            the all-gather path.
        cp_size: The CP world size.
        shard_start: First padded-sequence row of this rank's shard.
        sequence_parallel: The block stack's rows are further split over the TP group: each
            attention / feed-forward all-gathers them on entry and reduce-scatters on exit.
        tp_cols: This TP rank's slice of ``time_embed_dim`` for the row-parallel AdaLN
            projections, as a runtime input (sequence parallelism; see `sequence_parallel_inputs`).
    """

    key_start: torch.Tensor
    key_end: int
    cp_group: object = None
    ring_groups: tuple | None = None
    cp_size: int = 1
    shard_start: int = 0
    sequence_parallel: bool = False
    tp_cols: torch.Tensor | None = None


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    """Projects the shared timestep embedding into one block's six modulation tensors.

    ``(num_timesteps, time_embed_dim)`` -> six ``(num_timesteps * 3, hidden_size)``
    tensors in the diffusers ``shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
    gate_mlp`` order, the table rows laid out ``[t0_mod0, t0_mod1, t0_mod2, t1_mod0,
    ...]`` — what ``timestep_index * 3 + tag`` addresses.

    At 2688 -> 96768 this projection is 260M parameters, ~40% of the model once summed
    over 50 blocks, so it has to be sharded. It is row-parallel: ``temb`` is small and
    replicated, the input dim is split, and the all-reduce output is only
    ``(num_timesteps * 3, 6 * hidden_size)`` — a few hundred kilobytes, versus the
    full-width all-gather a column-parallel split would need.
    """

    def __init__(self, time_embed_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = parallel.tp_size()
        self.tp_size = tp_size
        in_per_rank = time_embed_dim // tp_size
        out_features = 6 * hidden_size * MINIMAX_H3_MODALITY_NUM

        self.linear_weight = nn.Parameter(torch.empty(in_per_rank, out_features))
        set_weight_loader(self.linear_weight, _row_weight_loader(in_per_rank, tp_size))
        self.linear_bias = nn.Parameter(torch.empty(out_features))
        if tp_size > 1:
            set_weight_loader(self.linear_bias, _row_bias_loader(tp_size))

    def forward(self, temb: torch.Tensor, cols: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        # The activation runs at `temb`'s own precision — float32, since `time_embedder`
        # is a float32 module in this mixed-precision checkpoint — and only its result is
        # cast down to the bfloat16 projection. Every block reads the same `temb`, so a
        # rounding applied before the activation biases every block's modulation
        # parameters identically at every sampling step, which accumulates coherently
        # over the denoising trajectory.
        # Row-parallel: each rank holds its slice of `time_embed_dim`, so the replicated
        # activation is sliced to match before the local matmul. `cols` (a runtime input under
        # sequence parallelism) keeps the slice out of the trace, so TP peers trace one graph.
        temb = self.partial(temb, cols)
        if self.tp_size > 1:
            dist.all_reduce(temb, op=dist.ReduceOp.SUM, group=parallel.tp_device_group())
        return self.split(temb)

    def partial(self, temb: torch.Tensor, cols: torch.Tensor | None = None) -> torch.Tensor:
        """This rank's row-parallel share of the projection, before the TP all-reduce."""
        activated = F.silu(temb).to(self.linear_weight.dtype)
        if cols is not None:
            activated = activated.index_select(1, cols)
        else:
            rank = parallel.tp_rank()
            in_per_rank = self.linear_weight.shape[0]
            activated = activated[:, rank * in_per_rank : (rank + 1) * in_per_rank]
        return torch.matmul(activated, self.linear_weight) + self.linear_bias

    def host_partial(self, activated: torch.Tensor) -> torch.Tensor:
        """`ADALN_ON_HOST`: this rank's partial on the host; ``activated`` is ``silu(temb)`` in the
        projection dtype, full width."""
        rank = parallel.tp_rank()
        weight = self._host_weight
        in_per_rank = weight.shape[0]
        return torch.matmul(activated[:, rank * in_per_rank : (rank + 1) * in_per_rank], weight) + self._host_bias

    def reduce(self, partial: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """`ADALN_ON_HOST`: the six modulation tensors from a host-computed partial."""
        if self.tp_size > 1:
            partial = partial.clone()
            dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=parallel.tp_device_group())
        return self.split(partial)

    def split(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The all-reduced projection as the six modulation tensors."""
        # `reshape`: a slice of the hoisted, concatenated projection is not contiguous.
        temb = temb.reshape(-1, 6 * self.hidden_size)
        return torch.tensor_split(temb, [self.hidden_size * i for i in range(1, 6)], dim=-1)


class MiniMaxH3AdaLayerNormOut(nn.Module):
    """Final norm of the media rows, shift/scale modulated per timestep.

    Same layout and checkpoint keys as diffusers' `AdaLayerNormContinuous` (a `norm`
    plus a `linear` to ``2 * hidden_size``), with MiniMax-H3's two specifics: the
    modulation table holds one row per *timestep* — not per batch item — and the two
    halves are ``shift`` then ``scale``.
    """

    def __init__(self, hidden_size: int, time_embed_dim: int, eps: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm = RMSNorm(hidden_size, eps=eps)
        # 2688 -> 10752 is small enough to leave replicated; sharding it would cost an
        # extra collective per step for ~30 MB of weights.
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        runs: tuple[tuple[int, int], ...],
        run_rows: torch.Tensor,
    ) -> torch.Tensor:
        # As in `MiniMaxH3AdaLayerNormModulation`: activate at `temb`'s precision, cast
        # to the projection's dtype after.
        shift, scale = self.linear(F.silu(temb).to(self.linear.weight.dtype)).chunk(2, dim=-1)
        shift = shift.index_select(0, run_rows).to(hidden_states.dtype)
        scale = scale.index_select(0, run_rows).to(hidden_states.dtype)
        # The modulation stays at the block stack's precision; the caller casts to the
        # output heads' dtype. The norm sees every media row, so it runs in row chunks.
        return _modulate_runs(_rowwise(self.norm, hidden_states), shift, scale, runs)


class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Plain pre-norm block refining the projected text stream. No AdaLN, no rotary."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
            # The refiner runs before the sequence split, on the replicated text stream, so
            # there is nothing to gather here — every rank already holds every text row.
            sequence_is_sharded=False,
        )
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.ff = MiniMaxH3FeedForward(hidden_size, ffn_dim)

    def forward(self, hidden_states: torch.Tensor, context: _AttentionContext) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), context=context)
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
    ):
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_size, eps=final_norm_eps)

    def forward(self, hidden_states: torch.Tensor, context: _AttentionContext) -> torch.Tensor:
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, context)
        return self.final_norm(hidden_states)


class MiniMaxH3TransformerBlock(nn.Module):
    """Pre-norm self-attention and feed-forward, each AdaLN-modulated per row run."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
        )
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.ff = MiniMaxH3FeedForward(hidden_size, ffn_dim)
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(
            time_embed_dim=time_embed_dim, hidden_size=hidden_size
        )

    def _half_modulators(self, runs, sequence_parallel):
        if sequence_parallel:
            # `run_rows` holds one table row per local row: broadcast over the batch.
            return (lambda h, shift, scale: h * (1.0 + scale) + shift), (lambda h, value: h * value)
        return (
            (lambda h, shift, scale: _modulate_runs(h, shift, scale, runs)),
            (lambda h, value: _gate_runs(h, value, runs)),
        )

    def attn_half(self, hidden_states, temb, runs, run_rows, rotary_emb, context, adaln_partial=None):
        """`MINIMAX_H3_SPLIT_BLOCK`: norm1 + modulate + attention + gated residual; returns the
        residual stream and the reduced modulation table for `ff_half`. Under sequence parallelism
        the gathered sequence then lives in only one of the two graphs at a time."""
        if adaln_partial is not None:
            modulation = self.adaln_proj.reduce(adaln_partial)
        else:
            modulation = self.adaln_proj(temb, context.tp_cols)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            tensor.index_select(0, run_rows).to(hidden_states.dtype) for tensor in modulation
        )
        sp = context.sequence_parallel
        modulate, gate = self._half_modulators(runs, sp)
        if FUSED_ADALN and not sp:
            norm_hidden_states = _fused_norm_modulate_runs(
                hidden_states, self.norm1.weight, self.norm1.eps, shift_msa, scale_msa, runs
            )
        else:
            norm_hidden_states = modulate(self.norm1(hidden_states), shift_msa, scale_msa)
        attn_output = self.attn(norm_hidden_states, rotary_emb, context)
        hidden_states = hidden_states + gate(attn_output, gate_msa)
        return hidden_states, torch.stack((shift_mlp, scale_mlp, gate_mlp))

    def attn_half_spf(self, hidden_states, temb, runs, run_rows, row_rows, rotary_emb, context, adaln_partial=None):
        """`SP_FUSED` attention half. ``hidden_states`` is this rank's ``[1, S/TP, H]`` residual
        shard, ``run_rows`` one table row per run (full sequence), ``row_rows`` one per local row."""
        if adaln_partial is not None:
            modulation = self.adaln_proj.reduce(adaln_partial)
        else:
            modulation = self.adaln_proj(temb, context.tp_cols)
        shift_msa, scale_msa = (t.index_select(0, run_rows).to(hidden_states.dtype) for t in modulation[:2])
        shift_mlp, scale_mlp = (t.index_select(0, run_rows).to(hidden_states.dtype) for t in modulation[3:5])
        gate_msa = modulation[2].index_select(0, row_rows).to(hidden_states.dtype)
        gate_mlp = modulation[5].index_select(0, row_rows).to(hidden_states.dtype)
        full = _tp_gather_rows(hidden_states)
        if FUSED_ADALN:
            norm_hidden_states = _fused_norm_modulate_runs(
                full, self.norm1.weight, self.norm1.eps, shift_msa, scale_msa, runs
            )
        else:
            norm_hidden_states = _modulate_runs(self.norm1(full), shift_msa, scale_msa, runs)
        if getattr(self.attn, "_heads_first", False):
            attn_partial = self.attn.heads_first_partial(norm_hidden_states, rotary_emb, context)
        else:
            raise NotImplementedError("MINIMAX_H3_SP_FUSED needs MINIMAX_H3_HEADS_FIRST=bmm")
        hidden_states = hidden_states + _tp_scatter_rows(attn_partial) * gate_msa
        return hidden_states, torch.stack((shift_mlp, scale_mlp)), gate_mlp

    def block_spf(self, hidden_states, temb, runs, run_rows, row_rows, rotary_emb, context, adaln_partial=None):
        """`SP_FUSED` without `SPLIT_BLOCK`: both halves in one graph."""
        hidden_states, mlp_mod, gate_mlp = self.attn_half_spf(
            hidden_states, temb, runs, run_rows, row_rows, rotary_emb, context, adaln_partial
        )
        return self.ff_half_spf(hidden_states, mlp_mod, gate_mlp, runs)

    def ff_half_spf(self, hidden_states, mlp_mod, gate_mlp, runs):
        full = _tp_gather_rows(hidden_states)
        if FUSED_ADALN:
            norm_hidden_states = _fused_norm_modulate_runs(
                full, self.norm2.weight, self.norm2.eps, mlp_mod[0], mlp_mod[1], runs
            )
        else:
            norm_hidden_states = _modulate_runs(self.norm2(full), mlp_mod[0], mlp_mod[1], runs)
        return hidden_states + _tp_scatter_rows(self.ff.partial(norm_hidden_states)) * gate_mlp

    def ff_half(self, hidden_states, mlp_mod, runs, sequence_parallel=False):
        shift_mlp, scale_mlp, gate_mlp = mlp_mod[0], mlp_mod[1], mlp_mod[2]
        modulate, gate = self._half_modulators(runs, sequence_parallel)
        if FUSED_ADALN and not sequence_parallel:
            norm_hidden_states = _fused_norm_modulate_runs(
                hidden_states, self.norm2.weight, self.norm2.eps, shift_mlp, scale_mlp, runs
            )
        else:
            norm_hidden_states = modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)
        return hidden_states + gate(self.ff(norm_hidden_states, sequence_parallel), gate_mlp)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        runs: tuple[tuple[int, int], ...],
        run_rows: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        context: _AttentionContext,
        modulation: tuple[torch.Tensor, ...] | None = None,
        adaln_partial: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if modulation is None and adaln_partial is not None:
            modulation = self.adaln_proj.reduce(adaln_partial)
        if modulation is None:
            modulation = self.adaln_proj(temb, context.tp_cols)
        # One row per run instead of one per sequence row: the six table lookups shrink
        # from `(seq_len, hidden_size)` to `(num_runs, hidden_size)`.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            tensor.index_select(0, run_rows).to(hidden_states.dtype) for tensor in modulation
        )

        if FUSED_ADALN and not context.sequence_parallel and (
            context.cp_group is None or os.environ.get("MINIMAX_H3_FUSED_ADALN_CP", "0") == "1"
        ):
            # CP=1 only by default: under CP8 the per-run launches on 4.7K-row shards measured +48 %.
            norm_hidden_states = _fused_norm_modulate_runs(
                hidden_states, self.norm1.weight, self.norm1.eps, shift_msa, scale_msa, runs
            )
            attn_output = self.attn(norm_hidden_states, rotary_emb, context)
            hidden_states, norm_hidden_states = _fused_norm_modulate_runs(
                attn_output, self.norm2.weight, self.norm2.eps, shift_mlp, scale_mlp, runs,
                residual=hidden_states, gate=gate_msa,
            )
            residual = hidden_states
            ff_output = self.ff(norm_hidden_states)
            return residual + _gate_runs(ff_output, gate_mlp, runs)

        if context.sequence_parallel:
            # `run_rows` holds one table row per local row: broadcast over the batch.
            def modulate(hidden, shift, scale):
                return hidden * (1.0 + scale) + shift

            def gate(hidden, value):
                return hidden * value
        else:
            def modulate(hidden, shift, scale):
                return _modulate_runs(hidden, shift, scale, runs)

            def gate(hidden, value):
                return _gate_runs(hidden, value, runs)

        residual = hidden_states
        norm_hidden_states = modulate(self.norm1(hidden_states), shift_msa, scale_msa)
        attn_output = self.attn(norm_hidden_states, rotary_emb, context)
        hidden_states = residual + gate(attn_output, gate_msa)

        residual = hidden_states
        norm_hidden_states = modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)
        ff_output = self.ff(norm_hidden_states, context.sequence_parallel)
        hidden_states = residual + gate(ff_output, gate_mlp)

        return hidden_states


def _pad_rows(tensor: torch.Tensor, dim: int, length: int) -> torch.Tensor:
    """``tensor`` extended along ``dim`` to ``length`` rows with zeros."""
    shape = list(tensor.shape)
    shape[dim] = length - tensor.shape[dim]
    return torch.cat(
        [tensor, torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)], dim=dim
    )


def _gate_runs(
    hidden_states: torch.Tensor, gate: torch.Tensor, runs: tuple[tuple[int, int], ...]
) -> torch.Tensor:
    """Scale each run of ``hidden_states`` by its ``(1, hidden_size)`` gate row."""
    if len(runs) == 1:
        return hidden_states * gate[0:1]
    parts = [
        hidden_states[:, start:end] * gate[index : index + 1]
        for index, (start, end) in enumerate(runs)
    ]
    return torch.cat(parts, dim=1)


class NeuronMiniMaxH3Transformer3DModel(nn.Module):
    """MiniMax-H3 joint video+audio transformer for Neuron.

    Raw ``nn.Parameter`` with TP weight loaders throughout. See the module docstring for
    the three deliberate departures from the diffusers reference.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.config = MiniMaxH3Config(
            **{k: v for k, v in kwargs.items() if k in MiniMaxH3Config.__dataclass_fields__}
        )
        config = self.config
        if isinstance(config.patch_size, list):
            config.patch_size = tuple(config.patch_size)

        video_patch_dim = config.in_channels * (
            config.patch_size[0] * config.patch_size[1] * config.patch_size[2]
        )
        self.video_patch_dim = video_patch_dim

        self.cp_size = parallel.cp_size()
        self.cp_rank = parallel.cp_rank()
        # Make the TP/CP groups resolvable to their full replica-group partitions so the
        # torch-native backend can legalize the collectives for SPMD compilation.
        register_replica_groups(tp_size=parallel.tp_size(), cp_size=self.cp_size)
        # Resolved here rather than in `forward`: a Python-side lookup Dynamo cannot trace.
        self.cp_group = get_cp_group() if self.cp_size > 1 else None
        self.cp_ring_groups = get_cp_replica_groups(parallel.tp_size(), self.cp_size) if self.cp_size > 1 else ()
        # Sequence parallelism over the TP group: between the attention and feed-forward
        # projections each TP rank keeps 1/tp of its CP shard (a multiple of `SHARD_ALIGN`, so
        # it divides), and the norms, modulation and residual run on those rows only.
        self.tp_size = parallel.tp_size()
        self.tp_rank = parallel.tp_rank()
        # At CP=1 opt-in (`MINIMAX_H3_SEQUENCE_PARALLEL=1`): the whole sequence is the one "shard".
        sp_env = os.environ.get(_SEQUENCE_PARALLEL_ENV, "1" if self.cp_size > 1 else "0")
        self.sequence_parallel = self.tp_size > 1 and sp_env != "0"
        self._sp_fused = SP_FUSED and self.sequence_parallel and self.cp_size == 1
        # Set by `compile_stages`; None runs (or traces) the whole forward as one graph.
        self._stage_compiler = None

        # 1. Per-modality input projections. Small and float32 in the checkpoint, so
        # left replicated.
        self.proj_in = nn.Linear(video_patch_dim, config.hidden_size, bias=True)
        self.audio_proj_in = nn.Linear(config.audio_in_channels, config.hidden_size, bias=True)
        self.context_embedder = nn.Linear(config.text_dim, config.hidden_size, bias=True)

        # 2. Timestep MLP, shared by every AdaLN projection. `(num_timesteps, freq_dim)`
        # -> `(num_timesteps, time_embed_dim)`; tiny, so replicated.
        self.time_embedder = nn.Module()
        self.time_embedder.linear_1 = nn.Linear(config.freq_dim, config.time_embed_hidden_dim)
        self.time_embedder.linear_2 = nn.Linear(
            config.time_embed_hidden_dim, config.time_embed_dim
        )

        # 3. Text stream refiner.
        self.token_refiner = MiniMaxH3TokenRefiner(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_refiner_layers,
            norm_eps=config.norm_eps,
            qk_norm_eps=config.qk_norm_eps,
            final_norm_eps=config.final_norm_eps,
        )

        # 4. The block stack.
        if ROPE_MATMUL:
            # The rotate matrix as a model constant: moved to the device with the weights and fed to
            # every block graph as an input (`prepare_rope_matrix`).
            self.register_buffer(
                "rope_rotate",
                _rope_matrix(config.attention_head_dim, 2 * 3 * config.rope_freq_dim, torch.bfloat16, "cpu").clone(),
                persistent=False,
            )
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.attention_head_dim,
                    ffn_dim=config.ffn_dim,
                    time_embed_dim=config.time_embed_dim,
                    norm_eps=config.norm_eps,
                    qk_norm_eps=config.qk_norm_eps,
                )
                for _ in range(config.num_layers)
            ]
        )

        # 5. Shared output norm and the two per-modality heads.
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            hidden_size=config.hidden_size,
            time_embed_dim=config.time_embed_dim,
            eps=config.final_norm_eps,
        )
        self.proj_out = nn.Linear(config.hidden_size, video_patch_dim, bias=True)
        self.audio_proj_out = nn.Linear(config.hidden_size, config.audio_in_channels, bias=True)

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the block stack — the model's nominal precision.

        Not `proj_in.weight.dtype`: the patch projections and the two heads stay float32
        in this mixed-precision checkpoint.
        """
        return self.transformer_blocks[0].attn.qkv_proj_weight.dtype

    def _time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embedding plus the MLP, in the timestep MLP's precision.

        Timesteps are consumed unscaled in ``[0, 1]``. This reproduces diffusers'
        `Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)` followed by
        `TimestepEmbedding`, written out rather than imported so the traced graph holds
        no diffusers control flow.
        """
        half_dim = self.config.freq_dim // 2
        exponent = -torch.log(torch.tensor(10000.0, dtype=torch.float32)) * torch.arange(
            half_dim, dtype=torch.float32, device=timestep.device
        )
        # `downscale_freq_shift=0`, so the divisor is `half_dim` rather than
        # `half_dim - shift`.
        emb = torch.exp(exponent / half_dim)
        emb = timestep.float()[:, None] * emb[None, :]
        # `flip_sin_to_cos=True`: cosine first.
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

        emb = emb.to(self.time_embedder.linear_1.weight.dtype)
        emb = self.time_embedder.linear_2(F.silu(self.time_embedder.linear_1(emb)))
        return emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        adaln_run_rows: torch.Tensor,
        norm_out_run_rows: torch.Tensor,
        num_text_tokens: torch.Tensor,
        row_runs: MiniMaxH3RowRuns,
        sp_rows: torch.Tensor | None = None,
        sp_run_index: torch.Tensor | None = None,
        tp_cols: torch.Tensor | None = None,
        return_dict: bool = False,
        adaln_partials: torch.Tensor | None = None,
    ):
        """Run one denoising step over the packed sequence.

        Args:
            hidden_states: ``(B, num_condition_rows + num_video_rows, in_channels *
                prod(patch_size))`` patchified video rows, conditioning rows first —
                the order the packed layout places them in.
            audio_hidden_states: ``(B, num_audio_rows, audio_in_channels)`` audio rows,
                channel-major.
            encoder_hidden_states: ``(B, num_text_rows, text_dim)`` text conditioning: padding up
                to the bucket, then the prompt.
            timestep: ``(num_timesteps,)`` distinct timestep values present in the
                sequence, in ``[0, 1]`` and unscaled, padded to
                ``row_runs.num_timesteps``.
            rotary_cos: ``(seq_len, 2 * 3 * rope_freq_dim)`` cosines of the packed
                ``(t, h, w)`` grid in the DiT's row order, precomputed on the host in float32. MiniMax-H3
                builds the grid in float64 because video and audio share one rotary
                clock, and Neuron has no float64 — so the grid is built and reduced to
                cos/sin off-device, where it costs one pass per request.
            rotary_sin: The matching sines.
            adaln_run_rows: ``(num_runs,)`` row of the ``(timestep, modality)`` AdaLN
                table each run of the sequence reads, i.e. ``timestep_index * 3 + tag``
                sampled once per run.
            norm_out_run_rows: ``(3,)`` row of the per-timestep `norm_out` table each
                media run reads.
            num_text_tokens: ``(1,)`` int32 number of real prompt rows; the rest of the
                ``num_text_rows`` bucket is the padding in front of them.
            row_runs: The static row geometry, baked into the trace.
            sp_rows, sp_run_index: Under sequence parallelism, this TP rank's rows of its CP
                shard and the shard-local run each reads (`sequence_parallel_inputs`).
            return_dict: Kept for signature parity with the reference; the Neuron
                pipeline always consumes the tuple.

        Returns:
            ``(video_output, audio_output)`` — the video velocity in the row order of
            ``hidden_states`` and the audio velocity in the row order of
            ``audio_hidden_states``.
        """
        if self._stage_compiler is not None:
            return self._forward_staged(
                hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
                rotary_cos, rotary_sin, adaln_run_rows, norm_out_run_rows, num_text_tokens, row_runs,
                sp_rows, sp_run_index, tp_cols, adaln_partials,
            )
        return self._forward_whole(
            hidden_states=hidden_states, audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states, timestep=timestep, rotary_cos=rotary_cos,
            rotary_sin=rotary_sin, adaln_run_rows=adaln_run_rows, norm_out_run_rows=norm_out_run_rows,
            num_text_tokens=num_text_tokens, row_runs=row_runs, sp_rows=sp_rows, sp_run_index=sp_run_index,
            tp_cols=tp_cols, adaln_partials=adaln_partials,
        )

    def _forward_whole(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        rotary_cos,
        rotary_sin,
        adaln_run_rows,
        norm_out_run_rows,
        num_text_tokens,
        row_runs: MiniMaxH3RowRuns,
        sp_rows=None,
        sp_run_index=None,
        tp_cols=None,
        adaln_partials=None,
    ):
        hidden_states, temb, adaln_run_rows, rotary_cos, rotary_sin, key_start = self._prologue(
            hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
            rotary_cos, rotary_sin, adaln_run_rows, num_text_tokens, row_runs, sp_rows, sp_run_index,
        )
        runs, context = self._block_geometry(row_runs, key_start, tp_cols=tp_cols)
        modulations = [None] * len(self.transformer_blocks)
        if adaln_partials is not None:
            # Host-computed partials (`ADALN_ON_HOST`): one all-reduce for all 50 blocks, which
            # also keeps the TP collectives in program order under sequence parallelism.
            num_layers = len(self.transformer_blocks)
            reduced = adaln_partials.permute(1, 0, 2).reshape(adaln_partials.shape[1], -1).contiguous()
            if self.tp_size > 1:
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=parallel.tp_device_group())
            width = adaln_partials.shape[-1]
            modulations = [
                block.adaln_proj.split(reduced[:, index * width : (index + 1) * width])
                for index, block in zip(range(num_layers), self.transformer_blocks)
            ]
        elif self.sequence_parallel:
            # Every block's AdaLN projection depends on `temb` alone, so in one graph its TP
            # all-reduce is free to interleave with the blocks' SP all-gathers and
            # reduce-scatters — which, past a few blocks, corrupts the result (NaN with the ring
            # kernel). Projecting all of them up front, with one all-reduce, keeps the TP
            # collectives in program order (and saves 49 of them per step).
            partials = [block.adaln_proj.partial(temb, context.tp_cols) for block in self.transformer_blocks]
            reduced = torch.cat(partials, dim=-1)
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=parallel.tp_device_group())
            width = partials[0].shape[-1]
            modulations = [
                block.adaln_proj.split(reduced[:, index * width : (index + 1) * width])
                for index, block in enumerate(self.transformer_blocks)
            ]
        for block, modulation in zip(self.transformer_blocks, modulations):
            hidden_states = block(
                hidden_states, temb, runs, adaln_run_rows, (rotary_cos, rotary_sin), context, modulation
            )
        return self._epilogue(hidden_states, temb, norm_out_run_rows, row_runs)

    def compile_stages(self, compile_fn) -> None:
        """Compile the DiT as a prologue, one graph per transformer block and an epilogue.

        The blocks are structurally identical and their weights are graph inputs, so they share
        one NEFF: a cold start compiles one block instead of a 50-block graph, and sequences
        whose whole-model graph does not finish compiling (~60K rows) become usable. Every
        launch returns a one-element probe that is read back, which is what retires a launch
        on the Lite runtime (see `_Probed`).
        """
        import torch._dynamo.config as dynamo_config

        # One trace per block module; the default cache limit per code object is 8.
        dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 4 * len(self.transformer_blocks))
        self._stage_compiler = {
            "prologue": compile_fn(_Probed(self._prologue)),
            "blocks": [compile_fn(_Probed(block)) for block in self.transformer_blocks],
            "halves": [
                (compile_fn(_Probed(block.attn_half)), compile_fn(_Probed(block.ff_half, probe_input=1)))
                for block in self.transformer_blocks
            ]
            if SPLIT_BLOCK and not self._sp_fused
            else None,
            "spf_halves": [
                (
                    compile_fn(_Probed(block.attn_half_spf, probe_input=3)),
                    compile_fn(_Probed(block.ff_half_spf, probe_input=1)),
                )
                for block in self.transformer_blocks
            ]
            if self._sp_fused and SPLIT_BLOCK
            else None,
            "spf_blocks": [compile_fn(_Probed(block.block_spf, probe_input=3)) for block in self.transformer_blocks]
            if self._sp_fused and not SPLIT_BLOCK
            else None,
            "epilogue": compile_fn(self._epilogue),
        }

    def _forward_staged(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        rotary_cos,
        rotary_sin,
        adaln_run_rows,
        norm_out_run_rows,
        num_text_tokens,
        row_runs: MiniMaxH3RowRuns,
        sp_rows=None,
        sp_run_index=None,
        tp_cols=None,
        adaln_partials=None,
    ):
        stages = self._stage_compiler
        run_level_rows = adaln_run_rows
        outputs, probe = stages["prologue"](
            hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
            rotary_cos, rotary_sin, adaln_run_rows, num_text_tokens, row_runs, sp_rows, sp_run_index,
        )
        pending = [probe]
        hidden_states, temb, adaln_run_rows, rotary_cos, rotary_sin, key_start = outputs
        # Per-block graphs keep every block on the all-gather path: launched as its own graph, a
        # ring block costs ~17 ms more than an all-gather one (2.325 vs 1.526 s/step at 37888 rows),
        # where inside the whole graph it saves 0.2 s/step.
        runs, context = self._block_geometry(row_runs, key_start, ring=False, tp_cols=tp_cols)
        for index, block in enumerate(stages["blocks"]):
            if stages.get("spf_blocks") is not None:
                partial = None if adaln_partials is None else adaln_partials[index]
                hidden_states, probe = stages["spf_blocks"][index](
                    hidden_states, temb, runs, run_level_rows, adaln_run_rows, (rotary_cos, rotary_sin),
                    context, partial,
                )
                pending.append(probe)
                while len(pending) > _MAX_IN_FLIGHT:
                    pending.pop(0).to("cpu")
                continue
            if stages.get("spf_halves") is not None:
                attn_half, ff_half = stages["spf_halves"][index]
                partial = None if adaln_partials is None else adaln_partials[index]
                (hidden_states, mlp_mod, gate_mlp), probe = attn_half(
                    hidden_states, temb, runs, run_level_rows, adaln_run_rows, (rotary_cos, rotary_sin),
                    context, partial,
                )
                pending.append(probe)
                hidden_states, probe = ff_half(hidden_states, mlp_mod, gate_mlp, runs)
                pending.append(probe)
                while len(pending) > _MAX_IN_FLIGHT:
                    pending.pop(0).to("cpu")
                continue
            if stages["halves"] is not None:
                attn_half, ff_half = stages["halves"][index]
                partial = None if adaln_partials is None else adaln_partials[index]
                (hidden_states, mlp_mod), probe = attn_half(
                    hidden_states, temb, runs, adaln_run_rows, (rotary_cos, rotary_sin), context, partial
                )
                pending.append(probe)
                hidden_states, probe = ff_half(hidden_states, mlp_mod, runs, context.sequence_parallel)
                pending.append(probe)
                while len(pending) > _MAX_IN_FLIGHT:
                    pending.pop(0).to("cpu")
                continue
            if adaln_partials is None:
                hidden_states, probe = block(hidden_states, temb, runs, adaln_run_rows, (rotary_cos, rotary_sin), context)
            else:
                hidden_states, probe = block(
                    hidden_states, temb, runs, adaln_run_rows, (rotary_cos, rotary_sin), context,
                    None, adaln_partials[index],
                )
            pending.append(probe)
            # Keep a few launches in flight, well under the runtime's queue depth (~8).
            while len(pending) > _MAX_IN_FLIGHT:
                pending.pop(0).to("cpu")
        for probe in pending:
            probe.to("cpu")
        return stages["epilogue"](hidden_states, temb, norm_out_run_rows, row_runs)

    def sequence_parallel_inputs(self, row_runs: MiniMaxH3RowRuns) -> dict[str, torch.Tensor]:
        """Host tensors for `forward` under sequence parallelism, else ``{}``.

        ``sp_rows``: this TP rank's rows of its CP shard. ``sp_run_index``: for each of them,
        the shard-local run it belongs to (an index into the shard's AdaLN run rows).
        ``tp_cols``: its slice of ``time_embed_dim`` (the row-parallel AdaLN input). All three
        are runtime inputs so the TP peers trace one graph.
        """
        if not self.sequence_parallel:
            return {}
        sequence_length = row_runs.sequence_length
        shard = parallel.sequence_shard(sequence_length, self.cp_size, self.cp_rank)
        local_runs, _ = parallel.shard_runs(
            row_runs.runs,
            torch.zeros(len(row_runs.runs), dtype=torch.long),
            shard.start,
            min(shard.end, sequence_length),
            pad_to=shard.shard_length,
        )
        run_index = torch.empty(shard.shard_length, dtype=torch.long)
        for index, (start, end) in enumerate(local_runs):
            run_index[start:end] = index
        window = shard.shard_length // self.tp_size
        rows = torch.arange(self.tp_rank * window, (self.tp_rank + 1) * window, dtype=torch.long)
        per_rank = self.config.time_embed_dim // self.tp_size
        cols = torch.arange(self.tp_rank * per_rank, (self.tp_rank + 1) * per_rank, dtype=torch.long)
        return {"sp_rows": rows, "sp_run_index": run_index[rows], "tp_cols": cols}

    def _block_geometry(
        self, row_runs: MiniMaxH3RowRuns, key_start: torch.Tensor, ring: bool = True, tp_cols=None
    ):
        """This rank's run spans and attention context: static geometry, computed in Python."""
        runs = row_runs.runs
        if self.cp_size <= 1:
            return runs, _AttentionContext(
                key_start=key_start, key_end=row_runs.sequence_length,
                sequence_parallel=self.sequence_parallel, tp_cols=tp_cols,
            )
        shard = parallel.sequence_shard(row_runs.sequence_length, self.cp_size, self.cp_rank)
        runs, _ = parallel.shard_runs(
            runs,
            torch.zeros(len(runs), dtype=torch.long),
            shard.start,
            min(shard.end, row_runs.sequence_length),
            pad_to=shard.shard_length,
        )
        return runs, _AttentionContext(
            key_start,
            row_runs.sequence_length,
            cp_group=self.cp_group,
            ring_groups=self.cp_ring_groups if ring else None,
            cp_size=self.cp_size,
            shard_start=shard.start,
            sequence_parallel=self.sequence_parallel,
            tp_cols=tp_cols,
        )

    def _prologue(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        rotary_cos,
        rotary_sin,
        adaln_run_rows,
        num_text_tokens,
        row_runs: MiniMaxH3RowRuns,
        sp_rows=None,
        sp_run_index=None,
    ):
        """Embed the three streams, refine the prompt, pack and (under CP) shard the sequence.

        Returns ``(hidden_states, temb, adaln_run_rows, rotary_cos, rotary_sin, key_start)``
        for this rank's rows.
        """
        rotary_emb = (rotary_cos, rotary_sin)
        num_condition_rows = row_runs.num_condition_rows

        # 1. Project each modality and build the packed sequence. The checkpoint is
        # mixed-precision (the two patch projections are float32 while
        # `context_embedder` and the block stack are bfloat16), so every input is
        # aligned with its projection's parameter dtype, mirroring the reference's
        # explicit casts. The text stream sets the dtype of the packed sequence.
        video_embeds = _rowwise(self.proj_in, hidden_states.to(self.proj_in.weight.dtype))
        audio_embeds = _rowwise(
            self.audio_proj_in, audio_hidden_states.to(self.audio_proj_in.weight.dtype)
        )
        text_embeds = self.context_embedder(
            encoder_hidden_states.to(self.context_embedder.weight.dtype)
        )
        num_text_rows = row_runs.num_text_rows
        key_start = num_text_rows - num_text_tokens.to(torch.int32)
        # The refiner attends over the prompt alone, whose bucket need not be a multiple of the
        # 128-query tile the attention kernel reads its bounds in; the materialized fallback
        # does not fit on-chip at reference-sized prompts (NCC_IBIR229). So it runs over a copy
        # left-padded to the tile, the extra rows masked like the bucket's own padding.
        refiner_pad = -num_text_rows % 128
        if refiner_pad:
            text_embeds = torch.cat([text_embeds.new_zeros((text_embeds.shape[0], refiner_pad, text_embeds.shape[2])), text_embeds], dim=1)
        text_embeds = self.token_refiner(
            text_embeds,
            _AttentionContext(key_start=key_start + refiner_pad, key_end=num_text_rows + refiner_pad),
        )
        if refiner_pad:
            text_embeds = text_embeds[:, refiner_pad:]

        # `[padding | text | conditions | audio | video]` (see `MiniMaxH3RowRuns`). The
        # reference scatters with three `index_copy` calls into a zero buffer; the row blocks
        # are contiguous and statically sized, so a `cat` builds the sequence instead.
        dtype = text_embeds.dtype
        hidden_states = torch.cat(
            [
                text_embeds,
                video_embeds[:, :num_condition_rows].to(dtype),
                audio_embeds.to(dtype),
                video_embeds[:, num_condition_rows:].to(dtype),
            ],
            dim=1,
        )

        # 2. One timestep embedding per distinct noise level. `temb` is shared by all
        # AdaLN projections, which are bfloat16 while `time_embedder` is float32, so it
        # stays at the time embedder's precision: each AdaLN module applies its own
        # activation and casts to its projection's dtype afterwards.
        temb = self._time_embedding(timestep)

        # 2b. Under context parallelism, take this rank's rows. Everything in the block stack
        # except attention is row-local, so the shard needs no communication until the ring.
        runs = row_runs.runs
        sequence_length = row_runs.sequence_length
        if self.cp_size > 1:
            shard = parallel.sequence_shard(sequence_length, self.cp_size, self.cp_rank)
            if shard.num_pad:
                hidden_states = _pad_rows(hidden_states, 1, sequence_length + shard.num_pad)
                rotary_emb = tuple(
                    _pad_rows(table, 0, sequence_length + shard.num_pad) for table in rotary_emb
                )
            hidden_states = hidden_states[:, shard.start : shard.end].contiguous()
            # The rotary tables are `(seq_len, ...)`, so they shard on dim 0 and have to cover
            # exactly the rows Q now holds.
            rotary_emb = tuple(table[shard.start : shard.end].contiguous() for table in rotary_emb)
            _, adaln_run_rows = parallel.shard_runs(
                runs,
                adaln_run_rows,
                shard.start,
                min(shard.end, sequence_length),
                pad_to=shard.shard_length,
            )
            if self.sequence_parallel:
                # This TP rank's rows, and one AdaLN table row per row. Both are runtime inputs
                # (see `sequence_parallel_inputs`) so the TP peers trace identical graphs: peers
                # whose collectives the compiler schedules differently fail to rendezvous.
                hidden_states = hidden_states.index_select(1, sp_rows)
                adaln_run_rows = adaln_run_rows.index_select(0, sp_run_index)
        elif self.sequence_parallel:
            # CP=1: the one shard is the whole (128-aligned) sequence; attention gathers it back.
            shard = parallel.sequence_shard(sequence_length, 1, 0)
            if shard.num_pad:
                hidden_states = _pad_rows(hidden_states, 1, sequence_length + shard.num_pad)
                rotary_emb = tuple(_pad_rows(table, 0, sequence_length + shard.num_pad) for table in rotary_emb)
            _, adaln_run_rows = parallel.shard_runs(
                runs, adaln_run_rows, 0, sequence_length, pad_to=shard.shard_length
            )
            hidden_states = hidden_states.index_select(1, sp_rows)
            adaln_run_rows = adaln_run_rows.index_select(0, sp_run_index)

        return hidden_states, temb, adaln_run_rows, rotary_emb[0], rotary_emb[1], key_start

    def _epilogue(self, hidden_states, temb, norm_out_run_rows, row_runs: MiniMaxH3RowRuns):
        """Gather the shards (under CP), normalize the media rows and run the two output heads."""
        num_text_rows = row_runs.num_text_rows
        num_condition_rows = row_runs.num_condition_rows
        if self.sequence_parallel:
            hidden_states = _tp_gather_rows(hidden_states)
        if self.cp_size > 1:
            # Back to full length before the output heads, so `norm_out`'s run spans stay in
            # absolute coordinates.
            hidden_states = self.cp_group.all_gather(hidden_states.contiguous(), dim=1)

        # 3. Only the media rows reach a head, and both heads are row-wise linear maps,
        # so slicing before the output stack is identical to the reference's "project
        # every row, then `index_select`" — and drops the text rows from `norm_out` too.
        media = hidden_states[:, num_text_rows : row_runs.sequence_length]
        media = self.norm_out(media, temb, row_runs.media_runs, norm_out_run_rows)

        condition_end = num_condition_rows
        audio_end = condition_end + row_runs.num_audio_rows
        video_rows = torch.cat([media[:, :condition_end], media[:, audio_end:]], dim=1)
        audio_rows = media[:, condition_end:audio_end]

        video_output = _rowwise(self.proj_out, video_rows.to(self.proj_out.weight.dtype))
        audio_output = _rowwise(self.audio_proj_out, audio_rows.to(self.audio_proj_out.weight.dtype))
        return video_output, audio_output

    # ---------------------------------------------------------------
    # Weight loading
    # ---------------------------------------------------------------

    def _weight_mappings(self) -> dict[str, str | list[str]]:
        """Model parameter name -> converted-diffusers checkpoint key(s).

        Keys the port did not rename are omitted: the loader maps those by identity.
        """
        mappings: dict[str, str | list[str]] = {}

        def attention_mappings(model_prefix: str, checkpoint_prefix: str) -> None:
            mappings[f"{model_prefix}.qkv_proj_weight"] = [
                f"{checkpoint_prefix}.to_q.weight",
                f"{checkpoint_prefix}.to_k.weight",
                f"{checkpoint_prefix}.to_v.weight",
            ]
            mappings[f"{model_prefix}.o_proj_weight"] = f"{checkpoint_prefix}.to_out.0.weight"

        def feedforward_mappings(model_prefix: str, checkpoint_prefix: str) -> None:
            # Both halves read the one fused `[value; gate]` tensor; the loaders pick
            # their half (see `_gated_half_weight_loader`).
            mappings[f"{model_prefix}.value_proj_weight"] = f"{checkpoint_prefix}.net.0.proj.weight"
            mappings[f"{model_prefix}.gate_proj_weight"] = f"{checkpoint_prefix}.net.0.proj.weight"
            mappings[f"{model_prefix}.down_proj_weight"] = f"{checkpoint_prefix}.net.2.weight"

        for i in range(self.config.num_refiner_layers):
            prefix = f"token_refiner.refiner_blocks.{i}"
            attention_mappings(f"{prefix}.attn", f"{prefix}.attn")
            feedforward_mappings(f"{prefix}.ff", f"{prefix}.ff")

        for i in range(self.config.num_layers):
            prefix = f"transformer_blocks.{i}"
            attention_mappings(f"{prefix}.attn", f"{prefix}.attn")
            feedforward_mappings(f"{prefix}.ff", f"{prefix}.ff")
            mappings[f"{prefix}.adaln_proj.linear_weight"] = f"{prefix}.adaln_proj.linear.weight"
            mappings[f"{prefix}.adaln_proj.linear_bias"] = f"{prefix}.adaln_proj.linear.bias"

        return mappings

    def _param_dtype(self, name: str) -> torch.dtype:
        """The dtype a parameter is held in, honouring the mixed-precision contract."""
        if any(pattern in name for pattern in _KEEP_IN_FP32):
            return torch.float32
        return self.dtype

    def load_weights(
        self,
        model_name_or_path: str,
        device: torch.device = torch.device("cpu"),
        cache_dir: str | None = None,
    ) -> None:
        """Load a rank-sharded checkpoint to ``device`` with pipelined data movement.

        ``dtype_override`` carries the mixed-precision contract *into* the loader. The
        loader otherwise casts each tensor to the dtype its parameter was built with, and
        every parameter here was built under `DiffusersPipelineLoader`'s
        ``set_default_torch_dtype(od_config.dtype)`` — so the `_KEEP_IN_FP32` modules
        would be downcast to bfloat16 and then cast back below, losing what they are kept
        in float32 for.

        Args:
            model_name_or_path: HuggingFace model id or a directory of weights.
            device: Device to load the weights to.
            cache_dir: Optional download cache when loading from the Hub.
        """
        # Weights are sharded along the head / ffn axis only, so the slice is selected by the
        # rank within its TP group: every CP shard holds the same weights.
        tp_size = parallel.tp_size()
        tp_rank = parallel.tp_rank()

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
        if ADALN_ON_HOST:
            self._move_adaln_to_host()

        ring = os.environ.get(_RING_ATTENTION_ENV, "1") != "0"
        for block in self.transformer_blocks:
            block.attn.use_ring = ring and ring_bound_holds(block.attn.norm_q.weight, block.attn.norm_k.weight)
        gathered = [index for index, block in enumerate(self.transformer_blocks) if not block.attn.use_ring]
        if self.cp_size > 1:
            logger.info("MiniMax-H3 blocks %s attend over gathered K/V; the rest over the K/V ring.", gathered)

        # After the ring check: the heads-first relayout folds 1/sqrt(D) into `norm_q`'s weight.
        # Heads-first is a per-chip win at CP=1 (-5.6 % at TP4) but measured +13 % at TP8 x CP8, where
        # the K/V all-gather runs on the heads-first layout; keep it to CP=1 unless forced.
        # Not under sequence parallelism: the heads-first path has no TP row gather / scatter.
        if HEADS_FIRST and (not self.sequence_parallel or self._sp_fused) and (
            self.cp_size == 1 or os.environ.get("MINIMAX_H3_HEADS_FIRST_CP", "0") == "1"
        ):
            for block in self.transformer_blocks:
                block.attn.relayout_for_heads_first()


def _move_adaln_to_host_impl(model) -> None:
    import copy

    for block in model.transformer_blocks:
        mod = block.adaln_proj
        # The bias stays pre-divided by TP (`_row_bias_loader`); the all-reduce sums it back.
        mod._host_weight = mod.linear_weight.detach().cpu().contiguous()
        mod._host_bias = mod.linear_bias.detach().cpu()
        del mod.linear_weight
        del mod.linear_bias
    # A float32 host copy of the (tiny) timestep MLP, taken before the pipeline moves the model.
    model._host_time_embedder = copy.deepcopy(model.time_embedder).to("cpu").float()
    model._adaln_cache = {}


@torch.no_grad()
def _adaln_host_partials_impl(model, timestep: torch.Tensor) -> torch.Tensor:
    values = timestep.detach().cpu().float().tolist()
    missing = sorted({v for v in values if v not in model._adaln_cache})
    if missing:
        from vllm_omni_neuron.diffusion.models.minimax_h3.text_encoder import host_threads

        saved = model.time_embedder
        model.time_embedder = model._host_time_embedder
        try:
            with host_threads():
                temb = model._time_embedding(torch.tensor(missing, dtype=torch.float32))
                activated = F.silu(temb).to(model.transformer_blocks[0].adaln_proj._host_weight.dtype)
                # (num_layers, len(missing), 18 * hidden): one GEMM per block for all values.
                table = torch.stack([block.adaln_proj.host_partial(activated) for block in model.transformer_blocks])
        finally:
            model.time_embedder = saved
        for index, value in enumerate(missing):
            model._adaln_cache[value] = table[:, index].clone()
    return torch.stack([model._adaln_cache[v] for v in values], dim=1)


def _prepare_rope_matrix_impl(model) -> None:
    """Publish the (device-resident) `rope_rotate` buffer to `_rope_by_matmul`. Call after `.to(device)`."""
    if hasattr(model, "rope_rotate"):
        _ROPE_ROTATE[model.rope_rotate.dtype] = model.rope_rotate


NeuronMiniMaxH3Transformer3DModel.prepare_rope_matrix = _prepare_rope_matrix_impl
NeuronMiniMaxH3Transformer3DModel._move_adaln_to_host = _move_adaln_to_host_impl
NeuronMiniMaxH3Transformer3DModel.adaln_host_partials = _adaln_host_partials_impl


class WholeGraph(nn.Module):
    """The DiT's whole forward as one module, for `torch.compile` to trace end to end.

    Defined here, after everything it traces, rather than next to the pipeline: the compiled
    graph's cache key covers its source locations, so this keeps pipeline edits from
    invalidating a ~12-minute compile.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, **kwargs):
        return self.transformer._forward_whole(**kwargs)

