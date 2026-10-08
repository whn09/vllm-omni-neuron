# SPDX-License-Identifier: Apache-2.0
"""H3-AudioVAE for Neuron — the BigVGAN waveform decoder.

Scope: decoder only
-------------------
This port covers ``dec_in_proj`` + the BigVGAN decoder, which is everything the
**t2va** task touches. The DAC-lineage encoder, the causal-attention projection
(``pre_block``) and the ``mean_proj`` / ``logs_proj`` posterior heads are deliberately
absent: nothing in text-to-video+audio encodes a waveform. They are what the
**ref2va** task needs for its voice reference, so they belong here later, alongside
`NeuronAutoencoderKLMiniMaxH3`'s currently-dormant video encoder.

Running on Neuron
-----------------
800x upsampling is what makes this decoder hard to compile: the reference's alias-free
resamplers run a 12-tap Kaiser ``conv_transpose1d`` along the upsampled axis, which neuronx-cc
lays out as one tensor 12x the width of the signal and cannot fit in on-chip SBUF at any
window size. `_polyphase_conv_transpose1d` computes the same thing as ``stride`` narrow
``conv1d``s interleaved, so no such tensor exists.

The clip is decoded in fixed 96-latent-frame windows (`audio_windows`) rather than in one
graph: a whole 5-second clip is 207 latent frames and ~15M instructions, which compiles very
slowly. The receptive field is 30.6 latent frames, so with 24 frames of real context on each
side a window's core is identical to the whole-clip decode (to fp32 rounding, measured), the
windows share one graph shape for any clip length, and they are spread over the world's ranks.

Precision: float32, not bfloat16
--------------------------------
The released checkpoint is float32 and this stack does not tolerate bfloat16 — the
reference lists ``encoder``/``decoder``/``pre_block``/``dec_in_proj``/``mean_proj``/
``logs_proj`` in ``_keep_in_fp32_modules`` and notes decodes come out roughly 20 dB
quieter. The suspects are structural: `MiniMaxH3AudioSnakeBeta` exponentiates its two
log-space parameters and takes a reciprocal of ``exp(beta) + 1e-9``, and every
convolution's weight is a ``weight_g * weight_v / ||weight_v||`` reparameterization
whose norm spans many orders of magnitude across 65M values. This port therefore runs
float32 end to end and does not offer a compute dtype.

Neuron-specific departures from ``diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio``
-------------------------------------------------------------------------------------------------
1. **Weight norm is folded at load time.** The reference wraps every convolution in
   ``torch.nn.utils.weight_norm``, which recomputes ``g * v / ||v||`` on every forward.
   That is a dozen extra reductions per convolution inside the traced graph for a
   value that never changes at inference, so the fold happens in the weight loader
   (`_weight_norm_loader`) and the graph sees a plain weight.
2. **Replicate padding is expressed as `cat` of repeated edge slices.** The alias-free
   resamplers pad with ``mode="replicate"``; the equivalent `cat` traces on Neuron.
3. **No negative-index slicing.** `MiniMaxH3AudioUpSample1d` trims with
   ``[..., pad_left : -pad_right]``; every bound here is computed from the shape and
   applied with `narrow`.
4. **The Kaiser resampling filters are built on the host and are non-persistent.**
   They are a deterministic function of ``(ratio, kernel_size)`` — and every one of
   them in this decoder is the same ``ratio=2, kernel_size=12`` filter — so they are
   computed once at construction rather than loaded, keeping `torch.kaiser_window` and
   `torch.sinc` out of both the graph and the checkpoint read.
"""

import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_minimax_h3 import (
    _as_float32,
)

# The decoder's alias-free activations all use these; `MiniMaxH3AudioActivation1d`
# defaults to them in the reference and nothing overrides them.
_ACTIVATION_RESAMPLE_RATIO = 2
_ACTIVATION_RESAMPLE_KERNEL = 12


def _replicate_pad_1d(hidden_states: torch.Tensor, before: int, after: int) -> torch.Tensor:
    """Replicate-pad the last axis by repeating its edge values.

    ``F.pad(..., mode="replicate")`` on a 3-D tensor asks the backend for a 1-D
    replicate lowering; a `cat` of expanded edge slices is the same tensor and traces
    on Neuron. Widths here are up to 6 (``kernel_size // 2`` for the length-12 Kaiser
    filters), so `expand` is used rather than a per-element `cat`.
    """
    if before == 0 and after == 0:
        return hidden_states
    length = hidden_states.shape[-1]
    parts = []
    if before:
        parts.append(hidden_states.narrow(-1, 0, 1).expand(*hidden_states.shape[:-1], before))
    parts.append(hidden_states)
    if after:
        parts.append(hidden_states.narrow(-1, length - 1, 1).expand(*hidden_states.shape[:-1], after))
    return torch.cat(parts, dim=-1)


def _polyphase_conv_transpose1d(
    hidden_states: torch.Tensor, filter_: torch.Tensor, stride: int, num_channels: int
) -> torch.Tensor:
    """A depthwise ``conv_transpose1d`` written as `stride` narrow ``conv1d``s.

    Exactly equal to::

        F.conv_transpose1d(hidden_states, filter_.expand(num_channels, -1, -1),
                           stride=stride, groups=num_channels)

    and the reason for not calling that is that it does not compile. Transposed convolution
    places tap ``j`` at output index ``stride * i + j``, so the compiler materializes a product
    tensor as wide as the *upsampled* axis times the *whole* kernel, and neuronx-cc runs out of
    SBUF laying it out: ``[NCC_INLA001] ... SB<0,0>(64x308640)`` against a 192 KB partition, i.e.
    1.57x over. That overflow is invariant to the latent window — measured identical at 8, 16,
    24, 32, 64 and 96 frames — so chunking does not reach it and this rewrite is the only lever.

    Polyphase decomposition is the standard identity. Output index ``stride * m + p`` collects
    only the taps congruent to ``p``, and does so as an ordinary correlation of the input against
    ``filter_[p::stride]`` reversed. Each of those sub-filters is ``kernel_size // stride`` long
    — 6 rather than 12 here — and the wide intermediate is never built at all.

    Requires ``stride`` to divide the kernel length, which makes every phase the same length and
    the interleaved result come out at exactly ``stride * (length - 1) + kernel_size`` with no
    trimming at all: 'full' correlation gives each phase ``length + kernel_size // stride - 1``
    samples, and ``stride`` of those interleave to precisely the transposed convolution's length.
    Every filter in this decoder is ``ratio=2, kernel_size=12``; a ragged decomposition would need
    per-phase offsets and is not written because nothing needs it.
    """
    kernel_size = filter_.shape[-1]
    if kernel_size % stride:
        raise ValueError(
            f"Polyphase needs `stride` ({stride}) to divide the kernel length ({kernel_size})."
        )

    phases = []
    for phase in range(stride):
        # `conv1d` correlates where the transposed convolution convolves, so each phase's taps
        # are reversed. `reshape` rather than `view`: the strided slice is not contiguous.
        taps = filter_.reshape(-1)[phase::stride].flip(-1)
        weight = taps.reshape(1, 1, -1).expand(num_channels, -1, -1)
        phases.append(
            F.conv1d(
                hidden_states,
                weight,
                # 'Full' correlation, which is what makes the interleave land without a trim.
                padding=taps.shape[-1] - 1,
                groups=num_channels,
            )
        )

    # Interleave: element `stride * m + p` is phase `p` at position `m`. A new trailing axis
    # flattened into the last one is exactly that, and avoids an index_put on the device.
    stacked = torch.stack(phases, dim=-1)
    return stacked.reshape(*stacked.shape[:-2], -1)


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """Kaiser-windowed sinc low-pass filter, shape ``(1, 1, kernel_size)``.

    Arithmetically identical to the ``alias-free-torch`` implementation the checkpoint
    was trained with. Host-side only — `torch.kaiser_window` and `torch.sinc` are
    evaluated at construction, never in the traced graph.
    """
    half_size = kernel_size // 2

    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False, dtype=torch.float32)

    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size, dtype=torch.float32) + 0.5
    else:
        time = torch.arange(kernel_size, dtype=torch.float32) - half_size

    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    # Normalized to sum 1 so a constant input does not leak through the resampler.
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


def _weight_norm_loader() -> SafetensorsWeightLoader:
    """Fold ``weight_g * weight_v / ||weight_v||`` into one weight at load time.

    ``weight_norm`` with the default ``dim=0`` normalizes over every axis but the
    first, for both `nn.Conv1d` (``(out, in, k)``) and `nn.ConvTranspose1d`
    (``(in, out, k)``), so one loader serves both. The norm is taken in float32
    regardless of the stored dtype — ``weight_v`` rows here span many orders of
    magnitude and the reciprocal is what the reparameterization is sensitive to.

    The loader receives lazy ``PySafeSlice`` handles, not tensors: `safetensors` only
    reads what is indexed. That supports ``[...]`` but no tensor methods, and both
    factors are needed whole here, so each is materialized with ``[:]`` first.
    """

    def transform(slices, rank):
        assert len(slices) == 2, "weight norm needs `weight_g` and `weight_v`."
        weight_g, weight_v = (sl[:] for sl in slices)
        norm = weight_v.float().pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
        return (weight_g.float() * weight_v.float() / norm).to(weight_v.dtype)

    return SafetensorsWeightLoader(transform=transform)


class _WeightNormConv1d(nn.Conv1d):
    """`nn.Conv1d` whose weight is loaded from a folded ``weight_g`` / ``weight_v`` pair."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        set_weight_loader(self.weight, _weight_norm_loader())


class _WeightNormConvTranspose1d(nn.ConvTranspose1d):
    """`nn.ConvTranspose1d` whose weight is loaded from a folded pair."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        set_weight_loader(self.weight, _weight_norm_loader())


class MiniMaxH3AudioSnakeBeta(nn.Module):
    """``x + (exp(beta) + 1e-9)^-1 * sin(exp(alpha) * x)^2``.

    BigVGAN's activation: separate frequency (``alpha``) and magnitude (``beta``)
    parameters, both stored in log space as ``(channels,)`` vectors.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha).view(1, -1, 1)
        beta = torch.exp(self.beta).view(1, -1, 1)
        return hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)


class MiniMaxH3AudioUpSample1d(nn.Module):
    """Anti-aliased ``ratio``x upsampler — transposed depthwise Kaiser-sinc convolution.

    ``filter`` is non-persistent: it is a deterministic function of the constructor
    arguments (see `kaiser_sinc_filter1d`), so it is rebuilt here rather than read from
    the checkpoint.
    """

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size),
            persistent=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = _replicate_pad_1d(hidden_states, self.pad, self.pad)
        upsampled = self.ratio * _polyphase_conv_transpose1d(
            hidden_states, self.filter, self.stride, num_channels
        )
        length = upsampled.shape[-1]
        return upsampled.narrow(-1, self.pad_left, length - self.pad_left - self.pad_right)


class MiniMaxH3AudioDownSample1d(nn.Module):
    """Anti-aliased ``ratio``x downsampler — strided depthwise Kaiser-sinc convolution.

    The reference nests this under a ``lowpass`` submodule holding the filter buffer.
    The filter is non-persistent here, so nothing in the checkpoint depends on that
    nesting and it is flattened away.
    """

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size),
            persistent=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = _replicate_pad_1d(hidden_states, self.pad_left, self.pad_right)
        return F.conv1d(
            hidden_states,
            self.filter.expand(num_channels, -1, -1),
            stride=self.stride,
            groups=num_channels,
        )


class MiniMaxH3AudioActivation1d(nn.Module):
    """Upsample -> SnakeBeta -> downsample: BigVGAN's alias-free activation wrapper.

    The activation is applied at ``ratio``x rate so that the harmonics ``sin(alpha * x)``
    introduces above the original Nyquist limit are filtered out on the way back down
    instead of folding back as aliases.
    """

    def __init__(
        self,
        activation: nn.Module,
        ratio: int = _ACTIVATION_RESAMPLE_RATIO,
        kernel_size: int = _ACTIVATION_RESAMPLE_KERNEL,
    ):
        super().__init__()
        self.act = activation
        self.upsample = MiniMaxH3AudioUpSample1d(ratio, kernel_size)
        self.downsample = MiniMaxH3AudioDownSample1d(ratio, kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(hidden_states)))


class MiniMaxH3AudioAMPBlock(nn.Module):
    """BigVGAN anti-aliased multi-periodicity block (``AMPBlock1``).

    Each dilation contributes a ``(dilated conv, dilation-1 conv)`` pair, and every
    convolution is preceded by its own alias-free SnakeBeta activation. The
    activations interleave in the checkpoint — even indices feed ``convs1``, odd
    indices ``convs2``.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: tuple[int, ...]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                _WeightNormConv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=d,
                    padding=(kernel_size * d - d) // 2,
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                _WeightNormConv1d(
                    channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2
                )
                for _ in dilation
            ]
        )
        self.activations = nn.ModuleList(
            [MiniMaxH3AudioActivation1d(MiniMaxH3AudioSnakeBeta(channels)) for _ in range(2 * len(dilation))]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            residual = conv2(act2(conv1(act1(hidden_states))))
            hidden_states = residual + hidden_states
        return hidden_states


class MiniMaxH3AudioBigVGANDecoder(nn.Module):
    """BigVGAN decoder: ``(B, latent_dim, F)`` -> ``(B, 1, F * 800)``.

    Each upsampling stage runs `num_kernels` AMP blocks in parallel over the same
    input and **averages** their outputs — the multi-receptive-field fusion that gives
    BigVGAN its name.
    """

    def __init__(
        self,
        in_channels: int,
        upsample_initial_channel: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = _WeightNormConv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)

        # Each upsampler is wrapped in a one-element `ModuleList` in the checkpoint
        # (`ups.<i>.0`); the extra nesting is kept so key mapping stays a passthrough.
        self.ups = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _WeightNormConvTranspose1d(
                            upsample_initial_channel // (2**i),
                            upsample_initial_channel // (2 ** (i + 1)),
                            kernel,
                            rate,
                            padding=(kernel - rate) // 2,
                        )
                    ]
                )
                for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
            ]
        )

        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            channels = upsample_initial_channel // (2 ** (i + 1))
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(MiniMaxH3AudioAMPBlock(channels, kernel, tuple(dilation)))

        self.activation_post = MiniMaxH3AudioActivation1d(MiniMaxH3AudioSnakeBeta(channels))
        self.conv_post = _WeightNormConv1d(channels, 1, 7, 1, padding=3, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_pre(hidden_states)

        for i in range(self.num_upsamples):
            hidden_states = self.ups[i][0](hidden_states)
            residual = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j](hidden_states)
                residual = block if residual is None else residual + block
            hidden_states = residual / self.num_kernels

        hidden_states = self.conv_post(self.activation_post(hidden_states))
        return torch.clamp(hidden_states, min=-1.0, max=1.0)


#: Latent frames per audio-decode window, and the real context kept on each side of the part of
#: it that is used. BigVGAN's receptive field is 30.6 latent frames end to end; measured against a
#: whole-clip decode, 16 frames of context leaves a 1e-4 seam and 24 matches to fp32 rounding
#: (6.7e-7). Every window has the same shape, so one graph serves any clip length.
AUDIO_WINDOW = 96
AUDIO_HALO = 24


def audio_windows(num_frames: int, window: int = AUDIO_WINDOW, halo: int = AUDIO_HALO):
    """``(window_start, core_start, core_end)`` covering ``[0, num_frames)``, in latent frames.

    Every window is ``window`` frames, placed so its core has ``halo`` frames of real context on
    each side or meets the clip's own boundary.
    """
    core = window - 2 * halo
    spans = []
    for core_start in range(0, num_frames, core):
        core_end = min(core_start + core, num_frames)
        start = min(max(core_start - halo, 0), num_frames - window)
        spans.append((start, core_start, core_end))
    return spans


class NeuronMiniMaxH3AudioDecoder(nn.Module):
    """``dec_in_proj`` + BigVGAN, the unit decoded on the host when nothing is compiled."""

    def __init__(self, dec_in_proj: nn.Module, decoder: MiniMaxH3AudioBigVGANDecoder):
        super().__init__()
        self.dec_in_proj = dec_in_proj
        self.decoder = decoder

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.dec_in_proj(latents))


class NeuronMiniMaxH3AudioDecoderStage(nn.Module):
    """One upsampling stage of BigVGAN (upsample, then the averaged AMP blocks): one NEFF.

    The decoder is traced stage by stage because a whole 96-frame window is one program too
    large to stage onto a NeuronCore (``dlr_kelf_stage: Failed to load subgraph``). The first
    stage also runs ``dec_in_proj`` and ``conv_pre``; the last also runs the output activation,
    ``conv_post`` and the clamp. Modules are shared with the decoder, not copied.
    """

    def __init__(self, dec_in_proj: nn.Module, decoder: MiniMaxH3AudioBigVGANDecoder, index: int):
        super().__init__()
        self.index = index
        self.first = index == 0
        self.last = index == decoder.num_upsamples - 1
        self.num_kernels = decoder.num_kernels
        self.dec_in_proj = dec_in_proj if self.first else None
        self.conv_pre = decoder.conv_pre if self.first else None
        self.upsample = decoder.ups[index][0]
        self.resblocks = nn.ModuleList(
            decoder.resblocks[index * decoder.num_kernels + j] for j in range(decoder.num_kernels)
        )
        self.activation_post = decoder.activation_post if self.last else None
        self.conv_post = decoder.conv_post if self.last else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.first:
            hidden_states = self.conv_pre(self.dec_in_proj(hidden_states))
        hidden_states = self.upsample(hidden_states)
        residual = None
        for block in self.resblocks:
            out = block(hidden_states)
            residual = out if residual is None else residual + out
        hidden_states = residual / self.num_kernels
        if self.last:
            hidden_states = torch.clamp(self.conv_post(self.activation_post(hidden_states)), -1.0, 1.0)
        return hidden_states


class NeuronAutoencoderKLMiniMaxH3Audio(nn.Module):
    r"""H3-AudioVAE for Neuron — waveform decoding only.

    The autoencoder is **mono**. MiniMax-H3 carries stereo as two *batch* items, so
    `decode` takes ``(2, 32, F)`` and returns ``(2, 1, F * 800)``; interleaving into a
    stereo waveform happens at the pipeline's output boundary.

    Latents are normalized with per-channel ``latents_mean`` / ``latents_std`` (32
    floats each) rather than a scalar ``scaling_factor``, so a caller applies
    `denormalize_latents` before `decode`.

    Only the decoder is built — see the module docstring. `encode` is absent rather
    than raising, so a task that needs it fails at attribute lookup during pipeline
    construction instead of mid-generation.
    """

    def __init__(
        self,
        encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5),
        latent_dim: int = 2048,
        latent_channels: int = 32,
        decoder_dim: int = 1024,
        decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2),
        decoder_kernel_sizes: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4),
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        sampling_rate: int = 32000,
        latents_mean: list[float] | None = None,
        latents_std: list[float] | None = None,
        **ignored,
    ):
        super().__init__()
        encoder_rates = tuple(int(rate) for rate in encoder_rates)
        decoder_rates = tuple(int(rate) for rate in decoder_rates)
        self.hop_length = math.prod(encoder_rates)
        if math.prod(decoder_rates) != self.hop_length:
            raise ValueError(
                f"`decoder_rates` must upsample by the encoder hop length {self.hop_length}, got "
                f"{math.prod(decoder_rates)}."
            )
        self.latent_channels = latent_channels
        self.sampling_rate = sampling_rate
        self._compiled_decoder = None
        self.latents_mean = tuple(latents_mean) if latents_mean is not None else None
        self.latents_std = tuple(latents_std) if latents_std is not None else None

        self.dec_in_proj = nn.Conv1d(latent_channels, latent_dim, 1)
        self.decoder = MiniMaxH3AudioBigVGANDecoder(
            in_channels=latent_dim,
            upsample_initial_channel=decoder_dim,
            upsample_rates=decoder_rates,
            upsample_kernel_sizes=tuple(int(kernel) for kernel in decoder_kernel_sizes),
            resblock_kernel_sizes=tuple(int(kernel) for kernel in resblock_kernel_sizes),
            resblock_dilation_sizes=tuple(tuple(int(d) for d in dilation) for dilation in resblock_dilation_sizes),
        )

    # ---------------------------------------------------------------
    # Compilation
    # ---------------------------------------------------------------

    def compile(self, *args, **compiler_kwargs):
        """Trace the decoder for Neuron, one graph per upsampling stage.

        The polyphase upsampler (`_polyphase_conv_transpose1d`) is what makes this possible:
        the reference's ``conv_transpose1d`` widens a 12-tap Kaiser filter across the
        upsampled axis and overflows on-chip SBUF at every window size. Float32 end to end,
        hence ``--auto-cast=none``. See `NeuronMiniMaxH3AudioDecoderStage` for the split.
        """
        options = dict(compiler_kwargs.pop("options", {}))
        options["compiler_args"] = [
            "--model-type=unet-inference",
            "--auto-cast=none",
            "-O1",
        ]
        compiler_kwargs.setdefault("fullgraph", True)
        self._compiled_decoder = [
            torch.compile(
                NeuronMiniMaxH3AudioDecoderStage(self.dec_in_proj, self.decoder, index),
                *args,
                options=dict(options),
                **compiler_kwargs,
            )
            for index in range(self.decoder.num_upsamples)
        ]
        return self

    def _decoder_module(self):
        return NeuronMiniMaxH3AudioDecoder(self.dec_in_proj, self.decoder)

    @property
    def device(self) -> torch.device:
        """Where the weights live, read off a parameter rather than tracked separately."""
        return self.dec_in_proj.weight.device

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def decode(self, latents: torch.Tensor, return_dict: bool = False):
        """Decode ``(B, 32, F)`` denormalized latents into ``(B, 1, F * 800)`` samples.

        In and out on the host in float32; the waveform is clamped to ``[-1, 1]`` by the
        decoder's final op. See `_decode_windows` for how the work is split.
        """
        if latents.ndim != 3:
            raise ValueError(
                f"`latents` must be (batch_size, latent_channels, num_frames), got {tuple(latents.shape)}."
            )
        latents = _as_float32(latents.to("cpu"), "`decode` input")
        if self._compiled_decoder is not None:
            decoded = self._decode_windows(latents)
        elif dist.is_initialized() and dist.get_rank() != 0:
            # On the host the whole clip is one fast pass, so rank 0 decodes it alone.
            return (None,) if not return_dict else None
        else:
            decoded = self._decode_on_host(latents)
        if not return_dict:
            return (decoded,)
        from diffusers.models.autoencoders.vae import DecoderOutput

        return DecoderOutput(sample=decoded)

    def _decode_on_host(self, latents: torch.Tensor) -> torch.Tensor:
        """The whole clip in one host pass, with the host's threads (workers run with one)."""
        previous = torch.get_num_threads()
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        try:
            return self._decoder_module()(latents)
        finally:
            torch.set_num_threads(previous)

    def _run(self, latents: torch.Tensor) -> torch.Tensor:
        if self._compiled_decoder is None:
            return self._decoder_module()(latents.to(self.device)).to("cpu")
        hidden_states = latents.to(self.device)
        for stage in self._compiled_decoder:
            hidden_states = stage(hidden_states)
        return hidden_states.to("cpu")

    def _decode_windows(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode fixed-size windows, spread over the world's ranks, and keep each core.

        A clip no longer than one window is decoded whole. Otherwise each rank decodes its share
        of the windows (stereo channels batched), the shares are gathered on the host and the
        cores are joined; the result is identical on every rank.
        """
        batch, _, num_frames = latents.shape
        if num_frames <= AUDIO_WINDOW:
            return self._run(latents)
        spans = audio_windows(num_frames)
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        per_rank = -(-len(spans) // world)
        mine = spans[rank * per_rank : (rank + 1) * per_rank]
        # Every rank runs a full share, padding it with copies of a real window, so all ranks
        # reach the (one-time, minutes-long) compile together instead of the idle ones timing
        # out in the gather below while the others compile.
        starts = [start for start, _, _ in mine] or [spans[0][0]]
        starts += [starts[-1]] * (per_rank - len(starts))
        batch_in = torch.cat([latents[..., a : a + AUDIO_WINDOW] for a in starts], dim=0)
        out = self._run(batch_in).reshape(len(starts), batch, 1, -1)
        decoded = {span: out[index] for index, span in enumerate(mine)}
        if world > 1:
            shares = [None] * world
            dist.all_gather_object(shares, decoded)
            decoded = {span: wave for share in shares if share for span, wave in share.items()}
        hop = decoded[spans[0]].shape[-1] // AUDIO_WINDOW
        pieces = [
            decoded[(start, lo, hi)][..., (lo - start) * hop : (hi - start) * hop]
            for start, lo, hi in spans
        ]
        return torch.cat(pieces, dim=-1)

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """``(latent - latents_mean) / latents_std``, per channel; identity if unset."""
        stats = self._latent_stats(latents)
        if stats is None:
            return latents
        mean, std = stats
        return (latents - mean) / std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """``latent * latents_std + latents_mean``, per channel; identity if unset."""
        stats = self._latent_stats(latents)
        if stats is None:
            return latents
        mean, std = stats
        return latents * std + mean

    def _latent_stats(self, latents: torch.Tensor):
        if self.latents_mean is None or self.latents_std is None:
            return None
        shape = (1, self.latent_channels, 1)
        mean = torch.tensor(self.latents_mean, device=latents.device, dtype=latents.dtype).view(shape)
        std = torch.tensor(self.latents_std, device=latents.device, dtype=latents.dtype).view(shape)
        return mean, std

    @staticmethod
    def to_interleaved_stereo(waveform: torch.Tensor) -> torch.Tensor:
        """``(2, 1, samples)`` mono-per-channel -> ``(2, samples)``.

        The decode boundary: MiniMax-H3's two stereo channels ride through the whole
        model as two batch items, and this is where they become a stereo waveform.
        """
        return waveform.squeeze(1)

    # ---------------------------------------------------------------
    # Weight loading
    # ---------------------------------------------------------------

    def _weight_mappings(self) -> dict[str, str | list[str]]:
        """Model parameter name -> checkpoint key(s).

        Module and parameter names are otherwise identical to the checkpoint, so the
        only entries are the weight-norm folds: every `_WeightNormConv1d` and
        `_WeightNormConvTranspose1d` weight reads its ``weight_g`` / ``weight_v`` pair
        (see `_weight_norm_loader`). Collected by walking the tree rather than
        enumerated, because the decoder holds ~130 of them.
        """
        mappings: dict[str, str | list[str]] = {}
        for name, module in self.named_modules():
            if isinstance(module, _WeightNormConv1d | _WeightNormConvTranspose1d):
                mappings[f"{name}.weight"] = [f"{name}.weight_g", f"{name}.weight_v"]
        return mappings

    def load_weights(
        self,
        model_name_or_path: str,
        device: torch.device = torch.device("cpu"),
        cache_dir: str | None = None,
    ) -> None:
        """Load the decoder to ``device``.

        Nothing here is sharded — see the module docstring — so this loads rank 0's
        (i.e. the whole) checkpoint on whichever rank runs the audio decode. Encoder
        keys in the checkpoint are ignored.

        ``dtype_override`` is what keeps this stack in float32. The loader casts each
        tensor to the *parameter's* dtype, and the parameters were built under
        `DiffusersPipelineLoader`'s ``set_default_torch_dtype(od_config.dtype)`` — i.e.
        bfloat16. Without the override the weights would be downcast and then cast back
        below, silently costing the precision this decoder is documented to need.
        """
        checkpoint = SafetensorsCheckpoint(model_name_or_path, cache_dir)
        dtype_override = {
            name: torch.float32
            for name, _ in list(self.named_parameters()) + list(self.named_buffers())
        }
        load_result = checkpoint.load_sharded_pipelined(
            0, 1, self, self._weight_mappings(), device, dtype_override=dtype_override
        )

        state_dict = {
            name: tensor.to(torch.float32) if tensor.dtype != torch.float32 else tensor
            for name, tensor in load_result.state_dict.items()
        }
        self.load_state_dict(state_dict, strict=False, assign=True)


__all__ = [
    "MiniMaxH3AudioBigVGANDecoder",
    "NeuronAutoencoderKLMiniMaxH3Audio",
    "kaiser_sinc_filter1d",
]
