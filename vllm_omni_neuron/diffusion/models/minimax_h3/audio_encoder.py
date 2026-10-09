# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3's audio VAE encoder, on the host: ``ref2va`` soundtracks into conditioning rows.

The DAC-lineage waveform encoder, the causal-attention projection (``pre_block``) and the
``mean_proj`` head of ``diffusers``' ``AutoencoderKLMiniMaxH3Audio``. MiniMax-H3 conditions on
the posterior *mean* of a reference soundtrack, so ``logs_proj`` is not built. It runs once per
request on at most 15 s of stereo audio, which takes well under a second on the host, so it does
not get Neuron graphs of its own (the decoder half lives in
`..distributed.autoencoders.autoencoder_minimax_h3_audio`).

Float32 throughout, as the release keeps it; weight norm is folded at load time.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni_neuron.diffusion.models.minimax_h3.packing import MINIMAX_H3_AUDIO_CHANNELS


class _Snake1d(nn.Module):
    """``x + (alpha + 1e-9)^-1 * sin(alpha * x)^2`` with a per-channel ``(1, C, 1)`` alpha."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


class _ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            _Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=3 * dilation),
            _Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.block(x)
        pad = (x.shape[-1] - residual.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + residual


class _EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            _ResidualUnit(dim // 2, dilation=1),
            _ResidualUnit(dim // 2, dilation=3),
            _ResidualUnit(dim // 2, dilation=9),
            _Snake1d(dim // 2),
            nn.Conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Encoder(nn.Module):
    """``(B, 1, samples)`` -> ``(B, latent_dim, samples / prod(strides))``."""

    def __init__(self, d_model: int, strides: tuple[int, ...], d_latent: int):
        super().__init__()
        block: list[nn.Module] = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block.append(_EncoderBlock(d_model, stride))
        block += [_Snake1d(d_model), nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _GeGluMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.w2(F.gelu(self.w0(x), approximate="tanh") * self.w1(x))


class _CausalAttention(nn.Module):
    """Causal attention whose heads are mean-pooled away and whose head dimension is then
    adaptively average-pooled down to ``out_dim``; the key bias is a fixed zero."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(in_dim))
        self.v_bias = nn.Parameter(torch.zeros(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        bias = torch.cat((self.q_bias, torch.zeros_like(self.q_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias)
        query, key, value = qkv.view(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        out = out.mean(dim=1)
        return self.proj(F.adaptive_avg_pool1d(out, self.out_dim))


class _AttnProjection(nn.Module):
    """``pre_block``: narrows ``latent_dim`` to ``latent_channels``."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = _CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = _GeGluMlp(out_dim, out_dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MiniMaxH3AudioEncoder(nn.Module):
    """Waveform -> normalized audio conditioning rows, on the host in float32."""

    def __init__(self, config: dict):
        super().__init__()
        self.strides = tuple(int(rate) for rate in config.get("encoder_rates", (2, 4, 4, 5, 5)))
        self.hop_length = math.prod(self.strides)
        latent_dim = int(config.get("latent_dim", 2048))
        self.latent_channels = int(config.get("latent_channels", 32))
        self.sampling_rate = int(config.get("sampling_rate", 32000))
        self.encoder = _Encoder(int(config.get("encoder_dim", 64)), self.strides, latent_dim)
        self.pre_block = _AttnProjection(
            latent_dim, self.latent_channels, int(config.get("num_attention_heads", 8))
        )
        self.mean_proj = nn.Conv1d(self.latent_channels, self.latent_channels, 1)
        self.latents_mean = torch.tensor(config["latents_mean"], dtype=torch.float32)
        self.latents_std = torch.tensor(config["latents_std"], dtype=torch.float32)

    def load_weights(self, path: str) -> None:
        """Read the encoder half of the audio VAE checkpoint, folding every weight-norm pair."""
        from safetensors import safe_open

        prefixes = ("encoder.", "pre_block.", "mean_proj.")
        files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
        tensors = {}
        for name in files:
            with safe_open(os.path.join(path, name), framework="pt") as handle:
                for key in handle.keys():
                    if key.startswith(prefixes):
                        tensors[key] = handle.get_tensor(key).float()
        state = {}
        for key, tensor in tensors.items():
            if key.endswith(".weight_v"):
                stem = key[: -len("_v")]
                weight_g = tensors[stem + "_g"]
                norm = tensor.pow(2).sum(dim=tuple(range(1, tensor.ndim)), keepdim=True).sqrt()
                state[stem] = weight_g * tensor / norm
            elif not key.endswith((".weight_g", "zero_k_bias")):
                state[key] = tensor
        self.load_state_dict(state, strict=True)
        self.float().eval()

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """``(2, samples)`` stereo at the VAE's rate -> ``(2 * latents, latent_channels)`` rows.

        The two channels are two batch items of the mono VAE and the rows are channel-major,
        normalized per channel, as `diffusers`' ``MiniMaxH3Ref2VAReferenceEncoderStep`` packs them.
        """
        sample = waveform.to(torch.float32)[:, None]
        right_pad = -sample.shape[-1] % self.hop_length
        if right_pad:
            sample = F.pad(sample, (0, right_pad))
        hidden = self.encoder(sample)
        hidden = self.pre_block(hidden.transpose(1, 2)).transpose(1, 2)
        latents = self.mean_proj(hidden).transpose(1, 2)
        rows = (latents - self.latents_mean.view(1, 1, -1)) / self.latents_std.view(1, 1, -1)
        assert rows.shape[0] == MINIMAX_H3_AUDIO_CHANNELS
        return rows.reshape(-1, self.latent_channels)
