# SPDX-License-Identifier: Apache-2.0
"""The conditioner's vision-tower blocks on every NeuronCore, the patch rows split over the world.

Qwen3-VL's vision tower is 27 identical ViT blocks over every patch of every image and video
reference — ~33K patches for one 2048-pixel image and a 1-second video, ~24 s on the host. The
blocks attend within each image / frame group only (block-diagonal, ``cu_seqlens``), which the
attention kernel's per-query KV bounds express exactly.

The rows are split over the world like the DiT's context parallelism: each rank projects and
runs the MLP for its own rows, and keys and values are all-gathered once per block. On one core
the whole 33K-row block is ~1M instructions and its DMA rings alone take ~3.5 GB of HBM, more
than is free beside the DiT; split 64 ways each rank's graph is small, at the cost of the
0.9 GB of weights replicated on every rank.

The host (rank 0) keeps the patch embedding, the interpolated position embeddings, the 2D rotary
tables and the patch mergers: cheap, and transformers' own code. Each block is its own graph
(they share one NEFF), launched with a one-element probe that retires it on the Lite runtime.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni_neuron.diffusion.models.minimax_h3.attention import _h3_nki_attention

#: Each rank's share of the patch rows is a multiple of this (the kernel's query tile).
ROW_TILE = 128
_MAX_IN_FLIGHT = 4


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _layer_norm(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    """LayerNorm accumulated in float32, as PyTorch's CPU kernel does for bfloat16 input."""
    return F.layer_norm(x.float(), norm.normalized_shape, norm.weight.float(), norm.bias.float(), norm.eps).to(x.dtype)


class _VisionBlock(nn.Module):
    """One Qwen3-VL vision block over this rank's rows; keys and values from every rank."""

    def __init__(self, hidden: int, heads: int, inter: int, group, world: int):
        super().__init__()
        self.heads, self.head_dim, self.group, self.world = heads, hidden // heads, group, world
        self.norm1 = nn.LayerNorm(hidden, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden, eps=1e-6)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.fc1 = nn.Linear(hidden, inter)
        self.fc2 = nn.Linear(inter, hidden)

    def forward(self, hidden, cos, sin, bound_min, bound_max):
        rows = hidden.shape[0]
        qkv = self.qkv(_layer_norm(hidden, self.norm1)).reshape(rows, 3, self.heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # As `apply_rotary_pos_emb_vision`: rotate in float32, back to the block's dtype.
        cos_, sin_ = cos[:, None, :], sin[:, None, :]
        q = (q.float() * cos_ + _rotate_half(q.float()) * sin_).to(hidden.dtype)
        k = (k.float() * cos_ + _rotate_half(k.float()) * sin_).to(hidden.dtype)
        key_value = torch.stack((k, v)).contiguous()
        if self.group is not None:
            # Rank order is row order, so the gathered rows are the whole (padded) sequence.
            key_value = self.group.all_gather(key_value, dim=1)
        q = (q * self.head_dim**-0.5).transpose(0, 1).contiguous()
        k = key_value[0].transpose(0, 1).contiguous()
        v = key_value[1].transpose(0, 1).contiguous()
        attn = _h3_nki_attention(q, k, v, bound_min, bound_max)
        hidden = hidden + self.proj(attn.transpose(0, 1).reshape(rows, -1))
        return hidden + self.fc2(F.gelu(self.fc1(_layer_norm(hidden, self.norm2)), approximate="tanh"))


class _GatherRows(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group

    def forward(self, x):
        return self.group.all_gather(x.contiguous(), dim=0)


class _Probed(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args):
        out = self.block(*args)
        return out, out.reshape(-1)[:1] * 1


class NeuronQwen3VLVisionBlocks(nn.Module):
    """The vision tower's 27 blocks, the patch rows split over ``group`` (a GroupCoordinator)."""

    def __init__(self, model_path: str, rank: int, world: int, group, subfolder: str = "text_encoder"):
        super().__init__()
        self.path = os.path.join(model_path, subfolder)
        config = json.load(open(os.path.join(self.path, "config.json")))["vision_config"]
        self.rank, self.world, self.group = rank, world, group
        self.heads = config["num_heads"]
        self.taps = tuple(config["deepstack_visual_indexes"])
        self.blocks = nn.ModuleList(
            _VisionBlock(config["hidden_size"], self.heads, config["intermediate_size"], group, world)
            for _ in range(config["depth"])
        )
        self._compiled = None
        self._gather = None

    @torch.no_grad()
    def load_weights(self, dtype=torch.bfloat16) -> None:
        from safetensors import safe_open

        index = json.load(open(os.path.join(self.path, "model.safetensors.index.json")))["weight_map"]
        handles: dict[str, object] = {}
        names = {
            "norm1": "norm1", "norm2": "norm2", "qkv": "attn.qkv", "proj": "attn.proj",
            "fc1": "mlp.linear_fc1", "fc2": "mlp.linear_fc2",
        }
        for number, block in enumerate(self.blocks):
            for attr, key in names.items():
                for param in ("weight", "bias"):
                    name = f"model.visual.blocks.{number}.{key}.{param}"
                    file = index[name]
                    if file not in handles:
                        handles[file] = safe_open(os.path.join(self.path, file), framework="pt")
                    getattr(getattr(block, attr), param).data = handles[file].get_tensor(name).to(dtype)

    def compile(self, compile_fn) -> None:
        import torch._dynamo.config as dynamo_config

        dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 4 * len(self.blocks))
        self._compiled = [compile_fn(_Probed(block)) for block in self.blocks]
        self._gather = compile_fn(_GatherRows(self.group)) if self.group is not None else None

    def _all_rows(self, local: torch.Tensor) -> torch.Tensor:
        """Every rank's rows of ``local``, on the host, in row order."""
        if self.group is None:
            return local.to("cpu")
        return self._gather(local).to("cpu")

    @torch.no_grad()
    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cu_seqlens: torch.Tensor):
        """Host ``(N, C)`` patches (identical on every rank) -> ``(final, {tap: hidden})`` on the host.

        ``cu_seqlens`` delimits the attention segments (one per image / frame group).
        """
        device = next(self.parameters()).device
        length = hidden.shape[0]
        unit = self.world * ROW_TILE
        bucket = -(-length // unit) * unit
        share = bucket // self.world
        starts = torch.empty(bucket, dtype=torch.int32)
        ends = torch.empty(bucket, dtype=torch.int32)
        edges = cu_seqlens.tolist()
        for start, end in zip(edges[:-1], edges[1:]):
            starts[start:end], ends[start:end] = start, end
        # Pad rows attend among themselves and are dropped.
        starts[length:], ends[length:] = length, bucket
        lo, hi = self.rank * share, (self.rank + 1) * share

        def mine(x, value=0.0):
            full = torch.cat([x, x.new_full((bucket - length, *x.shape[1:]), value)])
            return full[lo:hi].contiguous().to(device)

        bound_min = starts[lo:hi].view(1, share, 1).expand(self.heads, share, 1).contiguous().to(device)
        bound_max = ends[lo:hi].view(1, share, 1).expand(self.heads, share, 1).contiguous().to(device)
        x = mine(hidden)
        cos_d, sin_d = mine(cos.float(), 1.0), mine(sin.float())
        blocks = self._compiled or [_Probed(block) for block in self.blocks]
        taps, pending = {}, []
        for index, block in enumerate(blocks):
            x, probe = block(x, cos_d, sin_d, bound_min, bound_max)
            pending.append(probe)
            if index in self.taps:
                for probe in pending:
                    probe.to("cpu")
                pending.clear()
                taps[index] = self._all_rows(x)[:length]
            while len(pending) > _MAX_IN_FLIGHT:
                pending.pop(0).to("cpu")
        for probe in pending:
            probe.to("cpu")
        return self._all_rows(x)[:length], taps
