# SPDX-License-Identifier: Apache-2.0
"""The conditioner's decoder layers on NeuronCores, sharded over the whole world.

MiniMax-H3 conditions on ``hidden_states[50]`` of a Qwen3-VL-32B language model, i.e. the
output of its first 50 decoder layers before the final norm. On the host those layers take
~87 s for a reference-sized (~8K-token) ``ref2va`` presentation. Here they run on every
NeuronCore: query heads, the MLP width and (replicated per group of ranks) the KV heads are
split over the world, with two all-reduces per layer. At 64 ranks the weights are ~0.8 GB
per core, which fits beside the DiT; one whole-world group because the 26B layers do not
fit in a TP=8 group's HBM headroom (~4 GB/core at a 1344x768 ``ref2va`` request).

The host keeps what is cheap or data-dependent: the token embedding, the vision tower, the
3D rotary positions and DeepStack features (see `text_encoder`). Each layer is its own graph
(they are identical, so they share one NEFF; the three DeepStack layers are a second one),
launched with a one-element probe that retires it on the Lite runtime.

The prompt is right-padded to a multiple of `TEXT_BUCKET`: the attention is causal, so no
real row sees the padding and the first rows of the output are exact.
"""

from __future__ import annotations

import json
import os

import nki
import nki.language as nl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from nkilib.core.attention.attention_cte import attention_cte

from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import _kernel_lnc
from vllm_omni_neuron.lite_compat import nki_op

#: Prompt rows are padded to a multiple of this, so every prompt up to it shares one graph.
TEXT_BUCKET = 512
#: Layers whose output conditions the DiT (``hidden_states[50]``).
NUM_LAYERS = 50
_MAX_IN_FLIGHT = 4


@nki.jit
def _causal_attention_kernel(q, k, v):
    """Causal flash attention, token-major ``[BN, S, D]`` in and out; ``q`` is pre-scaled."""
    return attention_cte(
        q=q,
        k=k,
        v=v,
        scale=1.0,
        causal_mask=True,
        tp_q=True,
        tp_k=True,
        tp_out=False,
        cache_softmax=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
    )


@nki_op("minimax_h3_text_encoder::causal_attention")
def _causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(_causal_attention_kernel)[_kernel_lnc()](q, k, v)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3's RMSNorm: normalize in float32, cast back, then scale."""
    dtype = x.dtype
    x32 = x.to(torch.float32)
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _DecoderLayer(nn.Module):
    """One Qwen3 decoder layer, this rank's shard of it."""

    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, inter: int, eps: float, group):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim, self.eps, self.group = heads, kv_heads, head_dim, eps, group
        self.input_norm = nn.Parameter(torch.empty(hidden))
        self.post_norm = nn.Parameter(torch.empty(hidden))
        self.q_weight = nn.Parameter(torch.empty(hidden, heads * head_dim))
        self.k_weight = nn.Parameter(torch.empty(hidden, kv_heads * head_dim))
        self.v_weight = nn.Parameter(torch.empty(hidden, kv_heads * head_dim))
        self.q_norm = nn.Parameter(torch.empty(head_dim))
        self.k_norm = nn.Parameter(torch.empty(head_dim))
        self.o_weight = nn.Parameter(torch.empty(heads * head_dim, hidden))
        self.gate_weight = nn.Parameter(torch.empty(hidden, inter))
        self.up_weight = nn.Parameter(torch.empty(hidden, inter))
        self.down_weight = nn.Parameter(torch.empty(inter, hidden))

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.group is not None:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.group)
        return x

    def forward(self, hidden, cos, sin, deepstack=None):
        seq = hidden.shape[1]
        x = _rms_norm(hidden, self.input_norm, self.eps)
        q = torch.matmul(x, self.q_weight).view(1, seq, self.heads, self.head_dim)
        k = torch.matmul(x, self.k_weight).view(1, seq, self.kv_heads, self.head_dim)
        v = torch.matmul(x, self.v_weight).view(1, seq, self.kv_heads, self.head_dim)
        q = _rms_norm(q, self.q_norm, self.eps)
        k = _rms_norm(k, self.k_norm, self.eps)
        cos_, sin_ = cos[None, :, None, :], sin[None, :, None, :]
        q = q * cos_ + _rotate_half(q) * sin_
        k = k * cos_ + _rotate_half(k) * sin_
        # GQA: each KV head serves `heads // kv_heads` query heads; heads-first for the kernel.
        group = self.heads // self.kv_heads
        q = (q * self.head_dim**-0.5).transpose(1, 2).reshape(self.heads, seq, self.head_dim)
        k = k.transpose(1, 2).repeat_interleave(group, dim=1).reshape(self.heads, seq, self.head_dim)
        v = v.transpose(1, 2).repeat_interleave(group, dim=1).reshape(self.heads, seq, self.head_dim)
        attn = _causal_attention(q.contiguous(), k.contiguous(), v.contiguous())
        attn = attn.reshape(1, self.heads, seq, self.head_dim).transpose(1, 2).reshape(1, seq, -1)
        hidden = hidden + self._all_reduce(torch.matmul(attn, self.o_weight))
        x = _rms_norm(hidden, self.post_norm, self.eps)
        mlp = torch.matmul(F.silu(torch.matmul(x, self.gate_weight)) * torch.matmul(x, self.up_weight), self.down_weight)
        hidden = hidden + self._all_reduce(mlp)
        if deepstack is not None:
            # DeepStack: visual features added at the vision rows (zeros elsewhere).
            hidden = hidden + deepstack
        return hidden


class _Probed(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args):
        out = self.layer(*args)
        return out, out.reshape(-1)[:1] * 1


class NeuronQwen3VLTextLayers(nn.Module):
    """The first `NUM_LAYERS` decoder layers of the conditioner, sharded over ``group``."""

    def __init__(self, model_path: str, rank: int, world: int, group, subfolder: str = "text_encoder"):
        super().__init__()
        self.path = os.path.join(model_path, subfolder)
        config = json.load(open(os.path.join(self.path, "config.json")))["text_config"]
        heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
        self.head_dim = config.get("head_dim") or config["hidden_size"] // heads
        if heads % world or config["intermediate_size"] % world:
            raise ValueError(f"The conditioner's {heads} heads / MLP do not split over {world} ranks.")
        self.rank, self.world = rank, world
        self.heads_per_rank = heads // world
        # Below 8 ranks a rank holds several KV heads; above, groups of ranks share one.
        self.kv_per_rank = max(1, kv_heads // world)
        self.kv_start = (rank * kv_heads) // world if world >= kv_heads else rank * self.kv_per_rank
        self.inter = config["intermediate_size"] // world
        self.layers = nn.ModuleList(
            _DecoderLayer(
                config["hidden_size"], self.heads_per_rank, self.kv_per_rank, self.head_dim, self.inter,
                config["rms_norm_eps"], group,
            )
            for _ in range(NUM_LAYERS)
        )
        self._compiled = None

    @torch.no_grad()
    def load_weights(self, dtype=torch.bfloat16) -> None:
        """Read this rank's slice of every layer straight from the safetensors shards."""
        from safetensors import safe_open

        index = json.load(open(os.path.join(self.path, "model.safetensors.index.json")))["weight_map"]
        handles: dict[str, object] = {}

        def tensor(name: str, rows=None, cols=None) -> torch.Tensor:
            file = index[name]
            if file not in handles:
                handles[file] = safe_open(os.path.join(self.path, file), framework="pt")
            sl = handles[file].get_slice(name)
            if rows is not None:
                return sl[rows[0] : rows[1]]
            if cols is not None:
                return sl[:, cols[0] : cols[1]]
            return sl[:]

        d, q0, k0, i0 = self.head_dim, self.rank * self.heads_per_rank, self.kv_start, self.rank * self.inter
        for index_, layer in enumerate(self.layers):
            prefix = f"model.language_model.layers.{index_}."
            values = {
                "input_norm": tensor(prefix + "input_layernorm.weight"),
                "post_norm": tensor(prefix + "post_attention_layernorm.weight"),
                "q_norm": tensor(prefix + "self_attn.q_norm.weight"),
                "k_norm": tensor(prefix + "self_attn.k_norm.weight"),
                "q_weight": tensor(prefix + "self_attn.q_proj.weight", rows=(q0 * d, (q0 + self.heads_per_rank) * d)).T,
                "k_weight": tensor(prefix + "self_attn.k_proj.weight", rows=(k0 * d, (k0 + self.kv_per_rank) * d)).T,
                "v_weight": tensor(prefix + "self_attn.v_proj.weight", rows=(k0 * d, (k0 + self.kv_per_rank) * d)).T,
                "o_weight": tensor(prefix + "self_attn.o_proj.weight", cols=(q0 * d, (q0 + self.heads_per_rank) * d)).T,
                "gate_weight": tensor(prefix + "mlp.gate_proj.weight", rows=(i0, i0 + self.inter)).T,
                "up_weight": tensor(prefix + "mlp.up_proj.weight", rows=(i0, i0 + self.inter)).T,
                "down_weight": tensor(prefix + "mlp.down_proj.weight", cols=(i0, i0 + self.inter)).T,
            }
            for name, value in values.items():
                getattr(layer, name).data = value.to(dtype).contiguous()

    def compile(self, compile_fn) -> None:
        import torch._dynamo.config as dynamo_config

        dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 4 * NUM_LAYERS)
        self._compiled = [compile_fn(_Probed(layer)) for layer in self.layers]

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, deepstack: list[torch.Tensor]):
        """``(1, S, hidden)`` -> ``hidden_states[50]``; ``deepstack[i]`` is added after layer ``i``."""
        layers = self._compiled or [_Probed(layer) for layer in self.layers]
        pending = []
        for index, layer in enumerate(layers):
            if index < len(deepstack):
                hidden, probe = layer(hidden, cos, sin, deepstack[index])
            else:
                hidden, probe = layer(hidden, cos, sin)
            pending.append(probe)
            while len(pending) > _MAX_IN_FLIGHT:
                pending.pop(0).to("cpu")
        for probe in pending:
            probe.to("cpu")
        return hidden
