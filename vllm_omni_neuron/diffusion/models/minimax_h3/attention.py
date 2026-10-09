# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 self-attention cores: local flash attention, and K/V-gathered attention under CP.

Every attention call takes the key interval ``[key_start, key_end)``. The packed sequence holds
rows that must not act as keys — the prompt's left padding up to its length bucket, and the rows
that even out the CP shards at the end — and H3's attention otherwise has no mask. The kernel's
per-query KV bounds (``bound_min`` / ``bound_max``) exclude them exactly, so the gathered keys keep
their aligned length; ``key_start`` is a runtime tensor, so one graph serves every prompt length in
a bucket.

Context parallelism keeps each rank's query rows local and all-gathers keys and values, so every
rank attends its shard of queries to the whole sequence with the online-softmax ``attention_cte``
kernel. Keys and values travel in one collective rather than two: on Trn2 a collective's cost is
dominated by a fixed per-call overhead, not by its bytes.

The plugin's const-max ring kernel is deliberately not used. It replaces the running softmax max
with the per-row bound ``scale * |q| * max|k|``, which its own docstring notes silently zeroes rows
once the bound exceeds the true max by ~85 nats. H3's learned QK-norm scales put the bound far
past that in its late blocks (the last block's ``norm_q`` reaches 33.75, a bound of ~300 nats);
measured at 704x384, ring CP=4 lands 39% (relative L2) away from an fp32 reference against 7% for
this path, which is within the 9% a bf16 run of the reference itself shows.
"""

from __future__ import annotations

import nki
import nki.language as nl
import torch
from nkilib.core.attention.attention_cte import attention_cte

from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import (
    _kernel_lnc,
    can_run_kernel,
)
from vllm_omni_neuron.lite_compat import nki_op

# attention_cte tiling limits; see `wan2_2_transformer._can_use_wan_attention_kernel`.
_ATTN_MAX_BS = 512
_ATTN_MAX_SEQLEN = 131072
_ATTN_MAX_HEAD_DIM = 128


@nki.jit
def _h3_attention_kernel(q, k, v, bound_min, bound_max):
    """Non-causal flash attention over the keys ``[bound_min, bound_max)`` of every query.

    Token-major in and out: ``[BN, S, D]`` -> ``[BN, S, D]``; the bounds are ``[BN, S_q, 1]``.
    """
    return attention_cte(
        q=q,
        k=k,
        v=v,
        scale=1.0,
        causal_mask=False,
        tp_q=True,
        tp_k=True,
        tp_out=False,
        cache_softmax=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
        bound_min=bound_min,
        bound_max=bound_max,
    )


@nki_op("minimax_h3_transformer::attention_cte")
def _h3_nki_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bound_min: torch.Tensor,
    bound_max: torch.Tensor,
) -> torch.Tensor:
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(_h3_attention_kernel)[_kernel_lnc()](q, k, v, bound_min, bound_max)


def _can_use_attention_kernel(query: torch.Tensor, key: torch.Tensor) -> bool:
    if not can_run_kernel(query):
        return False
    B, S_q, N, D = query.shape
    return (
        B * N <= _ATTN_MAX_BS
        and max(S_q, key.shape[1]) <= _ATTN_MAX_SEQLEN
        and D <= _ATTN_MAX_HEAD_DIM
        # The KV bounds are read in 128-query tiles (`NCC_IBIR243` otherwise); the DiT pads the
        # refiner's rows and the CP shards to the tile, so in practice nothing falls back.
        and S_q % 128 == 0
    )


def _torch_attend(query, key, value, scale, key_start=None, key_end=None):
    """``softmax(scale Q K^T) V`` over the keys ``[key_start, key_end)``; token-major ``[B, S, N, D]``."""
    q, k, v = (t.transpose(1, 2).float() for t in (query, key, value))
    scores = torch.matmul(q * scale, k.transpose(-2, -1))
    if key_start is not None:
        index = torch.arange(key.shape[1], device=scores.device)
        keep = (index >= key_start) & (index < key_end)
        scores = scores.masked_fill(~keep, float("-inf"))
    out = torch.matmul(torch.softmax(scores, dim=-1), v)
    return out.transpose(1, 2).to(query.dtype)


def local_attention(query, key, value, scale, key_start: torch.Tensor, key_end: int):
    """Attention of ``query`` over the rows ``[key_start, key_end)`` of ``key``/``value``.

    Token-major ``[B, S, N, D]`` in and out; ``key_start`` is a ``(1,)`` int32 tensor.
    """
    if not _can_use_attention_kernel(query, key):
        return _torch_attend(query, key, value, scale, key_start, key_end)
    B, S, N, D = query.shape
    S_k = key.shape[1]

    def heads_first(t, length):
        return t.transpose(1, 2).reshape(B * N, length, D).contiguous()

    bound_min = key_start.to(torch.int32).view(1, 1, 1).expand(B * N, S, 1).contiguous()
    bound_max = torch.full_like(bound_min, key_end)
    out = _h3_nki_attention(
        heads_first(query * scale, S),
        heads_first(key, S_k),
        heads_first(value, S_k),
        bound_min,
        bound_max,
    )
    return out.reshape(B, N, S, D).transpose(1, 2)


def context_parallel_attention(query, key, value, scale, cp_group, key_start: torch.Tensor, key_end: int):
    """Attention of one CP shard's queries over the whole, gathered sequence.

    Token-major ``[B, local_S, N, D]`` in and out. Only the gathered rows ``[key_start, key_end)``
    act as keys.
    """
    key_value = cp_group.all_gather(torch.stack((key, value)).contiguous(), dim=2)
    return local_attention(query, key_value[0], key_value[1], scale, key_start, key_end)
