# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3's Qwen3-VL conditioner, run on the host by rank 0 and broadcast.

The conditioner is a 32B model read once per request; the DiT it conditions runs 30-50 steps of
a 33B model. Holding it on the NeuronCores would cost ~6 GiB of HBM per rank and its own set of
compiled graphs (one per prompt-length bucket) to save the ~2 s it takes on the host, so it stays
on the host: rank 0 owns the weights and every other rank receives the embeddings over the
world's CPU group.

The conditioning is ``hidden_states[50]`` of the language model, i.e. the output of decoder layer
50 *before* the final norm, for the prompt tokenized verbatim (no chat template, no special
tokens), as `diffusers`' ``get_qwen3vl_prompt_embeds`` reads it. Only the first 51 of the 64
decoder layers are built: 51 rather than 50 because the last hidden state of a stack is post-norm.
"""

from __future__ import annotations

import logging
import os
import time

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

#: Decoder layer whose output conditions the DiT.
TEXT_ENCODER_LAYER = 50

#: Host threads rank 0 uses while encoding. The Lite worker pins every rank to one thread, which
#: takes the 50-layer forward from ~2 s to ~23 s; the other ranks are idle in the broadcast.
_ENCODE_THREADS_ENV = "MINIMAX_H3_TEXT_ENCODER_THREADS"


def _world():
    from vllm_omni.diffusion.distributed.parallel_state import get_world_group

    return get_world_group()


class MiniMaxH3TextEncoder:
    """Owns the conditioner on rank 0; every rank calls `encode` and gets the embeddings."""

    def __init__(self, model_path: str, subfolder: str = "text_encoder", dtype=torch.bfloat16):
        self.path = os.path.join(model_path, subfolder)
        self.dtype = dtype
        self.model = None
        self.hidden_size = None

    @property
    def _is_owner(self) -> bool:
        return not dist.is_initialized() or _world().rank_in_group == 0

    def load_weights(self) -> None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self.path)
        self.hidden_size = config.text_config.hidden_size
        if not self._is_owner:
            return

        from transformers import Qwen3VLForConditionalGeneration

        config.text_config.num_hidden_layers = TEXT_ENCODER_LAYER + 1
        started = time.perf_counter()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.path, config=config, dtype=self.dtype, device_map="cpu"
        ).eval()
        logger.info("MiniMax-H3 conditioner loaded on the host in %.1f s", time.perf_counter() - started)

    @torch.no_grad()
    def _forward(self, token_ids: list[int]) -> torch.Tensor:
        threads = int(os.environ.get(_ENCODE_THREADS_ENV, "0")) or max(1, (os.cpu_count() or 2) // 2)
        previous = torch.get_num_threads()
        torch.set_num_threads(threads)
        try:
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                # Qwen-internal modality ids; a t2va presentation is all text.
                mm_token_type_ids=torch.zeros_like(input_ids),
                use_cache=False,
                output_hidden_states=True,
            )
            return outputs.hidden_states[TEXT_ENCODER_LAYER].to(self.dtype).contiguous()
        finally:
            torch.set_num_threads(previous)

    def encode(self, token_ids: list[int]) -> torch.Tensor:
        """``(1, len(token_ids), hidden_size)`` conditioning on the host, identical on every rank."""
        if self._is_owner:
            embeds = self._forward(token_ids)
        else:
            embeds = torch.empty((1, len(token_ids), self.hidden_size), dtype=self.dtype)
        if dist.is_initialized() and _world().world_size > 1:
            dist.broadcast(embeds, src=_world().ranks[0], group=_world().cpu_group)
        return embeds
