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

import contextlib
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


@contextlib.contextmanager
def host_threads():
    """Lift the Lite worker's one-thread pin while rank 0 runs host-side models."""
    threads = int(os.environ.get(_ENCODE_THREADS_ENV, "0")) or max(1, (os.cpu_count() or 2) // 2)
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def broadcast_from_owner(value):
    """Rank 0's picklable ``value`` on every rank, over the world's CPU group."""
    if not dist.is_initialized() or _world().world_size == 1:
        return value
    holder = [value]
    dist.broadcast_object_list(holder, src=_world().ranks[0], group=_world().cpu_group)
    return holder[0]


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
        self.image_token_id = None
        self.video_token_id = None

    @property
    def _is_owner(self) -> bool:
        return not dist.is_initialized() or _world().rank_in_group == 0

    def load_weights(self) -> None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self.path)
        self.hidden_size = config.text_config.hidden_size
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
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
    def _forward(self, token_ids: list[int], vision_inputs: dict | None = None) -> torch.Tensor:
        with host_threads():
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            vision_inputs = dict(vision_inputs or {})
            # Qwen-internal modality ids (`0` text, `1` image, `2` video), which drive its
            # per-modality rotary layout; not MiniMax-H3's own row tags.
            mm_token_type_ids = (input_ids == self.image_token_id).long()
            mm_token_type_ids[input_ids == self.video_token_id] = 2
            for name in ("pixel_values", "pixel_values_videos"):
                if name in vision_inputs:
                    vision_inputs[name] = vision_inputs[name].to(self.dtype)
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
                output_hidden_states=True,
                **vision_inputs,
            )
            return outputs.hidden_states[TEXT_ENCODER_LAYER].to(self.dtype).contiguous()

    def encode(self, token_ids: list[int], vision_inputs: dict | None = None) -> torch.Tensor:
        """``(1, len(token_ids), hidden_size)`` conditioning on the host, identical on every rank.

        ``vision_inputs`` are the conditioner's own image inputs (``pixel_values``,
        ``image_grid_thw``) for the vision blocks ``token_ids`` contains.
        """
        if self._is_owner:
            embeds = self._forward(token_ids, vision_inputs)
        else:
            embeds = torch.empty((1, len(token_ids), self.hidden_size), dtype=self.dtype)
        if dist.is_initialized() and _world().world_size > 1:
            dist.broadcast(embeds, src=_world().ranks[0], group=_world().cpu_group)
        return embeds
