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

#: World size at which `device="auto"` runs the decoder layers on NeuronCores.
_NEURON_AUTO_WORLD = 64


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

    def __init__(
        self, model_path: str, subfolder: str = "text_encoder", dtype=torch.bfloat16, device: str = "cpu"
    ):
        self.model_path = model_path
        self.path = os.path.join(model_path, subfolder)
        self.dtype = dtype
        # `neuron`: the 50 decoder layers run on every NeuronCore (see `text_encoder_neuron`);
        # rank 0 keeps only the embedding, the vision tower and the rotary/DeepStack inputs.
        # `auto` picks `neuron` on a 64-rank world (validated: ~0.8 GB/core) and the host
        # otherwise.
        self.device = device
        self.neuron_layers = None
        self.neuron_vision = None
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
        if self.device == "auto":
            world_size = _world().world_size if dist.is_initialized() else 1
            self.device = "neuron" if world_size == _NEURON_AUTO_WORLD else "cpu"
        if self.device == "neuron":
            from vllm_omni_neuron.diffusion.models.minimax_h3.text_encoder_neuron import (
                NeuronQwen3VLTextLayers,
            )

            world = _world() if dist.is_initialized() else None
            group = world.device_group if world is not None and world.world_size > 1 else None
            if group is not None:
                from vllm_omni_neuron.lite_compat import register_process_group_replica_groups

                register_process_group_replica_groups(group.group_name, [list(world.ranks)])
            self.neuron_layers = NeuronQwen3VLTextLayers(
                self.model_path,
                world.rank_in_group if world is not None else 0,
                world.world_size if world is not None else 1,
                group,
            )
            self.neuron_layers.load_weights(self.dtype)
            from vllm_omni_neuron.diffusion.models.minimax_h3.vision_neuron import (
                NeuronQwen3VLVisionBlocks,
            )

            self.neuron_vision = NeuronQwen3VLVisionBlocks(
                self.model_path,
                world.rank_in_group if world is not None else 0,
                world.world_size if world is not None else 1,
                world if group is not None else None,
            )
            self.neuron_vision.load_weights(self.dtype)
        if not self._is_owner:
            return

        from transformers import Qwen3VLForConditionalGeneration

        # On Neuron the host needs no decoder layer at all: the embedding, the vision tower and
        # the rotary embedding are what produce the layers' inputs.
        config.text_config.num_hidden_layers = 0 if self.device == "neuron" else TEXT_ENCODER_LAYER + 1
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

    @torch.no_grad()
    def _layer_inputs(self, token_ids: list[int], vision_inputs: dict | None):
        """Rank 0: what the language model's decoder layers receive, captured from transformers.

        ``(inputs_embeds (S, H), cos (S, D), sin (S, D), [DeepStack (S, H) per layer])`` — the
        embeddings with the vision features merged in, the 3D (mRoPE) rotary tables, and each
        DeepStack feature scattered onto the vision rows (zeros elsewhere).
        """

        class _Captured(Exception):
            pass

        language_model = self.model.model.language_model
        captured = {}

        def capture(*args, **kwargs):
            captured.update(kwargs)
            raise _Captured

        with host_threads():
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            vision_inputs = dict(vision_inputs or {})
            mm_token_type_ids = (input_ids == self.image_token_id).long()
            mm_token_type_ids[input_ids == self.video_token_id] = 2
            for name in ("pixel_values", "pixel_values_videos"):
                if name in vision_inputs:
                    vision_inputs[name] = vision_inputs[name].to(self.dtype)
            original = language_model.forward
            language_model.forward = capture
            try:
                self.model.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    mm_token_type_ids=mm_token_type_ids,
                    use_cache=False,
                    **vision_inputs,
                )
            except _Captured:
                pass
            finally:
                language_model.forward = original
            embeds = captured["inputs_embeds"]
            position_ids = captured.get("position_ids")
            # As `Qwen3VLTextModel.forward`: text-only input gets plain positions on all three
            # mRoPE axes, and a leading text-position row is dropped.
            if position_ids is None:
                position_ids = torch.arange(embeds.shape[1]).view(1, 1, -1).expand(3, embeds.shape[0], -1)
            elif position_ids.ndim == 2:
                position_ids = position_ids[None].expand(3, -1, -1)
            if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                position_ids = position_ids[1:]
            cos, sin = language_model.rotary_emb(embeds, position_ids)
            deepstack = []
            masks = captured.get("visual_pos_masks")
            for features in captured.get("deepstack_visual_embeds") or []:
                dense = torch.zeros_like(embeds[0])
                dense[masks[0]] = features.to(dense.dtype)
                deepstack.append(dense)
        return embeds[0], cos[0], sin[0], deepstack

    def compile(self, compile_fn) -> None:
        """Compile the on-device decoder layers and vision blocks."""
        if self.neuron_layers is not None:
            self.neuron_layers.compile(compile_fn)
        if self.neuron_vision is not None:
            self.neuron_vision.compile(compile_fn)

    @torch.no_grad()
    def _vision_block_inputs(self, vision_inputs: dict) -> list[tuple]:
        """Rank 0: each modality's vision-block inputs, as `Qwen3VLVisionModel.forward` builds them.

        ``[(grid_thw, hidden, cos, sin, cu_seqlens), ...]`` for the images, then the videos —
        the order transformers calls the vision tower in.
        """
        from transformers.vision_utils import (
            get_vision_attention_seqlens,
            get_vision_interpolation_indices_and_weights,
            get_vision_position_ids,
        )

        visual = self.model.model.visual
        prepared = []
        with host_threads():
            for pixels_key, grid_key in (("pixel_values", "image_grid_thw"), ("pixel_values_videos", "video_grid_thw")):
                if pixels_key not in vision_inputs:
                    continue
                grid_thw = vision_inputs[grid_key]
                indices, weights = get_vision_interpolation_indices_and_weights(
                    grid_thw,
                    num_grid_per_side=visual.num_grid_per_side,
                    mode=visual.interpolation_mode,
                    align_corners=visual.interpolation_align_corners,
                    spatial_merge_size=visual.config.spatial_merge_size,
                )
                position_ids = get_vision_position_ids(grid_thw, visual.spatial_merge_size)
                cu_seqlens, _ = get_vision_attention_seqlens(grid_thw, visual.config)
                hidden = visual.patch_embed(vision_inputs[pixels_key].to(self.dtype))
                hidden = hidden + (visual.pos_embed(indices) * weights[:, :, None]).sum(1).to(hidden.dtype)
                rotary = visual.rotary_pos_emb(position_ids).reshape(hidden.shape[0], -1)
                emb = torch.cat((rotary, rotary), dim=-1)
                prepared.append((grid_thw, hidden, emb.cos(), emb.sin(), cu_seqlens))
        return prepared

    def _run_vision_blocks(self, prepared: list[tuple]) -> list[tuple]:
        """Every rank: one pass of the vision blocks over all modalities' patches.

        Returns ``[(grid_thw, final, taps), ...]`` per modality, on the host.
        """
        offsets, hiddens, coss, sins, edges = [0], [], [], [], [0]
        for _, hidden, cos, sin, cu_seqlens in prepared:
            hiddens.append(hidden)
            coss.append(cos)
            sins.append(sin)
            edges.extend((cu_seqlens[1:] + offsets[-1]).tolist())
            offsets.append(offsets[-1] + hidden.shape[0])
        final, taps = self.neuron_vision(
            torch.cat(hiddens), torch.cat(coss), torch.cat(sins), torch.tensor(edges, dtype=torch.int32)
        )
        results = []
        for (grid_thw, *_), start, end in zip(prepared, offsets[:-1], offsets[1:]):
            results.append((grid_thw, final[start:end], {index: tap[start:end] for index, tap in taps.items()}))
        return results

    def _served_vision_forward(self, results: list[tuple]):
        """A `Qwen3VLVisionModel.forward` that returns the blocks' precomputed results."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            BaseModelOutputWithDeepstackFeatures,
        )

        visual = self.model.model.visual

        def forward(hidden_states, grid_thw, **kwargs):
            for grid, final, taps in results:
                if grid.shape == grid_thw.shape and torch.equal(grid, grid_thw):
                    break
            else:
                raise RuntimeError("The vision tower was called on inputs it was not prepared for.")
            deepstack = [
                visual.deepstack_merger_list[position](taps[index])
                for position, index in enumerate(visual.deepstack_visual_indexes)
            ]
            return BaseModelOutputWithDeepstackFeatures(
                last_hidden_state=final, pooler_output=visual.merger(final), deepstack_features=deepstack
            )

        return forward

    def _encode_on_neuron(self, token_ids: list[int], vision_inputs: dict | None) -> torch.Tensor:
        from vllm_omni_neuron.diffusion.models.minimax_h3.text_encoder_neuron import TEXT_BUCKET

        # The vision blocks run on every rank, so their inputs are broadcast first.
        on_neuron = self.neuron_vision is not None and os.environ.get("MINIMAX_H3_VISION_ON_HOST") != "1"
        prepared = None
        if on_neuron and self._is_owner and vision_inputs:
            prepared = self._vision_block_inputs(vision_inputs)
        prepared = broadcast_from_owner(prepared) if on_neuron else None
        visual, original = None, None
        if prepared:
            results = self._run_vision_blocks(prepared)
            if self._is_owner:
                visual = self.model.model.visual
                original = visual.forward
                visual.forward = self._served_vision_forward(results)
        try:
            inputs = self._layer_inputs(token_ids, vision_inputs) if self._is_owner else None
        finally:
            if visual is not None:
                visual.forward = original
        embeds, cos, sin, deepstack = broadcast_from_owner(inputs)
        length = embeds.shape[0]
        bucket = -(-length // TEXT_BUCKET) * TEXT_BUCKET

        def padded(x):
            # Right padding: the attention is causal, so no real row sees it.
            pad = x.new_zeros((bucket - length, *x.shape[1:]))
            return torch.cat([x, pad])[None].to(self._neuron_device())

        hidden = self.neuron_layers(
            padded(embeds), padded(cos)[0], padded(sin)[0], [padded(d) for d in deepstack]
        )
        return hidden.to("cpu")[:, :length].to(self.dtype).contiguous()

    def _neuron_device(self):
        return next(self.neuron_layers.parameters()).device

    def encode(self, token_ids: list[int], vision_inputs: dict | None = None) -> torch.Tensor:
        """``(1, len(token_ids), hidden_size)`` conditioning on the host, identical on every rank.

        ``vision_inputs`` are the conditioner's own image inputs (``pixel_values``,
        ``image_grid_thw``) for the vision blocks ``token_ids`` contains.
        """
        if self.neuron_layers is not None:
            return self._encode_on_neuron(token_ids, vision_inputs)
        if self._is_owner:
            embeds = self._forward(token_ids, vision_inputs)
        else:
            embeds = torch.empty((1, len(token_ids), self.hidden_size), dtype=self.dtype)
        if dist.is_initialized() and _world().world_size > 1:
            dist.broadcast(embeds, src=_world().ranks[0], group=_world().cpu_group)
        return embeds
