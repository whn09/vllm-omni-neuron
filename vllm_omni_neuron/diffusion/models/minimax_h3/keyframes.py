# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 ``fl2va``: first- and/or last-keyframe conditioning, on the host.

Mirrors the diffusers ``MiniMaxH3ResizeStep``, ``MiniMaxH3FL2VATextEncoderStep`` and
``encode_vae_condition``: every arithmetic choice here (the stretch, the cover crop's rounding and
centring, the posterior sample's seed and float16 rounding) is the released model's, because the
conditioning latents are reproduced only if they match it.
"""

from __future__ import annotations

import torch
from PIL import Image

from vllm_omni_neuron.diffusion.models.minimax_h3.packing import (
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    resolve_canvas_size,
)

#: Seed the keyframe posterior is sampled under, independently of the request's generator.
KEYFRAME_ENCODE_SEED = 42


def collect_keyframes(image, last_image) -> tuple[list[Image.Image], tuple[str, ...]]:
    """The keyframes in packed order and the end of the video each one is anchored to."""
    pairs = [(frame, anchor) for frame, anchor in ((image, "first"), (last_image, "last")) if frame is not None]
    return [frame.convert("RGB") for frame, _ in pairs], tuple(anchor for _, anchor in pairs)


def resolve_keyframe_canvas(keyframes, height: int | None, width: int | None) -> tuple[int, int]:
    """The canvas: MiniMax-H3's own geometry for the first keyframe's aspect ratio, unless given."""
    if (height is None) != (width is None):
        raise ValueError("`height` and `width` have to be passed together, or neither of them.")
    if height is not None:
        return height, width
    return resolve_canvas_size(*keyframes[0].size)


def fit_keyframes(keyframes, height: int, width: int) -> list[Image.Image]:
    """Put the keyframes onto the canvas: the first is stretched, a second is cover-cropped."""
    fitted = []
    for index, keyframe in enumerate(keyframes):
        if keyframe.size == (width, height):
            fitted.append(keyframe)
        elif index == 0:
            fitted.append(keyframe.resize((width, height), Image.Resampling.LANCZOS))
        else:
            scale = max(width / keyframe.size[0], height / keyframe.size[1])
            size = (max(width, round(keyframe.size[0] * scale)), max(height, round(keyframe.size[1] * scale)))
            left = max(0, (size[0] - width) // 2)
            top = max(0, (size[1] - height) // 2)
            resized = keyframe.resize(size, Image.Resampling.LANCZOS)
            fitted.append(resized.crop((left, top, left + width, top + height)))
    return fitted


def keyframe_presentation(tokenizer, image_processor, keyframes, prompt: str):
    """Tokenize ``fl2va``'s presentation: a ``"<Picture i>: "`` label and a vision block per
    keyframe, then the prompt verbatim — no chat template, no special tokens.

    Returns ``(token_ids, token_tags, vision_inputs)``; a vision block's rows are tagged as video.
    """
    vision_inputs, grid = {}, None
    if keyframes:
        vision = image_processor(images=keyframes, return_tensors="pt")
        grid = vision["image_grid_thw"]
        vision_inputs = {"pixel_values": vision["pixel_values"], "image_grid_thw": grid}

    token_ids: list[int] = []
    token_tags: list[int] = []
    merge = image_processor.merge_size**2
    for index in range(len(keyframes)):
        label = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
        block = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * (int(grid[index].prod()) // merge)
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        token_ids += label + block
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(label) + [MINIMAX_H3_VIDEO_TAG] * len(block)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_ids += prompt_ids
    token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)
    return token_ids, torch.tensor(token_tags, dtype=torch.long), vision_inputs


def keyframe_pixels(keyframe: Image.Image) -> torch.Tensor:
    """``(1, 3, 1, H, W)`` ImageNet-normalized float32 pixels, the video VAE's input convention."""
    pixels = torch.frombuffer(bytearray(keyframe.tobytes()), dtype=torch.uint8)
    pixels = pixels.view(keyframe.size[1], keyframe.size[0], 3).permute(2, 0, 1)[None, :, None]
    mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN).view(1, -1, 1, 1, 1)
    std = torch.tensor(MINIMAX_H3_PIXEL_STD).view(1, -1, 1, 1, 1)
    return ((pixels.float() / 255.0 - mean) / std).contiguous()


def sample_condition_latents(moments: torch.Tensor, latents_mean, latents_std) -> torch.Tensor:
    """Sample the encoder's posterior as the released model does, then normalize.

    Sampled (not the mode) under a fresh ``KEYFRAME_ENCODE_SEED`` generator, rounded to float16,
    then ``(latent - mean) / std`` per channel.
    """
    mean, logvar = moments.float().cpu().chunk(2, dim=1)
    std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
    noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(KEYFRAME_ENCODE_SEED), dtype=torch.float32)
    latents = (mean + std * noise).to(torch.float16).float()
    shape = (1, -1, 1, 1, 1)
    return (latents - torch.tensor(latents_mean).view(shape)) / torch.tensor(latents_std).view(shape)
