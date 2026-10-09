# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 ``ref2va``: image, video and audio references, on the host.

Mirrors `diffusers`' ``MiniMaxH3Ref2VASetupStep`` (normalization onto MiniMax-H3's rates and
resolutions), ``MiniMaxH3Ref2VATextEncoderStep`` (the presentation the conditioner reads) and the
reference half of ``MiniMaxH3Ref2VAReferenceEncoderStep``. As in `keyframes`, every resampling and
rounding choice here is the released model's.

A request carries its references as ``multi_modal_data["references"]``, a list **in the order the
model should read them** (the order labels them in the prompt and lays them out on the rotary
clock). Each entry is a dict — ``{"type": "image", "image": PIL.Image}``, ``{"type": "video",
"frames": (T, H, W, 3) uint8, "fps": float, "audio": (C, N) | None, "sample_rate": int | None}`` or
``{"type": "audio", "audio": (C, N), "sample_rate": int}`` — or any object with the same attributes
and a ``kind``, such as `diffusers`' ``MiniMaxH3*Reference`` dataclasses.

Any of them may instead carry a ``"path"`` to a local media file, decoded here on every rank. Prefer
that for videos: the request is copied to every worker, and decoded frames are tens of MB per
second of video — at 64 ranks the copy outright stalls the engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from vllm_omni_neuron.diffusion.models.minimax_h3.packing import (
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    resolve_canvas_size,
)

#: Limits MiniMax-H3 documents for the released checkpoint.
MAX_IMAGES, MAX_VIDEOS, MAX_AUDIOS, MAX_REFERENCES = 9, 3, 3, 12
#: Short edge an image reference is encoded at (upscaling included, no area cap).
REFERENCE_IMAGE_SHORT_EDGE = 2048
#: Rate the conditioner reads a video reference at.
VIDEO_SAMPLE_FPS = 2.0


@dataclass
class Reference:
    """A reference normalized onto MiniMax-H3's rates and resolutions."""

    kind: str
    image: Image.Image | None = None
    frames: np.ndarray | None = None
    audio: torch.Tensor | None = None

    @property
    def has_audio(self) -> bool:
        return self.audio is not None


def _field(entry, name, default=None):
    if isinstance(entry, dict):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _decode_audio(path: str):
    """``((channels, samples) float32, sample_rate)`` of a file's first audio stream, or ``(None, None)``."""
    import av

    with av.open(path) as container:
        if not container.streams.audio:
            return None, None
        chunks, rate = [], None
        for frame in container.decode(audio=0):
            rate = frame.sample_rate
            array = frame.to_ndarray()
            channels = frame.layout.nb_channels
            chunks.append(array.reshape(channels, -1) if frame.format.is_planar else array.reshape(-1, channels).T)
    return torch.from_numpy(np.concatenate(chunks, axis=1).astype(np.float32)), rate


def _decode_video(path: str):
    """``(frames (T, H, W, 3) uint8, fps, audio, sample_rate)`` of a video file."""
    import av

    with av.open(path) as container:
        fps = float(container.streams.video[0].average_rate)
        frames = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    audio, rate = _decode_audio(path)
    return frames, fps, audio, rate


def _load(kind: str, entry) -> dict:
    """The reference's media as a dict, decoding a ``path`` if it has one."""
    path = _field(entry, "path")
    if path is None:
        return {name: _field(entry, name) for name in ("image", "frames", "video", "fps", "audio", "sample_rate")}
    if kind == "image":
        return {"image": Image.open(path)}
    if kind == "video":
        frames, fps, audio, rate = _decode_video(path)
        return {"frames": frames, "fps": fps, "audio": audio, "sample_rate": rate}
    audio, rate = _decode_audio(path)
    if audio is None:
        raise ValueError(f"The audio reference {path!r} has no audio stream.")
    return {"audio": audio, "sample_rate": rate}


def parse_references(raw) -> list:
    """Validate a request's ``references`` list (kinds and MiniMax-H3's limits)."""
    if not raw:
        raise ValueError("`ref2va` needs at least one reference; use `t2va` for text-only requests.")
    kinds = []
    for index, entry in enumerate(raw):
        kind = _field(entry, "type") or _field(entry, "kind")
        if kind not in ("image", "video", "audio"):
            raise ValueError(f"`references[{index}]` must be an image, a video or an audio, got {kind!r}.")
        kinds.append(kind)
    for kind, limit in (("image", MAX_IMAGES), ("video", MAX_VIDEOS), ("audio", MAX_AUDIOS)):
        if kinds.count(kind) > limit:
            raise ValueError(f"MiniMax-H3 accepts at most {limit} {kind} references, got {kinds.count(kind)}.")
    if len(kinds) > MAX_REFERENCES:
        raise ValueError(f"MiniMax-H3 accepts at most {MAX_REFERENCES} references, got {len(kinds)}.")
    if set(kinds) == {"audio"}:
        raise ValueError("An audio reference has to be paired with at least one image or video reference.")
    return [(kind, _load(kind, entry)) for kind, entry in zip(kinds, raw)]


def _normalize_image(image) -> Image.Image:
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255.0).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        array = image.movedim(-3, -1).cpu().numpy()
        image = _normalize_image(array if array.dtype == np.uint8 else array.astype(np.float32))
    image = image.convert("RGB")
    width, height = image.size
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    scale = REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    size = (
        max(multiple, round(width * scale / multiple) * multiple),
        max(multiple, round(height * scale / multiple) * multiple),
    )
    return image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)


def _normalize_video(frames, fps: float, num_frames: int) -> np.ndarray:
    """Onto ``uint8`` THWC at 24 fps (whole frames held, as ffmpeg's ``fps`` filter does),
    truncated to the generated frame count, LANCZOS-rescaled onto its own aspect's canvas."""
    if isinstance(frames, list):
        frames = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
    if isinstance(frames, torch.Tensor):
        frames = frames.movedim(-3, -1).cpu().numpy()
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = (frames * 255.0).round().clip(0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"A reference video must be (T, H, W, 3) RGB frames, got {tuple(frames.shape)}.")
    if fps <= 0:
        raise ValueError(f"A reference video must have a positive frame rate, got {fps}.")
    if fps != MINIMAX_H3_FPS:
        scale = MINIMAX_H3_FPS / fps
        slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
        frames = np.repeat(frames, np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5)), axis=0)
    frames = frames[:num_frames]
    height, width = resolve_canvas_size(frames.shape[2], frames.shape[1])
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames]
    )


def _normalize_audio(waveform, sample_rate: int, target_rate: int, max_duration: float) -> torch.Tensor:
    """Stereo float32 at the audio VAE's rate, truncated at the source rate first."""
    waveform = torch.as_tensor(waveform)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
        raise ValueError(f"A reference soundtrack must be (channels, samples), got {tuple(waveform.shape)}.")
    waveform = waveform.to(torch.float32)[:, : int(max_duration * sample_rate)]
    if waveform.shape[0] != 2:
        waveform = waveform.expand(2, -1).contiguous()
    if sample_rate == target_rate:
        return waveform
    import torchaudio

    return torchaudio.transforms.Resample(sample_rate, target_rate)(waveform)


def normalize_references(parsed, num_frames: int, audio_sample_rate: int) -> list[Reference]:
    """The setup step's normalization, in packed order."""
    normalized = []
    for kind, entry in parsed:
        audio = _field(entry, "audio")
        if audio is not None:
            rate = _field(entry, "sample_rate") or audio_sample_rate
            audio = _normalize_audio(audio, int(rate), audio_sample_rate, num_frames / MINIMAX_H3_FPS)
        if kind == "image":
            normalized.append(Reference("image", image=_normalize_image(_field(entry, "image"))))
        elif kind == "video":
            frames = _field(entry, "frames")
            if frames is None:
                frames = _field(entry, "video")
            fps = float(_field(entry, "fps") or MINIMAX_H3_FPS)
            normalized.append(Reference("video", frames=_normalize_video(frames, fps, num_frames), audio=audio))
        else:
            if audio is None:
                raise ValueError("An audio reference needs an `audio` waveform.")
            normalized.append(Reference("audio", audio=audio))
    return normalized


def _sample_video_frames(frames: np.ndarray, temporal_patch: int):
    """The frames the conditioner reads (every ``24 / 2``-th, deduplicated) and the timestamp of
    every merged block of ``temporal_patch`` frames."""
    stride = MINIMAX_H3_FPS / VIDEO_SAMPLE_FPS
    indices, cursor = [], 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    if len(indices) < temporal_patch:
        minimum = round((temporal_patch - 1) * stride) + 1
        raise ValueError(
            f"A reference video must run at least {minimum} frames at {MINIMAX_H3_FPS} fps, got {frames.shape[0]}."
        )
    timestamps = [index / VIDEO_SAMPLE_FPS for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % temporal_patch)
    blocks = [
        (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2
        for index in range(0, len(timestamps), temporal_patch)
    ]
    return [frames[index] for index in indices], blocks


def reference_presentation(tokenizer, processor, references: list[Reference], prompt: str):
    """Tokenize ``ref2va``'s presentation: per reference, in packed order and numbered per
    modality, ``"<Audio j>: "`` (a soundtrack, before its video), ``"<Picture i>: "`` plus a
    vision block, or ``"<Video k>: "`` plus a ``"<t seconds>"`` label and vision block per merged
    frame pair; then the prompt verbatim.

    Returns ``(token_ids, token_tags, vision_inputs)``; vision rows are tagged as video.
    """
    merge = processor.image_processor.merge_size**2
    vision_inputs: dict = {}
    image_counts: list[int] = []
    images = [reference.image for reference in references if reference.kind == "image"]
    if images:
        features = processor.image_processor(images=images, return_tensors="pt")
        vision_inputs["pixel_values"] = features["pixel_values"]
        vision_inputs["image_grid_thw"] = features["image_grid_thw"]
        image_counts = [int(grid.prod()) // merge for grid in features["image_grid_thw"]]

    video_counts: list[int] = []
    video_timestamps: list[list[float]] = []
    videos = [reference for reference in references if reference.kind == "video"]
    if videos:
        temporal_patch = processor.video_processor.temporal_patch_size
        sampled = [_sample_video_frames(reference.frames, temporal_patch) for reference in videos]
        video_timestamps = [blocks for _, blocks in sampled]
        features = processor.video_processor(
            videos=[np.stack(frames) for frames, _ in sampled], do_sample_frames=False, return_tensors="pt"
        )
        vision_inputs["pixel_values_videos"] = features["pixel_values_videos"]
        vision_inputs["video_grid_thw"] = features["video_grid_thw"]
        video_counts = [int(grid[1]) * int(grid[2]) // merge for grid in features["video_grid_thw"]]
        for blocks, grid in zip(video_timestamps, features["video_grid_thw"]):
            if int(grid[0]) != len(blocks):
                raise ValueError(f"The processor merged a video into {int(grid[0])} blocks, not {len(blocks)}.")

    token_ids: list[int] = []
    token_tags: list[int] = []

    def text(value: str) -> None:
        ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        token_ids.extend(ids)
        token_tags.extend([MINIMAX_H3_TEXT_TAG] * len(ids))

    def vision(pad_token: str, count: int) -> None:
        ids = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids(pad_token)] * count
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        token_ids.extend(ids)
        token_tags.extend([MINIMAX_H3_VIDEO_TAG] * len(ids))

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            text(f"<Audio {counts['audio']}>: ")
        if reference.kind == "image":
            counts["image"] += 1
            text(f"<Picture {counts['image']}>: ")
            vision("<|image_pad|>", image_counts[counts["image"] - 1])
        elif reference.kind == "video":
            counts["video"] += 1
            text(f"<Video {counts['video']}>: ")
            for timestamp in video_timestamps[counts["video"] - 1]:
                # `"{:.1f}"` rounds half to even: a 2 fps pair's mean renders as "<0.2 seconds>".
                text(f"<{timestamp:.1f} seconds>")
                vision("<|video_pad|>", video_counts[counts["video"] - 1])
    text(prompt)
    return token_ids, torch.tensor(token_tags, dtype=torch.long), vision_inputs


def reference_pixels(reference: Reference) -> torch.Tensor:
    """``(1, 3, F, H, W)`` ImageNet-normalized float32 pixels of an image or video reference.

    A video is snapped *down* to ``17 * n + 5`` frames so the VAE encodes it without padding.
    """
    if reference.kind == "image":
        array = np.asarray(reference.image)[None]
    else:
        frames = reference.frames.shape[0]
        frames = (
            max(1, (frames - MINIMAX_H3_LATENTS_PER_CHUNK) // MINIMAX_H3_FRAMES_PER_CHUNK)
            * MINIMAX_H3_FRAMES_PER_CHUNK
            + MINIMAX_H3_LATENTS_PER_CHUNK
        )
        array = reference.frames[:frames]
    pixels = torch.from_numpy(np.ascontiguousarray(array)).permute(3, 0, 1, 2)[None]
    mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN).view(1, -1, 1, 1, 1)
    std = torch.tensor(MINIMAX_H3_PIXEL_STD).view(1, -1, 1, 1, 1)
    # Contiguous: the elementwise ops keep the permute's strides, which the VAE's slicing rejects.
    return ((pixels.float() / 255.0 - mean) / std).contiguous()
