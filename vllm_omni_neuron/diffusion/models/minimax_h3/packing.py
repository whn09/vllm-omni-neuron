# SPDX-License-Identifier: Apache-2.0
"""Packed-sequence geometry of MiniMax-H3.

MiniMax-H3 runs its transformer over a single packed 1-D sequence holding every
modality at once. For the text/keyframe tasks the row order is

    [ text (L) | keyframe conditions (C) | target audio (A) | target video (V) ]

and everything here exists to place a row in that sequence and give it its
``(t, h, w)`` rotary coordinate. Coordinates are built in float64 because video and
audio share one 40-units-per-second rotary clock — video advances ``5/3`` rotary
units per pixel frame at 24 fps, audio one unit per latent at 40 latents/s — and
that shared clock *is* the audio/video alignment. Rounding it to float32 while
building it drifts the two streams apart, so the grid is built in float64 on the
host and only cast down when it is handed to the traced graph.

This is a port of ``diffusers.modular_pipelines.minimax_h3.packing``, kept
bit-exact: two deliberate numpy-vs-torch choices in the reference
(``np.linspace(..., endpoint=False)`` and numpy's pairwise summation of the
keyframe anchor) are reproduced rather than "cleaned up", because they define the
released checkpoint's grid.

Everything in this module is host-side Python that runs once per request, before
the traced graph. None of it is traced, so plain numpy and data-dependent control
flow are fine here — and cheaper than making Neuron compute a grid whose shape is
already known.
"""

from dataclasses import dataclass

import numpy as np
import torch

# Per-row modality tags. They index the transformer's AdaLN table, so the values are
# a checkpoint contract.
MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2

# MiniMax-H3 generates at a fixed 24 fps and was released for a 768 pixel short edge
# only, with a soft area cap of 768x1344 and both axes rounded to a multiple of 32.
MINIMAX_H3_FPS = 24
MINIMAX_H3_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4
MINIMAX_H3_MAX_ASPECT_RATIO = 4
MINIMAX_H3_MIN_DURATION = 5.0
MINIMAX_H3_MAX_DURATION = 15.0

# The video VAE encodes 17 pixel frames per chunk and drops the 3 trailing latent
# frames of every chunk, so `17 * n + 5` pixel frames map to `5 * n + 2` latent frames.
MINIMAX_H3_FRAMES_PER_CHUNK = 17
MINIMAX_H3_LATENTS_PER_CHUNK = 5

# The pixel convention of the video VAE: ImageNet-normalized RGB over a `[0, 1]` base range.
MINIMAX_H3_PIXEL_MEAN = (0.485, 0.456, 0.406)
MINIMAX_H3_PIXEL_STD = (0.229, 0.224, 0.225)

# MiniMax-H3 conditions on the *unnormalized* hidden state its Qwen3-VL conditioner
# produces after the 50th of its 64 decoder layers, i.e. `hidden_states[50]`
# (`hidden_states[0]` being the embedding output).
MINIMAX_H3_TEXT_ENCODER_LAYER = 50

# The audio VAE hops 800 samples at 32 kHz, i.e. 40 latents per second. Stereo is
# carried as two channel-major blocks of audio rows (and as two batch items at the
# audio VAE boundary, which is mono).
MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40
MINIMAX_H3_AUDIO_CHANNELS = 2

# Conditioning rows are not fully clean: the released model noises keyframe latents
# to `t = 0.999` and runs them at that timestep for every denoising step.
MINIMAX_H3_KEYFRAME_NOISE_AUG = 0.999

# The seeded posterior sample of the keyframe VAE encode. Fixed at 42 independently
# of the request seed.
MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42

# Rotary-time constants. One latent frame spans `5/3 * frames_per_latent` rotary
# units, where the pattern `(1, 4, 4, 4, 4)` mirrors the VAE's 17-pixel-frames-to-
# 5-latent-frames grouping; the spatial axes are normalized by the square root of the
# latent area and scaled by 32.
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


@dataclass
class MiniMaxH3PackedSequence:
    """The structural description of one packed MiniMax-H3 sequence.

    Attributes:
        sequence_length: Total number of rows, ``L + C + A + V``.
        position_ids: ``(sequence_length, 3)`` float64 ``(t, h, w)`` rotary coordinates.
        token_tags: ``(sequence_length,)`` modality tag of every row.
        video_indices: Sequence positions of the video rows, conditioning rows first.
        audio_indices: Sequence positions of the audio rows, reference rows first.
        text_indices: Sequence positions of the text rows.
        num_condition_video_rows: How many leading ``video_indices`` entries are
            conditioning rows rather than generated rows.
        num_condition_audio_rows: How many leading ``audio_indices`` entries are
            reference rows rather than generated rows.
    """

    sequence_length: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    """Resolve a display aspect ratio into a MiniMax-H3 canvas.

    The short edge starts at 768, the area is capped at ``768 * 1344`` and both axes
    are then rounded to the nearest multiple of 32 — so the final area may end up
    slightly above the pre-rounding budget. Only the ratio of the two arguments
    matters; pass either the aspect ratio (``16, 9``) or a keyframe's dimensions.

    Returns:
        The ``(height, width)`` of the canvas.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")

    ratio = aspect_width / aspect_height
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got "
            f"{aspect_width}:{aspect_height} ({ratio:g})."
        )

    if ratio >= 1.0:
        width, height = MINIMAX_H3_SHORT_EDGE * ratio, float(MINIMAX_H3_SHORT_EDGE)
    else:
        width, height = float(MINIMAX_H3_SHORT_EDGE), MINIMAX_H3_SHORT_EDGE / ratio

    area = width * height
    if area > MINIMAX_H3_MAX_PIXELS:
        scale = (MINIMAX_H3_MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale

    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return (
        max(multiple, round(height / multiple) * multiple),
        max(multiple, round(width / multiple) * multiple),
    )


def align_num_frames(num_frames: int) -> int:
    """Snap a frame count up to the next ``17 * n + 5`` the video VAE can encode."""
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    while num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int) -> int:
    """The number of latent frames the video VAE produces for a ``17 * n + 5`` count."""
    if num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        raise ValueError(f"`num_frames` must be of the form 17 * n + 5, got {num_frames}.")
    return (
        num_frames - MINIMAX_H3_LATENTS_PER_CHUNK
    ) // MINIMAX_H3_FRAMES_PER_CHUNK * MINIMAX_H3_LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    """The number of audio latents covering ``num_frames`` video frames at 24 fps."""
    return int(round(num_frames / MINIMAX_H3_FPS * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND))


def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Pack video latents of shape ``(B, C, F, H, W)`` into transformer rows.

    Returns ``(B * num_patches, C * prod(patch_size))``, frame-major then row-major.
    """
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}."
        )

    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video_tokens(
    rows: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Unpack transformer rows back into video latents. The inverse of `patchify_video_latents`."""
    patch_t, patch_h, patch_w = patch_size
    rows = rows.reshape(
        -1,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(-1, channels, num_latent_frames, latent_height, latent_width).contiguous()


def unpack_audio_tokens(rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
    """Unpack channel-major audio rows into ``(2, latent_channels, num_audio_latents)``.

    One batch item per stereo channel, which is what the mono audio VAE consumes.
    """
    rows = rows.reshape(MINIMAX_H3_AUDIO_CHANNELS, num_audio_latents, rows.shape[-1])
    return rows.permute(0, 2, 1).contiguous()


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """One aspect-normalized spatial rotary axis.

    ``dim // patch`` coordinates centred on the unit interval, scaled up by 32. The
    right endpoint is excluded, so a square canvas spans ``[0, 32)``.
    """
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # Built with numpy: `np.linspace(..., endpoint=False)` is
    # `start + arange(num) * (stop - start) / num`, which is not what `torch.linspace`
    # computes, and the float64 grid has to be reproduced exactly.
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    """Rotary time of every latent frame from ``origin``; spacing ``5/3 * (1, 4, 4, 4, 4)``."""
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _temporal_position_span(num_latent_frames: int) -> float:
    """The rotary time spanned by ``num_latent_frames`` latent frames.

    Summed by numpy (pairwise summation) rather than sequentially: the reference
    computes the keyframe anchor this way and the two summation orders differ in the
    last ulp from 16 latent frames onwards.
    """
    spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
    for index in range(len(_ROPE_FRAMES_PER_LATENT)):
        spans[index :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[index]
    return float(spans.sum())


def build_packed_sequence(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    keyframe_anchors: tuple[str, ...] = (),
) -> MiniMaxH3PackedSequence:
    """Build the ``[text | keyframe conditions | target audio | target video]`` layout.

    Args:
        text_token_tags: ``(num_text_tokens,)`` modality tag of every text row. Text is
            tagged ``1``, except the rows of a keyframe's vision block, which
            MiniMax-H3 tags ``0`` (video).
        num_latent_frames: Number of target latent frames.
        latent_height: Target latent height.
        latent_width: Target latent width.
        num_audio_latents: Number of target audio latents per channel.
        patch_size: The transformer's ``(t, h, w)`` patch.
        keyframe_anchors: One entry per keyframe conditioning block, in packed order:
            ``"first"`` anchors the block at the first latent frame, ``"last"`` at the
            last one.
    """
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = text_token_tags.shape[0]
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    # 1. The (t, h, w) grid. Text rows sit on the time axis at their row index, and the
    # media rows continue the time axis from there, so text length shifts the whole
    # media clock.
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    frame_grid = torch.stack(
        [grid.reshape(-1) for grid in torch.meshgrid(height_grid, width_grid, indexing="ij")], -1
    )

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            anchor_time = (
                float(num_text_tokens)
                + _temporal_position_span(num_latent_frames)
                - _ROPE_FRAME_RESCALE
            )
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(
            condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame
        )
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    # Audio rows are channel-major and share the video's rotary clock: one unit per
    # latent at 40 latents/s equals 24 fps * 5/3. They carry no height coordinate and
    # are pinned to the two extremes of the width grid.
    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full(
                (num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64
            ),
        ]
    )

    video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(
        num_latent_frames, float(num_text_tokens)
    )[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    # 2. Row indices and modality tags.
    video_indices = torch.cat(
        [torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)]
    )
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_rows,
        num_condition_audio_rows=0,
    )


def _frame_position_grid(latent_height: int, latent_width: int, patch_h: int, patch_w: int):
    """The ``(h, w)`` rotary coordinates of one latent frame, and the width axis they came from."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _fill_audio_positions(position_ids, rows: slice, num_audio_latents: int, rotary_time: float, width_grid):
    """Place one channel-major audio block, pinned to the extremes of its own block's width grid."""
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[rows, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )


def build_ref2va_packed_sequence(
    text_token_tags: torch.Tensor,
    reference_kinds: list[tuple[str, bool]],
    condition_shapes: list[tuple[int, int, int]],
    audio_condition_rows: list[int],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
) -> MiniMaxH3PackedSequence:
    """Build the ``[text | reference blocks | target audio | target video]`` layout of ``ref2va``.

    Mirrors `diffusers`' ``MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence``.

    Args:
        text_token_tags: ``(num_text_tokens,)`` modality tag of every text row.
        reference_kinds: ``(kind, has_audio)`` per reference in packed order; ``kind`` is
            ``"image"``, ``"video"`` or ``"audio"``.
        condition_shapes: ``(latent_frames, latent_height, latent_width)`` of every image and
            video reference's latents, in packed order.
        audio_condition_rows: Row count of every audio-bearing reference's soundtrack, in
            packed order.
        num_latent_frames, latent_height, latent_width: The target's latent geometry.
        num_audio_latents: Target audio latents per channel.
        patch_size: The transformer's ``(t, h, w)`` patch.
    """
    _, patch_h, patch_w = patch_size
    num_text_tokens = text_token_tags.shape[0]
    num_target_video_rows = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    num_target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_reference_video_rows = sum(f * (h // patch_h) * (w // patch_w) for f, h, w in condition_shapes)
    num_reference_audio_rows = sum(audio_condition_rows)
    sequence_length = (
        num_text_tokens
        + num_reference_video_rows
        + num_reference_audio_rows
        + num_target_audio_rows
        + num_target_video_rows
    )

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    # Reference blocks in request order. `rotary_time` is the shared audio/video clock: it starts
    # where the text ends and every block pushes it forward by the time the block occupies.
    visual_geometry = iter(condition_shapes)
    audio_row_counts = iter(audio_condition_rows)
    video_indices, audio_indices = [], []
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for kind, has_audio in reference_kinds:
        if kind == "image":
            frames, height, width = next(visual_geometry)
            rows = slice(cursor, cursor + frames * (height // patch_h) * (width // patch_w))
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(height, width, patch_h, patch_w)
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            # An image takes one integer rotary slot, not a latent frame's 5/3 units.
            rotary_time += 1.0
        elif kind == "audio":
            num_rows = next(audio_row_counts)
            rows = slice(cursor, cursor + num_rows)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            latents = num_rows // MINIMAX_H3_AUDIO_CHANNELS
            _fill_audio_positions(position_ids, rows, latents, rotary_time, target_width_grid)
            rotary_time += float(latents)
        elif kind == "video":
            # A soundtrack's rows go right before its video's and share their origin.
            num_audio_rows = next(audio_row_counts) if has_audio else 0
            audio_latents = num_audio_rows // MINIMAX_H3_AUDIO_CHANNELS
            frames, height, width = next(visual_geometry)
            frame_grid, width_grid = _frame_position_grid(height, width, patch_h, patch_w)
            audio_rows = slice(cursor, cursor + num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + frames * frame_grid.shape[0])
            cursor = video_rows.stop
            audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))
            _fill_audio_positions(position_ids, audio_rows, audio_latents, rotary_time, width_grid)
            frame_time = _temporal_position_grid(frames, rotary_time)
            position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
            position_ids[video_rows, 1:] = frame_grid.repeat(frames, 1)
            # Summed sequentially, as the reference does at this call site (not pairwise, as
            # `_temporal_position_span` does for the keyframe anchor).
            video_span = sum(
                _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
                for index in range(frames)
            )
            rotary_time += max(float(audio_latents), video_span)
        else:
            raise ValueError(f"A reference must be an 'image', a 'video' or an 'audio', got {kind!r}.")

    # The generated rows share the origin the reference blocks left behind.
    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(
        position_ids, slice(audio_start, video_start), num_audio_latents, rotary_time, target_width_grid
    )
    frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

    video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
    audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_reference_video_rows,
        num_condition_audio_rows=num_reference_audio_rows,
    )


def build_rotary_tables(
    position_ids: torch.Tensor, rope_freq_dim: int, rope_theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the DiT's 3-axis MM-RoPE tables from a packed layout's coordinate grid.

    One ``rope_freq_dim``-wide frequency band per axis, concatenated ``(t, h, w)`` and then
    duplicated for the half-rotation `_apply_rotary_emb` performs — so the tables are
    ``2 * 3 * rope_freq_dim`` wide and cover only the leading channels of each head.

    Built here, on the host, rather than in the traced graph: ``position_ids`` is the
    float64 grid whose precision *is* the audio/video alignment (see the module docstring),
    and the outer product is one small matrix per request.

    Args:
        position_ids: ``(seq_len, 3)`` float64 ``(t, h, w)`` coordinates from
            `build_packed_sequence`.
        rope_freq_dim: Frequencies per axis, the transformer's ``rope_freq_dim``.
        rope_theta: The rotary base, the transformer's ``rope_theta``.

    Returns:
        Two ``(seq_len, 6 * rope_freq_dim)`` float32 tables, cosine and sine.
    """
    inv_freq = 1.0 / rope_theta ** (
        torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim)
    )
    freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)
    freqs_t, freqs_h, freqs_w = freqs.unbind(dim=1)
    freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()


def build_row_timesteps(
    layout: MiniMaxH3PackedSequence,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign a timestep to every row and reduce it to ``(timestep, timestep_indices)``.

    One forward serves rows at different noise levels: the generated video and audio
    rows step down their own schedules while the conditioning rows stay pinned at
    their noise-augmentation level. Text rows never reach an output head and inherit
    the video timestep.

    Returns:
        The distinct timesteps, sorted, and the index of every row into them.
    """
    row_timesteps = torch.full((layout.sequence_length,), video_timestep, dtype=torch.float32)
    row_timesteps[layout.video_indices[: layout.num_condition_video_rows]] = (
        condition_video_timestep
    )
    row_timesteps[layout.audio_indices[layout.num_condition_audio_rows :]] = audio_timestep
    row_timesteps[layout.audio_indices[: layout.num_condition_audio_rows]] = (
        condition_audio_timestep
    )
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


def build_run_table_rows(
    runs: tuple[tuple[int, int], ...],
    run_tags: tuple[int, ...],
    timestep_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce a step's per-row timestep indices to the per-run AdaLN table rows.

    Args:
        runs: The ``(start, end)`` spans from `DiTRowOrder.runs`.
        run_tags: The modality tag of each span.
        timestep_indices: ``(seq_len,)`` per-row index into the step's timestep table.

    Returns:
        ``(adaln_run_rows, timestep_run_rows)``: the row each run reads from the
        ``(timestep, modality)`` AdaLN table, i.e. ``timestep_index * 3 + tag``, and the
        row it reads from the per-timestep `norm_out` table.

    Raises:
        ValueError: If a run is not uniform in its timestep index, which would mean the
            structural decomposition no longer matches the timestep assignment.
    """
    indices = timestep_indices.tolist()
    adaln_rows, timestep_rows = [], []
    for (start, end), tag in zip(runs, run_tags):
        span = indices[start:end]
        if span and min(span) != max(span):
            raise ValueError(
                f"Rows [{start}, {end}) span more than one timestep index "
                f"({min(span)}..{max(span)}); the row-run decomposition assumes a run is "
                "uniform in its timestep."
            )
        timestep_index = span[0] if span else 0
        adaln_rows.append(timestep_index * 3 + max(tag, 0))
        timestep_rows.append(timestep_index)
    return (
        torch.tensor(adaln_rows, dtype=torch.long),
        torch.tensor(timestep_rows, dtype=torch.long),
    )


def pad_num_timesteps(
    timestep: torch.Tensor, timestep_indices: torch.Tensor, num_timesteps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the distinct-timestep table to a fixed length so the NEFF shape is static.

    `build_row_timesteps` returns as many distinct timesteps as the step happens to
    have, and that count is not constant across a schedule: the keyframe conditioning
    rows sit at ``max(t, 0.999)``, which coincides with the video timestep on the very
    first step and separates from it afterwards. A varying table length would retrace
    the graph, so the table is padded up to the worst case and the padding rows are
    made unreachable by leaving `timestep_indices` untouched — no row indexes them.

    Args:
        timestep: The distinct timesteps of one step.
        timestep_indices: Per-row index into ``timestep``.
        num_timesteps: The fixed table length to pad to.

    Returns:
        The padded ``(timestep, timestep_indices)`` pair.
    """
    present = timestep.shape[0]
    if present > num_timesteps:
        raise ValueError(
            f"The step has {present} distinct timesteps, more than the compiled "
            f"{num_timesteps}. Recompile with a larger `num_timesteps`."
        )
    if present == num_timesteps:
        return timestep, timestep_indices
    # Repeat the last live timestep rather than padding with zeros: an unreachable row
    # of the AdaLN table still runs through `time_embedder` and every `adaln_proj`, and
    # a duplicate keeps those activations in the range the checkpoint was trained on.
    pad = timestep[-1:].expand(num_timesteps - present)
    return torch.cat([timestep, pad]), timestep_indices


def text_bucket_length(num_text_tokens: int, media_length: int, align: int) -> int:
    """The prompt length the DiT graph is built for: the smallest ``>= num_text_tokens`` that
    makes ``media_length + bucket`` a multiple of ``align``.

    Every prompt length up to the bucket then shares one graph, and the packed sequence is
    ``align``-row aligned, which the attention kernel runs fastest on. The padding is a few
    hundred rows against tens of thousands of media rows. ``align <= 1`` disables bucketing.
    """
    if align <= 1:
        return num_text_tokens
    return num_text_tokens + (-(media_length + num_text_tokens)) % align


@dataclass(frozen=True)
class DiTRowOrder:
    """Map a released layout to the DiT's ``[padding | text | conditions | audio | video]``.

    The prompt is padded on the *left* up to its length bucket, so the rows that act as keys
    stay one interval — ``[num_text_pad, sequence_length)`` — which the attention kernel's KV
    bounds express; the rows that even out the CP shards trail the sequence and fall outside it
    too. The media keep the released block order with all visual conditions first, then every
    audio row (references first), then the target video — the order the DiT concatenates its
    two input streams in. Attention sees rows only through their rotary positions, which are
    built in the released order, so this is the reference's computation.

    Attributes:
        permutation: For each real DiT row, its index in the released order.
        num_text_rows: The prompt bucket, padding included.
        num_text_pad: Padding rows in front of the prompt.
    """

    permutation: torch.Tensor
    num_text_rows: int
    num_text_pad: int

    @classmethod
    def build(cls, layout: MiniMaxH3PackedSequence, num_text_rows: int) -> "DiTRowOrder":
        num_text = int(layout.text_indices.shape[0])
        if num_text_rows < num_text:
            raise ValueError(f"A {num_text_rows}-row text bucket cannot hold {num_text} tokens.")
        num_condition = layout.num_condition_video_rows
        permutation = torch.cat(
            [
                layout.text_indices,
                layout.video_indices[:num_condition],
                layout.audio_indices,
                layout.video_indices[num_condition:],
            ]
        )
        return cls(permutation, num_text_rows, num_text_rows - num_text)

    def rows(self, tensor: torch.Tensor, pad: str = "zeros") -> torch.Tensor:
        """``tensor``'s leading axis in DiT order; padding rows are zeros or repeat the first row."""
        tensor = tensor.index_select(0, self.permutation)
        if not self.num_text_pad:
            return tensor
        if pad == "zeros":
            filler = tensor.new_zeros((self.num_text_pad, *tensor.shape[1:]))
        else:
            filler = tensor[:1].expand(self.num_text_pad, *tensor.shape[1:])
        return torch.cat([filler, tensor])

    def runs(self, layout: MiniMaxH3PackedSequence) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
        """Decompose the DiT-ordered sequence into runs of constant ``(block, modality)``.

        The AdaLN table is addressed by ``timestep_index * 3 + tag``, and both are piecewise
        constant: the tag changes at a block edge or at a vision block inside the prompt, and
        the timestep is uniform per block by construction (`build_row_timesteps`). So the graph
        gathers one modulation row per run instead of one per sequence row.

        Boundaries are structural — at every block edge and tag change — rather than found by
        comparing AdaLN rows: the conditioning rows sit at ``max(t, 0.999)``, which equals the
        video timestep on the first step only, so an equality-merged decomposition would change
        its run count mid-schedule and retrace. The padding repeats the first prompt row, so it
        extends that row's run.
        """
        num_condition_video = layout.num_condition_video_rows
        num_condition_audio = layout.num_condition_audio_rows
        block = torch.zeros(layout.sequence_length, dtype=torch.long)
        block[layout.video_indices[:num_condition_video]] = 1
        block[layout.audio_indices[:num_condition_audio]] = 2
        block[layout.audio_indices[num_condition_audio:]] = 3
        block[layout.video_indices[num_condition_video:]] = 4
        tags = self.rows(layout.token_tags[:, None], pad="repeat")[:, 0]
        keys = self.rows(block[:, None], pad="repeat")[:, 0] * 8 + tags.clamp(min=0)
        changes = ((keys[1:] != keys[:-1]).nonzero().flatten() + 1).tolist()
        edges = [0, *changes, int(keys.numel())]
        runs = tuple(zip(edges[:-1], edges[1:]))
        return runs, tuple(int(tags[start]) for start, _ in runs)
