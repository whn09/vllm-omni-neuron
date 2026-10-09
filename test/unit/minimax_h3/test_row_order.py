# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the prompt bucket and the DiT's row order."""

import pytest
import torch

from vllm_omni_neuron.diffusion.models.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    DiTRowOrder,
    build_packed_sequence,
    build_ref2va_packed_sequence,
    text_bucket_length,
)


@pytest.mark.parametrize("num_text", [1, 29, 178, 179, 500])
def test_bucket_aligns_the_sequence(num_text):
    media = 37710
    bucket = text_bucket_length(num_text, media, 512)
    assert bucket >= num_text and (media + bucket) % 512 == 0 and bucket - num_text < 512
    assert text_bucket_length(num_text, media, 1) == num_text


def test_prompts_in_a_bucket_share_it():
    media = 37710
    assert {text_bucket_length(n, media, 512) for n in range(1, 179)} == {178}


def _layout(num_text):
    tags = torch.full((num_text,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    return build_packed_sequence(tags, 3, 8, 8, 6, (1, 2, 2))


def test_dit_order_left_pads_the_text():
    layout = _layout(5)
    order = DiTRowOrder.build(layout, 9)
    # t2va's released order is already `[text | audio | video]`.
    assert order.permutation.tolist() == list(range(layout.sequence_length))
    table = torch.arange(1, layout.sequence_length + 1, dtype=torch.float32)[:, None]
    rows = order.rows(table)
    assert rows.shape[0] == layout.sequence_length + 4 and rows[:4].abs().sum() == 0
    assert rows[4:, 0].tolist() == table[:, 0].tolist()
    assert order.rows(torch.arange(layout.sequence_length), pad="repeat")[:5].tolist() == [0] * 5


def test_dit_order_runs_tile_the_sequence():
    layout = _layout(5)
    order = DiTRowOrder.build(layout, 9)
    runs, tags = order.runs(layout)
    assert runs[0] == (0, 9) and tags[0] == MINIMAX_H3_TEXT_TAG
    assert runs[-1][1] == layout.sequence_length + 4
    for (_, end), (start, _) in zip(runs, runs[1:]):
        assert end == start
    assert [tag for tag in tags] == [MINIMAX_H3_TEXT_TAG, MINIMAX_H3_AUDIO_TAG, MINIMAX_H3_VIDEO_TAG]


def test_ref2va_dit_order_groups_conditions_and_audio():
    tags = torch.full((4,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    # image (1x8x8 latents -> 16 rows), video with soundtrack (2x8x8 -> 32 rows, 6 audio rows),
    # audio (4 rows); target 3x8x8 video, 6 audio latents.
    layout = build_ref2va_packed_sequence(
        tags, [("image", False), ("video", True), ("audio", False)], [(1, 8, 8), (2, 8, 8)], [6, 4],
        3, 8, 8, 6, (1, 2, 2),
    )
    assert layout.num_condition_video_rows == 48 and layout.num_condition_audio_rows == 10
    order = DiTRowOrder.build(layout, 4)
    dit_tags = order.rows(layout.token_tags)
    expected = (
        [MINIMAX_H3_TEXT_TAG] * 4 + [MINIMAX_H3_VIDEO_TAG] * 48 + [MINIMAX_H3_AUDIO_TAG] * 22
        + [MINIMAX_H3_VIDEO_TAG] * 48
    )
    assert dit_tags.tolist() == expected
    runs, _ = order.runs(layout)
    # text | conditions | reference audio | target audio | target video
    assert runs == ((0, 4), (4, 52), (52, 62), (62, 74), (74, 122))
    # The soundtrack shares its video's rotary origin; the image takes one integer slot.
    video_start = int(layout.video_indices[16])
    soundtrack_start = int(layout.audio_indices[0])
    assert layout.position_ids[video_start, 0] == layout.position_ids[soundtrack_start, 0] == 5.0
