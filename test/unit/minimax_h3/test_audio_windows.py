# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the windowed MiniMax-H3 audio decode layout."""

import pytest

from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_minimax_h3_audio import (
    AUDIO_HALO,
    AUDIO_WINDOW,
    audio_windows,
)


@pytest.mark.parametrize("num_frames", [AUDIO_WINDOW + 1, 207, 360, 600])
def test_windows_cover_the_clip_with_context(num_frames):
    spans = audio_windows(num_frames)
    assert spans[0][1] == 0 and spans[-1][2] == num_frames
    for (_, _, end), (_, start, _) in zip(spans, spans[1:]):
        assert end == start
    for window_start, core_start, core_end in spans:
        window_end = window_start + AUDIO_WINDOW
        assert 0 <= window_start and window_end <= num_frames
        assert core_start - window_start >= AUDIO_HALO or window_start == 0
        assert window_end - core_end >= AUDIO_HALO or window_end == num_frames
