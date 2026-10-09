# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ``ref2va`` reference parsing and normalization."""

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni_neuron.diffusion.models.minimax_h3.references import (
    normalize_references,
    parse_references,
    reference_pixels,
)


def test_limits_and_audio_only_are_rejected():
    with pytest.raises(ValueError):
        parse_references([])
    with pytest.raises(ValueError):
        parse_references([{"type": "audio", "audio": torch.zeros(1, 10)}])
    with pytest.raises(ValueError):
        parse_references([{"type": "image", "image": None}] * 10)


def test_image_is_put_on_a_2048_short_edge():
    parsed = parse_references([{"type": "image", "image": Image.new("RGB", (640, 360))}])
    (reference,) = normalize_references(parsed, 124, 32000)
    assert reference.image.size == (3648, 2048)
    assert reference_pixels(reference).shape == (1, 3, 1, 2048, 3648)


def test_video_is_resampled_to_24_fps_and_put_on_its_canvas():
    frames = np.zeros((30, 360, 640, 3), dtype=np.uint8)
    parsed = parse_references([{"type": "video", "frames": frames, "fps": 12.0}])
    (reference,) = normalize_references(parsed, 124, 32000)
    assert reference.frames.shape == (60, 768, 1344, 3)
    # Snapped down to 17 * n + 5 for the encoder.
    assert reference_pixels(reference).shape[2] == 56


def test_soundtrack_is_stereo_and_truncated():
    parsed = parse_references(
        [{"type": "image", "image": Image.new("RGB", (64, 64))}, {"type": "audio", "audio": torch.ones(1, 32000 * 9), "sample_rate": 32000}]
    )
    _, audio = normalize_references(parsed, 124, 32000)
    assert audio.audio.shape == (2, int(124 / 24 * 32000))


def test_references_can_be_paths(tmp_path):
    import wave

    image = tmp_path / "a.png"
    Image.new("RGB", (64, 48)).save(image)
    audio = tmp_path / "a.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(32000)
        handle.writeframes(np.zeros(32000, dtype=np.int16).tobytes())
    parsed = parse_references([{"type": "image", "path": str(image)}, {"type": "audio", "path": str(audio)}])
    image_ref, audio_ref = normalize_references(parsed, 124, 32000)
    assert image_ref.image.size == (2720, 2048)
    assert audio_ref.audio.shape == (2, 32000)
