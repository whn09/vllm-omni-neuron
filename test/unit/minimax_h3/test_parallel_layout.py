# SPDX-License-Identifier: Apache-2.0
"""CPU tests for MiniMax-H3's context-parallel sequence layout."""

import pytest
import torch

from vllm_omni_neuron.diffusion.models.minimax_h3.parallel import (
    SHARD_ALIGN,
    sequence_shard,
    shard_runs,
)


@pytest.mark.parametrize("length", [10211, 10216, 37739, 19313])
@pytest.mark.parametrize("num_shards", [2, 4, 8])
def test_sequence_shards_tile_the_padded_sequence(length, num_shards):
    shards = [sequence_shard(length, num_shards, rank) for rank in range(num_shards)]
    assert len({shard.shard_length for shard in shards}) == 1
    assert shards[0].shard_length % SHARD_ALIGN == 0
    assert shards[0].start == 0
    for left, right in zip(shards, shards[1:]):
        assert left.end == right.start
    assert shards[-1].end == length + shards[-1].num_pad
    assert sum(shard.real_rows for shard in shards) == length
    # The pad is shorter than a shard, so no rank holds only padding.
    assert all(shard.real_rows > 0 for shard in shards)


def test_shard_runs_clips_shifts_and_stretches_over_pad():
    runs = ((0, 29), (29, 443), (443, 10211))
    run_rows = torch.tensor([7, 3, 5])
    shard = sequence_shard(10211, 4, 3)
    local, rows = shard_runs(runs, run_rows, shard.start, min(shard.end, 10211), pad_to=shard.shard_length)
    assert local == ((0, shard.shard_length),)
    assert rows.tolist() == [5]

    shard = sequence_shard(10211, 4, 0)
    local, rows = shard_runs(runs, run_rows, shard.start, shard.end, pad_to=shard.shard_length)
    assert local == ((0, 29), (29, 443), (443, shard.shard_length))
    assert rows.tolist() == [7, 3, 5]
