# SPDX-License-Identifier: Apache-2.0
"""The staged video encoder computes exactly what the one-module encoder does."""

import torch
from torch import nn

from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_minimax_h3 import (
    MiniMaxH3VideoEncoder3d,
    NeuronMiniMaxH3VideoEncoder,
    encoder_stages,
)


def test_stages_match_the_whole_encoder():
    torch.manual_seed(0)
    encoder = MiniMaxH3VideoEncoder3d(block_out_channels=(32,) * 6, out_channels=8).eval()
    quant_conv = nn.Conv3d(8, 8, 1)
    x = torch.randn(1, 3, 17, 32, 32)
    with torch.no_grad():
        whole = NeuronMiniMaxH3VideoEncoder(encoder, quant_conv)(x)
        staged = x
        for stage in encoder_stages(encoder, quant_conv):
            staged = stage(staged)
    assert len(encoder_stages(encoder, quant_conv)) == 9
    torch.testing.assert_close(staged, whole, rtol=0, atol=0)
