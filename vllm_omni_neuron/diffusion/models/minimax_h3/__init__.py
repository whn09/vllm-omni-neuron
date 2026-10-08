# SPDX-License-Identifier: Apache-2.0
"""Neuron implementation of the MiniMax-H3 joint video+audio pipeline."""

from .pipeline_minimax_h3 import (
    PIPELINE_REGISTRY,
    MiniMaxH3Pipeline,
    get_minimax_h3_post_process_func,
)

__all__ = ["PIPELINE_REGISTRY", "MiniMaxH3Pipeline", "get_minimax_h3_post_process_func"]
