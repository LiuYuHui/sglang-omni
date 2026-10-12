# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch

from sglang_omni.models.personaplex.architecture import MIMI
from sglang_omni.models.personaplex.components.mimi import EuclideanCodebook, MimiCodec
from sglang_omni.models.personaplex.components.mimi_transformer import MimiAttention


@pytest.fixture
def random_codec() -> MimiCodec:
    """A full-size Mimi with small random weights and every codebook entry in use."""
    torch.manual_seed(0)
    codec = MimiCodec().eval()
    with torch.no_grad():
        for parameter in codec.parameters():
            parameter.normal_(std=0.05)
        for module in codec.modules():
            if hasattr(module, "embedding_sum"):
                module.embedding_sum.normal_()
                module.cluster_usage.fill_(1.0)
    return codec


@pytest.fixture
def small_codec() -> MimiCodec:
    """A small decoder with a short attention ring and nonzero codebooks."""
    torch.manual_seed(8)
    spec = replace(
        MIMI,
        dim=16,
        n_filters=4,
        ratios=(2, 2),
        num_heads=2,
        num_layers=2,
        ffn_dim=32,
        context=6,
        layer_scale=1.0,
        codebook_dim=8,
        codebook_size=32,
    )
    codec = MimiCodec(spec).eval()
    with torch.no_grad():
        for module in codec.modules():
            if isinstance(module, EuclideanCodebook):
                module.embedding_sum.normal_()
                module.cluster_usage.fill_(1.0)
            elif isinstance(module, MimiAttention):
                module.in_proj_weight.normal_(std=0.2)
            else:
                pass
    return codec
