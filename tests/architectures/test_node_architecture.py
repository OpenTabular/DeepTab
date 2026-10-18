"""Regression test for NODE's norm config option.

The declared `norm` config field was never consumed by NODE, so setting it
had no effect on the pooled tree output.
"""

from __future__ import annotations

import pytest
import torch

from deeptab.architectures.node import NODE
from deeptab.configs import NODEConfig
from deeptab.nn.blocks.node import ODST

NUM_INFO = {"f0": {"preprocessing": "", "dimension": 1, "categories": None}}


class TestNODENormWiring:
    @pytest.mark.parametrize("tree_dim", [1, 2, 3])
    @pytest.mark.parametrize("with_embeddings", [False, True])
    def test_response_channel_width_and_backward(self, tree_dim, with_embeddings):
        config = NODEConfig(
            d_model=8,
            num_layers=2,
            layer_dim=3,
            depth=2,
            tree_dim=tree_dim,
            norm="LayerNorm",
            use_embeddings=with_embeddings,
        )
        model = NODE((NUM_INFO, {}, {}), num_classes=3, config=config)
        predictions = model([torch.randn(5, 1)], [], [])
        assert predictions.shape == (5, 3)
        assert torch.isfinite(predictions).all()
        predictions.sum().backward()
        for layer in model.block:
            assert isinstance(layer, ODST)
            gradient = layer.response.grad
            assert gradient is not None
            assert torch.isfinite(gradient).all()

    def test_norm_is_applied_when_configured(self):
        config = NODEConfig(num_layers=2, layer_dim=8, depth=2, norm="LayerNorm")
        model = NODE(feature_information=(NUM_INFO, {}, {}), num_classes=2, config=config)
        assert model.norm_f is not None
        out = model([torch.randn(4, 1)], [], [])
        assert out.shape == (4, 2)

    def test_norm_defaults_to_none(self):
        config = NODEConfig(num_layers=2, layer_dim=8, depth=2)
        model = NODE(feature_information=(NUM_INFO, {}, {}), num_classes=2, config=config)
        assert model.norm_f is None
