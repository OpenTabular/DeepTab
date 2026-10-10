"""Regression tests for ENODE's tabular head and norm config options.

- The tabular head was a hardcoded nn.Sequential(Linear, ReLU, Dropout,
  Linear), so head_layer_sizes/head_skip_layers/head_activation and friends
  on ENODEConfig had no effect.
- The declared `norm` config field was never consumed, so setting it had no
  effect on the pooled output.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from deeptab.architectures.enode import ENODE
from deeptab.configs import ENODEConfig
from deeptab.nn.blocks.mlp import MLPhead
from deeptab.nn.blocks.node import ODSTE

NUM_INFO = {"f0": {"preprocessing": "", "dimension": 1, "categories": None}}


@pytest.mark.parametrize("flatten_output", [False, True])
def test_odste_preserves_response_channels(flatten_output):
    layer = ODSTE(in_features=4, num_trees=3, embed_dim=8, depth=2, tree_dim=3, flatten_output=flatten_output)
    with torch.no_grad():
        for channel in range(3):
            layer.response[:, channel].fill_(channel + 1)
    predictions = layer(torch.randn(5, 4, 8))
    expected = torch.arange(1, 4, dtype=predictions.dtype).view(1, 1, 3, 1).expand(5, 3, 3, 8)
    if flatten_output:
        expected = expected.flatten(1, 2)
    torch.testing.assert_close(predictions, expected)
    predictions.sum().backward()
    assert layer.response.grad is not None
    assert (layer.response.grad.abs().sum(dim=(0, 2, 3)) > 0).all()


@pytest.mark.parametrize("tree_dim", [2, 3])
def test_enode_supports_multiple_response_channels(tree_dim):
    model = ENODE(
        (NUM_INFO, {}, {}),
        num_classes=3,
        config=ENODEConfig(d_model=8, num_layers=2, layer_dim=3, depth=2, tree_dim=tree_dim),
    )
    predictions = model([torch.randn(5, 1)], [], [])
    assert predictions.shape == (5, 3)
    assert torch.isfinite(predictions).all()
    predictions.sum().backward()


class TestENODEHeadOptions:
    def test_tabular_head_is_mlphead_and_respects_head_layer_sizes(self):
        config = ENODEConfig(d_model=16, num_layers=2, layer_dim=8, depth=2, head_layer_sizes=[32])
        model = ENODE(feature_information=(NUM_INFO, {}, {}), num_classes=3, config=config)
        assert isinstance(model.tabular_head, MLPhead)
        out_features = [m.out_features for m in model.tabular_head.modules() if isinstance(m, nn.Linear)]
        assert 32 in out_features

    def test_forward_shape(self):
        config = ENODEConfig(d_model=16, num_layers=2, layer_dim=8, depth=2)
        model = ENODE(feature_information=(NUM_INFO, {}, {}), num_classes=3, config=config)
        batch_size = 4
        out = model([torch.randn(batch_size, 1)], [], [])
        assert out.shape == (batch_size, 3)


class TestENODENormWiring:
    def test_norm_is_applied_when_configured(self):
        config = ENODEConfig(d_model=16, num_layers=2, layer_dim=8, depth=2, norm="LayerNorm")
        model = ENODE(feature_information=(NUM_INFO, {}, {}), num_classes=2, config=config)
        assert model.norm_f is not None
        out = model([torch.randn(4, 1)], [], [])
        assert out.shape == (4, 2)
