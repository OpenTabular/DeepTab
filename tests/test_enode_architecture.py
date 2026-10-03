"""Regression tests for ENODE's tabular head and norm config options.

- The tabular head was a hardcoded nn.Sequential(Linear, ReLU, Dropout,
  Linear), so head_layer_sizes/head_skip_layers/head_activation and friends
  on ENODEConfig had no effect.
- The declared `norm` config field was never consumed, so setting it had no
  effect on the pooled output.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deeptab.architectures.enode import ENODE
from deeptab.configs import ENODEConfig
from deeptab.nn.blocks.mlp import MLPhead

NUM_INFO = {"f0": {"preprocessing": "", "dimension": 1, "categories": None}}


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
