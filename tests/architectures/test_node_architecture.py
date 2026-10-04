"""Regression test for NODE's norm config option.

The declared `norm` config field was never consumed by NODE, so setting it
had no effect on the pooled tree output.
"""

from __future__ import annotations

import torch

from deeptab.architectures.node import NODE
from deeptab.configs import NODEConfig

NUM_INFO = {"f0": {"preprocessing": "", "dimension": 1, "categories": None}}


class TestNODENormWiring:
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
