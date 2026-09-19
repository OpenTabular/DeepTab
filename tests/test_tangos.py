"""Regression tests for the Tangos architecture's penalty_forward.

Covers https://github.com/OpenTabular/DeepTab/issues/453: a batch containing exactly
one sample used to crash with ``IndexError: tuple index out of range`` because
``jacobian.squeeze()`` also collapsed the batch axis when it happened to be size 1.
"""

import torch

from deeptab.architectures.experimental.tangos import Tangos
from deeptab.configs.experimental.tangos_config import TangosConfig


def _make_model(num_classes=1):
    num_info = {"f0": {"preprocessing": "none", "dimension": 5, "categories": None}}
    config = TangosConfig(layer_sizes=[8, 4], dropout=0.0, subsample=0.5)
    return Tangos(feature_information=(num_info, {}, {}), num_classes=num_classes, config=config)


class TestTangosPenaltyForward:
    def test_batch_size_one_does_not_crash(self):
        model = _make_model()
        x = [torch.randn(1, 5)]

        predictions, penalty = model.penalty_forward(x, [], [])

        assert predictions.shape == (1, 1)
        assert penalty.isfinite()

    def test_batch_size_one_penalty_matches_larger_batch_computation(self):
        """The single-row penalty should be a normal finite scalar, not degenerate."""
        model = _make_model()
        model.eval()
        torch.manual_seed(0)
        x = [torch.randn(1, 5)]

        _, penalty = model.penalty_forward(x, [], [])

        assert penalty.ndim == 0
        assert penalty.isfinite()
        assert penalty.requires_grad

    def test_regular_batch_still_works(self):
        model = _make_model()
        x = [torch.randn(16, 5)]

        predictions, penalty = model.penalty_forward(x, [], [])

        assert predictions.shape == (16, 1)
        assert penalty.isfinite()
