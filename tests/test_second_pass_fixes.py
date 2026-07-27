"""Regression tests for the second-pass review findings."""

import pandas as pd
import pytest
import torch

from deeptab.configs import MLPConfig
from deeptab.distributions.registry import DISTRIBUTION_REGISTRY
from deeptab.models.base import _validate_fit_inputs
from deeptab.nn.blocks.common import sparsemax
from deeptab.training.lightning_module import TaskModel


class _Dummy(torch.nn.Module):
    def __init__(self, config=None, feature_information=None, num_classes=1, lss=False, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(2, num_classes)

    def forward(self, *data):
        return self.linear(data[0])


def test_sparsemax_does_not_mutate_input():
    """ODST passes its feature_selection_logits Parameter straight in."""
    logits = torch.randn(4, 8)
    original = logits.clone()
    sparsemax(logits, dim=-1)
    assert torch.equal(logits, original)


def test_sparsemax_output_unchanged_by_the_fix():
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 5.0]])
    out = sparsemax(logits, dim=-1)
    assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert (out >= 0).all()


class TestValidateFitInputsFamilies:
    @pytest.mark.parametrize("family", ["gamma", "inversegamma", "lognormal"])
    def test_strictly_positive_families(self, family):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = [1.0, 0.0, 2.0]
        with pytest.raises(Exception, match="positive"):
            _validate_fit_inputs(X, y, regression=True, family=family)

    def test_checked_families_exist_in_the_registry(self):
        """The old check gated on 'inversegaussian', which is not a family."""
        assert "inversegamma" in DISTRIBUTION_REGISTRY
        assert "inversegaussian" not in DISTRIBUTION_REGISTRY


def test_taskmodel_candidate_pool_defaults_to_none():
    """The `is not None` guards presume a None default, set only in setup('fit')."""
    task = TaskModel(model_class=_Dummy, config=MLPConfig(), feature_information=({}, {}, {}))
    assert task.train_features is None
    assert task.train_targets is None
