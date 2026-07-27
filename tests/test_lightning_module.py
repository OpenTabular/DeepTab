"""Regression tests for TaskModel wiring in deeptab.training.lightning_module."""

import numpy as np
import pandas as pd
import pytest
import torch.nn as nn

from deeptab.configs import MLPConfig
from deeptab.models import MLPRegressor
from deeptab.training.lightning_module import TaskModel


class _DummyEstimator(nn.Module):
    def __init__(self, config=None, feature_information=None, num_classes=1, lss=False, **kwargs):
        super().__init__()
        self.linear = nn.Linear(4, num_classes)

    def forward(self, *data):
        return self.linear(data[0])


def _make_task_model(**overrides):
    kwargs = {
        "model_class": _DummyEstimator,
        "config": MLPConfig(),
        "feature_information": ({}, {}, {}),
        "num_classes": 1,
    }
    kwargs.update(overrides)
    return TaskModel(**kwargs)


class TestLossSelection:
    def test_regression_custom_loss_respected(self):
        """Regression test: num_classes=1 must not overwrite a custom loss with MSE."""
        custom = nn.HuberLoss()
        task = _make_task_model(loss_fct=custom)
        assert task.loss_fct is custom

    def test_regression_defaults_to_mse(self):
        task = _make_task_model()
        assert isinstance(task.loss_fct, nn.MSELoss)

    def test_binary_custom_loss_respected(self):
        custom = nn.BCEWithLogitsLoss(pos_weight=None)
        task = _make_task_model(num_classes=2, loss_fct=custom)
        assert task.loss_fct is custom

    def test_multiclass_defaults_to_cross_entropy(self):
        task = _make_task_model(num_classes=3)
        assert isinstance(task.loss_fct, nn.CrossEntropyLoss)


class TestValLossTracking:
    def test_val_losses_excludes_sanity_check(self):
        """Regression test: the sanity-check validation must not enter val_losses.

        Recording it shifts epoch_val_loss_at(e) to return the loss of epoch
        e-1, which skews HPO pruning baselines.
        """
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"a": rng.randn(60), "b": rng.randn(60)})
        y = rng.randn(60)
        model = MLPRegressor()
        model.fit(X, y, max_epochs=2, batch_size=16, accelerator="cpu")
        assert len(model._task_model.val_losses) == 2

    def test_epoch_val_loss_at_out_of_range(self):
        task = _make_task_model()
        assert task.epoch_val_loss_at(0) == pytest.approx(float("inf"))
