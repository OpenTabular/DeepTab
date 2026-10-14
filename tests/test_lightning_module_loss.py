"""Regression tests for loss selection in TaskModel."""

import torch.nn as nn

from deeptab.configs import MLPConfig
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
        custom = nn.BCEWithLogitsLoss()
        task = _make_task_model(num_classes=2, loss_fct=custom)
        assert task.loss_fct is custom

    def test_multiclass_defaults_to_cross_entropy(self):
        task = _make_task_model(num_classes=3)
        assert isinstance(task.loss_fct, nn.CrossEntropyLoss)
