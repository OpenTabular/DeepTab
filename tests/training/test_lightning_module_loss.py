"""Regression tests for loss selection in TaskModel (GH-415)."""

import pytest
import torch
import torch.nn as nn

from deeptab.architectures.experimental.trompt import Trompt
from deeptab.architectures.tabm import TabM
from deeptab.configs import MLPConfig
from deeptab.configs.experimental.trompt_config import TromptConfig
from deeptab.configs.models.tabm_config import TabMConfig
from deeptab.distributions import get_distribution
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


@pytest.mark.parametrize("model_class", [TabM, Trompt])
@pytest.mark.parametrize(
    "family_name,options",
    [
        ("poisson", {}),
        ("tweedie", {}),
        ("quantile", {"quantiles": [0.5]}),
        ("normal", {}),
        ("categorical", {"num_classes": 3}),
        ("dirichlet", {"num_classes": 3}),
    ],
)
def test_ensemble_lss_loss_receives_parameter_columns(model_class, family_name, options):
    family = get_distribution(family_name, **options)
    config = (
        TabMConfig(d_model=8, layer_sizes=[8, 4], ensemble_size=3, dropout=0.0)
        if model_class is TabM
        else TromptConfig(d_model=8, n_cycles=3, P=4)
    )
    task = TaskModel(
        model_class=model_class,
        config=config,
        feature_information=({f"feature_{index}": {"dimension": 1} for index in range(3)}, {}, {}),
        num_classes=family.param_count,
        lss=True,
        family=family,
    )
    predictions = task([torch.randn(5, 1) for _ in range(3)], [], [])
    targets = (
        torch.tensor([[0.2, 0.3, 0.5]]).repeat(5, 1)
        if family_name == "dirichlet"
        else (torch.arange(5) % 3).reshape(-1, 1)
    )

    assert predictions.shape == (5, 3, family.param_count)
    loss = task.compute_loss(predictions, targets)
    assert isinstance(loss, torch.Tensor)
    expected = torch.stack(
        [family.compute_loss(predictions[:, member], targets.squeeze(-1)) for member in range(3)]
    ).sum()
    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(loss)
    transformed = family(predictions.mean(dim=1))
    assert transformed.shape == (5, family.param_count)
    assert torch.isfinite(transformed).all()
    loss.backward()
    gradients = [parameter.grad for parameter in task.estimator.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
