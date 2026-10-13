"""Regression tests for GH-443: fit(train_metrics=/val_metrics=) argument order and dtype.

``TaskModel.training_step``/``validation_step`` must call each metric as
``metric_fn(y_true, y_pred)`` (matching ``DeepTabMetric.__call__``'s documented
contract), with both arguments converted to detached, host-side numpy arrays
before the call.
"""

# pyright: reportOptionalMemberAccess=false
# pyright: reportArgumentType=false

import numpy as np
import pandas as pd
import pytest

from deeptab.configs import MLPConfig, TrainerConfig
from deeptab.metrics import DeepTabMetric, MeanAbsoluteError
from deeptab.models.mlp import MLPRegressor

N = 64
RNG = np.random.default_rng(0)
X_reg = pd.DataFrame(RNG.standard_normal((N, 4)), columns=[f"f{i}" for i in range(4)])
# Offset far outside the range an untrained network's small random-init weights could
# produce, so y_true and y_pred are trivially distinguishable regardless of swap-order.
y_reg = RNG.standard_normal(N) * 10 + 5000

_FAST_TRAINER = TrainerConfig(max_epochs=1, batch_size=32, patience=1)


class _ArgOrderProbeMetric(DeepTabMetric):
    """Records the raw arguments it was called with, without transforming them."""

    name = "probe"
    higher_is_better = False
    needs_raw = False

    def __init__(self):
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, y_true, y_pred):
        self.calls.append((np.asarray(y_true), np.asarray(y_pred)))
        return 0.0


class TestFitMetricsArgumentOrder:
    def test_train_metrics_receive_y_true_then_y_pred(self):
        probe = _ArgOrderProbeMetric()
        model = MLPRegressor(model_config=MLPConfig(layer_sizes=[16]), trainer_config=_FAST_TRAINER)
        model.fit(X_reg, y_reg, train_metrics={"probe": probe}, accelerator="cpu")

        assert probe.calls, "probe metric was never invoked"
        y_true, y_pred = probe.calls[0]
        # The true targets sit around 5000; an untrained network's output does not.
        assert abs(y_true.mean()) > 1000
        assert abs(y_pred.mean()) < 1000

    def test_val_metrics_receive_y_true_then_y_pred(self):
        probe = _ArgOrderProbeMetric()
        model = MLPRegressor(model_config=MLPConfig(layer_sizes=[16]), trainer_config=_FAST_TRAINER)
        model.fit(X_reg, y_reg, val_metrics={"probe": probe}, accelerator="cpu")

        assert probe.calls, "probe metric was never invoked"
        y_true, y_pred = probe.calls[0]
        assert abs(y_true.mean()) > 1000
        assert abs(y_pred.mean()) < 1000


class TestFitMetricsAcceptTorchTensors:
    """Regression test (GH-443): built-in numpy-based metrics used to crash here."""

    def test_train_metrics_with_builtin_metric_does_not_crash(self):
        model = MLPRegressor(model_config=MLPConfig(layer_sizes=[16]), trainer_config=_FAST_TRAINER)
        model.fit(X_reg, y_reg, train_metrics={"mae": MeanAbsoluteError()}, accelerator="cpu")
        assert np.isfinite(model._trainer.callback_metrics["train_mae"].item())

    def test_val_metrics_with_builtin_metric_does_not_crash(self):
        model = MLPRegressor(model_config=MLPConfig(layer_sizes=[16]), trainer_config=_FAST_TRAINER)
        model.fit(X_reg, y_reg, val_metrics={"mae": MeanAbsoluteError()}, accelerator="cpu")
        assert np.isfinite(model._trainer.callback_metrics["val_mae"].item())


class TestFitMetricsSupportTorchMetrics:
    """torchmetrics ``Metric`` objects must stay on-device and use (preds, target)."""

    def test_train_metrics_with_torchmetrics_object_does_not_crash(self):
        from torchmetrics.regression import MeanAbsoluteError as TorchMeanAbsoluteError

        model = MLPRegressor(model_config=MLPConfig(layer_sizes=[16]), trainer_config=_FAST_TRAINER)
        model.fit(X_reg, y_reg, train_metrics={"mae": TorchMeanAbsoluteError()}, accelerator="cpu")
        assert np.isfinite(model._trainer.callback_metrics["train_mae"].item())

    def test_val_metrics_with_torchmetrics_object_does_not_crash(self):
        from torchmetrics.regression import MeanAbsoluteError as TorchMeanAbsoluteError

        model = MLPRegressor(model_config=MLPConfig(layer_sizes=[16]), trainer_config=_FAST_TRAINER)
        model.fit(X_reg, y_reg, val_metrics={"mae": TorchMeanAbsoluteError()}, accelerator="cpu")
        assert np.isfinite(model._trainer.callback_metrics["val_mae"].item())


class _ClassificationProbabilityProbe(_ArgOrderProbeMetric):
    input_space = "probabilities"


@pytest.mark.parametrize("classes", [2, 3])
def test_classifier_fit_metric_columns_match_predict_proba_and_encoded_labels(classes, tmp_path):
    import torch

    from deeptab.configs import PreprocessingConfig
    from deeptab.models import MLPClassifier
    from deeptab.training import TaskModel

    class_labels = np.array([10, 30, 70])[:classes]
    encoded_targets = np.arange(N) % classes
    targets = class_labels[encoded_targets]
    train_probe = _ClassificationProbabilityProbe()
    val_probe = _ClassificationProbabilityProbe()
    model = MLPClassifier(
        model_config=MLPConfig(layer_sizes=[8]),
        preprocessing_config=PreprocessingConfig(numerical_method="standardization"),
        trainer_config=TrainerConfig(max_epochs=1, batch_size=8, checkpoint_path=str(tmp_path)),
        random_state=17,
    )
    model.build_model(
        X_reg.iloc[:48],
        targets[:48],
        X_val=X_reg.iloc[48:],
        y_val=targets[48:],
        train_metrics={"probe": train_probe},
        val_metrics={"probe": val_probe},
    )
    task = model._task_model
    assert isinstance(task, TaskModel)
    linear_layers = [module for module in task.estimator.modules() if isinstance(module, torch.nn.Linear)]
    bias = linear_layers[-1].bias
    assert bias is not None
    with torch.no_grad():
        for parameter in task.parameters():
            parameter.zero_()
        if classes == 2:
            bias.fill_(0.25)
        else:
            bias.copy_(torch.tensor([0.25, 0.75, -0.25]))
    model.fit(
        X_reg.iloc[:48],
        targets[:48],
        X_val=X_reg.iloc[48:],
        y_val=targets[48:],
        rebuild=False,
        lr=0,
        shuffle=False,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    np.testing.assert_array_equal(model.classes_, class_labels)
    received_targets, received = train_probe.calls[0]
    np.testing.assert_array_equal(received_targets.ravel(), encoded_targets[:8])
    expected = model.predict_proba(X_reg.iloc[:8])
    np.testing.assert_allclose(received, expected, rtol=0, atol=0)
    np.testing.assert_allclose(val_probe.calls[0][1], model.predict_proba(X_reg.iloc[48:56]), rtol=0, atol=0)
    np.testing.assert_array_equal(model.predict(X_reg.iloc[:8]), class_labels[received.argmax(axis=1)])
