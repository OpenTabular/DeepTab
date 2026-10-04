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
