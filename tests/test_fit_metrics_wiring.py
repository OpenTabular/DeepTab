"""Regression tests for user metrics passed to fit(train_metrics=/val_metrics=).

Two defects lived at the same two call sites: metrics were invoked as
``metric_fn(preds, labels)`` although DeepTabMetric is documented as
``__call__(y_true, y_pred)``, and they were handed live torch tensors although
every shipped metric is numpy-based.
"""

import numpy as np
import pandas as pd

from deeptab.metrics import DeepTabMetric, MeanAbsoluteError
from deeptab.models import MLPLSS, MLPRegressor

QUIET = {
    "accelerator": "cpu",
    "devices": 1,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "logger": False,
}


def _data(n=48, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    return X, X["a"].to_numpy() * 2.0


class _Probe(DeepTabMetric):
    name = "probe"
    higher_is_better = False
    needs_raw = False

    def __init__(self):
        self.calls = []

    def __call__(self, y_true, y_pred):
        self.calls.append((np.asarray(y_true).shape, np.asarray(y_pred).shape))
        return 0.0


def test_lss_metric_receives_targets_as_y_true():
    """The distribution parameters must arrive as y_pred, not as y_true."""
    X, y = _data()
    probe = _Probe()
    model = MLPLSS()
    model.fit(X, y, family="normal", val_metrics={"probe": probe}, max_epochs=1, batch_size=16, **QUIET)

    y_true_shape, y_pred_shape = probe.calls[0]
    assert y_true_shape[-1] == 1, f"y_true should be the (n, 1) targets, got {y_true_shape}"
    assert y_pred_shape[-1] == 2, f"y_pred should be the 2 Normal parameters, got {y_pred_shape}"


def test_builtin_metric_works_as_train_metric():
    """Numpy-based metrics must not receive grad-carrying device tensors."""
    X, y = _data()
    model = MLPRegressor()
    model.fit(X, y, train_metrics={"mae": MeanAbsoluteError()}, max_epochs=1, batch_size=16, **QUIET)
    assert model.predict(X).shape == (48,)


def test_builtin_metric_works_as_val_metric():
    X, y = _data()
    model = MLPRegressor()
    model.fit(X, y, val_metrics={"mae": MeanAbsoluteError()}, max_epochs=1, batch_size=16, **QUIET)
    assert model.predict(X).shape == (48,)
