"""Metric utilities for tabular model evaluation.

Every metric is a :class:`~deeptab.metrics.DeepTabMetric` subclass that
exposes three attributes the framework reads automatically:

* **``name``** -- short string identifier used as the dict key in
  ``model.evaluate()`` results and as the suffix in training-log entries
  (e.g. ``val_rmse``, ``val_crps``).

* **``higher_is_better``** -- ``True`` when a *larger* value is better
  (accuracy, AUROC, R2, log-score), ``False`` when a *smaller* value is
  better (MSE, MAE, NLL, deviances).  The training loop and HPO use this
  to set the optimisation direction automatically.

* **``needs_raw``** -- ``False`` (default) means the metric receives
  *already-transformed* distribution parameters from
  ``model.predict(X, raw=False)``, e.g. ``[mean, std]`` for a Normal model.
  ``True`` means the metric receives *raw model logits* and applies
  parameter transforms itself (only :class:`NegativeLogLikelihood` uses this).

Quick start
-----------
Import any metric and call it like a function::

    from deeptab.metrics import RootMeanSquaredError, CRPS, Accuracy
    import numpy as np

    rmse = RootMeanSquaredError()
    print(rmse.name)              # "rmse"
    print(rmse.higher_is_better)  # False -- lower RMSE is better

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.0, 2.9])
    print(rmse(y_true, y_pred))   # 0.0816...

    # Works with 2-D LSS parameter arrays too -- first column is the mean
    y_pred_lss = np.column_stack([y_pred, np.ones(3) * 0.5])  # [mean, std]
    print(rmse(y_true, y_pred_lss))   # same result

Pass metrics to ``model.fit()`` for live training logging::

    from deeptab.metrics import CRPS, MeanAbsoluteError
    from deeptab.models import MambularLSS

    model = MambularLSS()
    model.fit(
        X_train, y_train,
        val_metrics={
            "crps": CRPS(family="normal"),   # logged as "val_crps"
            "mae":  MeanAbsoluteError(),      # logged as "val_mae"
        },
    )

Pass metrics to ``model.evaluate()`` for post-hoc scoring::

    scores = model.evaluate(X_test, y_test)
    # Returns e.g. {"crps": 0.32, "rmse": 1.45}

Auto-select default metrics via the registry::

    from deeptab.metrics import get_default_metrics

    metrics = get_default_metrics("lss", family="normal")
    # [CRPS(family='normal'), RootMeanSquaredError(), MeanAbsoluteError()]

    metrics = get_default_metrics("regression")
    # [RootMeanSquaredError(), MeanAbsoluteError(), R2Score()]

    metrics = get_default_metrics("classification")
    # [Accuracy(), AUROC(), LogLoss()]
"""

from .base import DeepTabMetric

# Classification
from .classification import AUPRC, AUROC, Accuracy, BrierScore, ExpectedCalibrationError, F1Score, LogLoss

# Distributional / LSS
from .distributional import (
    CRPS,
    BetaBrierScore,
    CoverageProbability,
    DirichletError,
    EnergyScore,
    GammaDeviance,
    IntervalScore,
    InverseGammaDeviance,
    LogNormalNLL,
    LogScore,
    NegativeBinomialDeviance,
    NegativeLogLikelihood,
    PoissonDeviance,
    ProbabilityIntegralTransform,
    SharpnessScore,
    StudentTLoss,
    TweedieDeviance,
)

# Registry
from .registry import METRIC_REGISTRY, get_default_metrics, get_default_metrics_dict

# Regression
from .regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    PinballLoss,
    R2Score,
    RootMeanSquaredError,
)

__all__ = [
    "AUPRC",
    "AUROC",
    "CRPS",
    # Registry
    "METRIC_REGISTRY",
    # Classification
    "Accuracy",
    "BetaBrierScore",
    "BrierScore",
    "CoverageProbability",
    # Base
    "DeepTabMetric",
    "DirichletError",
    "EnergyScore",
    "ExpectedCalibrationError",
    "F1Score",
    "GammaDeviance",
    "IntervalScore",
    "InverseGammaDeviance",
    "LogLoss",
    "LogNormalNLL",
    "LogScore",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    # Regression
    "MeanSquaredError",
    "NegativeBinomialDeviance",
    # Distributional
    "NegativeLogLikelihood",
    "PinballLoss",
    "PoissonDeviance",
    "ProbabilityIntegralTransform",
    "R2Score",
    "RootMeanSquaredError",
    "SharpnessScore",
    "StudentTLoss",
    "TweedieDeviance",
    "get_default_metrics",
    "get_default_metrics_dict",
]
