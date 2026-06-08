"""Metric registry: maps (task, family) keys to default metric lists."""

from __future__ import annotations

from .base import DeepTabMetric
from .classification import AUROC, Accuracy, LogLoss
from .distributional import (
    CRPS,
    BetaBrierScore,
    DirichletError,
    GammaDeviance,
    InverseGammaDeviance,
    LogNormalNLL,
    NegativeBinomialDeviance,
    PoissonDeviance,
    StudentTLoss,
    TweedieDeviance,
)
from .regression import MeanAbsoluteError, PinballLoss, R2Score, RootMeanSquaredError

# ---------------------------------------------------------------------------
# Registry definition
# ---------------------------------------------------------------------------
# Keys follow the pattern "<task>" or "<task>:<family>".
# The first entry in each list is treated as the *primary* metric.
# All metrics here receive already-transformed distribution parameters
# (raw=False predictions).  NegativeLogLikelihood is intentionally excluded
# from this registry because it requires raw logits; use model.score() for NLL.

METRIC_REGISTRY: dict[str, list[DeepTabMetric]] = {
    # ---- Point-estimate tasks ----
    "regression": [RootMeanSquaredError(), MeanAbsoluteError(), R2Score()],
    "classification": [Accuracy(), AUROC(), LogLoss()],
    # ---- LSS families ----
    "lss:normal": [CRPS(family="normal"), RootMeanSquaredError(), MeanAbsoluteError()],
    "lss:lognormal": [LogNormalNLL(), CRPS(family="lognormal"), RootMeanSquaredError()],
    "lss:studentt": [StudentTLoss(), CRPS(family="studentt")],
    "lss:gamma": [GammaDeviance(), RootMeanSquaredError()],
    "lss:inversegamma": [InverseGammaDeviance(), GammaDeviance()],
    "lss:tweedie": [TweedieDeviance(), RootMeanSquaredError()],
    "lss:beta": [BetaBrierScore(), RootMeanSquaredError()],
    "lss:poisson": [PoissonDeviance(), RootMeanSquaredError()],
    "lss:zip": [PoissonDeviance(), RootMeanSquaredError()],
    "lss:negativebinom": [NegativeBinomialDeviance(), RootMeanSquaredError()],
    "lss:categorical": [Accuracy(), LogLoss()],
    "lss:dirichlet": [DirichletError()],
    "lss:multinomial": [LogLoss()],
    "lss:johnsonsu": [CRPS(family="johnsonsu"), RootMeanSquaredError()],
    "lss:mog": [CRPS(family="normal"), RootMeanSquaredError()],
    "lss:quantile": [PinballLoss(quantile=0.5)],
}


def get_default_metrics(task: str, family: str | None = None) -> list[DeepTabMetric]:
    """Return the default list of metrics for a given task and distribution family.

    Parameters
    ----------
    task : str
        One of ``"regression"``, ``"classification"``, or ``"lss"``.
    family : str, optional
        Distribution family key used for LSS tasks, e.g. ``"normal"``,
        ``"gamma"``, ``"poisson"``.  Ignored for non-LSS tasks.

    Returns
    -------
    list[DeepTabMetric]
        Ordered list of metric instances.  The first entry is the primary
        metric.  Returns an empty list when the combination is unknown.
    """
    if family is not None:
        key = f"{task}:{family}"
        if key in METRIC_REGISTRY:
            return METRIC_REGISTRY[key]
    return METRIC_REGISTRY.get(task, [])


def get_default_metrics_dict(task: str, family: str | None = None) -> dict[str, DeepTabMetric]:
    """Like :func:`get_default_metrics` but returns a ``{name: metric}`` dict.

    Convenience wrapper for code paths that store metrics as dicts.
    """
    return {m.name: m for m in get_default_metrics(task, family)}
