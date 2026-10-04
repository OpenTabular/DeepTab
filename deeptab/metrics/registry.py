"""Metric registry: maps (task, family) keys to default metric lists."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

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
    "lss:lognormal": [LogNormalNLL()],
    "lss:studentt": [StudentTLoss(), CRPS(family="studentt", loc_col=1, scale_col=2)],
    "lss:gamma": [GammaDeviance(shape_col=0, rate_col=1)],
    "lss:inversegamma": [InverseGammaDeviance()],
    "lss:tweedie": [TweedieDeviance(), RootMeanSquaredError()],
    "lss:beta": [BetaBrierScore(alpha_col=0, beta_col=1)],
    "lss:poisson": [PoissonDeviance(), RootMeanSquaredError()],
    "lss:zip": [PoissonDeviance(pi_col=0, rate_col=1)],
    "lss:negativebinom": [NegativeBinomialDeviance(), RootMeanSquaredError()],
    "lss:categorical": [Accuracy(), LogLoss()],
    "lss:dirichlet": [DirichletError()],
    "lss:multinomial": [LogLoss()],
    "lss:johnsonsu": [CRPS(family="johnsonsu", loc_col=2, scale_col=3)],
    "lss:mog": [CRPS(family="mog")],
    "lss:quantile": [PinballLoss(quantile=0.5, col=1)],
}


_FAMILY_CLASS_KEYS = {
    "BetaDistribution": "beta",
    "CategoricalDistribution": "categorical",
    "DirichletDistribution": "dirichlet",
    "GammaDistribution": "gamma",
    "InverseGammaDistribution": "inversegamma",
    "JohnsonSuDistribution": "johnsonsu",
    "LogNormalDistribution": "lognormal",
    "MixtureOfGaussiansDistribution": "mog",
    "MultinomialDistribution": "multinomial",
    "NormalDistribution": "normal",
    "PoissonDistribution": "poisson",
    "Quantile": "quantile",
    "StudentTDistribution": "studentt",
    "TweedieDistribution": "tweedie",
    "ZeroInflatedPoissonDistribution": "zip",
}

_DEFAULT_PARAM_NAMES = {
    "beta": ["alpha", "beta"],
    "gamma": ["shape", "rate"],
    "johnsonsu": ["skew", "shape", "loc", "scale"],
    "studentt": ["df", "loc", "scale"],
    "zip": ["pi", "rate"],
}


class _DistributionRMSE(RootMeanSquaredError):
    def __init__(self, mean_metric: GammaDeviance | PoissonDeviance) -> None:
        self.mean_metric = mean_metric

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return super().__call__(y_true, self.mean_metric.predicted_mean(y_pred))


def get_default_lss_metrics(
    family: str | object,
    *,
    param_names: Sequence[str] | None = None,
    quantiles: Sequence[float] | None = None,
    p: float | None = None,
) -> list[DeepTabMetric]:
    """Return default LSS metrics, using family metadata to select parameter columns.

    ``family`` may be a registered family key or a distribution instance. When
    a family instance is supplied, its ``param_names`` and ``quantiles`` are
    used unless overridden by keyword arguments. The same metadata can be
    supplied with a family key, for example ``quantiles=[0.1, 0.5, 0.9]``.
    """
    family_object = None if isinstance(family, str) else family
    if isinstance(family, str):
        family_key = family.lower().removeprefix("lss:")
    else:
        family_key = _FAMILY_CLASS_KEYS.get(
            type(family).__name__, str(getattr(family, "name", "")).lower().replace(" ", "")
        )
        family_key = {
            "zeroinflatedpoisson": "zip",
            "negativebinomial": "negativebinom",
            "mixtureofgaussians": "mog",
        }.get(family_key, family_key)

    metric_list = METRIC_REGISTRY.get(f"lss:{family_key}")
    if metric_list is None:
        return []

    if param_names is None:
        param_names = getattr(family_object, "param_names", None)
    if param_names is None:
        param_names = _DEFAULT_PARAM_NAMES.get(family_key, [])
    param_indexes = {str(name): index for index, name in enumerate(param_names)}

    if family_key == "quantile":
        quantile_levels: Sequence[float] = (
            quantiles if quantiles is not None else getattr(family_object, "quantiles", [0.25, 0.5, 0.75])
        )
        quantile_values = [float(value) for value in quantile_levels]
        if not quantile_values:
            return []
        median_index = min(range(len(quantile_values)), key=lambda index: abs(quantile_values[index] - 0.5))
        median_quantile = quantile_values[median_index]
        return [PinballLoss(quantile=median_quantile, col=median_index)]

    if family_key == "studentt":
        return [
            StudentTLoss(
                df_col=param_indexes.get("df", 0),
                loc_col=param_indexes.get("loc", 1),
                scale_col=param_indexes.get("scale", 2),
            ),
            CRPS(
                family="studentt",
                loc_col=param_indexes.get("loc", 1),
                scale_col=param_indexes.get("scale", 2),
            ),
        ]
    if family_key == "johnsonsu":
        return [
            CRPS(
                family="johnsonsu",
                loc_col=param_indexes.get("loc", 2),
                scale_col=param_indexes.get("scale", 3),
            )
        ]
    if family_key == "gamma":
        deviance = GammaDeviance(shape_col=param_indexes.get("shape", 0), rate_col=param_indexes.get("rate", 1))
        return [deviance, _DistributionRMSE(deviance)]
    if family_key == "zip":
        deviance = PoissonDeviance(pi_col=param_indexes.get("pi", 0), rate_col=param_indexes.get("rate", 1))
        return [deviance, _DistributionRMSE(deviance)]
    if family_key == "tweedie":
        power = float(p if p is not None else getattr(family_object, "p", 1.5))
        return [TweedieDeviance(p=power), RootMeanSquaredError()]
    if family_key == "beta":
        return [BetaBrierScore(alpha_col=param_indexes.get("alpha", 0), beta_col=param_indexes.get("beta", 1))]

    return list(metric_list)


def get_default_metrics(
    task: str,
    family: str | object | None = None,
    *,
    param_names: Sequence[str] | None = None,
    quantiles: Sequence[float] | None = None,
    p: float | None = None,
) -> list[DeepTabMetric]:
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
    if task == "lss" and family is not None:
        return get_default_lss_metrics(family, param_names=param_names, quantiles=quantiles, p=p)
    if isinstance(family, str):
        key = f"{task}:{family}"
        if key in METRIC_REGISTRY:
            return METRIC_REGISTRY[key]
    return METRIC_REGISTRY.get(task, [])


def get_default_metrics_dict(
    task: str,
    family: str | object | None = None,
    *,
    param_names: Sequence[str] | None = None,
    quantiles: Sequence[float] | None = None,
    p: float | None = None,
) -> dict[str, DeepTabMetric]:
    """Like :func:`get_default_metrics` but returns a ``{name: metric}`` dict.

    Convenience wrapper for code paths that store metrics as dicts.
    """
    return {m.name: m for m in get_default_metrics(task, family, param_names=param_names, quantiles=quantiles, p=p)}
