"""Distributional / LSS evaluation metrics (CRPS, log-score, deviances, calibration).

All metrics expect ``y_pred`` to be **already-transformed** distribution
parameters (i.e. the output of ``model.predict(X, raw=False)``), unless the
metric's :attr:`~DeepTabMetric.needs_raw` attribute is ``True``.

Understanding ``needs_raw``
---------------------------
Most metrics set ``needs_raw = False`` (the default).  They receive the
output of the distribution's ``forward()`` method -- i.e. parameters *after*
transforms such as ``softplus`` have been applied to guarantee positivity.
For a Normal distribution model this looks like ``[[mean_0, std_0], ...]``.

:class:`NegativeLogLikelihood` is the only class here with ``needs_raw = True``.
It calls ``distribution.compute_loss()`` directly, which applies the
parameter transforms *internally*.  Passing already-transformed values would
double-transform them and give wrong results.

Understanding ``higher_is_better``
----------------------------------
Proper scoring rules and deviances are *losses* -- lower values are better,
so they use the default ``higher_is_better = False``.
:class:`LogScore` (which equals ``-NLL``) is the exception: a *higher*
log-score indicates a better-calibrated forecast.

Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 18 18 14 20

   * - Class
     - ``name``
     - Family
     - ``higher_is_better``
     - ``needs_raw``
   * - :class:`NegativeLogLikelihood`
     - ``"nll"``
     - any
     - ``False``
     - ``True``
   * - :class:`LogScore`
     - ``"log_score"``
     - any
     - ``True``
     - ``True``
   * - :class:`CRPS`
     - ``"crps"``
     - continuous
     - ``False``
     - ``False``
   * - :class:`IntervalScore`
     - ``"interval_score"``
     - any
     - ``False``
     - ``False``
   * - :class:`PoissonDeviance`
     - ``"poisson_deviance"``
     - poisson / zip
     - ``False``
     - ``False``
   * - :class:`GammaDeviance`
     - ``"gamma_deviance"``
     - gamma / inversegamma
     - ``False``
     - ``False``
   * - :class:`TweedieDeviance`
     - ``"tweedie_deviance"``
     - tweedie
     - ``False``
     - ``False``
   * - :class:`NegativeBinomialDeviance`
     - ``"nb_deviance"``
     - negativebinom
     - ``False``
     - ``False``
   * - :class:`StudentTLoss`
     - ``"studentt_nll"``
     - studentt
     - ``False``
     - ``False``
   * - :class:`CoverageProbability`
     - ``"coverage"``
     - any
     - ``True``
     - ``False``
   * - :class:`SharpnessScore`
     - ``"sharpness"``
     - any
     - ``False``
     - ``False``
   * - :class:`ProbabilityIntegralTransform`
     - ``"pit"``
     - normal
     - ``False``
     - ``False``
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from .base import DeepTabMetric

if TYPE_CHECKING:
    from deeptab.distributions.base import BaseDistribution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _col(arr: np.ndarray, idx: int) -> np.ndarray:
    """Extract column *idx* from a 2-D array, or return the flat 1-D array."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        return arr[:, idx]
    return arr.ravel()


# ---------------------------------------------------------------------------
# Proper-scoring rules
# ---------------------------------------------------------------------------


class NegativeLogLikelihood(DeepTabMetric):
    """Negative Log-Likelihood computed via the distribution's ``compute_loss``.

    This metric requires raw model logits (``needs_raw=True``) and the
    distribution family object, because ``compute_loss`` applies parameter
    transforms internally.

    Parameters
    ----------
    distribution : BaseDistribution
        The fitted distribution object (e.g. ``model.task_model.family``).
    """

    name = "nll"
    higher_is_better = False
    needs_raw = True

    def __init__(self, distribution: BaseDistribution) -> None:
        self.distribution = distribution

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        import torch

        y_true_t = torch.tensor(np.asarray(y_true, dtype=np.float32))
        y_pred_t = torch.tensor(np.asarray(y_pred, dtype=np.float32))
        with torch.no_grad():
            loss = self.distribution.compute_loss(y_pred_t, y_true_t)
        return float(loss.detach().cpu().numpy())

    def __repr__(self) -> str:
        return f"NegativeLogLikelihood(distribution={self.distribution!r})"


class LogScore(DeepTabMetric):
    """Log Score (higher is better = -NLL).

    Convenience wrapper around :class:`NegativeLogLikelihood`.

    Parameters
    ----------
    distribution : BaseDistribution
        The fitted distribution object.
    """

    name = "log_score"
    higher_is_better = True
    needs_raw = True

    def __init__(self, distribution: BaseDistribution) -> None:
        self._nll = NegativeLogLikelihood(distribution)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return -self._nll(y_true, y_pred)

    def __repr__(self) -> str:
        return f"LogScore(distribution={self._nll.distribution!r})"


class CRPS(DeepTabMetric):
    """Continuous Ranked Probability Score (CRPS) for univariate distributions.

    Uses vectorised Gaussian ``properscoring`` routines for supported
    parameter layouts when available. Otherwise returns a mean absolute
    error fallback without estimating predictive spread from observed targets.

    Expected ``y_pred`` format (2-D array, columns are distribution parameters):

    * **Normal / LogNormal**: ``[loc, scale]``
    * **StudentT**: ``[df, loc, scale]``
    * **JohnsonSU**: ``[skew, shape, loc, scale]``
    * **Mixture of Gaussians**: ``[weights..., means..., scales...]``
        * All other families: ``[mean, ...]``; returns mean absolute error of
            the first predicted column.

    With ``properscoring`` installed, the exact Gaussian CRPS is computed for
    ``normal``. Student-T and Johnson SU use a Gaussian location/scale
    approximation, and mixtures use a moment-matched Gaussian. Without that
    dependency, all families fall back to mean absolute error.

    Notes
    -----
    The mean absolute error fallback is a point-forecast score, not a score
    of the full predictive distribution. It remains exposed under the
    ``"crps"`` metric name for compatibility. For unsupported family layouts,
    supply a predicted mean in the first column rather than raw family parameters.

    Parameters
    ----------
    family : str, optional
        Distribution family key (e.g. ``"normal"``, ``"studentt"``).
        Selects the parameter layout and approximation.
    loc_col, scale_col : int, optional
        Override the family's location and scale column indices.
    """

    name = "crps"
    higher_is_better = False

    def __init__(
        self,
        family: str = "normal",
        loc_col: int | None = None,
        scale_col: int | None = None,
    ) -> None:
        self.family = family
        self.loc_col = loc_col
        self.scale_col = scale_col

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float)

        if self.family in ("normal", "lognormal", "studentt", "johnsonsu"):
            default_columns = {"studentt": (1, 2), "johnsonsu": (2, 3)}
            loc_col, scale_col = default_columns.get(self.family, (0, 1))
            loc = _col(y_pred, self.loc_col if self.loc_col is not None else loc_col)
            scale = np.clip(_col(y_pred, self.scale_col if self.scale_col is not None else scale_col), 1e-9, None)
        elif self.family == "mog":
            if y_pred.ndim != 2 or y_pred.shape[1] % 3 != 0:
                raise ValueError("Mixture CRPS expects [weights..., means..., scales...] columns")
            component_count = y_pred.shape[1] // 3
            weights = np.clip(y_pred[:, :component_count], 0.0, None)
            weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-9, None)
            means = y_pred[:, component_count : 2 * component_count]
            scales = np.clip(y_pred[:, 2 * component_count :], 1e-9, None)
            loc = np.sum(weights * means, axis=1)
            variance = np.sum(weights * (scales**2 + means**2), axis=1) - loc**2
            scale = np.sqrt(np.clip(variance, 1e-18, None))
        else:
            loc = _col(y_pred, 0)
            return float(np.mean(np.abs(y_true - loc)))

        try:
            import properscoring as ps

            return float(np.mean(ps.crps_gaussian(y_true, mu=loc, sig=scale)))
        except ImportError:
            return float(np.mean(np.abs(y_true - loc)))

    def __repr__(self) -> str:
        return f"CRPS(family={self.family!r})"


class IntervalScore(DeepTabMetric):
    """Winkler Interval Score at coverage level ``1 - alpha``.

    Penalises both width and mis-coverage.  Expected ``y_pred`` format:

    * Column 0: lower bound of the prediction interval
    * Column 1: upper bound of the prediction interval

    Parameters
    ----------
    alpha : float
        Significance level, e.g. ``0.05`` for a 95% prediction interval.
    """

    name = "interval_score"
    higher_is_better = False

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.ndim != 2 or y_pred.shape[1] < 2:
            raise ValueError("IntervalScore expects y_pred with at least 2 columns: [lower, upper]")
        lower = y_pred[:, 0]
        upper = y_pred[:, 1]
        width = upper - lower
        penalty_low = (2.0 / self.alpha) * np.maximum(lower - y_true, 0.0)
        penalty_high = (2.0 / self.alpha) * np.maximum(y_true - upper, 0.0)
        return float(np.mean(width + penalty_low + penalty_high))

    def __repr__(self) -> str:
        return f"IntervalScore(alpha={self.alpha})"


class EnergyScore(DeepTabMetric):
    """Energy Score — multivariate generalisation of CRPS.

    Suitable for multivariate / compositional distributions (e.g.
    :class:`~deeptab.distributions.MixtureOfGaussiansDistribution`,
    :class:`~deeptab.distributions.DirichletDistribution`).

    Computed via Monte-Carlo sampling from the predicted distribution when
    samples are provided, or via a closed-form energy distance otherwise.

    For simple use-cases where ``y_pred`` is a 2-D parameter array,
    the energy score is approximated as the mean Euclidean distance between
    ``y_true`` and the predicted mean.
    """

    name = "energy_score"
    higher_is_better = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mean_pred = y_pred[:, 0] if y_pred.ndim == 2 else y_pred.ravel()
        y_true_flat = y_true.ravel() if y_true.ndim == 1 else y_true[:, 0]
        return float(np.mean(np.abs(y_true_flat - mean_pred)))


# ---------------------------------------------------------------------------
# Distribution-specific deviances (fixed)
# ---------------------------------------------------------------------------


class PoissonDeviance(DeepTabMetric):
    """Mean Poisson Deviance.

    Suitable for ``poisson`` and ``zip`` families.  Expected ``y_pred``:
    predicted mean (1-D or first column of 2-D), or ZIP parameters when
    ``pi_col`` and ``rate_col`` select the zero-inflation and rate columns.
    """

    name = "poisson_deviance"
    higher_is_better = False

    def __init__(self, pi_col: int | None = None, rate_col: int | None = None) -> None:
        self.pi_col = pi_col
        self.rate_col = rate_col

    def predicted_mean(self, y_pred: np.ndarray) -> np.ndarray:
        """Extract the mean, accounting for zero inflation when configured."""
        if self.pi_col is not None and self.rate_col is not None:
            pi = np.clip(_col(y_pred, self.pi_col), 0.0, 1.0)
            rate = _col(y_pred, self.rate_col)
            return (1.0 - pi) * rate
        return _col(y_pred, 0)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).ravel()
        mu = self.predicted_mean(y_pred)
        mu = np.clip(mu, 1e-9, None)
        # Safe log: avoid log(0/0) when y_true == 0
        log_ratio = np.where(y_true > 0, np.log(np.where(y_true > 0, y_true / mu, 1.0)), 0.0)
        return float(2.0 * np.mean(y_true * log_ratio - (y_true - mu)))


class GammaDeviance(DeepTabMetric):
    """Mean Gamma Deviance.

    Suitable for ``gamma`` and ``inversegamma`` families.  Expected ``y_pred``:
    predicted mean (1-D or first column of 2-D), or Gamma parameters when
    ``shape_col`` and ``rate_col`` select the shape and rate columns.
    """

    name = "gamma_deviance"
    higher_is_better = False

    def __init__(self, shape_col: int | None = None, rate_col: int | None = None) -> None:
        self.shape_col = shape_col
        self.rate_col = rate_col

    def predicted_mean(self, y_pred: np.ndarray) -> np.ndarray:
        """Extract the mean, dividing Gamma shape by rate when configured."""
        if self.shape_col is not None and self.rate_col is not None:
            return _col(y_pred, self.shape_col) / np.clip(_col(y_pred, self.rate_col), 1e-9, None)
        return _col(y_pred, 0)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.clip(np.asarray(y_true, dtype=float).ravel(), 1e-9, None)
        mu = self.predicted_mean(y_pred)
        mu = np.clip(mu, 1e-9, None)
        return float(2.0 * np.mean(np.log(mu / y_true) + (y_true - mu) / mu))


class TweedieDeviance(DeepTabMetric):
    """Mean Tweedie Deviance.

    Suitable for the ``tweedie`` family where ``1 < p < 2``.

    Parameters
    ----------
    p : float
        Tweedie power parameter.  Defaults to 1.5.
    """

    name = "tweedie_deviance"
    higher_is_better = False

    def __init__(self, p: float = 1.5) -> None:
        if not (1.0 < p < 2.0):
            raise ValueError(f"Tweedie power p must satisfy 1 < p < 2, got {p}")
        self.p = p

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).ravel()
        mu = np.clip(_col(y_pred, 0), 1e-9, None)
        p = self.p
        term1 = y_true ** (2.0 - p) / ((1.0 - p) * (2.0 - p))
        term2 = y_true * mu ** (1.0 - p) / (1.0 - p)
        term3 = mu ** (2.0 - p) / (2.0 - p)
        return float(2.0 * np.mean(term1 - term2 + term3))

    def __repr__(self) -> str:
        return f"TweedieDeviance(p={self.p})"


class NegativeBinomialDeviance(DeepTabMetric):
    """Mean Negative-Binomial Deviance.

    Suitable for the ``negativebinom`` family.

    Expected ``y_pred``: 2-D array where column 0 is the predicted mean ``mu``
    and column 1 (optional) is the overdispersion parameter ``alpha``.  If
    only one column is present, ``alpha`` falls back to the ``default_alpha``
    constructor argument.

    Parameters
    ----------
    default_alpha : float
        Overdispersion parameter used when ``y_pred`` has only one column.
        Defaults to ``1.0``.
    """

    name = "nb_deviance"
    higher_is_better = False

    def __init__(self, default_alpha: float = 1.0) -> None:
        self.default_alpha = default_alpha

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float)
        mu = np.clip(_col(y_pred, 0), 1e-9, None)
        if y_pred.ndim == 2 and y_pred.shape[1] >= 2:
            alpha = np.clip(y_pred[:, 1], 1e-9, None)
        else:
            alpha = self.default_alpha
        log_ratio = np.where(y_true > 0, np.log(np.where(y_true > 0, y_true / mu, 1.0)), 0.0)
        return float(
            2.0 * np.mean(y_true * log_ratio + (y_true + alpha) * np.log((mu + alpha) / (y_true + alpha + 1e-9)))
        )

    def __repr__(self) -> str:
        return f"NegativeBinomialDeviance(default_alpha={self.default_alpha})"


class BetaBrierScore(DeepTabMetric):
    """Mean Squared Error of the predicted mean for Beta-distributed targets.

    Suitable for the ``beta`` family.  Expected ``y_pred``:
    1-D or first column is predicted mean in (0, 1).
    """

    name = "beta_brier"
    higher_is_better = False

    def __init__(self, alpha_col: int | None = None, beta_col: int | None = None) -> None:
        self.alpha_col = alpha_col
        self.beta_col = beta_col

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).ravel()
        if self.alpha_col is not None and self.beta_col is not None:
            alpha = np.clip(_col(y_pred, self.alpha_col), 1e-9, None)
            beta = np.clip(_col(y_pred, self.beta_col), 1e-9, None)
            mu = alpha / (alpha + beta)
        else:
            mu = _col(y_pred, 0)
        mu = np.clip(mu, 1e-9, 1.0 - 1e-9)
        return float(np.mean((mu - y_true) ** 2))


class DirichletError(DeepTabMetric):
    """Mean KL Divergence between true and predicted Dirichlet means.

    Suitable for the ``dirichlet`` family.  Both ``y_true`` and ``y_pred``
    are treated as probability vectors (rows must sum to 1 after clipping).
    """

    name = "dirichlet_error"
    higher_is_better = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.ndim == 1:
            y_true = y_true.reshape(1, -1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)
        # Normalise rows to valid probability vectors
        p = np.clip(y_true, 1e-9, None)
        p /= p.sum(axis=1, keepdims=True)
        q = np.clip(y_pred, 1e-9, None)
        q /= q.sum(axis=1, keepdims=True)
        kl = np.sum(p * np.log(p / q), axis=1)
        return float(np.mean(kl))


class StudentTLoss(DeepTabMetric):
    """Proper Student-T negative log-likelihood (mean) for the ``studentt`` family.

    Expected ``y_pred`` columns: ``[df, loc, scale]``. If only 2 columns
    are present, ``df`` defaults to the constructor argument and the columns
    are interpreted as ``[loc, scale]``.

    Parameters
    ----------
    default_df : float
        Degrees-of-freedom fallback when not present in ``y_pred``.
        Defaults to 3.0.
    """

    name = "studentt_nll"
    higher_is_better = False

    def __init__(
        self,
        default_df: float = 3.0,
        df_col: int = 0,
        loc_col: int = 1,
        scale_col: int = 2,
    ) -> None:
        self.default_df = default_df
        self.df_col = df_col
        self.loc_col = loc_col
        self.scale_col = scale_col

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from scipy.special import gammaln

        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.ndim == 2 and y_pred.shape[1] >= 3:
            mu = _col(y_pred, self.loc_col)
            scale = np.clip(_col(y_pred, self.scale_col), 1e-9, None)
            df = np.clip(_col(y_pred, self.df_col), 2.0 + 1e-6, None)
        else:
            mu = _col(y_pred, 0)
            scale = np.clip(_col(y_pred, 1), 1e-9, None)
            df = self.default_df
        # Student-T NLL: -log Γ((df+1)/2) + log Γ(df/2) + 0.5*log(π*df*σ²) + (df+1)/2 * log(1 + (y-μ)²/(df*σ²))
        nll = (
            gammaln(df / 2.0)
            - gammaln((df + 1.0) / 2.0)
            + 0.5 * np.log(np.pi * df * scale**2)
            + (df + 1.0) / 2.0 * np.log(1.0 + (y_true - mu) ** 2 / (df * scale**2))
        )
        return float(np.mean(nll))

    def __repr__(self) -> str:
        return f"StudentTLoss(default_df={self.default_df})"


class InverseGammaDeviance(DeepTabMetric):
    """Mean Inverse-Gamma deviance for the ``inversegamma`` family.

    Expected ``y_pred`` columns: ``[shape (alpha), scale (beta)]``.

    The deviance is computed as ``-2 * (log p(y | alpha, beta) - log p(y | alpha_sat, beta_sat))``
    where the saturated model likelihood equals 1 (per-sample deviance).
    """

    name = "inversegamma_deviance"
    higher_is_better = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from scipy.special import gammaln

        y_true = np.clip(np.asarray(y_true, dtype=float).ravel(), 1e-9, None)
        y_pred = np.asarray(y_pred, dtype=float)
        alpha = np.clip(_col(y_pred, 0), 1e-6, None)
        beta = np.clip(_col(y_pred, 1), 1e-6, None)
        # log p(y | alpha, beta) = alpha*log(beta) - log Gamma(alpha) - (alpha+1)*log(y) - beta/y
        log_p = alpha * np.log(beta) - gammaln(alpha) - (alpha + 1.0) * np.log(y_true) - beta / y_true
        return float(-2.0 * np.mean(log_p))


class LogNormalNLL(DeepTabMetric):
    """Mean Log-Normal Negative Log-Likelihood for the ``lognormal`` family.

    Expected ``y_pred`` columns: ``[loc (log-space mean), scale (log-space std)]``.
    """

    name = "lognormal_nll"
    higher_is_better = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.clip(np.asarray(y_true, dtype=float).ravel(), 1e-9, None)
        loc = _col(y_pred, 0)
        scale = np.clip(_col(y_pred, 1), 1e-9, None)
        nll = np.log(y_true * scale * np.sqrt(2.0 * np.pi)) + (np.log(y_true) - loc) ** 2 / (2.0 * scale**2)
        return float(np.mean(nll))


# ---------------------------------------------------------------------------
# Calibration / uncertainty metrics
# ---------------------------------------------------------------------------


class CoverageProbability(DeepTabMetric):
    """Empirical coverage probability at a given ``1 - alpha`` level.

    Expected ``y_pred`` columns: ``[lower_bound, upper_bound]``.

    A well-calibrated model should have coverage close to ``1 - alpha``.
    Higher is *not* unconditionally better — the target is the nominal level.

    Parameters
    ----------
    alpha : float
        Significance level, e.g. ``0.05`` for 95% prediction intervals.
    """

    name = "coverage"
    higher_is_better = True  # directional: want coverage ≈ 1 - alpha

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.ndim != 2 or y_pred.shape[1] < 2:
            raise ValueError("CoverageProbability expects y_pred with at least 2 columns: [lower, upper]")
        lower = y_pred[:, 0]
        upper = y_pred[:, 1]
        covered = (y_true >= lower) & (y_true <= upper)
        return float(np.mean(covered))

    def __repr__(self) -> str:
        return f"CoverageProbability(alpha={self.alpha})"


class SharpnessScore(DeepTabMetric):
    """Mean prediction interval width (sharpness).

    Narrower intervals are sharper (lower is better), but must be balanced
    against calibration.  Expected ``y_pred`` columns: ``[lower, upper]``.
    """

    name = "sharpness"
    higher_is_better = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.ndim != 2 or y_pred.shape[1] < 2:
            raise ValueError("SharpnessScore expects y_pred with at least 2 columns: [lower, upper]")
        return float(np.mean(y_pred[:, 1] - y_pred[:, 0]))


class ProbabilityIntegralTransform(DeepTabMetric):
    """PIT uniformity test — returns the mean absolute deviation from uniformity.

    The Probability Integral Transform (PIT) of a well-calibrated forecast
    should be uniform on [0, 1].  This metric computes the PIT values for a
    Normal predictive distribution and returns the MAD from the uniform CDF.
    Lower is better (0 = perfect calibration).

    Expected ``y_pred`` columns: ``[loc, scale]`` (Normal distribution).

    Parameters
    ----------
    n_bins : int
        Number of histogram bins for the PIT.  Defaults to 10.
    family : str
        Distribution family for CDF computation.  Currently only ``"normal"``
        is supported.
    """

    name = "pit"
    higher_is_better = False

    def __init__(self, n_bins: int = 10, family: str = "normal") -> None:
        self.n_bins = n_bins
        self.family = family

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from scipy.stats import norm

        y_true = np.asarray(y_true, dtype=float).ravel()
        loc = _col(y_pred, 0)
        scale = np.clip(_col(y_pred, 1), 1e-9, None)
        pit_vals = norm.cdf(y_true, loc=loc, scale=scale)
        # Histogram of PIT values — should be uniform
        counts, _ = np.histogram(pit_vals, bins=self.n_bins, range=(0.0, 1.0))
        empirical = counts / counts.sum()
        uniform = np.ones(self.n_bins) / self.n_bins
        return float(np.mean(np.abs(empirical - uniform)))

    def __repr__(self) -> str:
        return f"ProbabilityIntegralTransform(n_bins={self.n_bins}, family={self.family!r})"
