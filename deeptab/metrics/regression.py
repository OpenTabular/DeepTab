"""Regression metrics (MSE, MAE, RMSE, R2, MAPE, PinballLoss).

All standard metrics delegate to :mod:`sklearn.metrics` internally.
The wrapper classes exist for three reasons:

1. **Uniform interface** -- each class carries ``name``, ``higher_is_better``,
   and ``needs_raw`` so the training loop and registry can inspect them
   without hard-coding metric names.
2. **LSS compatibility** -- ``model.predict()`` returns a 2-D array of shape
   ``(n_samples, n_params)`` for distributional models.  The helper
   :func:`_extract_mean` pulls the first column (predicted mean) so sklearn
   functions receive the expected 1-D array.
3. **Consistent API** -- all metrics share the same
   ``metric(y_true, y_pred) -> float`` call signature regardless of their
   source.

Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 12 20 46

   * - Class
     - ``name``
     - ``higher_is_better``
     - Notes
   * - :class:`MeanSquaredError`
     - ``"mse"``
     - ``False``
     - Standard MSE; lower = better
   * - :class:`RootMeanSquaredError`
     - ``"rmse"``
     - ``False``
     - Same units as target; lower = better
   * - :class:`MeanAbsoluteError`
     - ``"mae"``
     - ``False``
     - Robust to outliers; lower = better
   * - :class:`R2Score`
     - ``"r2"``
     - ``True``
     - 1.0 = perfect; **higher = better**
   * - :class:`MeanAbsolutePercentageError`
     - ``"mape"``
     - ``False``
     - % scale; avoid when targets are near zero
   * - :class:`PinballLoss`
     - ``"pinball"``
     - ``False``
     - Quantile regression; lower = better
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error as _mae
from sklearn.metrics import mean_absolute_percentage_error as _mape
from sklearn.metrics import mean_squared_error as _mse
from sklearn.metrics import r2_score as _r2

from .base import DeepTabMetric


def _extract_mean(y_pred: np.ndarray) -> np.ndarray:
    """Return the first column of a 2-D array, or the flat 1-D array.

    LSS models return ``(n_samples, n_params)`` arrays; the first column is
    always the predicted mean / location parameter.
    """
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2:
        return y_pred[:, 0]
    return y_pred.ravel()


class MeanSquaredError(DeepTabMetric):
    """Mean Squared Error -- delegates to :func:`sklearn.metrics.mean_squared_error`.

    Accepts both point-prediction vectors and 2-D parameter arrays (uses
    the first column as the predicted mean).
    """

    name = "mse"
    higher_is_better = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(_mse(np.asarray(y_true).ravel(), _extract_mean(y_pred)))


class RootMeanSquaredError(DeepTabMetric):
    """Root Mean Squared Error -- sqrt of :func:`sklearn.metrics.mean_squared_error`."""

    name = "rmse"
    higher_is_better = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(_mse(np.asarray(y_true).ravel(), _extract_mean(y_pred))))


class MeanAbsoluteError(DeepTabMetric):
    """Mean Absolute Error -- delegates to :func:`sklearn.metrics.mean_absolute_error`."""

    name = "mae"
    higher_is_better = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(_mae(np.asarray(y_true).ravel(), _extract_mean(y_pred)))


class R2Score(DeepTabMetric):
    """Coefficient of Determination (R2) -- delegates to :func:`sklearn.metrics.r2_score`.

    Higher is better; perfect prediction gives R2 = 1.
    """

    name = "r2"
    higher_is_better = True

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(_r2(np.asarray(y_true).ravel(), _extract_mean(y_pred)))


class MeanAbsolutePercentageError(DeepTabMetric):
    """Mean Absolute Percentage Error -- delegates to
    :func:`sklearn.metrics.mean_absolute_percentage_error`.

    sklearn clips the denominator to ``np.finfo(np.float64).eps`` internally.
    """

    name = "mape"
    higher_is_better = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(_mape(np.asarray(y_true).ravel(), _extract_mean(y_pred)))


class PinballLoss(DeepTabMetric):
    """Pinball (Quantile) Loss -- delegates to
    :func:`sklearn.metrics.mean_pinball_loss`.

    Measures calibration at a single quantile level ``tau in (0, 1)``.

    For LSS ``quantile`` family predictions, ``y_pred`` is a 2-D array where
    each column is a predicted quantile.  Pass ``col`` to select the relevant
    column (default 0).

    Parameters
    ----------
    quantile : float
        The quantile level, e.g. 0.5 for the median.
    col : int
        Column of ``y_pred`` to use when predictions are 2-D.  Default 0.
    """

    name = "pinball"
    higher_is_better = False

    def __init__(self, quantile: float = 0.5, col: int = 0) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.quantile = quantile
        self.col = col

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import mean_pinball_loss

        y_pred_arr = np.asarray(y_pred, dtype=float)
        q_pred = y_pred_arr[:, self.col] if y_pred_arr.ndim == 2 else y_pred_arr.ravel()
        return float(mean_pinball_loss(np.asarray(y_true).ravel(), q_pred, alpha=self.quantile))

    def __repr__(self) -> str:
        return f"PinballLoss(quantile={self.quantile}, col={self.col})"
