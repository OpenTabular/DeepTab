"""Classification metrics (Accuracy, F1, AUROC, AUPRC, LogLoss, ECE, BrierScore).

All standard metrics delegate to :mod:`sklearn.metrics` internally.
The wrapper classes add the :class:`DeepTabMetric` interface (``name``,
``higher_is_better``, ``needs_raw``) and normalise DeepTab-specific
prediction formats (2-D probability arrays vs 1-D label arrays).

:class:`ExpectedCalibrationError` is the only class without a sklearn
equivalent and is therefore implemented from scratch.

Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 14 20 38

   * - Class
     - ``name``
     - ``higher_is_better``
     - Notes
   * - :class:`Accuracy`
     - ``"accuracy"``
     - ``True``
     - Fraction correct; **higher = better**
   * - :class:`F1Score`
     - ``"f1"``
     - ``True``
     - Harmonic mean precision/recall; **higher = better**
   * - :class:`AUROC`
     - ``"auroc"``
     - ``True``
     - Needs probability scores; **higher = better**
   * - :class:`AUPRC`
     - ``"auprc"``
     - ``True``
     - Better than AUROC for imbalanced data; **higher = better**
   * - :class:`LogLoss`
     - ``"log_loss"``
     - ``False``
     - Cross-entropy; lower = better
   * - :class:`BrierScore`
     - ``"brier"``
     - ``False``
     - MSE of probability; lower = better
   * - :class:`ExpectedCalibrationError`
     - ``"ece"``
     - ``False``
     - 0 = perfectly calibrated; lower = better
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import accuracy_score as _accuracy
from sklearn.metrics import average_precision_score as _auprc
from sklearn.metrics import brier_score_loss as _brier
from sklearn.metrics import f1_score as _f1
from sklearn.metrics import log_loss as _log_loss
from sklearn.metrics import roc_auc_score as _auroc

from .base import DeepTabMetric


class Accuracy(DeepTabMetric):
    """Classification accuracy -- delegates to :func:`sklearn.metrics.accuracy_score`.

    Accepts 1-D integer labels or 2-D probability arrays (argmax is taken).
    """

    name = "accuracy"
    higher_is_better = True

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred)
        labels = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else (y_pred.ravel() >= 0.5).astype(int)
        return float(_accuracy(y_true, labels))


class F1Score(DeepTabMetric):
    """F1 Score -- delegates to :func:`sklearn.metrics.f1_score`.

    Parameters
    ----------
    average : str
        Averaging strategy: ``"binary"`` (default), ``"macro"``, or
        ``"weighted"``.
    """

    name = "f1"
    higher_is_better = True

    def __init__(self, average: str = "binary") -> None:
        if average not in ("binary", "macro", "weighted"):
            raise ValueError(f"average must be 'binary', 'macro', or 'weighted', got {average!r}")
        self.average = average

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred)
        labels = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else (y_pred.ravel() >= 0.5).astype(int)
        return float(_f1(y_true, labels, average=self.average, zero_division=0))  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"F1Score(average={self.average!r})"


class AUROC(DeepTabMetric):
    """Area Under the ROC Curve -- delegates to :func:`sklearn.metrics.roc_auc_score`.

    Parameters
    ----------
    average : str
        ``"macro"`` (default) or ``"weighted"``.  Ignored for binary tasks.
    """

    name = "auroc"
    higher_is_better = True

    def __init__(self, average: str = "macro") -> None:
        self.average = average

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred)
        try:
            if y_pred.ndim == 2 and y_pred.shape[1] == 2:
                return float(_auroc(y_true, y_pred[:, 1]))
            elif y_pred.ndim == 2:
                return float(_auroc(y_true, y_pred, multi_class="ovr", average=self.average))
            else:
                return float(_auroc(y_true, y_pred.ravel()))
        except ValueError:
            return float("nan")

    def __repr__(self) -> str:
        return f"AUROC(average={self.average!r})"


class AUPRC(DeepTabMetric):
    """Area Under the Precision-Recall Curve -- delegates to
    :func:`sklearn.metrics.average_precision_score`.
    """

    name = "auprc"
    higher_is_better = True

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred)
        scores = y_pred[:, 1] if y_pred.ndim == 2 else y_pred.ravel()
        try:
            return float(_auprc(y_true, scores))
        except ValueError:
            return float("nan")


class LogLoss(DeepTabMetric):
    """Cross-Entropy / Log Loss -- delegates to :func:`sklearn.metrics.log_loss`."""

    name = "log_loss"
    higher_is_better = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(_log_loss(np.asarray(y_true).ravel(), np.asarray(y_pred)))


class BrierScore(DeepTabMetric):
    """Brier Score -- delegates to :func:`sklearn.metrics.brier_score_loss`.

    Accepts 1-D probability scores or a 2-D array (second column is used).
    """

    name = "brier"
    higher_is_better = False

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred, dtype=float)
        probs = y_pred[:, 1] if y_pred.ndim == 2 else y_pred.ravel()
        return float(_brier(y_true, probs))


class ExpectedCalibrationError(DeepTabMetric):
    """Expected Calibration Error (ECE).

    sklearn does not provide ECE natively, so this is a custom implementation.
    Bins predictions by confidence and measures the gap between mean confidence
    and accuracy per bin.

    Parameters
    ----------
    n_bins : int
        Number of confidence bins.  Default 10.
    """

    name = "ece"
    higher_is_better = False

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.ndim == 2:
            confidence = y_pred.max(axis=1)
            preds = y_pred.argmax(axis=1)
        else:
            confidence = np.where(y_pred >= 0.5, y_pred, 1.0 - y_pred).ravel()
            preds = (y_pred.ravel() >= 0.5).astype(int)
        correct = (preds == y_true).astype(float)

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece = 0.0
        n = len(y_true)
        for lo, hi in itertools.pairwise(bin_edges):
            mask = (confidence >= lo) & (confidence < hi)
            if mask.sum() == 0:
                continue
            acc = correct[mask].mean()
            conf = confidence[mask].mean()
            ece += mask.sum() / n * abs(acc - conf)
        return float(ece)

    def __repr__(self) -> str:
        return f"ExpectedCalibrationError(n_bins={self.n_bins})"
