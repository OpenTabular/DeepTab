"""Base class for DeepTab evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DeepTabMetric(ABC):
    """Abstract base class for all DeepTab evaluation metrics.

    Every metric in ``deeptab.metrics`` subclasses this ABC and exposes three
    class-level attributes that the training loop and registry read
    automatically — you never need to set them yourself when *using* a metric,
    only when *writing* a custom one.

    Attributes
    ----------
    name : str
        A short, machine-readable identifier for the metric.  It is used as:

        * the key in the dict returned by ``model.evaluate()``
        * the suffix in training-log entries (e.g. ``val_rmse``)
        * the registry lookup key in :data:`~deeptab.metrics.METRIC_REGISTRY`

        Examples: ``"rmse"``, ``"crps"``, ``"auroc"``.

    higher_is_better : bool
        Tells the framework whether a *larger* or *smaller* value is
        preferable.  This matters in two places:

        * **HPO** — hyperparameter search uses it to set the optimisation
          direction (maximise vs. minimise) when a metric is chosen as the
          objective.
        * **Early stopping / model selection** — callbacks can use it to
          decide whether a new checkpoint is an improvement.

        ``False`` (default) means *lower is better* — appropriate for loss
        functions and error metrics (MSE, MAE, NLL, deviances).
        ``True`` means *higher is better* — appropriate for scores like R²,
        accuracy, AUROC, and CRPS variants where a higher value is desirable.

    needs_raw : bool
        Controls *which* form of ``y_pred`` the training loop passes to this
        metric.

        * ``False`` (default) — the metric receives **already-transformed**
          distribution parameters, i.e. the output of
          ``model.predict(X, raw=False)``.  For example, a Normal distribution
          model returns ``[mean, std]`` where ``std > 0`` is guaranteed.  This
          is the right choice for almost every metric.
        * ``True`` — the metric receives **raw model logits** before the
          distribution's parameter transforms are applied.
          :class:`~deeptab.metrics.NegativeLogLikelihood` sets this to
          ``True`` because it calls ``distribution.compute_loss()`` which
          applies the transforms itself; passing already-transformed values
          would double-transform and produce wrong results.

    Examples
    --------
    Using a built-in metric directly:

    >>> from deeptab.metrics import RootMeanSquaredError
    >>> import numpy as np
    >>> metric = RootMeanSquaredError()
    >>> metric.name
    'rmse'
    >>> metric.higher_is_better
    False
    >>> metric(np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.0, 2.9]))
    0.08164965809277261

    Passing metrics to ``model.fit()`` for live training logging:

    >>> from deeptab.metrics import CRPS, MeanAbsoluteError
    >>> model.fit(X_train, y_train,
    ...           val_metrics={"crps": CRPS(family="normal"),
    ...                        "mae": MeanAbsoluteError()})
    # Logs val_crps and val_mae each epoch.

    Writing a custom metric:

    >>> from deeptab.metrics import DeepTabMetric
    >>> import numpy as np
    >>> class MedianAbsoluteError(DeepTabMetric):
    ...     name = "mdae"
    ...     higher_is_better = False          # lower error = better
    ...     needs_raw = False                 # use transformed predictions
    ...
    ...     def __call__(self, y_true, y_pred):
    ...         y_pred = np.asarray(y_pred)
    ...         mean_pred = y_pred[:, 0] if y_pred.ndim == 2 else y_pred.ravel()
    ...         return float(np.median(np.abs(np.asarray(y_true).ravel() - mean_pred)))
    """

    name: str
    higher_is_better: bool = False
    needs_raw: bool = False

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the metric value.

        Parameters
        ----------
        y_true : np.ndarray, shape (n,) or (n, d)
            Ground-truth target values.
        y_pred : np.ndarray, shape (n,) or (n, p)
            Model predictions.

            * When ``needs_raw=False`` (default): already-transformed
              distribution parameters from ``model.predict(X, raw=False)``.
              For a Normal distribution this is ``[[mean_0, std_0], ...]``.
            * When ``needs_raw=True``: raw logits from the model's final
              linear layer, before any parameter transform (e.g. softplus)
              is applied.

        Returns
        -------
        float
            Scalar metric value.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
