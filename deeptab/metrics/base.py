"""Base class for DeepTab evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


class DeepTabMetric(ABC):
    """Abstract base class for all DeepTab evaluation metrics.

    Every metric in ``deeptab.metrics`` subclasses this ABC. The training loop
    and registry read its metadata automatically; users declare metadata only
    when writing their own metrics.

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

    input_space : {"logits", "probabilities", "labels"} or None
        Declares classification inputs during training. Built-in classifiers
        request ``"probabilities"``: sigmoid outputs become two ordered columns,
        and multi-column logits receive softmax. Declared ensembles are averaged
        over member axis 1 before activation. ``"labels"`` requests encoded
        class indices, and ``"logits"`` keeps original outputs. ``None`` retains
        the existing custom-callable contract without guessing semantics from
        the metric name. ``needs_raw=True`` takes precedence. Distributional
        metrics continue to use family parameter transforms instead.

        Direct calls to a metric do not convert logits; callers must supply
        the documented labels or probabilities themselves.

      Notes
      -----
      During training, NumPy callables are evaluated once on detached,
      concatenated inputs for a complete epoch or validation pass. Their buffers
      are bounded by TaskModel's per-stage limit. Train/validation and sanity
      passes are isolated; native TorchMetrics use update/compute/reset instead.

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
    >>> from deeptab.models import MambularLSS
    >>> model = MambularLSS()
    >>> model.fit(X_train, y_train, family="normal",
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
    input_space: Literal["logits", "probabilities", "labels"] | None = None

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
