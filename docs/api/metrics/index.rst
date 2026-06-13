.. -*- mode: rst -*-

.. currentmodule:: deeptab.metrics

Metrics
=======

Evaluation metrics for all three DeepTab task types: regression, classification,
and distributional (LSS) regression.

Every metric is a :class:`DeepTabMetric` subclass with three attributes the
framework reads automatically:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Attribute
     - Type
     - Purpose
   * - ``name``
     - ``str``
     - Key in ``model.evaluate()`` results and training-log suffix
       (e.g. ``val_rmse``, ``val_crps``).
   * - ``higher_is_better``
     - ``bool``
     - ``True`` for scores (accuracy, AUROC, R²); ``False`` for losses/errors
       (MSE, NLL, deviances). Used by HPO to set the optimisation direction.
   * - ``needs_raw``
     - ``bool``
     - ``False`` (default): metric receives already-transformed distribution
       parameters. ``True``: metric receives raw model logits and applies
       transforms itself. Only :class:`NegativeLogLikelihood` uses ``True``.

Quick Start
-----------

.. code-block:: python

    from deeptab.metrics import RootMeanSquaredError, CRPS, Accuracy

    rmse = RootMeanSquaredError()
    print(rmse.name)              # "rmse"
    print(rmse.higher_is_better)  # False

    # Pass to model.fit() for live training logging
    from deeptab.models import MambularLSS
    model = MambularLSS()
    model.fit(
        X_train, y_train,
        val_metrics={
            "crps": CRPS(family="normal"),   # logged as "val_crps"
            "rmse": RootMeanSquaredError(),   # logged as "val_rmse"
        },
    )

    # Post-hoc evaluation
    scores = model.evaluate(X_test, y_test)
    # Returns e.g. {"crps": 0.32, "rmse": 1.45}

    # Auto-select default metrics via the registry
    from deeptab.metrics import get_default_metrics
    metrics = get_default_metrics("lss", family="normal")
    # [CRPS(family='normal'), RootMeanSquaredError(), MeanAbsoluteError()]

Available Metrics
-----------------

Regression Metrics
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 20 35

   * - Class
     - ``name``
     - ``higher_is_better``
     - Notes
   * - :class:`MeanSquaredError`
     - ``mse``
     - ``False``
     - sklearn-backed; lower = better
   * - :class:`RootMeanSquaredError`
     - ``rmse``
     - ``False``
     - Same units as target; default for regression
   * - :class:`MeanAbsoluteError`
     - ``mae``
     - ``False``
     - Robust to outliers
   * - :class:`R2Score`
     - ``r2``
     - ``True``
     - 1.0 = perfect; **higher = better**
   * - :class:`MeanAbsolutePercentageError`
     - ``mape``
     - ``False``
     - % scale; avoid when targets near zero
   * - :class:`PinballLoss`
     - ``pinball``
     - ``False``
     - Quantile regression; tau in (0, 1)

All regression metrics accept 2-D LSS parameter arrays and extract the first
column (predicted mean) automatically.

Classification Metrics
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Class
     - ``name``
     - ``higher_is_better``
     - Notes
   * - :class:`Accuracy`
     - ``accuracy``
     - ``True``
     - sklearn-backed; argmax of probability array
   * - :class:`F1Score`
     - ``f1``
     - ``True``
     - ``average`` param: binary / macro / weighted
   * - :class:`AUROC`
     - ``auroc``
     - ``True``
     - Requires probability scores
   * - :class:`AUPRC`
     - ``auprc``
     - ``True``
     - Better than AUROC for imbalanced data
   * - :class:`LogLoss`
     - ``log_loss``
     - ``False``
     - Cross-entropy; requires probability scores
   * - :class:`BrierScore`
     - ``brier``
     - ``False``
     - MSE of probability; binary only
   * - :class:`ExpectedCalibrationError`
     - ``ece``
     - ``False``
     - 0 = perfectly calibrated; custom implementation

Distributional / LSS Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 22 17 14 17

   * - Class
     - ``name``
     - ``higher_is_better``
     - ``needs_raw``
     - Notes
   * - :class:`NegativeLogLikelihood`
     - ``nll``
     - ``False``
     - ``True``
     - Requires distribution object; passes raw logits
   * - :class:`LogScore`
     - ``log_score``
     - ``True``
     - ``True``
     - = -NLL; **higher = better**
   * - :class:`CRPS`
     - ``crps``
     - ``False``
     - ``False``
     - Vectorised via ``properscoring``; all continuous families
   * - :class:`IntervalScore`
     - ``interval_score``
     - ``False``
     - ``False``
     - Winkler score; expects [lower, upper] columns
   * - :class:`EnergyScore`
     - ``energy_score``
     - ``False``
     - ``False``
     - Multivariate CRPS generalisation
   * - :class:`PoissonDeviance`
     - ``poisson_deviance``
     - ``False``
     - ``False``
     - poisson, zip families
   * - :class:`GammaDeviance`
     - ``gamma_deviance``
     - ``False``
     - ``False``
     - gamma, inversegamma families
   * - :class:`TweedieDeviance`
     - ``tweedie_deviance``
     - ``False``
     - ``False``
     - tweedie family; ``p`` param (1 < p < 2)
   * - :class:`NegativeBinomialDeviance`
     - ``nb_deviance``
     - ``False``
     - ``False``
     - negativebinom family
   * - :class:`BetaBrierScore`
     - ``beta_brier``
     - ``False``
     - ``False``
     - beta family (proportions)
   * - :class:`DirichletError`
     - ``dirichlet_error``
     - ``False``
     - ``False``
     - dirichlet family; KL divergence
   * - :class:`StudentTLoss`
     - ``studentt_nll``
     - ``False``
     - ``False``
     - studentt family; proper NLL
   * - :class:`InverseGammaDeviance`
     - ``inversegamma_deviance``
     - ``False``
     - ``False``
     - inversegamma family
   * - :class:`LogNormalNLL`
     - ``lognormal_nll``
     - ``False``
     - ``False``
     - lognormal family
   * - :class:`CoverageProbability`
     - ``coverage``
     - ``True``
     - ``False``
     - Fraction of targets inside prediction interval
   * - :class:`SharpnessScore`
     - ``sharpness``
     - ``False``
     - ``False``
     - Mean interval width; lower = sharper
   * - :class:`ProbabilityIntegralTransform`
     - ``pit``
     - ``False``
     - ``False``
     - MAD from uniform CDF; 0 = perfectly calibrated

Registry
--------

The registry maps ``(task, family)`` keys to ordered lists of default metrics.
The first entry in each list is the primary metric used by HPO and model selection.

.. code-block:: python

    from deeptab.metrics import get_default_metrics, get_default_metrics_dict

    # Returns list of DeepTabMetric instances
    get_default_metrics("regression")
    # [RootMeanSquaredError(), MeanAbsoluteError(), R2Score()]

    get_default_metrics("classification")
    # [Accuracy(), AUROC(), LogLoss()]

    get_default_metrics("lss", family="gamma")
    # [GammaDeviance(), RootMeanSquaredError()]

    # Returns {name: metric} dict, useful for model.evaluate()
    get_default_metrics_dict("lss", family="normal")
    # {"crps": CRPS(...), "rmse": RootMeanSquaredError(), "mae": MeanAbsoluteError()}

Choosing a Distribution-Specific Metric
----------------------------------------

**For continuous point-estimate regression**: use RMSE (default) or MAE for
outlier-robustness.

**For distributional (LSS) models**: use CRPS as the primary metric. CRPS is
a *proper scoring rule*: it rewards both accuracy and calibration, so it cannot
be gamed by reporting an over-wide predictive distribution.

**For count data** (poisson, zip, negativebinom): use the appropriate deviance.
Deviances are equivalent to twice the log-likelihood ratio against the saturated
model and are the standard criterion for GLM-type models.

**For probability / composition** (beta, dirichlet): use BetaBrierScore or
DirichletError.

**For uncertainty quantification**: combine CRPS with CoverageProbability and
SharpnessScore to get a complete picture of calibration and precision.

Writing a Custom Metric
-----------------------

Subclass :class:`DeepTabMetric`, set ``name`` and ``higher_is_better``, then
implement ``__call__``:

.. code-block:: python

    from deeptab.metrics import DeepTabMetric
    import numpy as np

    class MedianAbsoluteError(DeepTabMetric):
        name = "mdae"
        higher_is_better = False    # lower = better
        needs_raw = False           # use transformed predictions

        def __call__(self, y_true, y_pred):
            y_pred = np.asarray(y_pred)
            mean_pred = y_pred[:, 0] if y_pred.ndim == 2 else y_pred.ravel()
            return float(np.median(np.abs(np.asarray(y_true).ravel() - mean_pred)))

    # Use it anywhere a standard metric is accepted
    model.fit(X_train, y_train, val_metrics={"mdae": MedianAbsoluteError()})
    scores = model.evaluate(X_test, y_test, metrics={"mdae": MedianAbsoluteError()})

See Also
--------

- :doc:`../../core_concepts/training_and_evaluation`: training loop and evaluation guide
- :doc:`../../tutorials/uncertainty_quantification`: LSS model tutorial with metric examples
- :doc:`../distributions/index`: distribution families reference

API Reference
-------------

.. toctree::
   :maxdepth: 1

   metrics_ref
