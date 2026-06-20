.. -*- mode: rst -*-

.. currentmodule:: deeptab.distributions

Distributions
=============

Distribution families for Location, Scale, and Shape (LSS) regression. Each distribution defines
a parametric family and methods for computing negative log-likelihood loss.

Overview
--------

DeepTab's LSS models can predict full probability distributions instead of point estimates.
This is useful for uncertainty quantification, probabilistic forecasting, and heteroskedastic regression.

Available Distributions
-----------------------

Continuous Distributions
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Distribution
     - Use Case
   * - :class:`NormalDistribution`
     - General continuous targets; default choice.
   * - :class:`LogNormalDistribution`
     - Strictly positive targets with multiplicative noise (prices, incomes).
   * - :class:`StudentTDistribution`
     - Robust to outliers; heavy-tailed data.
   * - :class:`GammaDistribution`
     - Positive continuous targets (durations, amounts).
   * - :class:`InverseGammaDistribution`
     - Positive targets with right skew.
   * - :class:`BetaDistribution`
     - Bounded targets in (0, 1) interval (proportions, rates).
   * - :class:`JohnsonSuDistribution`
     - Flexible shape; can model skewness and kurtosis.
   * - :class:`TweedieDistribution`
     - Zero-inflated positive targets (insurance claims, rainfall).

Discrete Distributions
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Distribution
     - Use Case
   * - :class:`PoissonDistribution`
     - Count data (non-negative integers).
   * - :class:`ZeroInflatedPoissonDistribution`
     - Count data with excess zeros.
   * - :class:`NegativeBinomialDistribution`
     - Overdispersed count data.
   * - :class:`CategoricalDistribution`
     - Multiclass classification with uncertainty.

Multivariate / Compositional Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Distribution
     - Use Case
   * - :class:`DirichletDistribution`
     - Compositional data (proportions that sum to 1).
   * - :class:`MultinomialDistribution`
     - Multi-category count targets.
   * - :class:`MixtureOfGaussiansDistribution`
     - Multimodal continuous targets (bimodal price distributions etc.).

Quantile Regression
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Distribution
     - Use Case
   * - :class:`Quantile`
     - Predict arbitrary percentiles; distribution-free.

Quick Example
-------------

.. code-block:: python

    from deeptab.models import MambularLSS

    # Fit a distributional model
    model = MambularLSS()
    model.fit(X_train, y_train, family="normal")

    # Predict distribution parameters as an array of shape (n_samples, n_params).
    # For the normal family the columns are (loc, scale).
    params = model.predict(X_test)

    # Score with distribution-aware metrics such as CRPS and NLL
    scores = model.evaluate(X_test, y_test)

For worked examples that turn these parameters into prediction intervals and
calibration plots, see the :doc:`../../tutorials/uncertainty_quantification` tutorial.

Choosing a Distribution
------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - ``family=``
     - Target type
     - Use when
   * - ``"normal"``
     - Continuous
     - Default starting point; symmetric noise around a mean.
   * - ``"studentt"``
     - Continuous
     - Outliers are present; need heavier tails than Normal.
   * - ``"lognormal"``
     - Positive continuous
     - Multiplicative noise; targets span multiple orders of magnitude (prices, incomes).
   * - ``"gamma"``
     - Positive continuous
     - Strictly positive targets with right skew (durations, rainfall amounts).
   * - ``"inversegamma"``
     - Positive continuous
     - Positive targets with a longer right tail than Gamma.
   * - ``"beta"``
     - (0, 1) bounded
     - Proportions, rates, probabilities that must stay in (0, 1).
   * - ``"johnsonsu"``
     - Continuous
     - Need to model both skewness and excess kurtosis simultaneously.
   * - ``"tweedie"``
     - Zero-inflated positive
     - Mix of exact zeros and positive values (insurance claims, rainfall).
   * - ``"poisson"``
     - Count
     - Non-negative integer counts with mean ≈ variance.
   * - ``"zip"``
     - Count
     - Count data with more zeros than Poisson predicts.
   * - ``"negativebinom"``
     - Count
     - Overdispersed counts (variance > mean).
   * - ``"categorical"``
     - Multiclass
     - Classification with calibrated class probabilities.
   * - ``"dirichlet"``
     - Compositional
     - Vectors of proportions that must sum to 1.
   * - ``"multinomial"``
     - Multi-category count
     - Integer-valued compositional targets.
   * - ``"mog"``
     - Continuous multimodal
     - Targets with multiple distinct peaks (mixture of regimes).
   * - ``"quantile"``
     - Distribution-free
     - Predict specific percentiles without assuming a parametric family.

See Also
--------

- :doc:`../../tutorials/uncertainty_quantification`: Complete LSS examples
- :class:`deeptab.models.MambularLSS`: LSS model reference

API Reference
-------------

.. toctree::
   :maxdepth: 1

   distributions_ref
