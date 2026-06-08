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

=======================================    =======================================================================================================
Distribution                                Use Case
=======================================    =======================================================================================================
:class:`NormalDistribution`                General continuous targets, default choice.
:class:`LogNormalDistribution`             Strictly positive targets with multiplicative noise (e.g. prices, incomes).
:class:`StudentTDistribution`              Robust to outliers; heavy-tailed data.
:class:`GammaDistribution`                 Positive continuous targets (durations, amounts).
:class:`InverseGammaDistribution`          Positive targets with right skew.
:class:`BetaDistribution`                  Bounded targets in (0, 1) interval (proportions, rates).
:class:`JohnsonSuDistribution`             Flexible shape; can model skewness and kurtosis.
:class:`TweedieDistribution`               Zero-inflated positive targets (insurance claims, rainfall).
=======================================    =======================================================================================================

Discrete Distributions
~~~~~~~~~~~~~~~~~~~~~~

=======================================    =======================================================================================================
Distribution                                Use Case
=======================================    =======================================================================================================
:class:`PoissonDistribution`               Count data (non-negative integers).
:class:`ZeroInflatedPoissonDistribution`   Count data with excess zeros.
:class:`NegativeBinomialDistribution`      Overdispersed count data.
:class:`CategoricalDistribution`           Multiclass classification with uncertainty.
=======================================    =======================================================================================================

Multivariate / Compositional Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

=======================================    =======================================================================================================
Distribution                                Use Case
=======================================    =======================================================================================================
:class:`DirichletDistribution`             Compositional data (proportions that sum to 1).
:class:`MultinomialDistribution`           Multi-category count targets.
:class:`MixtureOfGaussiansDistribution`    Multimodal continuous targets (e.g. bimodal price distributions).
=======================================    =======================================================================================================

Quantile Regression
~~~~~~~~~~~~~~~~~~~

=======================================    =======================================================================================================
Distribution                                Use Case
=======================================    =======================================================================================================
:class:`Quantile`                          Predict arbitrary percentiles; distribution-free.
=======================================    =======================================================================================================

Quick Example
-------------

.. code-block:: python

    from deeptab.models import MambularLSS

    # Fit a distributional model
    model = MambularLSS()
    model.fit(X_train, y_train, family="normal")

    # Predict distribution parameters
    params = model.predict(X_test)  # Returns dict with 'loc' and 'scale'

    # Sample from predicted distributions
    samples = model.sample(X_test, n_samples=100)

    # Get prediction intervals
    lower, upper = model.predict_quantiles(X_test, quantiles=[0.025, 0.975])

Choosing a Distribution
------------------------

**For regression (continuous targets):**

- Start with ``normal`` (default)
- Use ``studentt`` if you have outliers
- Use ``lognormal`` for strictly positive targets with multiplicative noise
- Use ``gamma`` if targets are strictly positive
- Use ``beta`` if targets are in (0, 1)
- Use ``tweedie`` for zero-inflated positive targets (e.g. insurance claims)

**For count data:**

- Use ``poisson`` for counts without overdispersion
- Use ``zip`` for counts with excess zeros
- Use ``negativebinom`` for overdispersed counts

**For compositional data:**

- Use ``dirichlet`` for proportions that sum to 1
- Use ``multinomial`` for multi-category count targets

**For multimodal data:**

- Use ``mog`` (Mixture of Gaussians) for targets with multiple peaks

See Also
--------

- :doc:`../../core_concepts/distributional_regression` — LSS regression guide
- :doc:`../../tutorials/distributional` — Complete LSS examples
- :class:`deeptab.models.MambularLSS` — LSS model reference

API Reference
-------------

.. toctree::
   :maxdepth: 1

   distributions_ref
