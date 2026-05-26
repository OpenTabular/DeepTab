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
:class:`StudentTDistribution`              Robust to outliers, heavy-tailed data.
:class:`GammaDistribution`                 Positive continuous targets (durations, amounts).
:class:`InverseGammaDistribution`          Positive targets with right skew.
:class:`BetaDistribution`                  Bounded targets in (0, 1) interval (proportions, rates).
:class:`JohnsonSuDistribution`             Flexible shape, can model skewness and kurtosis.
=======================================    =======================================================================================================

Discrete Distributions
~~~~~~~~~~~~~~~~~~~~~~

=======================================    =======================================================================================================
Distribution                                Use Case
=======================================    =======================================================================================================
:class:`PoissonDistribution`               Count data (non-negative integers).
:class:`NegativeBinomialDistribution`      Overdispersed count data.
:class:`CategoricalDistribution`           Multiclass classification with uncertainty.
=======================================    =======================================================================================================

Multivariate Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

=======================================    =======================================================================================================
Distribution                                Use Case
=======================================    =======================================================================================================
:class:`DirichletDistribution`             Compositional data (proportions that sum to 1).
:class:`Quantile`                          Quantile regression (predict percentiles).
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
- Use ``gamma`` if targets are strictly positive
- Use ``beta`` if targets are in (0, 1)

**For count data:**

- Use ``poisson`` for counts without overdispersion
- Use ``negativebinomial`` for overdispersed counts

**For compositional data:**

- Use ``dirichlet`` for proportions that sum to 1

See Also
--------

- :doc:`../../core_concepts/distributional_regression` — LSS regression guide
- :doc:`../../tutorials/distributional` — Complete LSS examples
- :class:`deeptab.models.MambularLSS` — LSS model reference

Reference
---------

.. toctree::
   :maxdepth: 1
   :hidden:

   distributions_ref
