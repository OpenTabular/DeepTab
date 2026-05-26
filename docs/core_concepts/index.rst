Core Concepts
=============

This section explains the fundamental concepts you need to understand before using DeepTab effectively.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   sklearn_api
   model_tiers
   config_system
   preprocessing
   classification
   regression
   distributional_regression
   training_and_evaluation

Overview
--------

The Core Concepts section covers eight key topics that form the foundation of working with DeepTab:

scikit-learn API
~~~~~~~~~~~~~~~~

:doc:`sklearn_api` explains how DeepTab implements the familiar scikit-learn interface with ``fit``, ``predict``, ``predict_proba``, and ``evaluate`` methods. Learn about input formats, method signatures, and integration with scikit-learn tools like ``GridSearchCV`` and ``Pipeline``.

Model Tiers
~~~~~~~~~~~

:doc:`model_tiers` describes the difference between stable and experimental models. Stable models have frozen APIs under semantic versioning, while experimental models may change without deprecation. Learn when to use each tier and how models graduate from experimental to stable.

Config System
~~~~~~~~~~~~~

:doc:`config_system` introduces DeepTab's split-config design with three independent config classes: ``ModelConfig`` for architecture, ``PreprocessingConfig`` for feature engineering, and ``TrainerConfig`` for training loops. Understand how to customize each aspect independently and integrate with hyperparameter search.

Preprocessing
~~~~~~~~~~~~~

:doc:`preprocessing` covers automatic feature type detection, numerical preprocessing strategies (standard, quantile, minmax, ple, binning), categorical encoding, and handling missing values. Learn how to customize preprocessing and work with pre-computed embeddings.

Classification
~~~~~~~~~~~~~~

:doc:`classification` focuses on classification-specific concepts including binary vs multiclass, class imbalance handling, stratified splits (automatic in v2.0), probability outputs, and evaluation metrics. Learn how to handle imbalanced data and interpret model outputs.

Regression
~~~~~~~~~~

:doc:`regression` explains regression-specific topics including continuous predictions, target preprocessing, evaluation metrics (RMSE, MAE, R²), residual analysis, and handling different target distributions. Learn best practices for regression modeling.

Distributional Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`distributional_regression` introduces LSS (Location, Scale, and Shape) models that predict full probability distributions instead of point estimates. Learn about distribution families (normal, poisson, gamma, beta, etc.), prediction intervals, quantile predictions, and uncertainty quantification.

Training and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`training_and_evaluation` explains what happens during ``fit()``, including the training loop, early stopping, learning rate scheduling, gradient clipping, optimization, and monitoring. Learn how to evaluate models, handle GPU training, and troubleshoot common issues.

Reading Guide
-------------

**For beginners:**

1. Start with :doc:`sklearn_api` to understand the interface
2. Read :doc:`model_tiers` to choose appropriate models
3. Skim :doc:`config_system` to see what's configurable
4. Jump to task-specific pages (:doc:`classification` or :doc:`regression`)

**For advanced users:**

1. Review :doc:`config_system` for full customization options
2. Read :doc:`preprocessing` for preprocessing control
3. Explore :doc:`distributional_regression` for uncertainty quantification
4. Study :doc:`training_and_evaluation` for training optimization

**For specific tasks:**

- **Classification problems** → :doc:`classification`
- **Regression problems** → :doc:`regression`
- **Need uncertainty** → :doc:`distributional_regression`
- **Custom preprocessing** → :doc:`preprocessing`
- **Training issues** → :doc:`training_and_evaluation`

Prerequisites
-------------

This section assumes you have:

- Installed DeepTab (see :doc:`../getting_started/installation`)
- Basic Python and NumPy knowledge
- Familiarity with scikit-learn (helpful but not required)
- Understanding of supervised learning (classification/regression)

Next Steps
----------

After reading the core concepts:

- **Try the examples** — :doc:`../examples/classification`, :doc:`../examples/regression`
- **Explore the API** — :doc:`../api/models/index`, :doc:`../api/configs/index`
- **Ask questions** — Check the :doc:`../getting_started/faq` or open a GitHub issue
