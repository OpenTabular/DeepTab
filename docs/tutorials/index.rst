Tutorials
=========

This section provides hands-on tutorials demonstrating how to use DeepTab for various tabular learning tasks.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   classification
   regression
   distributional
   experimental

Overview
--------

Each tutorial follows a consistent structure:

1. **Setup** — Import statements and data preparation
2. **Basic workflow** — Train and evaluate a model with defaults
3. **Customization** — Configure model architecture, preprocessing, and training
4. **Advanced patterns** — Hyperparameter tuning, cross-validation, ensembles
5. **Model comparison** — Table of all available models for the task

Classification
~~~~~~~~~~~~~~

:doc:`classification` demonstrates binary and multiclass classification with DeepTab. Learn how to:

- Train a classifier with default settings
- Customize model architecture, preprocessing, and training configs
- Handle class imbalance with stratified splits and class weights
- Get probability outputs and use scikit-learn tools like GridSearchCV
- Compare all stable classification models

Regression
~~~~~~~~~~

:doc:`regression` shows how to predict continuous targets. Learn how to:

- Train a regressor with default settings
- Configure preprocessing for numerical and categorical features
- Customize optimization and early stopping
- Perform hyperparameter tuning with cross-validation
- Analyze residuals and feature importance
- Compare all stable regression models

Distributional Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`distributional` introduces LSS (Location, Scale, and Shape) models for uncertainty quantification. Learn how to:

- Train an LSS model to predict full distributions
- Choose the right distribution family for your data
- Extract distribution parameters and generate prediction intervals
- Visualize uncertainty and validate coverage
- Compare all stable LSS models

Experimental Models
~~~~~~~~~~~~~~~~~~~

:doc:`experimental` explains how to use cutting-edge models from ``deeptab.models.experimental``. Learn how to:

- Import and use experimental models safely
- Understand the differences from stable models
- Pin versions to avoid breaking changes
- Switch to stable imports when models are promoted

Prerequisites
-------------

These tutorials assume you have:

- Installed DeepTab (see :doc:`../getting_started/installation`)
- Read the :doc:`../getting_started/quickstart`
- Basic familiarity with Python, NumPy, and pandas

Next Steps
----------

After completing the tutorials:

- **Deep dive** → Read :doc:`../core_concepts/index` to understand internal workings
- **Customize** → Explore the :doc:`../api/configs/index` for full configuration options
- **Contribute** → See :doc:`../developer_guide/contributing` to add new models or features
