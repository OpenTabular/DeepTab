API Reference
=============

Complete API documentation for DeepTab. All public classes and functions are documented here.

.. toctree::
   :maxdepth: 2
   :caption: API Modules

   models/index
   configs/index
   data/index
   distributions/index
   training/index

Overview
--------

DeepTab's API is organized into the following modules:

**Models** (:doc:`models/index`)
   Scikit-learn compatible estimators for classification, regression, and distributional regression.
   All models come in three variants: ``Classifier``, ``Regressor``, and ``LSS``.

**Configs** (:doc:`configs/index`)
   Configuration dataclasses for model architecture, preprocessing, and training.
   DeepTab uses a split-config system for maximum flexibility.

**Data** (:doc:`data/index`)
   Dataset and data module classes for loading and preprocessing tabular data.
   Includes schema definitions and batch representations.

**Distributions** (:doc:`distributions/index`)
   Distribution families for Location, Scale, and Shape (LSS) regression.
   Supports Normal, Beta, Gamma, Poisson, and many other families.

**Training** (:doc:`training/index`)
   Lightning modules and pretraining utilities for advanced workflows.
   For most users, the high-level model API is sufficient.

Quick Links
-----------

**Most Common Classes:**

- :class:`deeptab.models.MambularClassifier` — Flagship model for classification
- :class:`deeptab.models.MambularRegressor` — Flagship model for regression
- :class:`deeptab.models.MambularLSS` — Flagship model for distributional regression
- :class:`deeptab.data.TabularDataset` — Dataset class for tabular data
- :class:`deeptab.data.TabularDataModule` — Data module for training
- :class:`deeptab.configs.PreprocessingConfig` — Preprocessing configuration
- :class:`deeptab.configs.TrainerConfig` — Training configuration

**See Also:**

- :doc:`../getting_started/quickstart` — Quick start guide
- :doc:`../tutorials/index` — Hands-on tutorials
- :doc:`../model_zoo/index` — Model selection guide
