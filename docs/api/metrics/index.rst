.. -*- mode: rst -*-

.. currentmodule:: deeptab.metrics

Metrics
=======

.. note::
   This module is currently under active development. Metric classes will be available in a future release.

Evaluation metrics for tabular models. This module will provide:

- Classification metrics (accuracy, F1, ROC-AUC, etc.)
- Regression metrics (MSE, MAE, R², etc.)
- Distributional metrics (NLL, CRPS, etc.)
- Custom metric implementations

Status
------

The metrics module is currently a placeholder. For now, use:

- ``model.evaluate()`` method for built-in evaluation
- ``sklearn.metrics`` for scikit-learn compatible metrics
- Lightning's ``torchmetrics`` for low-level metric computation

See Also
--------

- :doc:`../../core_concepts/training_and_evaluation` — Evaluation guide
- :doc:`../../tutorials/classification` — Evaluation examples
