NODE
====

Neural Oblivious Decision Ensembles. Each NODE layer is a differentiable
ensemble of oblivious decision trees — trees where the same splitting feature
and threshold is used at every node of a given depth. The trees are made
end-to-end differentiable via entmax transformations, allowing gradient-based
training.

When to Use
-----------

When you want the inductive bias of gradient-boosted decision trees inside a
neural framework. Often competitive with gradient boosting on structured tabular
benchmarks while remaining composable as a standard PyTorch layer.

Limitations
-----------

- High memory consumption, especially at larger tree depths.
- Slower to train than MLP-based models.
- Sensitive to the ``depth`` hyperparameter; too shallow loses expressiveness,
  too deep causes memory and overfitting issues.

API Reference
-------------

.. currentmodule:: deeptab.models

.. autoclass:: NODERegressor
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: NODEClassifier
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: NODELSS
   :members:
   :undoc-members:
   :noindex:
