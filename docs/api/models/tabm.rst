TabM
====

Batch ensembling applied to an MLP. TabM trains multiple ensemble members that
share most of their weights, with only lightweight per-member scaling factors
making each head distinct. This delivers ensemble-level accuracy at near
single-model memory and compute cost.

When to Use
-----------

When you want ensembling diversity without the cost of training multiple
independent models. A strong regularised baseline that often outperforms plain
MLP with minimal extra overhead.

Limitations
-----------

- Slightly higher memory footprint than a plain MLP due to the per-member factors.
- The number of ensemble members is an additional hyperparameter to tune.
- Gains diminish beyond a moderate number of members.

API Reference
-------------

.. currentmodule:: deeptab.models

.. autoclass:: TabMRegressor
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: TabMClassifier
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: TabMLSS
   :members:
   :undoc-members:
   :noindex:
