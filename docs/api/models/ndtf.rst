NDTF
====

Neural Decision Tree Forest. An ensemble of differentiable soft decision trees
where routing probabilities at each node are learned via sigmoid activations.
A path-probability regularisation term (controlled by ``lamda``) penalises
over-confident or imbalanced routing, encouraging diverse tree usage across the
forest.

When to Use
-----------

When interpretability through decision paths is desirable alongside neural
gradient optimisation. Useful as an alternative to NODE when a forest structure
(multiple independent trees) is preferred over oblivious ensembles.

Limitations
-----------

- Sensitive to the ``temperature`` and ``lamda`` regularisation hyperparameters.
- Can underfit with too few trees (``n_ensembles``) or overfit with too many.
- Less effective for very high-dimensional data where feature selection at each
  split becomes noisy.

API Reference
-------------

.. currentmodule:: deeptab.models

.. autoclass:: NDTFRegressor
   :members:
   :undoc-members:

.. autoclass:: NDTFClassifier
   :members:
   :undoc-members:

.. autoclass:: NDTFLSS
   :members:
   :undoc-members:
