TabR
====

Retrieval-augmented tabular model. At inference time, TabR retrieves the most
similar training examples from a stored memory of embeddings and uses them as
additional context when computing the prediction. This gives the model access to
local neighbourhood information beyond what is encoded in its weights.

When to Use
-----------

Datasets where local similarity structure is informative — rows that are similar
in feature space tend to share similar targets. Effective on low-to-medium-size
datasets where a full nearest-neighbour memory can be maintained affordably.

Limitations
-----------

- Inference time scales with training set size as the model must search the
  memory store.
- Not suitable for very large datasets (>100 k rows) without approximate
  nearest-neighbour indexing.
- Requires keeping the training set in memory during inference.

API Reference
-------------

.. currentmodule:: deeptab.models

.. autoclass:: TabRRegressor
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: TabRClassifier
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: TabRLSS
   :members:
   :undoc-members:
   :noindex:
