TabTransformer
==============

Transformer for tabular data with a focus on categorical feature embeddings.
Categorical features are embedded and passed through Transformer encoder layers
to capture inter-categorical dependencies, while numerical features bypass the
attention mechanism and are concatenated at the prediction head.

When to Use
-----------

Datasets dominated by high-cardinality categorical features where relationships
between categories are informative. Commonly used in click-through-rate
prediction and entity-heavy tabular problems.

Limitations
-----------

- Limited benefit for datasets with mostly numerical features.
- Slower than MLP-based models.
- FTTransformer typically outperforms TabTransformer on mixed datasets because
  it tokenises all features uniformly.

API Reference
-------------

.. currentmodule:: deeptab.models

.. autoclass:: TabTransformerRegressor
   :members:
   :undoc-members:

.. autoclass:: TabTransformerClassifier
   :members:
   :undoc-members:

.. autoclass:: TabTransformerLSS
   :members:
   :undoc-members:
