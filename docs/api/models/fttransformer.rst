FTTransformer
=============

Feature Tokenizer + Transformer. Each input feature — numerical or categorical —
is mapped to a dense token embedding, and the resulting sequence of tokens is
processed through a stack of standard Transformer encoder layers. A ``[CLS]``
token is prepended and used to produce the final prediction.

When to Use
-----------

Strong general-purpose model. Particularly effective on mixed datasets with both
numerical and categorical features where pairwise feature interactions are
important. Typically the first Transformer baseline to try.

Limitations
-----------

- Higher memory and compute cost relative to MLP and ResNet.
- Tends to overfit on very small datasets (under ~500 samples); consider adding
  dropout or reducing depth.
- Longer training time than simpler architectures.

API Reference
-------------

.. currentmodule:: deeptab.models

.. autoclass:: FTTransformerRegressor
   :members:
   :undoc-members:

.. autoclass:: FTTransformerClassifier
   :members:
   :undoc-members:

.. autoclass:: FTTransformerLSS
   :members:
   :undoc-members:
