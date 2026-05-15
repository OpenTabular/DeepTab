SAINT
=====

Self-Attention and Intersample Attention Transformer. SAINT augments the
standard column-wise attention of a Transformer with a second attention
mechanism that operates across rows — allowing each sample to attend to other
samples in the batch. This enables the model to leverage inter-sample
relationships during training.

When to Use
-----------

When inter-sample relationships are informative, such as in recommendation or
retrieval tasks. Reported strong performance on semi-supervised tabular
benchmarks. Consider SAINT when FTTransformer leaves significant headroom and
more expressive attention is warranted.

Limitations
-----------

- Quadratic memory complexity in batch size due to intersample attention.
- Significantly slower than single-sample Transformer models on large batches.
- Gains over simpler models are dataset-dependent; not always worth the extra cost.

API Reference
-------------

.. currentmodule:: deeptab.models

.. autoclass:: SAINTRegressor
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: SAINTClassifier
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: SAINTLSS
   :members:
   :undoc-members:
   :noindex:
