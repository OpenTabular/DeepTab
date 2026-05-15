MLP
===

A fully-connected feedforward network with configurable depth and width. The
simplest and fastest deep learning baseline for tabular data. Each hidden layer
applies a linear transformation followed by an activation function and optional
dropout.

When to Use
-----------

Start here before trying more complex architectures. Works well on most datasets
as a fast, low-cost baseline. Ideal for smaller datasets or when compute budget
is limited. Also useful as a sanity-check model to verify the data pipeline.

Limitations
-----------

- Cannot model complex feature interactions without explicit feature engineering.
- May underfit on datasets with strong structural or sequential patterns.
- Performance plateaus with depth due to vanishing gradients (use ResNet if this
  is a concern).

API Reference
-------------

.. currentmodule:: deeptab.models

.. autoclass:: MLPRegressor
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: MLPClassifier
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: MLPLSS
   :members:
   :undoc-members:
   :noindex:
