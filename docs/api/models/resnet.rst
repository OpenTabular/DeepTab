ResNet
======

A deep residual network adapted for tabular data. Skip connections let gradients
flow through deeper stacks without vanishing, enabling more representational
capacity than a plain MLP at the same depth. Each residual block applies two
linear layers with batch normalisation and a skip connection.

When to Use
-----------

Choose ResNet when a plain MLP fails to converge well or produces unstable
training curves, or when you need more depth without gradient issues. A good
second step after benchmarking MLP.

Limitations
-----------

- More hyperparameters than plain MLP (block size, number of blocks).
- Skip connections add memory overhead.
- May not outperform MLP on small datasets where depth is not beneficial.

API Reference
-------------

.. currentmodule:: deeptab.models

.. autoclass:: ResNetRegressor
   :members:
   :undoc-members:

.. autoclass:: ResNetClassifier
   :members:
   :undoc-members:

.. autoclass:: ResNetLSS
   :members:
   :undoc-members:
