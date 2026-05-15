MambaTab
========

A lightweight Mamba-based architecture that applies a single Mamba SSM block to
a joint representation of all input features. Rather than tokenising each
feature individually, MambaTab concatenates all feature embeddings into one
vector, making it the most computationally efficient model in the Mamba family.

When to Use
-----------

Efficiency-focused scenarios where a fast Mamba-based baseline is needed before
scaling to the more expressive :doc:`mambular` architecture. Useful when
training or inference speed is a hard constraint.

Limitations
-----------

- The joint input representation loses per-feature granularity compared to
  token-level models (FTTransformer, Mambular).
- Less expressive than multi-layer Mambular for complex datasets.

API Reference
-------------

.. currentmodule:: deeptab.models

.. autoclass:: MambaTabRegressor
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: MambaTabClassifier
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: MambaTabLSS
   :members:
   :undoc-members:
   :noindex:
