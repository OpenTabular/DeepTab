MambAttention
=============

Hybrid Mamba + Attention architecture. MambAttention interleaves Mamba SSM
layers with multi-head self-attention layers, allowing the model to capture both
local sequential patterns (via Mamba's linear-time recurrence) and global
dependencies across all features simultaneously (via attention).

When to Use
-----------

When you need the memory efficiency of Mamba for local patterns and the
expressiveness of attention for global feature interactions. A natural upgrade
from either :doc:`mambular` or :doc:`fttransformer` when neither alone is
sufficient.

Limitations
-----------

- More hyperparameters than either Mambular or FTTransformer alone.
- Higher compute and memory cost than a pure Mamba or pure attention model.
- Fewer community benchmarks available; expect more tuning effort.

API Reference
-------------

.. currentmodule:: deeptab.models

.. autoclass:: MambAttentionRegressor
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: MambAttentionClassifier
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: MambAttentionLSS
   :members:
   :undoc-members:
   :noindex:
