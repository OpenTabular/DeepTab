Mambular
========

Sequential Mamba Structured State Space Model (SSM) blocks adapted for tabular
data. Each feature is embedded as a token and the resulting sequence is
processed by stacked Mamba layers, which use efficient linear-time recurrence
rather than quadratic attention. This allows Mambular to scale to longer feature
sequences while keeping memory costs linear.

When to Use
-----------

Ordered feature sets or large-scale datasets where Transformer memory costs are
prohibitive. Particularly compelling as an attention-free alternative when the
feature sequence has inherent order (e.g., time-step columns, sensor channels).

Limitations
-----------

- Newer architecture with less empirical validation than MLP/ResNet baselines.
- May require more epochs to converge compared to Transformer-based models.
- Performance can be sensitive to the Mamba-specific hyperparameters
  (``d_state``, ``expand_factor``).

API Reference
-------------

.. currentmodule:: deeptab.models

.. autoclass:: MambularRegressor
   :members:
   :undoc-members:

.. autoclass:: MambularClassifier
   :members:
   :undoc-members:

.. autoclass:: MambularLSS
   :members:
   :undoc-members:
