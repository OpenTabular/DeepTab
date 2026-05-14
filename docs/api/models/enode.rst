ENODE
=====

Extended Neural Oblivious Decision Ensembles. ENODE builds on :doc:`node` by
adding explicit feature embedding layers before the decision ensemble. These
embedding layers transform raw input features into richer representations before
they are fed into the differentiable decision trees, improving performance when
the raw feature space is noisy or heterogeneous.

When to Use
-----------

Upgrade from NODE when raw feature quality is poor, the data is heterogeneous,
or vanilla NODE underfits. The embedding layers add a small representational
overhead that often pays off on real-world datasets.

Limitations
-----------

- Inherits the same fundamental limitations as NODE (high memory, slow training).
- Increased model size compared to plain NODE.
- May be harder to interpret than NODE because the input to the decision
  ensemble is no longer the raw feature space.

API Reference
-------------

.. currentmodule:: deeptab.models

.. autoclass:: ENODERegressor
   :members:
   :undoc-members:

.. autoclass:: ENODEClassifier
   :members:
   :undoc-members:

.. autoclass:: ENODELSS
   :members:
   :undoc-members:
