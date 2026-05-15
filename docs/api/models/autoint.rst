AutoInt
=======

Automatic feature Interaction learning via multi-head self-attention on feature
embeddings. Each input feature is projected into an embedding and the
embeddings are passed through stacked multi-head attention layers. Residual
connections allow the model to combine the original feature representation with
the interaction-augmented representation, making the learned interactions
explicitly additive.

When to Use
-----------

When capturing explicit pairwise and higher-order feature interactions is the
primary modelling goal. Historically strong in click-through-rate prediction
and recommendation system benchmarks.

Limitations
-----------

- Performance is generally comparable to FTTransformer on most generic tabular
  benchmarks; FTTransformer is often a simpler first choice.
- Less effective for very high-dimensional sparse feature spaces compared to
  factorisation-machine-based methods.
- The additional residual interaction terms add minor overhead vs plain
  Transformer models.

API Reference
-------------

.. currentmodule:: deeptab.models

.. autoclass:: AutoIntRegressor
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: AutoIntClassifier
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: AutoIntLSS
   :members:
   :undoc-members:
   :noindex:
