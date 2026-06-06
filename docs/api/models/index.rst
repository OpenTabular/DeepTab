.. -*- mode: rst -*-

.. currentmodule:: deeptab.models

Models
======

Scikit-learn compatible estimators for tabular deep learning. All models implement
the ``BaseEstimator`` interface and come in three task variants:

- **Classifier** — Multi-class and binary classification
- **Regressor** — Standard regression (point estimates)
- **LSS** — Distributional regression (Location, Scale, Shape)

Quick Example
-------------

.. code-block:: python

    from deeptab.models import MambularClassifier

    # Instantiate
    model = MambularClassifier()

    # Fit
    model.fit(X_train, y_train, max_epochs=50)

    # Predict
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Evaluate
    metrics = model.evaluate(X_test, y_test)

Model Selection
---------------

For detailed model comparisons, use cases, and configuration guidance, see the
:doc:`../../model_zoo/stable/index`.

Quick recommendations:

- **Best overall**: :class:`MambularClassifier`, :class:`MambularRegressor`, :class:`MambularLSS`
- **Fast baseline**: :class:`ResNetClassifier`, :class:`ResNetRegressor`, :class:`ResNetLSS`
- **Interpretable**: :class:`NODEClassifier`, :class:`NODERegressor`, :class:`NODELSS`
- **Categorical-heavy**: :class:`TabTransformerClassifier`, :class:`TabTransformerRegressor`, :class:`TabTransformerLSS`

Stable Models
-------------

**State Space Models**

=======================================    =======================================================================================================
Class                                       Description
=======================================    =======================================================================================================
:class:`MambularClassifier`                 Multi-layer Mamba architecture. Best overall performance. See :doc:`../../model_zoo/stable/mambular`.
:class:`MambularRegressor`
:class:`MambularLSS`
:class:`MambaTabClassifier`                 Single Mamba block. Fast and efficient. See :doc:`../../model_zoo/stable/mambatab`.
:class:`MambaTabRegressor`
:class:`MambaTabLSS`
:class:`MambAttentionClassifier`            Hybrid Mamba + Attention. Complex patterns. See :doc:`../../model_zoo/stable/mambattention`.
:class:`MambAttentionRegressor`
:class:`MambAttentionLSS`
=======================================    =======================================================================================================

**Transformer-Based**

=======================================    =======================================================================================================
Class                                       Description
=======================================    =======================================================================================================
:class:`FTTransformerClassifier`            Feature Tokenizer Transformer. Strong baseline. See :doc:`../../model_zoo/stable/fttransformer`.
:class:`FTTransformerRegressor`
:class:`FTTransformerLSS`
:class:`TabTransformerClassifier`           Specialized for categorical features. See :doc:`../../model_zoo/stable/tabtransformer`.
:class:`TabTransformerRegressor`
:class:`TabTransformerLSS`
:class:`SAINTClassifier`                    Row and column attention. Semi-supervised. See :doc:`../../model_zoo/stable/saint`.
:class:`SAINTRegressor`
:class:`SAINTLSS`
=======================================    =======================================================================================================

**MLP-Based**

=======================================    =======================================================================================================
Class                                       Description
=======================================    =======================================================================================================
:class:`ResNetClassifier`                   Residual MLP. Fast and simple. See :doc:`../../model_zoo/stable/resnet`.
:class:`ResNetRegressor`
:class:`ResNetLSS`
:class:`MLPClassifier`                      Standard MLP. Fastest baseline. See :doc:`../../model_zoo/stable/mlp`.
:class:`MLPRegressor`
:class:`MLPLSS`
:class:`TabMClassifier`                     Batch ensembling MLP. See :doc:`../../model_zoo/stable/tabm`.
:class:`TabMRegressor`
:class:`TabMLSS`
:class:`AutoIntClassifier`                  Automatic feature interactions. See :doc:`../../model_zoo/stable/autoint`.
:class:`AutoIntRegressor`
:class:`AutoIntLSS`
=======================================    =======================================================================================================

**Tree-Based**

=======================================    =======================================================================================================
Class                                       Description
=======================================    =======================================================================================================
:class:`NODEClassifier`                     Neural Oblivious Decision Ensembles. Interpretable. See :doc:`../../model_zoo/stable/node`.
:class:`NODERegressor`
:class:`NODELSS`
:class:`ENODEClassifier`                    Enhanced NODE. See :doc:`../../model_zoo/stable/enode`.
:class:`ENODERegressor`
:class:`ENODELSS`
:class:`NDTFClassifier`                     Neural Decision Tree Forest. See :doc:`../../model_zoo/stable/ndtf`.
:class:`NDTFRegressor`
:class:`NDTFLSS`
=======================================    =======================================================================================================

**Specialized**

=======================================    =======================================================================================================
Class                                       Description
=======================================    =======================================================================================================
:class:`TabRClassifier`                     Retrieval-augmented model. Large datasets. See :doc:`../../model_zoo/stable/tabr`.
:class:`TabRRegressor`
:class:`TabRLSS`
:class:`TabulaRNNClassifier`                RNN for sequential features. See :doc:`../../model_zoo/stable/tabularnn`.
:class:`TabulaRNNRegressor`
:class:`TabulaRNNLSS`
=======================================    =======================================================================================================

Experimental Models
-------------------

.. warning::

   Experimental models are available from ``deeptab.models.experimental``.
   Their API may change without a deprecation cycle. Always pin your DeepTab version
   when using experimental models.

.. currentmodule:: deeptab.models.experimental

=======================================    =======================================================================================================
Class                                       Description
=======================================    =======================================================================================================
:class:`ModernNCAClassifier`                Modern Neighborhood Component Analysis. See :doc:`../../model_zoo/experimental/modernnca`.
:class:`ModernNCARegressor`
:class:`ModernNCALSS`
:class:`TangosClassifier`                   Tangent-based optimization. See :doc:`../../model_zoo/experimental/tangos`.
:class:`TangosRegressor`
:class:`TangosLSS`
:class:`TromptClassifier`                   Transformer with prompts. See :doc:`../../model_zoo/experimental/trompt`.
:class:`TromptRegressor`
:class:`TromptLSS`
=======================================    =======================================================================================================

Base Classes
------------

.. currentmodule:: deeptab.models

=======================================    =======================================================================================================
Class                                       Description
=======================================    =======================================================================================================
:class:`SklearnBaseClassifier`              Abstract base class for all classification models.
:class:`SklearnBaseRegressor`               Abstract base class for all regression models.
:class:`SklearnBaseLSS`                     Abstract base class for all distributional regression models.
=======================================    =======================================================================================================

See Also
--------

- :doc:`../../model_zoo/stable/index` — Detailed model descriptions and selection guide
- :doc:`../../model_zoo/comparison_tables` — Performance comparisons
- :doc:`../../model_zoo/recommended_configs` — Hyperparameter recipes
- :doc:`../../tutorials/imbalance_classification` — Hands-on classification example

Reference
---------

.. toctree::
   :maxdepth: 1
   :caption: Model Reference

   Models
   autoint
   enode
   fttransformer
   mambatab
   mambattention
   mambular
   mlp
   ndtf
   node
   resnet
   saint
   tabm
   tabr
   tabtransformer
   tabularrnn
