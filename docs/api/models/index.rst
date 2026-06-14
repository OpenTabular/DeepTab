.. -*- mode: rst -*-

.. currentmodule:: deeptab.models

Models
======

Scikit-learn compatible estimators for tabular deep learning. Every model
implements the ``BaseEstimator`` interface and ships in three task variants:

- **Classifier**: binary and multi-class classification
- **Regressor**: point-estimate regression
- **LSS**: distributional regression (Location, Scale, Shape)

.. code-block:: python

    from deeptab.models import MambularClassifier

    model = MambularClassifier()
    model.fit(X_train, y_train, max_epochs=50)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    metrics = model.evaluate(X_test, y_test)

For model descriptions, comparisons, and tuned configurations, see the
:doc:`../../model_zoo/stable/index`.

Stable Models
-------------

Each architecture provides ``Classifier``, ``Regressor``, and ``LSS`` variants.

==================  ====================================================
Architecture        Summary
==================  ====================================================
``Mambular``        Multi-layer Mamba. Strong default.
``MambaTab``        Single Mamba block. Fast.
``MambAttention``   Hybrid Mamba and attention.
``FTTransformer``   Feature-tokenizer transformer.
``TabTransformer``  Transformer for categorical-heavy data.
``SAINT``           Row and column attention.
``ResNet``          Residual MLP.
``MLP``             Plain MLP baseline.
``TabM``            Batch-ensembling MLP.
``AutoInt``         Automatic feature interactions.
``NODE``            Neural oblivious decision ensembles.
``ENODE``           Enhanced NODE.
``NDTF``            Neural decision tree forest.
``TabR``            Retrieval-augmented model.
``TabulaRNN``       RNN over feature sequences.
==================  ====================================================

Experimental Models
-------------------

.. warning::

   Experimental models live in ``deeptab.models.experimental``. Their API may
   change without a deprecation cycle, so pin your DeepTab version when using
   them.

==================  ====================================================
Architecture        Summary
==================  ====================================================
``ModernNCA``       Modern neighborhood component analysis.
``Tangos``          Tangent-based regularization.
``Trompt``          Prompt-based transformer.
==================  ====================================================

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
