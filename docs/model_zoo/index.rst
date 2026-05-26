Model Zoo
=========

The DeepTab Model Zoo contains detailed documentation for all available architectures. Each model page includes a concise overview, key characteristics, recommended use cases, configuration options, and usage examples.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   comparison_tables
   recommended_configs
   stable/index
   experimental/index

Overview
--------

DeepTab provides 15 stable and 3 experimental deep learning architectures for tabular data. All models support:

- **Three task types**: Classification, Regression, LSS (distributional regression)
- **Automatic preprocessing**: Numerical and categorical feature detection
- **Unified API**: Same fit/predict interface across all models
- **Config system**: Independent ModelConfig, PreprocessingConfig, TrainerConfig
- **sklearn integration**: GridSearchCV, Pipeline, cross-validation

Model Categories
----------------

Transformer-based Models
~~~~~~~~~~~~~~~~~~~~~~~~

Models using attention mechanisms for feature interactions:

- :doc:`stable/fttransformer` — Feature Tokenizer Transformer (strong general-purpose)
- :doc:`stable/tabtransformer` — Transformer on categorical embeddings
- :doc:`stable/saint` — Self-attention and intersample attention

State Space Models
~~~~~~~~~~~~~~~~~~

Models using Mamba architecture for efficient sequence modeling:

- :doc:`stable/mambular` — Stacked Mamba SSM (flagship model)
- :doc:`stable/mambatab` — Single Mamba block (lightweight)
- :doc:`stable/mambattention` — Mamba + attention hybrid

MLP-based Models
~~~~~~~~~~~~~~~~

Feedforward and residual architectures:

- :doc:`stable/mlp` — Simple feedforward baseline
- :doc:`stable/resnet` — Residual MLP for deeper networks
- :doc:`stable/tabm` — Batch-ensembling MLP

Tree-based Neural Models
~~~~~~~~~~~~~~~~~~~~~~~~~

Models combining neural networks with decision tree structures:

- :doc:`stable/node` — Neural Oblivious Decision Ensembles
- :doc:`stable/enode` — Extended NODE with feature embeddings
- :doc:`stable/ndtf` — Neural Decision Tree Forest

Specialized Architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`stable/tabr` — Retrieval-augmented learning
- :doc:`stable/tabularnn` — RNN/LSTM/GRU for sequential features
- :doc:`stable/autoint` — Attention-based feature interactions

Experimental Models
~~~~~~~~~~~~~~~~~~~

Cutting-edge models under evaluation:

- :doc:`experimental/modernnca` — Modern Neighborhood Component Analysis
- :doc:`experimental/trompt` — Transformer with prompting
- :doc:`experimental/tangos` — Tangent-based optimization

Quick Start
-----------

All models follow the same usage pattern:

.. code-block:: python

   from deeptab.models import MambularClassifier  # or any model

   model = MambularClassifier()
   model.fit(X_train, y_train, max_epochs=50)
   predictions = model.predict(X_test)

See :doc:`comparison_tables` for performance comparisons and :doc:`recommended_configs` for suggested hyperparameters.

Choosing a Model
----------------

**Quick recommendations:**

- **Best general-purpose**: :doc:`stable/mambular`, :doc:`stable/fttransformer`
- **Fastest training**: :doc:`stable/mlp`, :doc:`stable/resnet`
- **Categorical-heavy data**: :doc:`stable/tabtransformer`
- **Small datasets**: :doc:`stable/tabm`, :doc:`stable/mambatab`
- **Large datasets**: :doc:`stable/mambular`, :doc:`stable/tabr`
- **Interpretability**: :doc:`stable/node`, :doc:`stable/ndtf`

See individual model pages for detailed characteristics and use cases.
