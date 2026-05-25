.. -*- mode: rst -*-

.. currentmodule:: deeptab.configs

Configurations
==============

DeepTab uses a **split-config API**: model hyperparameters are divided across three
separate dataclasses so that architecture choices, data preprocessing, and training
settings can be managed, versioned, and shared independently.

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Config class
     - Controls
     - Typical fields
   * - :class:`<Model>Config` |br| (e.g. :class:`MLPConfig`)
     - Neural architecture
     - ``d_model``, ``n_layers``, ``dropout``, ``activation``, …
   * - :class:`PreprocessingConfig`
     - Feature engineering
     - ``numerical_preprocessing``, ``n_bins``, ``scaling_strategy``, …
   * - :class:`TrainerConfig`
     - Training loop
     - ``max_epochs``, ``lr``, ``batch_size``, ``patience``, …

.. |br| raw:: html

   <br/>

----

Quick-start by task
-------------------

All three model variants — **Classifier**, **Regressor**, and **LSS** — accept the same
config objects. The only difference is the class you import.

Classification
~~~~~~~~~~~~~~

.. code-block:: python

    from deeptab.configs import MLPConfig, PreprocessingConfig, TrainerConfig
    from deeptab.models import MLPClassifier

    model = MLPClassifier(
        model_config=MLPConfig(d_model=128, dropout=0.1),
        preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
        trainer_config=TrainerConfig(max_epochs=50, lr=1e-3),
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)          # class labels
    proba = model.predict_proba(X_test)    # class probabilities

Regression
~~~~~~~~~~

.. code-block:: python

    from deeptab.configs import ResNetConfig, TrainerConfig
    from deeptab.models import ResNetRegressor

    model = ResNetRegressor(
        model_config=ResNetConfig(d_model=256, n_layers=4),
        trainer_config=TrainerConfig(max_epochs=100, lr=5e-4, patience=10),
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)          # continuous values

Distributional regression (LSS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``LSS`` models predict the *full distribution* of the target, not just a point estimate.
Pass ``family`` to ``fit`` to select the output distribution.

.. code-block:: python

    from deeptab.configs import MambularConfig, TrainerConfig
    from deeptab.models import MambularLSS

    model = MambularLSS(
        model_config=MambularConfig(d_model=64, n_layers=6),
        trainer_config=TrainerConfig(max_epochs=100, lr=1e-3),
    )
    model.fit(X_train, y_train, family="normal")   # learns μ and σ per row
    dist_params = model.predict(X_test)            # shape (N, n_params)

Common families: ``"normal"``, ``"poisson"``, ``"gamma"``, ``"beta"``, ``"dirichlet"``.

----

Scikit-learn compatibility
--------------------------

Every config dataclass extends ``sklearn.base.BaseEstimator``, so the full
scikit-learn parameter protocol is available.

get_params
~~~~~~~~~~

Returns a flat dictionary of all hyperparameters — identical to the behaviour of
any scikit-learn estimator:

.. code-block:: python

    from deeptab.configs import MLPConfig, TrainerConfig

    cfg = MLPConfig(d_model=128, dropout=0.2)
    print(cfg.get_params())
    # {'d_model': 128, 'dropout': 0.2, 'layer_sizes': [256, 128, 32], ...}

    trainer = TrainerConfig(max_epochs=50)
    print(trainer.get_params())
    # {'max_epochs': 50, 'lr': 0.0001, 'batch_size': 128, ...}

set_params
~~~~~~~~~~

Updates parameters in-place and returns ``self``, enabling scikit-learn pipeline
and grid-search integration:

.. code-block:: python

    cfg = MLPConfig()
    cfg.set_params(d_model=256, dropout=0.3)

    trainer = TrainerConfig()
    trainer.set_params(max_epochs=200, lr=5e-4)

Hyperparameter search with GridSearchCV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the estimator itself also follows ``get_params`` / ``set_params``, you can
tune any config field via ``GridSearchCV`` using the ``<config_attr>__<field>``
double-underscore notation:

.. code-block:: python

    from sklearn.model_selection import GridSearchCV
    from deeptab.configs import MLPConfig, TrainerConfig
    from deeptab.models import MLPClassifier

    model = MLPClassifier(
        model_config=MLPConfig(),
        trainer_config=TrainerConfig(max_epochs=20),
    )

    param_grid = {
        "model_config__d_model": [64, 128, 256],
        "model_config__dropout": [0.1, 0.3],
        "trainer_config__lr": [1e-3, 5e-4],
    }

    search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")
    search.fit(X_train, y_train)
    print(search.best_params_)

sklearn ``clone``
~~~~~~~~~~~~~~~~~

Configs can be deep-copied with ``sklearn.base.clone``:

.. code-block:: python

    from sklearn.base import clone

    original = MLPConfig(d_model=128)
    copy = clone(original)   # fully independent copy

----

Sharing and versioning configs
-------------------------------

Because configs are plain dataclasses they serialise trivially:

.. code-block:: python

    import dataclasses, json

    cfg = MLPConfig(d_model=128, dropout=0.1)
    # serialise
    blob = json.dumps(dataclasses.asdict(cfg))
    # restore
    cfg2 = MLPConfig(**json.loads(blob))

----

Available model configs
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Config class
     - Model family
   * - :class:`AutoIntConfig`
     - AutoInt — Automatic Feature Interaction Learning via Self-Attentive Neural Networks
   * - :class:`ENODEConfig`
     - ENODE — Extended Neural Oblivious Decision Ensembles
   * - :class:`FTTransformerConfig`
     - FT-Transformer — Feature Tokenizer Transformer
   * - :class:`MambaTabConfig`
     - MambaTab — Mamba-based tabular model
   * - :class:`MambAttentionConfig`
     - MambAttention — Mamba + self-attention hybrid
   * - :class:`MambularConfig`
     - Mambular — general-purpose Mamba backbone
   * - :class:`MLPConfig`
     - MLP — multilayer perceptron baseline
   * - :class:`ModernNCAConfig`
     - ModernNCA — Modern Neural Context-Aware model *(experimental)*
   * - :class:`NDTFConfig`
     - NDTF — Neural Decision Tree Forest
   * - :class:`NODEConfig`
     - NODE — Neural Oblivious Decision Ensembles
   * - :class:`ResNetConfig`
     - ResNet — residual network for tabular data
   * - :class:`SAINTConfig`
     - SAINT — Self-Attention and Intersample Attention Transformer
   * - :class:`TabMConfig`
     - TabM — Batch-Ensembling MLP
   * - :class:`TabRConfig`
     - TabR — Retrieval-Augmented Tabular model
   * - :class:`TabTransformerConfig`
     - TabTransformer — transformer with categorical embeddings
   * - :class:`TabulaRNNConfig`
     - TabulaRNN — LSTM / GRU recurrent baseline
   * - :class:`TangosConfig`
     - Tangos — Targeted Regularisation *(experimental)*
   * - :class:`TromptConfig`
     - Trompt — tree-inspired tabular model *(experimental)*

----

.. toctree::
   :maxdepth: 1

   Configurations
