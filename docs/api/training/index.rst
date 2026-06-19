.. -*- mode: rst -*-

.. currentmodule:: deeptab.training

Training
========

Low-level training utilities and Lightning modules. Most users should use the high-level
model API (``MambularClassifier``, etc.) instead of these classes directly.

Core Classes
------------

=======================================    =======================================================================================================
Class                                       Description
=======================================    =======================================================================================================
:class:`TaskModel`                         PyTorch Lightning module wrapping DeepTab architectures for training.
:class:`ContrastivePretrainer`             Self-supervised pretraining using contrastive learning on tabular data.
:func:`pretrain_embeddings`                Convenience function for pretraining feature embeddings.
=======================================    =======================================================================================================

When to Use
-----------

**Use the high-level API** (recommended):

.. code-block:: python

    from deeptab.models import MambularClassifier

    model = MambularClassifier()
    model.fit(X_train, y_train, max_epochs=50)

**Use these classes** when you need:

- Custom training loops with PyTorch Lightning
- Self-supervised pretraining before supervised training
- Integration with Lightning callbacks and loggers
- Multi-GPU or TPU training beyond the built-in support

TaskModel
---------

``TaskModel`` is the Lightning module used internally by all DeepTab estimators.
It wraps the base architecture and handles:

- Forward pass and loss computation
- Optimizer and scheduler configuration
- Metric logging

.. code-block:: python

    from deeptab.training import TaskModel
    from deeptab.architectures import Mambular
    from deeptab.configs import MambularConfig
    import pytorch_lightning as pl

    # Manual Lightning workflow
    config = MambularConfig(d_model=128, n_layers=6)
    backbone = Mambular(config)

    model = TaskModel(
        model=backbone,
        task="classification",
        num_classes=3,
    )

    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(model, datamodule=datamodule)

Contrastive Pretraining
------------------------

Contrastive pretraining warm-starts a model's **embedding layer** before
supervised training by pulling together rows that are close in target space and
pushing apart rows that are far. Pairs are built from the labels: same-class rows
(classification) or nearest-in-target rows (regression) form positives, and the
rest form negatives. It is most useful on **small or label-scarce datasets**,
where good embeddings are hard to learn from the supervised signal alone.

Only embedding-based architectures support it. The backbone must expose
``embedding_layer``, ``encode()``, ``pool_sequence()``, and
``get_embedding_state_dict()`` (for example ``FTTransformerClassifier``,
``TabTransformerClassifier``, ``MambularClassifier``). Architectures without an
embedding layer (MLP, ResNet) raise ``ArchitectureRequirementError``.

High-level API (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every embedding-based estimator exposes ``pretrain()``. Build the model first so
the backbone and data pipeline exist, warm-start the embeddings in place, then
fit as usual:

.. code-block:: python

    from deeptab.models import FTTransformerClassifier

    model = FTTransformerClassifier()
    model.build_model(X_train, y_train)          # build backbone + data pipeline
    model.pretrain(pretrain_epochs=15, k_neighbors=10)
    model.fit(X_train, y_train, max_epochs=50)   # supervised fine-tuning

``pretrain()`` updates the live model's embeddings, so the following ``fit()``
continues from the pretrained weights. It also writes the embedding weights to
``save_path`` for reuse.

Low-level API
~~~~~~~~~~~~~~

:func:`pretrain_embeddings` runs the same procedure on a raw architecture
instance and a PyTorch ``DataLoader`` that yields ``(data, labels)`` batches,
saving the learned embedding weights to ``save_path``:

.. code-block:: python

    import torch
    from deeptab.training import pretrain_embeddings

    # train_dataloader yields (data, labels): ``data`` is whatever the backbone's
    # encode() expects; ``labels`` drive the positive/negative pairing.
    pretrain_embeddings(
        base_model,
        train_dataloader,
        pretrain_epochs=5,
        k_neighbors=5,
        save_path="pretrained_embeddings.pth",
    )

    # Reuse the weights in a model that shares the same architecture.
    base_model.load_embedding_state_dict(torch.load("pretrained_embeddings.pth"))

For full control over the objective (margin, positive/negative terms, sequence
pooling), use :class:`ContrastivePretrainer` directly.

.. note::

   Pairs are formed **within each batch**. For classification, a batch must
   contain at least two classes; pretraining raises a ``ValueError`` if any
   sample has no same-class or no different-class neighbor. Use a batch size
   large enough to cover the classes, or a stratified sampler.

See Also
--------

- :doc:`../../core_concepts/training_and_evaluation`: Training guide
- :doc:`../models/index`: High-level model API
- `PyTorch Lightning docs <https://lightning.ai/docs/pytorch/stable/>`_

Reference
---------

.. toctree::
   :maxdepth: 1
   :hidden:

   training_ref
