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

Self-supervised pretraining can improve performance on small datasets by learning
better feature representations before supervised training.

:func:`pretrain_embeddings` operates on a base architecture (an ``nn.Module`` with an
``embedding_layer`` and an ``encode`` method) and a PyTorch ``DataLoader`` that yields
``(numerical_features, categorical_features)`` batches. It trains the embedding layer
with a contrastive objective and saves the learned weights to ``save_path``.

.. code-block:: python

    from deeptab.training import pretrain_embeddings

    # ``base_model`` is a DeepTab architecture instance; ``train_dataloader`` yields
    # (numerical_features, categorical_features) batches.
    pretrain_embeddings(
        base_model,
        train_dataloader,
        pretrain_epochs=5,
        save_path="pretrained_embeddings.pth",
    )

The saved embedding weights can then be loaded into a model that shares the same
architecture before supervised fine-tuning. For finer control over the contrastive
objective, use :class:`ContrastivePretrainer` directly.

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
