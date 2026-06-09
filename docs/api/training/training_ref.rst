deeptab.training
================

The classes below are the internal Lightning modules used by all DeepTab
estimators. Most users interact with these indirectly through the high-level
model API (e.g. ``MambularClassifier``).

``TaskModel``
-------------

The PyTorch Lightning module that wraps every DeepTab architecture.
Responsible for the forward pass, loss computation, optimizer/scheduler
configuration, and metric logging. Constructed automatically by each
estimator; users only need it for custom Lightning workflows.

``ContrastivePretrainer``
-------------------------

Self-supervised pretraining module using contrastive learning on tabular
data. Used via the ``pretrain_embeddings`` convenience function.

``pretrain_embeddings``
-----------------------

Convenience function that wraps ``ContrastivePretrainer`` for pretraining
feature embeddings before supervised training.
