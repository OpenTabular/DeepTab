.. -*- mode: rst -*-

.. currentmodule:: deeptab.base_models

BaseModels
==========

This module provides foundational classes and architectures for deeptab models, including various neural network architectures tailored for tabular data.

=========================================    =======================================================================================================
Modules                                       Description
=========================================    =======================================================================================================
:class:`Mambular`                             Flexible neural network model leveraging the Mamba architecture with configurable normalization techniques for tabular data.
:class:`MLP`                                  Multi-layer perceptron (MLP) model designed for tabular tasks, initialized with a custom configuration.
:class:`ResNet`                               Deep residual network (ResNet) model optimized for structured/tabular datasets.
:class:`FTTransformer`                        Feature Tokenizer (FTTransformer) model for tabular tasks, incorporating advanced embedding and normalization techniques.
:class:`TabTransformer`                       TabTransformer model leveraging attention mechanisms for tabular data processing.
:class:`NODE`                                 Neural Oblivious Decision Ensembles (NODE) for tabular tasks, combining decision tree logic with deep learning.
:class:`TabM`                                 TabM architecture designed for tabular data, implementing batch-ensembling MLP techniques.
:class:`NDTF`                                 Neural Decision Tree Forest (NDTF) model for tabular tasks, blending decision tree concepts with neural networks.
:class:`TabulaRNN`                            Recurrent neural network (RNN) model, including LSTM and GRU architectures, tailored for sequential or time-series tabular data.
:class:`MambAttention`                        Attention-based architecture for tabular tasks, combining feature importance weighting with advanced normalization techniques.
:class:`SAINT`                                SAINT model. Transformer based model using row and column attention.
:class:`MambaTab`                             Tabular model using a Mamba-Block on a joint input representation.
:class:`AutoInt`                              Automatic Feature Interaction model for tabular data.
:class:`ENODE`                                Embedding Neural Oblivious Decision Ensembles for tabular tasks.
:class:`ModernNCA`                            Modern Nearest Centroid Approach for tabular deep learning.
:class:`Tangos`                               Tangos model for tabular data.
:class:`Trompt`                               Trompt model for tabular data.
=========================================    =======================================================================================================


.. toctree::
   :maxdepth: 1

   BaseModels
