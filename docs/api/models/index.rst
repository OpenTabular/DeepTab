.. -*- mode: rst -*-

.. currentmodule:: deeptab.models

Models
======

This module provides classes for the Mambular models that adhere to scikit-learn's `BaseEstimator` interface.

Mambular
--------
=======================================    =======================================================================================================
Modules                                     Description
=======================================    =======================================================================================================
:class:`MambularClassifier`                 Multi-class and binary classification tasks with a sequential Mambular Model.
:class:`MambularRegressor`                  Regression tasks with a sequential Mambular Model.
:class:`MambularLSS`                        Various statistical distribution families for different types of regression and classification tasks.
=======================================    =======================================================================================================

FTTransformer
-------------
=======================================    =======================================================================================================
Modules                                     Description
=======================================    =======================================================================================================
:class:`FTTransformerClassifier`            FT transformer for classification tasks.
:class:`FTTransformerRegressor`            FT transformer for regression tasks.
:class:`FTTransformerLSS`                  Various statistical distribution families for different types of regression and classification tasks.
=======================================    =======================================================================================================

MLP Models
----------
=======================================    =======================================================================================================
Modules                                     Description
=======================================    =======================================================================================================
:class:`MLPClassifier`                      Multi-class and binary classification tasks.
:class:`MLPRegressor`                       MLP for regression tasks.
:class:`MLPLSS`                             Various statistical distribution families for different types of regression and classification tasks.
=======================================    =======================================================================================================

TabTransformer
--------------
=======================================    =======================================================================================================
Modules                                     Description
=======================================    =======================================================================================================
:class:`TabTransformerClassifier`           TabTransformer for classification tasks.
:class:`TabTransformerRegressor`            TabTransformer for regression tasks.
:class:`TabTransformerLSS`                  TabTransformer for distributional tasks.
=======================================    =======================================================================================================

ResNet
------
=======================================    =======================================================================================================
Modules                                     Description
=======================================    =======================================================================================================
:class:`ResNetClassifier`                   Multi-class and binary classification tasks using ResNet.
:class:`ResNetRegressor`                    Regression tasks using ResNet.
:class:`ResNetLSS`                          Distributional tasks using ResNet.
=======================================    =======================================================================================================

MambaTab
--------
=======================================    =======================================================================================================
Modules                                     Description
=======================================    =======================================================================================================
:class:`MambaTabClassifier`                 Multi-class and binary classification tasks using MambaTab.
:class:`MambaTabRegressor`                  Regression tasks using MambaTab.
:class:`MambaTabLSS`                        Distributional tasks using MambaTab.
=======================================    =======================================================================================================

MambaAttention
--------------
=======================================    =======================================================================================================
Modules                                     Description
=======================================    =======================================================================================================
:class:`MambAttentionClassifier`            Multi-class and binary classification tasks using a Combination between Mamba and Attention layers.
:class:`MambAttentionRegressor`             Regression tasks using sing a Combination between Mamba and Attention layers.
:class:`MambAttentionLSS`                   Distributional tasks using sing a Combination between Mamba and Attention layers.
=======================================    =======================================================================================================

RNN Models Including LSTM and GRU
---------------------------------
=======================================    =======================================================================================================
Modules                                     Description
=======================================    =======================================================================================================
:class:`TabulaRNNClassifier`                Multi-class and binary classification tasks using a RNN.
:class:`TabulaRNNRegressor`                 Regression tasks using a RNN.
:class:`TabulaRNNLSS`                       Distributional tasks using a RNN.
=======================================    =======================================================================================================

TabM
----
=======================================    =======================================================================================================
Modules                                     Description
=======================================    =======================================================================================================
:class:`TabMClassifier`                     Multi-class and binary classification tasks using TabM - Batch Ensembling MLP.
:class:`TabMRegressor`                      Regression tasks using TabM - Batch Ensembling MLP.
:class:`TabMLSS`                            Distributional tasks using TabM - Batch Ensembling MLP.
=======================================    =======================================================================================================

NODE
----
=======================================    =======================================================================================================
Modules                                     Description
=======================================    =======================================================================================================
:class:`NODEClassifier`                     Multi-class and binary classification tasks using Neural Oblivious Decision Ensembles.
:class:`NODERegressor`                      Regression tasks using Neural Oblivious Decision Ensembles.
:class:`NODELSS`                            Distributional tasks using Neural Oblivious Decision Ensembles.
=======================================    =======================================================================================================

NDTF
----
=======================================    =======================================================================================================
Modules                                     Description
=======================================    =======================================================================================================
:class:`NDTFClassifier`                     Multi-class and binary classification tasks using a Neural Decision Forest.
:class:`NDTFRegressor`                      Regression tasks using a Neural Decision Forest
:class:`NDTFLSS`                            Distributional tasks using a Neural Decision Forest.
=======================================    =======================================================================================================

SAINT
-----
=======================================    =======================================================================================================
Modules                                     Description
=======================================    =======================================================================================================
:class:`SAINTClassifier`                    Multi-class and binary classification tasks using SAINT.
:class:`SAINTRegressor`                     Regression tasks using SAINT.
:class:`SAINTLSS`                           Distributional tasks using SAINT.
=======================================    =======================================================================================================

Base Classes
------------
=======================================    =======================================================================================================
Modules                                     Description
=======================================    =======================================================================================================
:class:`SklearnBaseClassifier`              Base class for classification tasks.
:class:`SklearnBaseLSS`                     Base class for distributional tasks.
:class:`SklearnBaseRegressor`               Base class for regression tasks.
=======================================    =======================================================================================================

.. toctree::
   :maxdepth: 1

   Models
