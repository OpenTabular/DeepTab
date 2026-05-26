.. -*- mode: rst -*-

.. currentmodule:: deeptab.data

Data
====

Dataset and data module classes for tabular data loading, preprocessing, and schema management.

Core Classes
------------

=======================================    =======================================================================================================
Class                                       Description
=======================================    =======================================================================================================
:class:`TabularDataset`                    Dataset class for loading and preprocessing tabular data with automatic feature detection.
:class:`TabularDataModule`                 Lightning DataModule for train/val/test splits, batching, and data loading.
:class:`FeatureSchema`                     Schema definition containing feature types, names, and metadata.
:class:`FeatureInfo`                       Individual feature information (name, type, cardinality, etc.).
:class:`TabularBatch`                      Typed batch representation with numerical, categorical, and target tensors.
=======================================    =======================================================================================================

Quick Example
-------------

.. code-block:: python

    from deeptab.data import TabularDataset, TabularDataModule

    # Create dataset
    dataset = TabularDataset(
        X=X_train,
        y=y_train,
        categorical_features=["col1", "col2"],
        numerical_features=["col3", "col4"],
    )

    # Create data module
    datamodule = TabularDataModule(
        dataset=dataset,
        batch_size=256,
        num_workers=4,
    )

See Also
--------

- :doc:`../../core_concepts/preprocessing` — Preprocessing guide
- :doc:`../../tutorials/classification` — Complete workflow example

.. toctree::
   :maxdepth: 1
   :hidden:

   data_ref
