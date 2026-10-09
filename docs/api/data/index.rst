.. -*- mode: rst -*-

.. currentmodule:: deeptab.data

Data
=====

The data API provides low-level control over data loading, batching, and feature inspection. **Most users don't need this.** The sklearn-compatible interface (``model.fit(X, y)``) handles data management automatically.

Use the data API when you need:

* **Custom training loops** outside the sklearn interface
* **Feature schema inspection** to understand preprocessing applied to each feature
* **Fine-grained control** over batching and data loading
* **Integration with Lightning** for advanced training workflows

Core Classes
------------

=======================================    =======================================================================================================
Class                                       Description
=======================================    =======================================================================================================
:class:`FeatureSchema`                     Inspect feature types, preprocessing, and dimensions after fitting a model
:class:`FeatureInfo`                       Metadata for individual features (type, cardinality, preprocessing method)
:class:`TabularBatch`                      Typed container for batches (numerical, categorical features, labels); new in v2.0
:class:`TabularDataModule`                 Lightning DataModule for train/val/test splits and batching (internal use)
:class:`TabularDataset`                    PyTorch Dataset for preprocessed tensors (internal use)
=======================================    =======================================================================================================

Common Use Cases
----------------

Inspecting Feature Schema
~~~~~~~~~~~~~~~~~~~~~~~~~~

After fitting a model, inspect how features were preprocessed:

.. code-block:: python

    from deeptab.models import MambularClassifier

    model = MambularClassifier()
    model.fit(X_train, y_train)

    # Access feature schema
    schema = model.feature_schema

    # Inspect numerical features
    for name, info in schema.numerical_features.items():
        print(f"{name}: {info.preprocessing}, dim={info.dimension}")

    # Inspect categorical features
    for name, info in schema.categorical_features.items():
        print(f"{name}: {len(info.categories)} categories, dim={info.dimension}")

    # Get totals
    print(f"Total numerical dim: {schema.total_numerical_dim}")
    print(f"Total categorical dim: {schema.total_categorical_dim}")

**When to use:** Debugging feature preprocessing, understanding model input dimensions, verifying feature detection.

Working with TabularBatch
~~~~~~~~~~~~~~~~~~~~~~~~~~

The new ``TabularBatch`` replaces raw tuples for cleaner code:

.. code-block:: python

    from deeptab.data import TabularBatch

    # In custom training loops
    for batch in dataloader:
        if isinstance(batch, tuple):
            # Convert legacy format
            batch = TabularBatch.from_tuple(batch)

        # Move to device
        batch = batch.to('cuda')

        # Access features
        num_feats = batch.numerical_features
        cat_feats = batch.categorical_features
        labels = batch.labels

**When to use:** Custom training loops, cleaner code for batch processing, device management.

Custom Data Loading
~~~~~~~~~~~~~~~~~~~

For advanced workflows, create data modules directly:

Constructor ``X_val`` and ``y_val`` are used when ``preprocess_data()`` omits
both arguments. An explicit method pair overrides the constructor pair.
Both values must be supplied together; an incomplete pair raises ``ValueError``.
For automatic splits, the method uses the constructor's ``val_size`` unless
an explicit method value overrides it.

.. code-block:: python

    from deeptab.data import TabularDataModule

    datamodule = TabularDataModule(
        preprocessor=model.preprocessor,
        batch_size=512,
        shuffle=True,
        regression=False,
        X_val=X_val,
        y_val=y_val,
    )

    datamodule.preprocess_data(X_train, y_train)
    datamodule.setup("fit")

    # Access dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

**When to use:** Custom training loops, hyperparameter tuning with fixed preprocessing, integration with PyTorch Lightning.

.. note::

   Classification label shape uses the full training-target class count before
   splitting. A rare class falling entirely into validation with ``stratify=False``
   does not turn a multiclass task into a binary one. This prevents a loss-shape
   mismatch; it does not teach the model a class absent from its training rows.

Tuple datasets without embeddings work with PyTorch's standard ``DataLoader``:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader
    from deeptab.data import TabularDataset

    dataset = TabularDataset(
        [], [torch.arange(6, dtype=torch.float32).reshape(6, 1)],
        labels=torch.arange(6),
    )
    features, labels = next(iter(DataLoader(dataset, batch_size=2)))
    assert features[2] == []

Absent embeddings are represented by an empty list in tuple samples, including
indexed samples. ``TabularBatch`` object samples retain ``None`` for absent
embeddings and require a suitable custom collation strategy.

Key Design Principles
---------------------

**Automatic vs. Manual:**
    The sklearn interface (``fit(X, y)``) creates data modules automatically. Only use the data API directly for custom workflows.

**Internal Representation:**
    Features are stored as lists of tensors (one per feature), not single concatenated tensors. This supports heterogeneous preprocessing per feature.

**Typed Containers:**
    ``TabularBatch`` and ``FeatureSchema`` provide type hints and IDE autocompletion, replacing raw tuples and dictionaries.

See Also
--------

- :doc:`../../core_concepts/training_and_evaluation`: How preprocessing works under the hood
- :doc:`../../core_concepts/sklearn_api`: Standard sklearn interface (recommended for most users)
- :doc:`../../tutorials/imbalance_classification`: End-to-end workflow example

.. toctree::
   :maxdepth: 1
   :hidden:

   data_ref
