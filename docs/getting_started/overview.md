# Overview

DeepTab is a Python library that brings modern deep learning architectures to tabular data. Instead of writing boilerplate PyTorch code, defining data loaders, or managing training loops, you get a clean scikit-learn-style interface that handles all of this automatically.

## What is DeepTab?

The library ships with over a dozen architectures optimized for tabular data, including:

- **Sequential models** like Mambular and TabulaRNN that process features in sequence
- **Attention-based models** like FTTransformer and TabTransformer that learn feature interactions
- **Ensemble methods** like TabM that combine multiple predictions
- **Tree-based neural models** like NODE and NDTF that mimic decision tree behavior
- **Hybrid architectures** like MambAttention that combine multiple paradigms

All models support three types of tasks without changing the core workflow:

- **Classification** (binary or multiclass)
- **Regression** (predicting continuous values)
- **Distributional regression** (predicting full probability distributions)

## Design philosophy

DeepTab is built around three core principles:

### 1. Familiar interface

If you've used scikit-learn, you already know how to use DeepTab. Every model follows the same pattern:

```python
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=100)
predictions = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
```

No need to define datasets, write training loops, or manage optimizers manually. Pass a DataFrame (or NumPy array) and labels, and the library handles the rest.

### 2. Sensible defaults with full control

DeepTab takes care of the tedious parts automatically:

- **Feature type detection** — Automatically identifies numerical vs categorical columns
- **Preprocessing** — Applies appropriate encoding and scaling based on feature types
- **Missing values** — Handles missing data internally without manual imputation
- **Device management** — Uses GPU automatically if available
- **Checkpointing** — Saves best models during training with early stopping

But when you need fine-grained control, everything is configurable through three independent config objects:

- `ModelConfig` — Architecture hyperparameters
- `PreprocessingConfig` — Feature engineering strategy
- `TrainerConfig` — Training loop parameters

### 3. Production-ready from day one

DeepTab is designed for real-world tabular data with all its messiness:

- Mixed data types (numerical, categorical, text embeddings)
- Class imbalance (automatic stratified splitting)
- Variable scales (multiple preprocessing strategies)
- Missing values (built-in handling)
- Large datasets (efficient batching and data loading)

The library uses PyTorch Lightning under the hood for training, which provides:

- Automatic gradient clipping
- Learning rate scheduling
- Early stopping with patience
- Progress bars and logging
- Multi-GPU support (when needed)

But you never need to interact with Lightning directly unless you're building custom training workflows.

## What's new in v2.0

Version 2.0 introduces a fully typed data layer that makes it easier to work with tabular data at a lower level if you need custom training loops or want to integrate DeepTab components into your own PyTorch code.

### New data API components

All of these are importable from `deeptab.data`:

#### TabularDataset

A PyTorch `Dataset` for tabular data that handles:

- Feature lists (numerical, categorical, embeddings)
- Optional batch object returns via `return_batch_object=True`
- Automatic dtype conversion (numerical → float32, categorical → long)
- Support for unlabeled data (prediction mode)

```python
from deeptab.data import TabularDataset

dataset = TabularDataset(
    cat_feature_list=[cat_tensors],
    num_feature_list=[num_tensors],
    embedding_feature_list=None,
    y=labels,
    return_batch_object=False,  # Returns tuple by default
)
```

#### TabularDataModule

A Lightning `DataModule` that encapsulates:

- Preprocessing with pretab (categorical encoding, numerical scaling)
- Train/validation splitting with automatic stratification for classification
- DataLoader creation with configurable batch size and shuffling
- Schema generation via the `.schema` property

```python
from deeptab.data import TabularDataModule
from pretab.preprocessor import Preprocessor

datamodule = TabularDataModule(
    preprocessor=Preprocessor(),
    batch_size=256,
    shuffle=True,
    regression=False,  # Enables stratified splits
)
datamodule.preprocess_data(X_train, y_train, X_val, y_val)
```

#### FeatureSchema

A typed container that tracks feature metadata:

- Feature names and types (numerical, categorical, embedding)
- Preprocessing strategies applied to each feature
- Embedding dimensions and categorical cardinalities
- Total input dimensionality

```python
from deeptab.data import FeatureSchema

# Usually created automatically from preprocessor
schema = datamodule.schema

print(f"Numerical features: {schema.num_numerical_features}")
print(f"Categorical features: {schema.num_categorical_features}")
print(f"Total dimensions: {schema.total_numerical_dims + schema.total_embedding_dims}")
```

#### TabularBatch

A strongly typed batch container with:

- Named attributes: `.numerical_features`, `.categorical_features`, `.embeddings`, `.labels`
- Device movement: `.to(device)` moves all tensors
- Tuple conversion: `.from_tuple()` and `.to_tuple()` for backward compatibility

```python
from deeptab.data import TabularBatch

batch = TabularBatch(
    numerical_features=[tensor1, tensor2],
    categorical_features=[cat_tensor],
    embeddings=None,
    labels=target_tensor,
)

# Move entire batch to GPU
batch_gpu = batch.to("cuda")

# Convert to tuple format for legacy code
features, labels = batch.to_tuple()
```

### Why these changes matter

The high-level estimator API (e.g., `MambularClassifier`) remains unchanged and is still the recommended interface for most users. These new components are primarily used internally, but they're exposed in the public API for advanced use cases:

- **Custom training loops** — Use `TabularDataset` and `TabularDataModule` directly with your own PyTorch training code
- **Model integration** — Embed DeepTab's preprocessing and data loading into larger ML pipelines
- **Research and experimentation** — Access feature schemas and batch structures for analysis or visualization
- **Type safety** — Get better IDE autocomplete and type checking when working with tabular batches

The new data layer is fully tested and production-ready, with comprehensive contract tests covering all public methods and properties.

## When to use DeepTab

DeepTab is a good fit when you have:

- **Tabular data** with mixed feature types (numerical and categorical)
- **Moderate to large datasets** where deep learning can outperform linear models or gradient boosting
- **Complex feature interactions** that benefit from learned representations
- **Need for uncertainty** via distributional regression
- **Integration requirements** with existing scikit-learn pipelines

DeepTab may not be the best choice for:

- **Very small datasets** (< 1000 samples) — simpler models often work better
- **Extremely large datasets** that don't fit in memory — consider XGBoost or LightGBM with out-of-core training
- **Pure categorical data** — tree-based methods may be more efficient
- **Low-latency inference** requirements — neural networks are slower than tree ensembles

For most real-world tabular problems, DeepTab provides a strong baseline with minimal code.

## Next steps

- **[Why DeepTab](why_deeptab)** — Learn about specific advantages and use cases
- **[Installation](installation)** — Set up DeepTab in your environment
- **[Quickstart](quickstart)** — Run your first model in 5 minutes
- **[FAQ](faq)** — Common questions and troubleshooting
