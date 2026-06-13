# Overview

DeepTab brings modern deep learning to tabular data with a clean scikit-learn interface. No boilerplate PyTorch code, no manual data loaders, just `fit`, `predict`, and `evaluate`.

## What is DeepTab?

DeepTab provides 15 stable neural architectures for tabular data:

| Family                 | Models                               | Notes                                                    |
| ---------------------- | ------------------------------------ | -------------------------------------------------------- |
| **State Space Models** | Mambular, MambaTab, MambAttention    | Flagship models; linear feature-sequence scaling         |
| **Transformers**       | FTTransformer, TabTransformer, SAINT | Full feature or row attention                            |
| **Tree-inspired**      | NODE, ENODE, NDTF                    | Differentiable soft-tree structures                      |
| **Residual networks**  | ResNet, TabR                         | Skip-connection MLP and retrieval-augmented              |
| **Sequential**         | TabulaRNN, TabM                      | RNN feature processing and parameter-efficient ensembles |
| **Attention-based**    | AutoInt                              | Automatic feature interaction learning                   |
| **Baseline**           | MLP                                  | Fast dense baseline                                      |

**Plus 3 experimental models:** ModernNCA, Trompt, Tangos

```{important}
**All models support three tasks:**

- Classification (binary/multiclass)
- Regression (continuous)
- Distributional regression (uncertainty quantification)
```

**Example:**

```python
from deeptab.models import FTTransformerClassifier

model = FTTransformerClassifier()
model.fit(X_train, y_train, max_epochs=100)
predictions = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
```

## Design Philosophy

### Familiar Interface

If you know scikit-learn, you know DeepTab. Standard `fit`/`predict` API with seamless integration:

```python
from sklearn.model_selection import GridSearchCV
from deeptab.models import FTTransformerClassifier

search = GridSearchCV(FTTransformerClassifier(), param_grid, cv=5)
search.fit(X, y)
```

### Smart Defaults, Full Control

```{note}
**Automatic preprocessing:**

- Feature type detection (numerical/categorical)
- Missing value handling
- Scaling and encoding
- GPU utilization
- Early stopping with checkpointing
```

**Configure when needed:**

```python
from deeptab.configs import ResNetConfig, PreprocessingConfig, TrainerConfig

model = ResNetClassifier(
    model_config=ResNetConfig(d_model=128),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(lr=1e-3, batch_size=256)
)
```

### Production-Ready

Built for real-world messiness:

- ✅ Mixed data types (numerical, categorical, embeddings)
- ✅ Class imbalance (automatic stratified splits)
- ✅ Missing values (built-in handling)
- ✅ Large datasets (efficient batching)
- ✅ Multi-GPU support via Lightning

## When to Use DeepTab

```{tip}
**Good fit when you have:**

- Tabular data with mixed feature types
- 1000+ samples where deep learning excels
- Complex feature interactions
- Need for uncertainty (distributional regression)
- Integration with scikit-learn pipelines
```

```{warning}
**Consider alternatives for:**

- Very small datasets (<1000 samples) → try simpler models
- Out-of-core datasets → consider XGBoost/LightGBM
- Pure categorical data → tree methods may be faster
- Strict latency requirements → trees are faster at inference
```

## What's New in v2.0

```{important}
**Key improvements:**

- Fully typed data layer with `TabularBatch`, `TabularDataset`, `FeatureSchema`
- Automatic stratified splits for classification
- Enhanced preprocessing with `pretab` integration
- Better type safety and IDE support
- Structured exception hierarchy with descriptive error messages
- Registry-backed optimizer and LR-scheduler system (see below)
```

### Training system upgrades

`TrainerConfig` now exposes the full optimizer and scheduler surface without
requiring you to subclass `TaskModel`:

```python
from deeptab.configs import TrainerConfig

# Switch to AdamW with custom beta values
tc = TrainerConfig(
    optimizer_type="AdamW",
    optimizer_kwargs={"betas": (0.9, 0.95)},
    weight_decay=1e-2,
)

# Cosine annealing instead of ReduceLROnPlateau
tc = TrainerConfig(
    scheduler_type="CosineAnnealingLR",
    scheduler_kwargs={"T_max": 100},
)

# Align early stopping AND scheduler to the same metric/direction
tc = TrainerConfig(
    monitor="val_auroc",
    mode="max",          # Both early stopping and ReduceLROnPlateau now maximise
)

# Disable the scheduler entirely
tc = TrainerConfig(scheduler_type=None)

# Exempt bias and LayerNorm weights from weight decay (recommended for transformers)
tc = TrainerConfig(
    optimizer_type="AdamW",
    weight_decay=1e-2,
    no_weight_decay_for_bias_and_norm=True,
)
```

You can also inspect and extend the registries directly:

```python
from deeptab.training.optimizers import available_optimizers, register_optimizer
from deeptab.training.schedulers import available_schedulers, register_scheduler

print(available_optimizers())   # ['adadelta', 'adagrad', 'adam', 'adamw', ...]
print(available_schedulers())   # ['constantlr', 'cosineannealinglr', ...]

# Plug in a third-party optimizer
register_optimizer("muon", MyMuonOptimizer)
```

For advanced use cases (custom training loops, model integration), v2.0 exposes low-level components:

- **TabularDataset**: PyTorch Dataset with batch object support
- **TabularDataModule**: Lightning DataModule with preprocessing
- **FeatureSchema**: Typed feature metadata container
- **TabularBatch**: Strongly typed batch with device management

See [API docs](../api/data/index) for details. Most users can ignore these, since the high-level estimator API (e.g., `MambularClassifier`) is unchanged.

## Next Steps

- [Why DeepTab](why_deeptab): Key advantages and use cases
- [Installation](installation): Set up in 2 minutes
- [Quickstart](quickstart): First model in 5 minutes
- [FAQ](faq): Common questions
