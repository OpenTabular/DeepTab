# Overview

DeepTab brings modern deep learning to tabular data with a clean scikit-learn interface. No boilerplate PyTorch code, no manual data loaders—just `fit`, `predict`, and `evaluate`.

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
from deeptab.configs import ModelConfig, PreprocessingConfig, TrainerConfig

model = ResNetClassifier(
    model_config=ModelConfig(d_model=128, n_layers=8),
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
```

For advanced use cases (custom training loops, model integration), v2.0 exposes low-level components:

- **TabularDataset** — PyTorch Dataset with batch object support
- **TabularDataModule** — Lightning DataModule with preprocessing
- **FeatureSchema** — Typed feature metadata container
- **TabularBatch** — Strongly typed batch with device management

See [API docs](../api/data/index) for details. Most users can ignore these—the high-level estimator API (e.g., `MambularClassifier`) is unchanged.

## Next Steps

- [Why DeepTab](why_deeptab) — Key advantages and use cases
- [Installation](installation) — Set up in 2 minutes
- [Quickstart](quickstart) — First model in 5 minutes
- [FAQ](faq) — Common questions
