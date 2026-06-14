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

## One model, three tasks

Every architecture comes in three variants. Change the suffix to change the task:

| Class         | Task                      | Output                   |
| ------------- | ------------------------- | ------------------------ |
| `*Classifier` | Classification            | Labels and probabilities |
| `*Regressor`  | Regression                | Continuous values        |
| `*LSS`        | Distributional regression | Distribution parameters  |

```python
from deeptab.models import MambularClassifier, MambularRegressor, MambularLSS

clf = MambularClassifier()   # labels and probabilities
reg = MambularRegressor()    # point estimates
lss = MambularLSS()          # full predictive distribution
```

The interface is identical across all three, so you can move between tasks, or swap architectures, without rewriting your pipeline.

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

DeepTab targets the data encountered in practice, not only clean benchmarks:

- Mixed numerical, categorical, and precomputed embedding features
- Automatic stratified splits for classification, preserving class proportions
- Built-in imputation of missing values during preprocessing
- Mini-batch training that scales to large datasets
- Single-device GPU acceleration by default, with other Lightning strategies (including multi-device training) available by forwarding trainer arguments to `fit()`

## When to Use DeepTab

DeepTab is well suited to:

- Tabular data with a mix of numerical and categorical features
- Datasets large enough for neural networks to be competitive, typically from a few thousand samples upward
- Problems with complex feature interactions
- Applications that require calibrated uncertainty through distributional regression
- Workflows that integrate with the scikit-learn ecosystem

Gradient-boosted trees (XGBoost, LightGBM, CatBoost) remain a strong baseline and are often preferable for:

- Small datasets, where neural networks are prone to overfitting
- Data that does not fit in memory
- Latency-critical inference, where tree ensembles are typically faster

## Next Steps

- [Installation](installation): Set up in a couple of minutes
- [Quickstart](quickstart): Train your first model in a few minutes
- [Tutorials](../tutorials/imbalance_classification): End-to-end workflows
- [FAQ](faq): Common questions
