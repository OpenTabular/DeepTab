# Overview

DeepTab is a library for deep learning on tabular data with a scikit-learn compatible interface. It handles feature preprocessing, batching, and the training loop, so the workflow stays `fit`, `predict`, and `evaluate` instead of hand-written PyTorch code and data loaders. Every architecture supports three tasks: classification, regression, and distributional regression for uncertainty quantification.

## What is DeepTab?

DeepTab provides 15 stable neural architectures for tabular data:

| Family                 | Models                                        | Notes                                                      |
| ---------------------- | --------------------------------------------- | ---------------------------------------------------------- |
| **State Space Models** | Mambular, MambaTab, MambAttention             | Mamba-inspired; linear feature-sequence scaling            |
| **Transformers**       | FTTransformer, TabTransformer, SAINT, AutoInt | Feature, row, and self-attention over feature interactions |
| **Residual networks**  | ResNet, TabR                                  | Skip-connection MLP and retrieval-augmented                |
| **Tree-inspired**      | NODE, ENODE, NDTF                             | Differentiable soft-tree structures                        |
| **General baselines**  | MLP, TabM, TabulaRNN                          | Dense, parameter-efficient ensemble, and recurrent         |

**Plus 3 experimental models:** ModernNCA, Tangos, Trompt

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

### Defaults and Configuration

With no configuration, DeepTab detects feature types, encodes and scales them, and imputes missing values during preprocessing. Training runs on a GPU when one is available, with early stopping and checkpointing enabled.

Each layer is configurable when you need it. Architecture, preprocessing, and training settings live in separate config objects, so you can change one without touching the others:

```python
from deeptab.configs import ResNetConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import ResNetClassifier

model = ResNetClassifier(
    model_config=ResNetConfig(d_model=128),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(lr=1e-3, batch_size=256),
)
```

### Built for real datasets

Beyond the defaults above, DeepTab handles details that come up with real data:

- Mixed numerical, categorical, and precomputed embedding features in a single model
- Automatic stratified splits for classification, preserving class proportions
- Mini-batch training that scales to datasets larger than a single batch
- Multi-device and other Lightning training strategies, enabled by forwarding trainer arguments to `fit()`

## When to Use DeepTab

DeepTab and gradient-boosted trees (XGBoost, LightGBM, CatBoost) are complementary tools. The table below shows where each tends to be the stronger starting point.

| Situation                                                                  | Stronger starting point           |
| -------------------------------------------------------------------------- | --------------------------------- |
| Mixed numerical and categorical features                                   | DeepTab or gradient-boosted trees |
| A few thousand samples or more                                             | DeepTab                           |
| Complex feature interactions                                               | DeepTab                           |
| Calibrated uncertainty through distributional regression                   | DeepTab (`*LSS` variants)         |
| Classification, regression, and distributional models behind one interface | DeepTab                           |
| Integration with scikit-learn pipelines and `GridSearchCV`                 | DeepTab or gradient-boosted trees |
| Small datasets where overfitting is a risk                                 | Gradient-boosted trees            |
| Data that does not fit in memory                                           | Gradient-boosted trees            |
| Latency-critical inference                                                 | Gradient-boosted trees            |

```{note}
These are general guidelines, not strict rules. Results depend on the dataset, so when performance matters it is worth benchmarking DeepTab against a gradient-boosted baseline.
```

## Next Steps

- [Installation](installation): Set up in a couple of minutes
- [Quickstart](quickstart): Train your first model in a few minutes
- [Tutorials](../tutorials/imbalance_classification): End-to-end workflows
- [FAQ](faq): Common questions
