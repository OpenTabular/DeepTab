# Why DeepTab

DeepTab is your **one-stop shop for tabular deep learning**. Every model supports classification, regression, and distributional regression—15 stable architectures, all in one place, with consistent APIs.

## One Library, All Tasks

```{important}
**DeepTab is unique:** It provides implementations of most major tabular deep learning models (Transformers, State Space Models, Tree-inspired, and more) with **all three task types** in a single library. New models are continuously evaluated and promoted from experimental to stable.
```

**All 15 stable models support all 3 tasks:**

| Model          | Classification | Regression | LSS (Distributional) | Type          |
| -------------- | -------------- | ---------- | -------------------- | ------------- |
| Mambular       | ✅             | ✅         | ✅                   | State Space   |
| MambaTab       | ✅             | ✅         | ✅                   | State Space   |
| MambAttention  | ✅             | ✅         | ✅                   | Hybrid        |
| FTTransformer  | ✅             | ✅         | ✅                   | Transformer   |
| TabTransformer | ✅             | ✅         | ✅                   | Transformer   |
| SAINT          | ✅             | ✅         | ✅                   | Transformer   |
| ResNet         | ✅             | ✅         | ✅                   | Residual      |
| TabR           | ✅             | ✅         | ✅                   | Residual      |
| NODE           | ✅             | ✅         | ✅                   | Tree-inspired |
| ENODE          | ✅             | ✅         | ✅                   | Tree-inspired |
| NDTF           | ✅             | ✅         | ✅                   | Tree-inspired |
| TabM           | ✅             | ✅         | ✅                   | Sequential    |
| TabulaRNN      | ✅             | ✅         | ✅                   | Sequential    |
| AutoInt        | ✅             | ✅         | ✅                   | Attention     |
| MLP            | ✅             | ✅         | ✅                   | Baseline      |

**Plus 3 experimental models** (ModernNCA, Trompt, Tangos) undergoing evaluation for promotion.

## Familiar API

```{tip}
If you know `fit()` and `predict()`, you're ready to use DeepTab.
```

```python
from deeptab.models import SAINTClassifier

model = SAINTClassifier()
model.fit(X_train, y_train, max_epochs=100)
predictions = model.predict(X_test)
```

**Works with existing tools:**

```python
from sklearn.model_selection import GridSearchCV
from deeptab.models import TabRRegressor

# Drop-in replacement for any sklearn estimator
search = GridSearchCV(
    TabRRegressor(),
    param_grid={"model_config__d_model": [64, 128, 256]},
    cv=5
)
search.fit(X, y)
```

## One Model, Three Tasks

Every architecture comes in three variants—just change the suffix:

| Class         | Task                      | Output                  |
| ------------- | ------------------------- | ----------------------- |
| `*Classifier` | Classification            | Labels & probabilities  |
| `*Regressor`  | Regression                | Continuous values       |
| `*LSS`        | Distributional regression | Distribution parameters |

```python
from deeptab.models import MambularClassifier, MambularRegressor, MambularLSS

# Same architecture, different tasks
clf = MambularClassifier()
reg = MambularRegressor()
lss = MambularLSS()

# Identical API for all three
clf.fit(X_train, y_train, max_epochs=50)
```

```{note}
**All 15+ architectures support all three tasks.** Try different models by changing one import.
```

## Automatic Preprocessing

```{important}
DeepTab handles preprocessing automatically:

- Feature type detection (numerical/categorical)
- Encoding and scaling
- Missing value handling
- Batching and device placement
```

**Example with mixed types:**

```python
import pandas as pd

df = pd.DataFrame({
    "age": [25, 32, 47],           # Numerical → scaled
    "city": ["NYC", "LA", "CHI"],  # Categorical → encoded + embedded
    "income": [35000, 48000, 72000],
})

model = TabTransformerClassifier()
model.fit(df, y, max_epochs=50)  # Just works
```

**Override when needed:**

```python
from deeptab.configs import PreprocessingConfig

model = NODEClassifier(
    preprocessing_config=PreprocessingConfig(
        numerical_preprocessing="quantile",  # For outliers
        n_bins=50
    )
)
```

## Uncertainty Quantification

LSS models predict full distributions, not just point estimates:

```python
from deeptab.models import SAINTLSS

model = SAINTLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)

# Returns (mean, std) for each sample
params = model.predict(X_test)

# 95% confidence intervals
lower = params[:, 0] - 1.96 * params[:, 1]
upper = params[:, 0] + 1.96 * params[:, 1]
```

```{tip}
**Use distributional regression when:**

- You need prediction intervals
- Uncertainty varies across the input space
- Risk-aware decisions are important
- You're modeling count data or bounded outcomes
```

**Supported families:** `normal`, `poisson`, `gamma`, `beta`, `negative_binomial`, `student_t`, and more.

## Fast Experimentation

**Quick baseline:**

```python
from deeptab.models import AutoIntClassifier

model = AutoIntClassifier()
model.fit(X_train, y_train, max_epochs=50)
print(model.evaluate(X_test, y_test))
```

**Compare architectures:**

```python
from deeptab.models import *

models = [
    FTTransformerClassifier(),
    ResNetClassifier(),
    NODEClassifier(),
    MambularClassifier(),
]

for model in models:
    model.fit(X_train, y_train, max_epochs=50)
    acc = model.evaluate(X_test, y_test)["accuracy"]
    print(f"{model.__class__.__name__}: {acc:.3f}")
```

## Production Ready

✅ **Mixed data:** Numerical, categorical, pre-computed embeddings  
✅ **Class imbalance:** Automatic stratified splits (v2.0+)  
✅ **Large datasets:** Efficient batching with multi-worker data loading  
✅ **GPU support:** Automatic detection and usage  
✅ **Early stopping:** Best model checkpointing with patience

```python
from deeptab.configs import TrainerConfig

# Large dataset configuration
model = TabulaRNNRegressor(
    trainer_config=TrainerConfig(
        batch_size=512,
        num_workers=4,      # Parallel data loading
        max_epochs=100,
        patience=10,        # Early stopping
    )
)
```

## When to Choose DeepTab

```{tip}
**Great fit:**

- Tabular data with mixed types (numerical + categorical)
- 1000+ samples where deep learning shines
- Complex feature interactions
- Need uncertainty quantification
- scikit-learn integration required
```

```{warning}
**Consider alternatives:**

- <1000 samples → simpler models
- Out-of-core datasets → XGBoost/LightGBM
- Pure categorical → tree methods
- Strict latency needs → trees are faster
```

## Next Steps

- [Installation](installation) — Get started in 2 minutes
- [Quickstart](quickstart) — First model in 5 minutes
- [Tutorials](../tutorials/imbalance_classification) — End-to-end workflows
