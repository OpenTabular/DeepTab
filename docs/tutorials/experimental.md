# Using Experimental Models

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <a href="https://colab.research.google.com/github/basf/DeepTab/blob/main/docs/tutorials/notebooks/experimental.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/basf/DeepTab/blob/main/docs/tutorials/notebooks/experimental.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on-GitHub-181717?logo=github&logoColor=white" alt="View on GitHub"/>
  </a>
</div>

Experimental models live in `deeptab.models.experimental`. They implement cutting-edge architectures that are still being refined. While fully functional, their APIs may change without a deprecation cycle.

```{tip}
Click the badges above to run this tutorial in Google Colab or view the notebook on GitHub!
```

## What are experimental models?

Experimental models are:

- **Fully functional** — Same `fit` / `predict` / `evaluate` interface as stable models
- **Cutting-edge** — Latest architectures from recent research papers
- **Under evaluation** — Being tested for promotion to stable tier
- **Not semantically versioned** — May change in minor releases

```{warning}
Experimental models are not covered by semantic versioning. Pin your DeepTab version (`deeptab==x.y.z`) if you use them in production to avoid breaking changes.
```

## Import path

### Stable models

```python
from deeptab.models import MambularClassifier, ResNetRegressor, FTTransformerLSS
```

### Experimental models

```python
from deeptab.models.experimental import (
    TromptClassifier,
    ModernNCARegressor,
    TangosLSS,
)
```

### Deprecated import (still works but warns)

```python
# Raises DeprecationWarning — update your imports
from deeptab.models import TromptClassifier
```

## Why use experimental models?

1. **Access latest research** — Try state-of-the-art architectures before they're stable
2. **Early feedback** — Help improve models by reporting issues
3. **Performance gains** — May outperform stable models for your use case
4. **Exploration** — Experiment with different approaches

## Version pinning

Always pin DeepTab version when using experimental models:

```bash
# In requirements.txt or pyproject.toml
deeptab==2.0.0  # Pin exact version
```

Why?

- Experimental APIs may change in minor releases (e.g., 2.0.0 → 2.1.0)
- Stable models follow semantic versioning and won't break
- Pinning prevents unexpected failures after upgrades

## Classification tutorial

### Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deeptab.models.experimental import TromptClassifier
```

### Generate data

```python
np.random.seed(42)

n_samples, n_features, n_classes = 1000, 6, 3
X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, n_classes, size=n_samples)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)
```

### Train

```python
model = TromptClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

### Evaluate

```python
metrics = model.evaluate(X_test, y_test)
print(metrics)
# {'accuracy': 0.87, 'loss': 0.38}
```

### Predict

```python
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(predictions[:5])
print(probabilities[:3])
```

### Save and load

```python
model.save("trompt_classifier.pkl")

from deeptab.models.experimental import TromptClassifier
loaded_model = TromptClassifier.load("trompt_classifier.pkl")
```

## Regression tutorial

### Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deeptab.models.experimental import ModernNCARegressor
```

### Generate data

```python
np.random.seed(42)

n_samples, n_features = 1000, 5
X = np.random.randn(n_samples, n_features)
y = X @ np.random.randn(n_features) + np.random.randn(n_samples) * 0.1

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42
)
```

### Train

```python
model = ModernNCARegressor()
model.fit(X_train, y_train, max_epochs=50)
```

### Evaluate

```python
metrics = model.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"MAE: {metrics['mae']:.3f}")

r2 = model.score(X_test, y_test)
print(f"R²: {r2:.3f}")
```

### Predict

```python
predictions = model.predict(X_test)
print(predictions[:10])
```

## LSS (distributional) tutorial

### Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deeptab.models.experimental import TangosLSS
```

### Generate data

```python
np.random.seed(42)

n_samples, n_features = 1000, 5
X = np.random.randn(n_samples, n_features)
y = np.dot(X, np.random.randn(n_features)) + np.random.randn(n_samples)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=[]), y, test_size=0.2, random_state=42
)
```

### Train

```python
model = TangosLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
```

### Predict distribution parameters

```python
params = model.predict(X_test)
print(params.shape)
# (200, 2) for normal distribution

mean = params[:, 0]
log_std = params[:, 1]
std = np.exp(log_std)
```

### Generate prediction intervals

```python
from scipy import stats

# 90% prediction interval
lower = stats.norm.ppf(0.05, loc=mean, scale=std)
upper = stats.norm.ppf(0.95, loc=mean, scale=std)

coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"90% interval coverage: {coverage:.3f}")
```

## Customization with configs

Experimental models support the same config system as stable models:

```python
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.models.experimental import TromptClassifier

model_cfg = MambularConfig(
    d_model=256,
    n_layers=8,
    dropout=0.3,
)

prep_cfg = PreprocessingConfig(
    numerical_preprocessing="quantile",
    use_ple=True,
)

trainer_cfg = TrainerConfig(
    lr=1e-3,
    batch_size=256,
    patience=15,
)

model = TromptClassifier(
    model_config=model_cfg,
    preprocessing_config=prep_cfg,
    trainer_config=trainer_cfg,
)

model.fit(X_train, y_train, max_epochs=100)
```

## Integration with scikit-learn

Experimental models are fully compatible with scikit-learn tools:

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from deeptab.models.experimental import TromptClassifier

param_grid = {
    "model_config__d_model": [128, 256],
    "model_config__n_layers": [4, 6, 8],
    "trainer_config__lr": [5e-4, 1e-3],
}

model = TromptClassifier()

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=1,
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
```

### Cross-validation

```python
from sklearn.model_selection import cross_val_score
from deeptab.models.experimental import ModernNCARegressor

model = ModernNCARegressor()

scores = cross_val_score(
    model, X_train, y_train,
    cv=5,
    scoring="neg_mean_squared_error",
)

rmse_scores = np.sqrt(-scores)
print(f"CV RMSE: {rmse_scores.mean():.3f} (+/- {rmse_scores.std():.3f})")
```

## Available experimental models

### Classification

```python
from deeptab.models.experimental import (
    TromptClassifier,
    ModernNCAClassifier,
    TangosClassifier,
)
```

### Regression

```python
from deeptab.models.experimental import (
    TromptRegressor,
    ModernNCARegressor,
    TangosRegressor,
)
```

### LSS (Distributional)

```python
from deeptab.models.experimental import (
    TromptLSS,
    ModernNCALSS,
    TangosLSS,
)
```

## Switching to stable imports

When a model is promoted to stable (announced in release notes), update imports:

### Before promotion

```python
from deeptab.models.experimental import TromptClassifier

model = TromptClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

### After promotion

```python
# Only the import changes — everything else stays the same
from deeptab.models import TromptClassifier

model = TromptClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

No other code changes needed!

## Model promotion criteria

Experimental models graduate to stable when they meet these criteria:

1. **Performance** — Competitive with existing stable models
2. **Stability** — No known bugs or crashes
3. **Testing** — Comprehensive unit and integration tests
4. **Documentation** — Full API documentation and examples
5. **Community feedback** — Positive user experience
6. **Production use** — Successfully used in real-world projects

See [Model Promotion Policy](../developer_guide/model_promotion_policy) for details.

## Comparing experimental and stable

```python
from deeptab.models import MambularClassifier  # Stable
from deeptab.models.experimental import TromptClassifier  # Experimental

# Same API — different import paths
for ModelClass in [MambularClassifier, TromptClassifier]:
    model = ModelClass()
    model.fit(X_train, y_train, max_epochs=50)
    accuracy = model.score(X_test, y_test)
    print(f"{ModelClass.__name__}: {accuracy:.3f}")
```

## Best practices

1. **Pin versions** — Always use `deeptab==x.y.z` with experimental models
2. **Monitor releases** — Check release notes for API changes
3. **Test thoroughly** — Validate experimental models on your data
4. **Report issues** — File GitHub issues if you encounter problems
5. **Stay updated** — Update imports when models are promoted to stable
6. **Use stable for production** — Prefer stable models for critical applications

## Checking model tier at runtime

```python
from deeptab.models import MambularClassifier
from deeptab.models.experimental import TromptClassifier

# Check if a model is experimental
print(hasattr(TromptClassifier, "_experimental"))  # True for experimental

# Stable models don't have this attribute
print(hasattr(MambularClassifier, "_experimental"))  # False for stable
```

## Using your own data

### Classification

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from deeptab.models.experimental import TromptClassifier

df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = TromptClassifier()
model.fit(X_train, y_train, max_epochs=100)

metrics = model.evaluate(X_test, y_test)
print(metrics)
```

### Regression

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from deeptab.models.experimental import ModernNCARegressor

df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = ModernNCARegressor()
model.fit(X_train, y_train, max_epochs=100)

metrics = model.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.3f}")
```

### LSS

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from deeptab.models.experimental import TangosLSS

df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = TangosLSS()
model.fit(X_train, y_train, family="normal", max_epochs=100)

# Get distribution parameters and intervals
params = model.predict(X_test)
# Generate prediction intervals as shown in distributional tutorial
```

## Next steps

- **Understand model tiers** → Read [Model Tiers](../core_concepts/model_tiers) for tier definitions
- **See promotion policy** → Check [Model Promotion Policy](../developer_guide/model_promotion_policy)
- **Try stable models** → Use [Classification](classification), [Regression](regression), or [Distributional](distributional) tutorials
- **Report feedback** → Open GitHub issues for bugs or feature requests
