# Key Concepts

This page explains the mental model behind DeepTab before you write any code.

## scikit-learn-compatible API

Every DeepTab model implements the scikit-learn `BaseEstimator` interface. If you have used scikit-learn before, the workflow is identical:

```python
model = MambularClassifier()      # 1. instantiate
model.fit(X_train, y_train)       # 2. fit
predictions = model.predict(X_test)  # 3. predict
metrics = model.evaluate(X_test, y_test)  # 4. evaluate
```

`X` can be a pandas `DataFrame` or a NumPy array. DeepTab handles the conversion internally.

## Task variants

Each model ships in three variants selected by the class suffix:

| Suffix       | Task                      | Output                         |
| ------------ | ------------------------- | ------------------------------ |
| `Classifier` | Classification            | Class labels and probabilities |
| `Regressor`  | Regression                | Continuous point estimates     |
| `LSS`        | Distributional regression | Full distribution parameters   |

Switching tasks requires only changing the import — the rest of the code is identical:

```python
from deeptab.models import MambularClassifier   # classification
from deeptab.models import MambularRegressor    # regression
from deeptab.models import MambularLSS          # distributional regression
```

## Stable vs experimental models

DeepTab ships models at two tiers:

| Tier             | Import path                                   | Guarantee                                   |
| ---------------- | --------------------------------------------- | ------------------------------------------- |
| **Stable**       | `from deeptab.models import ...`              | Public API frozen under semantic versioning |
| **Experimental** | `from deeptab.models.experimental import ...` | May change without a deprecation cycle      |

Always use the explicit experimental import path to signal that you accept the instability:

```python
# stable
from deeptab.models import FTTransformerClassifier

# experimental — explicit path required
from deeptab.models.experimental import TromptClassifier
```

See [Using experimental models](examples/experimental) for a full worked example.

## Split-config API

DeepTab separates hyperparameters into three independent config dataclasses, each
passed explicitly to the model constructor:

| Config                             | Controls                                                        |
| ---------------------------------- | --------------------------------------------------------------- |
| `<Model>Config` (e.g. `MLPConfig`) | Neural architecture — `d_model`, `dropout`, `n_layers`, …       |
| `PreprocessingConfig`              | Feature engineering — `numerical_preprocessing`, `n_bins`, …    |
| `TrainerConfig`                    | Training loop — `lr`, `max_epochs`, `batch_size`, `patience`, … |

```python
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MambularClassifier

model = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=6, dropout=0.1),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(max_epochs=100, lr=1e-3, batch_size=256),
)
model.fit(X_train, y_train)
```

Omitting any config applies all defaults.

### Scikit-learn `get_params` / `set_params`

All three config classes implement the scikit-learn parameter protocol, so you can
inspect or update them at any time:

```python
cfg = MambularConfig(d_model=64)
print(cfg.get_params())         # {'d_model': 64, 'dropout': 0.2, ...}
cfg.set_params(d_model=128)     # update in-place, returns self
```

The estimator itself also delegates to the configs via double-underscore notation,
which makes grid search straightforward:

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    MambularClassifier(
        model_config=MambularConfig(),
        trainer_config=TrainerConfig(max_epochs=20),
    ),
    param_grid={
        "model_config__d_model": [64, 128],
        "trainer_config__lr": [1e-3, 5e-4],
    },
    cv=3,
)
search.fit(X_train, y_train)
```

## Distributional regression (LSS)

`LSS` models predict the parameters of a parametric distribution rather than a single value. Specify the output family via the `family` argument of `fit`:

```python
from deeptab.configs import MambularConfig, TrainerConfig
from deeptab.models import MambularLSS

model = MambularLSS(
    model_config=MambularConfig(d_model=64),
    trainer_config=TrainerConfig(max_epochs=100),
)
model.fit(X_train, y_train, family="normal")  # learns μ and σ per sample
```

Common families: `"normal"`, `"poisson"`, `"gamma"`, `"beta"`. See the API reference for the full list.

## Data preprocessing

DeepTab detects column types automatically from the DataFrame and applies appropriate preprocessing:

- **Numerical columns** — standardised by default.
- **Categorical columns** — ordinally encoded and embedded.
- **Missing values** — handled internally; no need to impute before passing data.

Override the default strategy via `PreprocessingConfig`:

```python
from deeptab.configs import PreprocessingConfig

cfg = PreprocessingConfig(
    numerical_preprocessing="ple",   # piecewise-linear encoding
    n_bins=32,
    scaling_strategy="standard",
)
```
