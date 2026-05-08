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

## Configuring hyperparameters

Every model has a corresponding config class in `deeptab.configs` that documents all available hyperparameters. You can either pass hyperparameters directly to the constructor or via a config object:

```python
from deeptab.configs import MambularConfig
from deeptab.models import MambularClassifier

# Option A: keyword arguments
model = MambularClassifier(d_model=64, n_layers=4, dropout=0.1)

# Option B: config object — same result, easier to version and share
config = MambularConfig(d_model=64, n_layers=4, dropout=0.1)
model = MambularClassifier(config=config)
```

## Fit arguments

Training arguments such as learning rate, batch size, and epochs are passed to `fit`, not the constructor. This keeps architecture hyperparameters separate from training hyperparameters:

```python
model.fit(
    X_train,
    y_train,
    max_epochs=100,
    lr=1e-3,
    batch_size=256,
)
```

## Distributional regression (LSS)

`LSS` models predict the parameters of a parametric distribution rather than a single value. Specify the output family via the `family` argument of `fit`:

```python
model = MambularLSS()
model.fit(X_train, y_train, family="normal")  # learns μ and σ per sample
```

Common families: `"normal"`, `"poisson"`, `"gamma"`, `"beta"`. See the API reference for the full list.

## Data preprocessing

DeepTab detects column types automatically from the DataFrame and applies appropriate preprocessing:

- **Numerical columns** — standardised by default.
- **Categorical columns** — ordinally encoded and embedded.
- **Missing values** — handled internally; no need to impute before passing data.

You can override the preprocessing strategy via config parameters if needed.
