# Regression

This example walks through a complete regression workflow using DeepTab — from generating data to evaluating a trained model.

## Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deeptab.models import MambularRegressor
```

## Generate data

We create a synthetic tabular dataset with 1 000 samples and 5 numeric features. The target is a continuous value derived from a linear combination of the features plus Gaussian noise.

```python
np.random.seed(42)

n_samples, n_features = 1000, 5
X = np.random.randn(n_samples, n_features)
y = np.dot(X, np.random.randn(n_features)) + np.random.randn(n_samples)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
df["target"] = y
```

## Split

```python
X = df.drop(columns=["target"])
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## Train

Instantiate `MambularRegressor` with default hyperparameters and fit on the training split.

```python
model = MambularRegressor()
model.fit(X_train, y_train, max_epochs=10)
```

## Evaluate

```python
metrics = model.evaluate(X_test, y_test)
print(metrics)
```

```{note}
Replace `MambularRegressor` with any other regressor from `deeptab.models`
(e.g. `ResNetRegressor`, `FTTransformerRegressor`) without changing any other line.
```

## Using your own data

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from deeptab.models import MambularRegressor

df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MambularRegressor()
model.fit(X_train, y_train, max_epochs=50)
print(model.evaluate(X_test, y_test))
```

## Next steps

- [Key Concepts](../key_concepts) — learn how to tune hyperparameters via config objects.
- [Distributional regression](distributional) — predict a full output distribution instead of a point estimate.
- [API reference](../api/models/index) — full parameter documentation for all regressors.
