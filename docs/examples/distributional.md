# Distributional Regression

Distributional regression predicts the full conditional distribution of the target rather than a single point estimate. This is useful when you need uncertainty estimates or when the target distribution is asymmetric or heavy-tailed.

## Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deeptab.models import MambularLSS
```

## Generate data

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

Pass `family` to specify the output distribution. Use `"normal"` for continuous symmetric targets. Other supported families include `"poisson"`, `"gamma"`, `"beta"`, and more.

```python
model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=10)
```

## Evaluate

```python
metrics = model.evaluate(X_test, y_test)
print(metrics)
```

```{note}
The `family` argument controls which distribution parameters the model learns.
For count data try `"poisson"`, for strictly positive targets try `"gamma"`.
See the API reference for the full list of supported families.
```

## Using your own data

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from deeptab.models import MambularLSS

df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
print(model.evaluate(X_test, y_test))
```

## Next steps

- [Key Concepts](../key_concepts) — understand the `LSS` task variant and available distribution families.
- [Regression example](regression) — use a point-estimate regressor instead.
- [API reference](../api/models/index) — full parameter documentation.
