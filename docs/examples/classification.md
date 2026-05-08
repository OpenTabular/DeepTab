# Classification

This example walks through a complete binary/multi-class classification workflow using DeepTab — from generating data to evaluating a trained model.

## Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deeptab.models import MambularClassifier
```

## Generate data

We create a synthetic tabular dataset with 1 000 samples and 5 numeric features. The continuous target is bucketed into four quartile classes to form a multi-class classification problem.

```python
np.random.seed(42)

n_samples, n_features = 1000, 5
X = np.random.randn(n_samples, n_features)
y_continuous = np.dot(X, np.random.randn(n_features)) + np.random.randn(n_samples)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
df["target"] = pd.qcut(y_continuous, q=4, labels=False)
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

Instantiate `MambularClassifier` with default hyperparameters and fit on the training split. `max_epochs` is kept small here for illustration.

```python
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=10)
```

## Evaluate

```python
metrics = model.evaluate(X_test, y_test)
print(metrics)
```

```{note}
Replace `MambularClassifier` with any other classifier from `deeptab.models`
(e.g. `ResNetClassifier`, `FTTransformerClassifier`) without changing any other line.
```

## Using your own data

Replace the synthetic data block with your own DataFrame. DeepTab detects column types automatically — no manual encoding needed:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from deeptab.models import MambularClassifier

df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)
print(model.evaluate(X_test, y_test))
```

## Next steps

- [Key Concepts](../key_concepts) — learn how to tune hyperparameters via config objects.
- [Regression example](regression) — adapt this workflow to continuous targets.
- [API reference](../api/models/index) — full parameter documentation for all classifiers.
