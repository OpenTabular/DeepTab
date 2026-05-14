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

## All stable LSS models

Swap `MambularLSS` for any class below — pass `family=` to `.fit()` to select the output distribution:

| Class | Architecture | Notes |
|---|---|---|
| `MLPLSS` | Feedforward MLP | Fastest baseline |
| `ResNetLSS` | Residual MLP | Better than MLP for deeper networks |
| `FTTransformerLSS` | Feature-Tokenizer Transformer | Strong general-purpose model |
| `TabTransformerLSS` | Transformer on categorical embeddings | Best for categorical-heavy data |
| `SAINTLSS` | Self + intersample attention | Good for semi-supervised settings |
| `TabMLSS` | Batch-ensembling MLP | Ensemble accuracy at low cost |
| `TabRLSS` | Retrieval-augmented | Strong when local similarity matters |
| `NODELSS` | Differentiable decision trees | Gradient-boosting inductive bias |
| `NDTFLSS` | Neural decision tree forest | Use `n_ensembles` and `max_depth` |
| `TabulaRNNLSS` | RNN / LSTM / GRU | Use `model_type` to select cell |
| `MambularLSS` | Stacked Mamba SSM | Efficient sequence model |
| `MambaTabLSS` | Single Mamba block | Lightest Mamba variant |
| `MambAttentionLSS` | Mamba + attention hybrid | Local + global patterns |
| `ENODELSS` | Extended NODE | NODE with feature embeddings |
| `AutoIntLSS` | Attention-based interaction | Explicit feature crossing |

Experimental LSS models (`ModernNCALSS`, `TromptLSS`, `TangosLSS`) are available from `deeptab.models.experimental`. See [Experimental models](experimental).

## Next steps

- [Key Concepts](../key_concepts) — understand the `LSS` task variant and available distribution families.
- [Regression example](regression) — use a point-estimate regressor instead.
- [API reference](../api/models/index) — full parameter documentation.
