# Using Experimental Models

Experimental models live in `deeptab.models.experimental`. Their API may change
without a deprecation cycle, but they are otherwise fully functional and follow
the same `fit` / `predict` / `evaluate` interface as stable models.

```{warning}
Experimental models are not covered by semantic versioning guarantees.
Pin your DeepTab version (`deeptab==x.y.z`) if you use them in production code
to avoid unexpected breakage after upgrades.
```

## Import path

```python
# stable models — imported directly from deeptab.models
from deeptab.models import MambularClassifier

# experimental models — always import from deeptab.models.experimental
from deeptab.models.experimental import TromptClassifier, ModernNCARegressor, TangosLSS
```

Importing an experimental class directly from `deeptab.models` (the old path)
still works but raises a `DeprecationWarning`:

```python
# raises DeprecationWarning — update the import
from deeptab.models import TromptClassifier
```

---

## End-to-end example — Trompt for classification

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

n_samples, n_features, n_classes = 800, 6, 3
X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, n_classes, size=n_samples)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
```

### Train

```python
model = TromptClassifier()
model.fit(X_train, y_train, max_epochs=10)
```

### Evaluate

```python
metrics = model.evaluate(X_test, y_test)
print(metrics)
```

### Predict

```python
preds = model.predict(X_test)
proba = model.predict_proba(X_test)
```

---

## End-to-end example — ModernNCA for regression

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deeptab.models.experimental import ModernNCARegressor

np.random.seed(0)
n_samples, n_features = 800, 5
X = np.random.randn(n_samples, n_features)
y = X @ np.random.randn(n_features) + np.random.randn(n_samples) * 0.1

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

model = ModernNCARegressor(d_model=64, n_layers=4)
model.fit(X_train, y_train, max_epochs=10)

metrics = model.evaluate(X_test, y_test)
print(metrics)
```

---

## Switching between experimental and stable

The API is identical — only the import path changes. When a model is promoted to
stable, update the import and nothing else:

```python
# Before promotion
from deeptab.models.experimental import TromptClassifier

# After promotion (no other code changes needed)
from deeptab.models import TromptClassifier
```

See [Model Promotion Policy](../developer_guide/model_promotion_policy) for the
criteria a model must meet before it moves to stable.
