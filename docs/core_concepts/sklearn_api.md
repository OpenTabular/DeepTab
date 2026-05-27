# scikit-learn Compatible API

DeepTab estimators follow the scikit-learn pattern while training PyTorch models under the hood. You instantiate an estimator, call `fit`, then use `predict`, `evaluate`, `score`, `save`, and `load`.

## Basic Workflow

```python
from deeptab.configs import MambularConfig, TrainerConfig
from deeptab.models import MambularClassifier

model = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=4),
    trainer_config=TrainerConfig(max_epochs=50, patience=10),
    random_state=101,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
```

## Estimator Families

Most architectures expose three task variants:

| Suffix | Task | Example |
| --- | --- | --- |
| `Classifier` | Binary or multiclass classification | `MambularClassifier` |
| `Regressor` | Point-estimate regression | `MambularRegressor` |
| `LSS` | Distributional regression | `MambularLSS` |

Stable models are imported from `deeptab.models`. Experimental models are imported from `deeptab.models.experimental`.

## Accepted Inputs

Use pandas DataFrames when possible:

```python
import pandas as pd

X = pd.DataFrame({
    "age": [25, 32, 47],
    "city": pd.Series(["NYC", "Boston", "Chicago"], dtype="category"),
    "income": [50000.0, 75000.0, 90000.0],
})
```

NumPy arrays are accepted, but they lose column names and dtype semantics:

```python
import numpy as np

X = np.random.randn(1000, 10)
```

For mixed numerical/categorical data, DataFrames are strongly preferred.

## Constructor Pattern

```python
from deeptab.configs import MLPConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MLPRegressor

model = MLPRegressor(
    model_config=MLPConfig(layer_sizes=[256, 128, 32], dropout=0.2),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="standard"),
    trainer_config=TrainerConfig(lr=1e-3, batch_size=256, max_epochs=100),
    random_state=101,
)
```

The split-config API is the recommended style for new code.

## Fit

```python
model.fit(
    X_train,
    y_train,
    X_val=X_val,
    y_val=y_val,
)
```

Useful fit arguments:

| Argument | Use |
| --- | --- |
| `X`, `y` | Training features and targets. |
| `X_val`, `y_val` | Explicit validation set. If omitted, DeepTab creates one. |
| `embeddings`, `embeddings_val` | Optional external embeddings for train/validation data. |
| `max_epochs`, `batch_size`, `lr`, `patience` | Legacy fit-time overrides; prefer `TrainerConfig` for reusable experiments. |
| `train_metrics`, `val_metrics` | Optional Lightning metrics logged during training. |
| `**trainer_kwargs` | Additional Lightning trainer keyword arguments. |

For LSS models, `family` is required:

```python
from deeptab.models import MambularLSS

model = MambularLSS()
model.fit(X_train, y_train, family="normal")
```

## Predict

```python
labels = classifier.predict(X_test)
values = regressor.predict(X_test)
params = lss_model.predict(X_test)
```

For classifiers:

```python
probabilities = classifier.predict_proba(X_test)
```

For external embeddings at inference:

```python
predictions = model.predict(X_test, embeddings=test_embeddings)
```

## Evaluate

Default metric names are implementation-defined:

```python
classifier.evaluate(X_test, y_test)
# {"Accuracy": ...}

regressor.evaluate(X_test, y_test)
# {"Mean Squared Error": ...}
```

Use explicit metrics in tutorials and papers:

```python
from sklearn.metrics import accuracy_score, log_loss

classifier.evaluate(
    X_test,
    y_test,
    metrics={
        "accuracy": (accuracy_score, False),
        "log_loss": (log_loss, True),
    },
)
```

## Score

`score()` exists for sklearn compatibility, but its defaults are not the same as all sklearn estimators:

| Estimator | Current default |
| --- | --- |
| Classifier | `log_loss` on probabilities |
| Regressor | mean squared error |
| LSS | negative log-likelihood |

Pass a metric explicitly if you need accuracy, F1, R2, or another convention:

```python
from sklearn.metrics import accuracy_score

accuracy = classifier.score(X_test, y_test, metric=(accuracy_score, False))
```

## Save and Load

```python
model.fit(X_train, y_train)
model.save("model.pt")

loaded = type(model).load("model.pt")
predictions = loaded.predict(X_test)
```

The saved bundle includes preprocessing state, model metadata, config, and weights.

## Model Inspection

DeepTab estimators expose a small inspection layer for understanding a configured or fitted model.

| Method | Returns | When to use |
| --- | --- | --- |
| `describe()` | Dictionary with estimator, architecture, task, feature counts, config classes, and parameter counts when available | Programmatic metadata for reports and experiment tracking |
| `summary()` | Compact human-readable string | Notebook/log output before or after training |
| `parameter_table()` | `pandas.DataFrame` with parameter name, module, shape, count, trainability, dtype, and device | Auditing model size and trainable layers |
| `runtime_info()` | Dictionary with device, dtype, precision, accelerator, strategy, batch size, optimizer, and trainer state | Checking how the model is actually running |

```python
model.fit(X_train, y_train)

print(model.summary())
metadata = model.describe()
params = model.parameter_table()
runtime = model.runtime_info()
```

`describe()`, `summary()`, and `runtime_info()` are safe to call before fitting. `parameter_table()` requires a built or fitted model because the PyTorch modules do not exist until DeepTab has seen the feature schema.

```python
model = MambularClassifier()

print(model.describe()["built"])
print(model.runtime_info()["batch_size"])

# Raises ValueError until fit() or build_model() has created the network.
model.parameter_table()
```

```{tip}
Use `runtime_info()` in benchmark notebooks and experiment logs. It records the resolved runtime state, which can differ from what you intended if Lightning chooses a different accelerator or if the model was loaded on CPU.
```

## scikit-learn Integration

DeepTab implements `get_params` and `set_params`, including nested config parameters:

```python
model.get_params()

model.set_params(
    model_config__d_model=128,
    trainer_config__lr=3e-4,
)
```

This enables `GridSearchCV`:

```python
from sklearn.model_selection import GridSearchCV
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MambularClassifier

estimator = MambularClassifier(
    model_config=MambularConfig(),
    preprocessing_config=PreprocessingConfig(),
    trainer_config=TrainerConfig(max_epochs=30, patience=5),
)

search = GridSearchCV(
    estimator=estimator,
    param_grid={
        "model_config__d_model": [32, 64],
        "trainer_config__lr": [1e-3, 3e-4],
    },
    cv=3,
    n_jobs=1,
)
```

## Practical Differences From sklearn

DeepTab models train neural networks, so `fit()` is slower than fitting most classical sklearn estimators. Validation data, early stopping, checkpoints, GPU settings, and random seeds matter.

For reproducible research:

1. Use explicit train/validation/test splits.
2. Set `random_state` on the estimator and split functions.
3. Save model, preprocessing, and config choices.
4. Report the exact DeepTab version.

## Next Steps

- [Config System](config_system)
- [Preprocessing](preprocessing)
- [Training and Evaluation](training_and_evaluation)
