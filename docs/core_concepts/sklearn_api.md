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

`score()` follows a consistent default per estimator family:

| Estimator | Current default |
| --- | --- |
| Classifier | accuracy |
| Regressor | mean squared error |
| LSS | negative log-likelihood |

Pass a metric explicitly if you need F1, R2, log loss, or another convention:

```python
from sklearn.metrics import log_loss

loss = classifier.score(X_test, y_test, metric=(log_loss, True))
```

## Learned Attributes

After `fit()` or `build_model()`, DeepTab estimators expose common sklearn-style fitted attributes:

| Attribute | Available on | Meaning |
| --- | --- | --- |
| `n_features_in_` | Classifier, regressor, LSS | Number of input columns seen during fitting. |
| `feature_names_in_` | Estimators fitted with string-named DataFrame columns | Feature names and order seen during fitting. |
| `classes_` | Classifiers and categorical LSS | Class labels seen during fitting. |

Prediction inputs are checked against the fitted feature count. When the model was fitted with named DataFrame columns, prediction DataFrames must use the same feature names in the same order. This catches accidental column drops, additions, and reordering before inference.

## Save and Load

DeepTab has two persistence layers:

| Method | Scope | Use case |
| --- | --- | --- |
| `model.save(...)` / `Estimator.load(...)` | Full fitted estimator artifact | Reuse a trained classifier, regressor, or LSS model for inference or reproducible experiments. |
| `BaseModel.save_model(...)` / `load_model(...)` | Raw PyTorch architecture weights only | Low-level architecture work where you already know how to rebuild the model and preprocessing pipeline. |

For normal user workflows, prefer the estimator-level API:

```python
model.fit(X_train, y_train)
model.save("model.pt")

loaded = type(model).load("model.pt")
predictions = loaded.predict(X_test)
```

The saved estimator bundle is designed as a fitted inference artifact. It includes:

| Artifact field | Why it matters |
| --- | --- |
| Architecture metadata | Stores the model class, module, registry status, config class, and resolved config values. |
| Trained weights | Restores the fitted `TaskModel` state. |
| Fitted preprocessing state | Reuses the exact fitted preprocessing object instead of refitting on future data. |
| Feature schema | Stores column order, numerical/categorical/embedding feature groups, dimensions, and feature preprocessing metadata. |
| Task metadata | Stores the task type, regression/LSS flags, distribution family for LSS, number of output classes, and `classes_` for classifiers. |
| Runtime/debug metadata | Stores Python, platform, DeepTab, PyTorch, Lightning, pandas, NumPy, scikit-learn, pretab, and related dependency versions. |

Using pandas DataFrames is recommended because the saved schema can preserve meaningful column names. NumPy inputs are supported, but their inferred column order is positional.

```python
loaded = MambularClassifier.load("model.pt")

loaded.input_columns_
loaded.feature_schema_
loaded.task_info_
loaded.versions_
```

`load()` keeps backward compatibility with older DeepTab artifacts that do not contain the richer metadata block, but newer artifacts are easier to audit and debug across environments.

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
