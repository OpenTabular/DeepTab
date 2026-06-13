# Inference Model

`InferenceModel` is a deployment-only wrapper for a fitted DeepTab artifact. It provides a strict, minimal surface for production use: load → validate → predict.

Training, hyper-parameter optimisation, and inspection methods are intentionally absent, so deployment code cannot accidentally trigger a fit or mutate model state.

---

## Why use `InferenceModel` instead of `estimator.load()` + `predict()`?

Both paths load the same artifact and call the same underlying neural network. The difference is the **contract** you code against and the **guardrails** available at the boundary.

| Concern                   | `estimator.load()` + `predict()`                                                      | `InferenceModel`                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Interface surface**     | Full estimator API: `fit`, `optimize_hparams`, `build_model`, etc.                    | Only `predict`, `predict_proba`, `predict_params`, `validate_input`, `describe`, `runtime_info`  |
| **Schema validation**     | `validate_input_features` checks count and name equality, but column order must match | `validate_input` checks missing columns, extra columns, and silently re-orders to training order |
| **Missing-column error**  | Raises a generic sklearn-style message                                                | Raises with the exact list of missing column names                                               |
| **Extra-column handling** | Raises                                                                                | Configurable: raises by default, or drops with a warning when `allow_extra_columns=True`         |
| **Column reordering**     | Not performed                                                                         | Always reorders to match training order before calling the estimator                             |
| **Production intent**     | Signals "research / local experimentation"                                            | Signals "deployment": the code reviewer and the type checker both see a narrower type            |
| **Task-aware API**        | One `predict()` for all tasks                                                         | `predict_proba()` and `predict_params()` raise `TypeError` when called on the wrong task type    |

```{tip}
Use the normal estimator API for research, notebook exploration, and retraining.
Use `InferenceModel` when writing a service, pipeline step, or batch job where the model should never be modified after loading.
```

---

## Step 1: Load from a saved artifact

```python
from deeptab import InferenceModel

model = InferenceModel.from_path("my_model.deeptab")
```

`from_path` calls the estimator's own `load()` classmethod internally, so the artifact format is identical to what `estimator.load()` reads. Any `.deeptab` file saved by `model.save()` is valid input.

```{note}
A `UserWarning` is emitted when the file does not end with `.deeptab`. The file is still loaded correctly; the warning is advisory only.
```

### Wrap an already-fitted estimator

When the estimator is already in memory (e.g. you just finished training in a notebook), skip the file round-trip:

```python
clf = MLPClassifier()
clf.fit(X_train, y_train)

model = InferenceModel.from_estimator(clf)
```

Passing an unfitted estimator raises immediately:

```python
InferenceModel.from_estimator(MLPClassifier())
# ValueError: Cannot wrap an unfitted estimator in InferenceModel.
```

---

## Step 2: Inspect what was loaded

Before routing data through the model, check that the artifact matches your expectations.

### Task and feature schema

```python
model.task          # "classification" | "regression" | "distributional_regression"
model.n_features    # 10
model.feature_names # ["age", "income", "score", ...]  (None when artifact has no column names)
model.classes_      # array([0, 1])  (None for regression)
model.task_info     # {"task": "classification", "regression": False, "num_classes": 2, ...}
model.feature_schema  # full feature-schema dict from the artifact
```

### Structured summary

```python
info = model.describe()
# {
#   "estimator":      "MLPClassifier",
#   "architecture":   "MLP",
#   "task":           "classification",
#   "built":          True,
#   "fitted":         True,
#   "feature_counts": {"numerical": 8, "categorical": 2, "embedding": 0, "total": 10},
#   "parameters":     {"total": 45312, "trainable": 45312, "non_trainable": 0},
#   "inference_task": "classification",   # ← added by InferenceModel
#   ...
# }
```

### Device and runtime

```python
info = model.runtime_info()
# {"built": True, "fitted": True, "device": "cpu", "dtype": "float32", ...}
```

### Parameter table

```python
df = model.parameter_table()
# name                            module       shape      num_params  trainable  dtype    device
# estimator.num_embeddings.weight estimator... (20, 64)   1280        True       float32  cpu
# ...
```

---

## Step 3: Validate input

`validate_input` enforces the column contract against training data before prediction. Call it explicitly to get a clear error before handing data to the model, or rely on the fact that `predict`, `predict_proba`, and `predict_params` all call it internally.

```python
X_validated = model.validate_input(X_new)
predictions  = model.predict(X_validated)
```

### What is checked

| Check                        | Behaviour                                                                |
| ---------------------------- | ------------------------------------------------------------------------ |
| **Missing columns**          | `ValueError` listing every missing column name                           |
| **Extra columns**            | `ValueError` by default                                                  |
| **Extra columns (lenient)**  | Pass `allow_extra_columns=True` to drop them with a `UserWarning`        |
| **Column order**             | Always silently reordered to match training order                        |
| **Feature count (no names)** | `ValueError` when count does not match and no column names are available |

### Missing columns

```python
X_bad = X_new.drop(columns=["income"])
model.validate_input(X_bad)
# ValueError: Input is missing 1 column(s) that were present during training: ['income'].
```

### Extra columns

```python
X_extra = X_new.copy()
X_extra["debug_flag"] = 0

# Default: raise
model.validate_input(X_extra)
# ValueError: Input has 1 unexpected column(s) not seen during training: ['debug_flag'].
# To drop them automatically, pass allow_extra_columns=True.

# Lenient: drop with a warning
X_clean = model.validate_input(X_extra, allow_extra_columns=True)
# UserWarning: Input has 1 column(s) not seen during training (['debug_flag']); they will be dropped.
```

### Column reordering

The returned DataFrame always uses the column order from training, regardless of the order in the input. This is handled silently and requires no action from the caller.

```python
X_shuffled = X_new[["score", "income", "age"]]   # wrong order
X_correct  = model.validate_input(X_shuffled)     # reordered automatically
print(list(X_correct.columns))
# ['age', 'income', 'score']
```

### No column names in the artifact

Artifacts saved from models that were fitted on arrays (not DataFrames) may not store column names. In that case only a feature-count check is performed:

```python
model.n_features    # 10
model.feature_names # None

model.validate_input(X_wrong_shape)
# ValueError: Expected 10 feature(s) (no column names available for
# detailed validation), got 7.
```

---

## Step 4: Predict

### Classification

```python
# Hard class labels
predictions = model.predict(X_new)
# array([0, 1, 1, 0, ...])

# Class probabilities
proba = model.predict_proba(X_new)
# array([[0.82, 0.18], [0.11, 0.89], ...])  shape (n_samples, n_classes)
```

`predict_proba` raises `TypeError` when called on a regression or LSS model:

```python
model.predict_proba(X_new)
# TypeError: predict_proba() is only available for classification models,
# but this model's task is 'regression'.
```

### Regression

```python
predictions = model.predict(X_new)
# array([23.4, 18.1, 31.7, ...])  shape (n_samples,)
```

### Distributional regression (LSS)

```python
# Distribution mean / mode (default)
predictions = model.predict(X_new)

# Raw distribution parameters (before inverse-link transform)
params = model.predict_params(X_new, raw=False)
# array([...])  shape (n_samples, n_params)
```

`predict_params` raises `TypeError` on non-LSS models:

```python
model.predict_params(X_new)
# TypeError: predict_params() is only available for distributional regression
# (LSS) models, but this model's task is 'classification'.
```

---

## Full production example

```python
import pandas as pd
from deeptab import InferenceModel

# --- Load once at service startup ---
model = InferenceModel.from_path("models/churn_v3.deeptab")

print(model)
# InferenceModel(task='classification', estimator='MLPClassifier',
#                n_features=12, features=['age', 'tenure', ...], n_classes=2)

# --- Per-request inference ---
def score_request(payload: dict) -> dict:
    X = pd.DataFrame([payload])

    # Validate schema, raises immediately on mismatch
    X_clean = model.validate_input(X, allow_extra_columns=True)

    proba   = model.predict_proba(X_clean)
    label   = model.predict(X_clean)

    return {
        "churn_probability": float(proba[0, 1]),
        "label":             int(label[0]),
    }
```

---

## Comparison: `estimator.load()` vs `InferenceModel`

```python
# --- Standard estimator path ---
from deeptab.models import MLPClassifier

loaded = MLPClassifier.load("model.deeptab")
# Nothing stops this:
loaded.fit(X_new, y_new)        # accidentally retrains
loaded.optimize_hparams(...)    # runs Bayesian search in production

# --- InferenceModel path ---
from deeptab import InferenceModel

model = InferenceModel.from_path("model.deeptab")
# These don't exist on InferenceModel:
model.fit(...)              # AttributeError
model.optimize_hparams(...) # AttributeError
```

The `InferenceModel` surface makes it structurally impossible to call training methods, giving you a clear separation between the training codebase and the inference codebase.

---

## `repr` at a glance

```python
print(model)
# InferenceModel(task='classification', estimator='MLPClassifier',
#                n_features=12, features=['age', 'tenure', 'monthly_charges', ...],
#                n_classes=2)
```

---

## Next Steps

- [Model Operations](model_operations): saving, loading, and inspecting estimators
- [sklearn API](sklearn_api): the full estimator interface for research and training
- [Training and Evaluation](training_and_evaluation): fit pipeline, configs, and callbacks
