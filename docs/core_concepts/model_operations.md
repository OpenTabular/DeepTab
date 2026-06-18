# Model Operations

This page covers what you can do with a fitted DeepTab model beyond training: how to save and reload artifacts, and how to inspect any model's architecture, parameters, device, and runtime characteristics.

---

## Serialisation

DeepTab models save the complete artifact needed for inference: weights, fitted preprocessor, feature schema, model config, task metadata, and package versions.

### Saving and loading

The recommended extension is `.deeptab`. DeepTab emits a `UserWarning` when a different extension is used (e.g. `.pt`), but any path is accepted.

```python
# Save
model.save("my_model.deeptab")

# Load (returns a fully ready estimator, no re-fitting needed)
from deeptab.models import MLPClassifier

loaded = MLPClassifier.load("my_model.deeptab")
predictions = loaded.predict(X_test)
```

```{tip}
Use the class that matches the saved model type. Using the wrong class will raise an error with a clear message pointing to the mismatch.
```

### What is inside the artifact

The bundle saved to disk is a PyTorch-serialised dictionary containing:

| Key                     | Contents                                                                  |
| ----------------------- | ------------------------------------------------------------------------- |
| `task_model_state_dict` | Neural network weights (Lightning module state dict)                      |
| `preprocessor`          | Fitted `pretab.Preprocessor` object                                       |
| `feature_info`          | Numerical, categorical, and embedding feature metadata                    |
| `config`                | Model config dataclass used during training                               |
| `artifact_metadata`     | Architecture, schema, preprocessing, task, and version sub-blocks         |
| `input_columns`         | Ordered list of column names, for feature-name validation at predict time |
| `classes_`              | Class labels for classifiers                                              |
| `versions`              | Python, PyTorch, Lightning, NumPy, pandas, scikit-learn versions          |

### Why everything lives in one bundle

A trained model is more than its weights. To turn raw input into a prediction you also need the fitted preprocessor that scaled and encoded the features, the feature schema that says which columns belong where, the architecture and its config to rebuild the network, and the task metadata that decides whether an output is a class label, a point estimate, or distribution parameters. If any of these travel separately, a reload can silently go wrong: a column in the wrong position or a re-fitted scaler will produce confident but incorrect predictions.

DeepTab keeps all of it together so that one file is enough to reproduce the exact model you trained. The saved package versions make that promise auditable, so when a colleague loads your artifact a year later they can see the environment it was built in.

```{note}
The metadata is tiny next to the weights. Schema, config, task info, and version stamps add a few kilobytes and grow with the number of features, not the number of training rows. A model trained on ten rows and one trained on ten million carry the same metadata footprint.
```

### How `.deeptab` compares to raw formats

`.deeptab` is not a new on-disk format. It is a PyTorch-serialised dictionary with a clear name, and the value it adds over saving raw weights is everything wrapped around those weights.

| Capability                                         | `.pt` (state dict) | `.pkl` (pickled estimator) | `.h5` | `.deeptab` |
| -------------------------------------------------- | :----------------: | :------------------------: | :---: | :--------: |
| Model weights                                      |         ✓          |             ✓              |   ✓   |     ✓      |
| Rebuilds the correct architecture automatically    |         ✗          |      depends on class      |   ✗   |     ✓      |
| Fitted preprocessor (scalers, encoders)            |         ✗          |         sometimes          |   ✗   |     ✓      |
| Feature schema for predict-time validation         |         ✗          |             ✗              |   ✗   |     ✓      |
| Task metadata (regression, LSS family, `classes_`) |         ✗          |         sometimes          |   ✗   |     ✓      |
| Environment version stamps                         |         ✗          |             ✗              |   ✗   |     ✓      |
| Self-contained: predict with no extra glue code    |         ✗          |             ✗              |   ✗   |     ✓      |

With a bare `.pt` file you have to recreate the architecture by hand and re-attach a preprocessor before the weights mean anything. A pickled estimator can capture more, but it stores a live Python object graph that breaks the moment a class is renamed or a dependency shifts, and unpickling it runs arbitrary code. `.deeptab` sidesteps both problems by storing structured metadata alongside the weights and reconstructing the model through DeepTab's own loader.

```{important}
The self-contained reload is a feature of the DeepTab package, not of the file on its own. Loading a `.deeptab` artifact needs `deeptab` installed, ideally at a compatible version, which is exactly why the version snapshot is saved. The file is not a framework-independent interchange format. If you need a model that runs in a non-Python or non-DeepTab runtime, export to ONNX or TorchScript instead.
```

```{warning}
Because the artifact is pickle-backed under the hood, only load `.deeptab` files from sources you trust, the same caution that applies to any `torch.load` or pickle file.
```

### Verifying a round-trip

```python
model.save("my_model.deeptab")
loaded = MLPClassifier.load("my_model.deeptab")

# Hard predictions must be bit-identical
assert (model.predict(X_test) == loaded.predict(X_test)).all()

# Probabilities within floating-point tolerance
import numpy as np
np.testing.assert_allclose(
    model.predict_proba(X_test),
    loaded.predict_proba(X_test),
    atol=1e-5,
)
print("Round-trip verified ✓")
```

### Metadata attributes after loading

After `load()` the estimator exposes several read-only metadata attributes:

```python
loaded.artifact_metadata_      # full metadata dict
loaded.architecture_metadata_  # architecture sub-block
loaded.feature_schema_         # feature schema sub-block
loaded.task_info_              # {"task": "classification", "num_classes": 2, ...}
loaded.classes_                # class labels
loaded.versions_               # package version snapshot
loaded.n_features_in_          # number of input features
loaded.input_columns_          # ordered feature names
```

### The feature schema

`feature_schema_` is the model's data contract. When the preprocessor fits, DeepTab records each feature's name, the preprocessing applied to it, its output dimension, and, for categorical columns, its category list. It tracks numerical, categorical, and embedding features separately.

```python
loaded.feature_schema_
# {
#   "numerical_features":   {"age": {...}, "income": {...}},
#   "categorical_features": {"city": {..., "categories": ["NYC", "Boston", ...]}},
#   "embedding_features":   None,
#   "dimensions": {"num_numerical_features": 2, "num_categorical_features": 1, ...},
# }
```

This single description does several jobs. The architecture reads it to size its input and embedding layers, so you never wire feature counts by hand. It records which columns the model expects, in what order and of what type, which is what lets `InferenceModel.validate_input()` reject a mismatched request at serving time. Because it is saved inside the artifact, a reloaded model knows its feature layout without re-fitting.

```{note}
The schema grows with the number of features, not the number of rows. It is the piece that lets a saved model carry "how to feed me" alongside its weights, so think of it as the bridge between preprocessing, the network, and deployment.
```

---

## Model Inspection

All DeepTab estimators inherit `InspectionMixin`, which provides four read-only methods and one dry-run profiler. They are safe to call before or after fitting.

### `describe()`: structured dict

Returns a structured snapshot of the estimator and its fitted state:

```python
info = model.describe()
# {
#   "estimator":      "MLPClassifier",
#   "architecture":   "MLP",
#   "task":           "classification",
#   "built":          True,
#   "fitted":         True,
#   "model_config":   "MLPConfig",
#   "feature_counts": {"numerical": 8, "categorical": 2, "embedding": 0, "total": 10},
#   "num_classes":    2,
#   "parameters":     {"total": 45312, "trainable": 45312, "non_trainable": 0},
# }
```

Safe to call before fitting: parameter and feature metadata are omitted when the model is not yet built.

### `summary()`: human-readable string

Compact text report combining `describe()` and `runtime_info()`:

```python
print(model.summary())
# MLPClassifier summary
#   Architecture: MLP
#   Task: classification
#   Built: True
#   Fitted: True
#   Model config: MLPConfig
#   Features: 10 total (8 numerical, 2 categorical, 0 embedding)
#   Parameters: 45,312 total, 45,312 trainable, 0 non-trainable
#   Device: cpu
#   Precision: None
#   Accelerator: None
```

### `parameter_table()`: per-parameter DataFrame

Returns one row per parameter:

```python
df = model.parameter_table()
df.head()
# name                       module               shape       num_params  trainable  dtype    device
# estimator.embedding.weight estimator.embedding  (50, 32)    1600        True       float32  cpu
# ...

# Trainable only
df_train = model.parameter_table(trainable_only=True)
```

### `runtime_info()`: device and training setup

```python
info = model.runtime_info()
# {
#   "built":        True,
#   "fitted":       True,
#   "device":       "cpu",
#   "dtype":        "float32",
#   "precision":    None,
#   "accelerator":  None,
#   "max_epochs":   100,
#   "current_epoch": 87,
#   "batch_size":   64,
#   "lr":           0.0001,
#   "weight_decay": 1e-06,
#   ...
# }
```

### `profile()`: pre-training dry run

`profile()` builds the model on a small sample, runs a forward pass, and returns a complete picture of what training will look like, without any gradient updates.

```python
result = model.profile(X, y)   # dry_run=True by default
# {
#   "builds":             True,
#   "error":              None,
#   "device":             "cpu",
#   "dtype":              "float32",
#   "total_params":       45312,
#   "trainable_params":   45312,
#   "memory_mb":          0.173,
#   "batch_shape":        {"num_features": [[64, 20], ...], "cat_features": [], "labels": [64, 1]},
#   "output_shape":       [64, 1],
#   "loss_fct":           "BCEWithLogitsLoss",
#   "forward_ms_median":  1.4,
#   "forward_ms_min":     1.1,
#   "describe":           {...},
#   "runtime":            {...},
# }
```

Key parameters:

| Parameter          | Default | Effect                                                                        |
| ------------------ | ------- | ----------------------------------------------------------------------------- |
| `dry_run`          | `True`  | Discard temporary build after profiling; leaves estimator unfitted            |
| `n_forward_passes` | `3`     | Number of passes used to estimate timing; median is reported                  |
| `batch_size`       | `None`  | Override batch size for timing (defaults to `TrainerConfig.batch_size` or 64) |
| `random_state`     | `0`     | Seed for the dry-run build                                                    |

When `dry_run=False`, the estimator is left built after the call and can proceed directly to `fit()`.

If the build fails for any reason, `result["builds"]` is `False` and `result["error"]` contains the exception message, while all other keys are still present.

---

## Next Steps

- [Training and Evaluation](training_and_evaluation)
- [sklearn API](sklearn_api)
- [Imbalanced Classification Tutorial](../tutorials/imbalance_classification)
