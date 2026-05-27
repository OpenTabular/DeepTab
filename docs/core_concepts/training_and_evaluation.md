# Training and Evaluation

DeepTab estimators train PyTorch models through Lightning while exposing a scikit-learn style API. This page explains what happens during `fit()`, how validation and checkpointing work, and how to evaluate models correctly.

## Fit Pipeline

```text
model.fit(X, y)
  -> create or reuse configs
  -> convert inputs to DataFrames when needed
  -> split train/validation if X_val/y_val are not provided
  -> fit preprocessing on training data only
  -> transform train/validation data with fitted preprocessing
  -> build the neural architecture from feature metadata
  -> train with Lightning
  -> save best checkpoint
  -> restore best checkpoint after training
```

Classification splits are stratified automatically when DeepTab creates the validation split. Regression splits are random.

## TrainerConfig

```python
from deeptab.configs import TrainerConfig

trainer_config = TrainerConfig(
    max_epochs=100,
    batch_size=128,
    val_size=0.2,
    patience=15,
    monitor="val_loss",
    mode="min",
    lr=1e-4,
    lr_patience=10,
    lr_factor=0.1,
    weight_decay=1e-6,
    optimizer_type="Adam",
    checkpoint_path="model_checkpoints",
)
```

`TrainerConfig` does not contain device, precision, logging, or gradient-clipping fields. Those can be passed as Lightning trainer keyword arguments to `fit(...)` where supported:

```python
model.fit(
    X_train,
    y_train,
    accelerator="gpu",
    devices=1,
    precision="32-true",
)
```

## Validation Sets

If no validation data is supplied, DeepTab creates a validation split:

```python
model.fit(X_train, y_train)
```

For research, prefer explicit validation data so every model sees the same split:

```python
model.fit(
    X_train,
    y_train,
    X_val=X_val,
    y_val=y_val,
)
```

Use temporal or grouped validation splits outside DeepTab when the data is ordered or clustered.

## Early Stopping and Checkpointing

Early stopping monitors `TrainerConfig.monitor`, which defaults to `"val_loss"`. The best checkpoint is saved under `checkpoint_path` and loaded back after training.

```python
TrainerConfig(
    patience=20,
    monitor="val_loss",
    mode="min",
    checkpoint_path="model_checkpoints",
)
```

Checkpointing currently uses `"val_loss"` for the checkpoint callback. If you monitor another metric for early stopping, verify that the checkpoint behavior still matches your intended selection criterion.

## Optimizer and Scheduler

The optimizer is selected by name:

```python
TrainerConfig(
    optimizer_type="AdamW",
    lr=3e-4,
    weight_decay=1e-4,
)
```

DeepTab's `TaskModel.configure_optimizers()` creates a `ReduceLROnPlateau` scheduler using `lr_factor` and `lr_patience`.

```python
TrainerConfig(
    lr=1e-3,
    lr_patience=5,
    lr_factor=0.5,
)
```

## Evaluation

The default `evaluate()` outputs are task-specific and use the current implementation's metric names:

```python
classification_metrics = classifier.evaluate(X_test, y_test)
# {"Accuracy": ...}

regression_metrics = regressor.evaluate(X_test, y_test)
# {"Mean Squared Error": ...}

lss_metrics = lss_model.evaluate(X_test, y_test)
# depends on family, for example {"MSE": ..., "CRPS": ...} for normal
```

For reports, pass explicit metrics so names and behavior are clear:

```python
from sklearn.metrics import accuracy_score, f1_score, log_loss

metrics = classifier.evaluate(
    X_test,
    y_test,
    metrics={
        "accuracy": (accuracy_score, False),
        "f1_macro": (lambda y, pred: f1_score(y, pred, average="macro"), False),
        "log_loss": (log_loss, True),
    },
)
```

Regression:

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

metrics = regressor.evaluate(
    X_test,
    y_test,
    metrics={
        "rmse": lambda y, pred: np.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error,
        "r2": r2_score,
    },
)
```

## Score Method

`score()` is available for scikit-learn compatibility. The default is consistent by estimator family:

| Estimator | Default `score()` |
| --- | --- |
| Classifier | accuracy |
| Regressor | `sklearn.metrics.mean_squared_error` |
| LSS | Negative log-likelihood through the fitted distribution family |

For F1, R2, log loss, or another convention, pass an explicit metric or use sklearn metrics on predictions.

```python
from sklearn.metrics import log_loss

loss = classifier.score(X_test, y_test, metric=(log_loss, True))
```

## Custom Metrics During Training

The `fit()` method accepts `train_metrics` and `val_metrics` dictionaries. These are passed to the Lightning task model.

```python
from torchmetrics.classification import MulticlassAccuracy

model.fit(
    X_train,
    y_train,
    train_metrics={"train_acc": MulticlassAccuracy(num_classes=3)},
    val_metrics={"val_acc": MulticlassAccuracy(num_classes=3)},
)
```

Use metric objects compatible with the tensors produced by the task.

## Saving and Loading

```python
model.fit(X_train, y_train)
model.save("model.pt")

loaded = type(model).load("model.pt")
predictions = loaded.predict(X_test)
```

The saved bundle includes the fitted preprocessor, feature schema and column order, task metadata, model config, weights, and version metadata needed for inference and debugging.

## Troubleshooting

| Symptom | First checks |
| --- | --- |
| Training is slow | Reduce `max_epochs`, reduce model size, increase `batch_size`, use GPU through Lightning trainer kwargs. |
| Validation loss unstable | Lower `lr`, increase `batch_size`, simplify preprocessing, inspect outliers. |
| Overfitting | Increase dropout/model regularization, lower capacity, use explicit validation, increase data. |
| Poor regression scale | Transform the target manually and inverse-transform predictions. |
| Unexpected metric names | Pass explicit `metrics=` to `evaluate()`. |

## Next Steps

- [Config System](config_system)
- [Classification](classification)
- [Regression](regression)
- [Distributional Regression](distributional_regression)
