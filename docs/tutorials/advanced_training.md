# Advanced Training and Production Inference

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <a href="https://colab.research.google.com/github/basf/DeepTab/blob/main/docs/tutorials/notebooks/advanced_training.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/basf/DeepTab/blob/main/docs/tutorials/notebooks/advanced_training.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on-GitHub-181717?logo=github&logoColor=white" alt="View on GitHub"/>
  </a>
</div>

This end-to-end tutorial covers three topics that come up after the basics: choosing
and customising the optimizer and scheduler, extending the built-in registries with
your own implementations, and deploying a trained model with `InferenceModel`.

```{note}
The notebook linked above mirrors this tutorial.  Use the markdown page for
reading; use the notebook when you want to execute cells directly.
```

## What You Will Learn

- How to discover available optimizers and schedulers at runtime.
- How to pass `optimizer_type`, `optimizer_kwargs`, and scheduler fields through
  `TrainerConfig`.
- What `no_weight_decay_for_bias_and_norm` does and when to use it.
- How to register a custom optimizer or scheduler so it works with the same config
  interface.
- How to use `InferenceModel` for schema-validated, deployment-friendly inference.
- How `validate_input`, `predict_proba`, and `predict_params` behave in production.

## Setup

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from deeptab import InferenceModel
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MambularClassifier
from deeptab.training import (
    available_optimizers,
    available_schedulers,
    register_optimizer,
    register_scheduler,
)
```

## Data

All examples in this tutorial share a single binary classification dataset.

```python
RANDOM_STATE = 42

X_num, y = make_classification(
    n_samples=1500,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    random_state=RANDOM_STATE,
)

X = pd.DataFrame(X_num, columns=[f"feat_{i}" for i in range(X_num.shape[1])])

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
)
```

---

## Part 1 — Optimizers

### Discovering available optimizers

`available_optimizers()` returns a sorted list of all names registered in the
optimizer registry. All standard `torch.optim` classes are pre-registered at
import time.

```python
opts = available_optimizers()
print(opts)
# ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'LBFGS',
#  'NAdam', 'RAdam', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
```

### Using AdamW instead of the default Adam

Pass `optimizer_type` to `TrainerConfig`. Any additional optimizer constructor
arguments go in `optimizer_kwargs`:

```python
trainer = TrainerConfig(
    max_epochs=40,
    batch_size=128,
    lr=3e-4,
    patience=10,
    optimizer_type="AdamW",
    optimizer_kwargs={
        "betas": (0.9, 0.98),   # custom momentum coefficients
        "eps": 1e-8,            # numerical stability term
    },
    weight_decay=1e-2,          # passed as a top-level TrainerConfig field
)

clf = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=trainer,
    random_state=RANDOM_STATE,
)
clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
print("AdamW AUROC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
```

```{note}
`lr` and `weight_decay` are top-level `TrainerConfig` fields because they
are also used by the early-stopping monitor and parameter-group logic.
All other optimizer-specific arguments go in `optimizer_kwargs`.
```

### Weight-decay exemption for bias and normalisation layers

Setting `no_weight_decay_for_bias_and_norm=True` splits model parameters into
two groups: one with `weight_decay` as configured and one (biases and
normalisation weights) with `weight_decay=0`. This is the recommended practice
for transformer-style architectures.

```python
trainer_wd = TrainerConfig(
    max_epochs=40,
    batch_size=128,
    lr=3e-4,
    patience=10,
    optimizer_type="AdamW",
    weight_decay=1e-2,
    no_weight_decay_for_bias_and_norm=True,   # <-- enable split
)

clf_wd = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=trainer_wd,
    random_state=RANDOM_STATE,
)
clf_wd.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

### Using SGD with momentum

```python
clf_sgd = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(
        max_epochs=40,
        batch_size=128,
        lr=5e-3,
        patience=10,
        optimizer_type="SGD",
        optimizer_kwargs={"momentum": 0.9, "nesterov": True},
        weight_decay=1e-4,
    ),
    random_state=RANDOM_STATE,
)
clf_sgd.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

---

## Part 2 — Schedulers

### Discovering available schedulers

```python
scheds = available_schedulers()
print(scheds)
# ['CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'CyclicLR',
#  'ExponentialLR', 'LambdaLR', 'LinearLR', 'MultiStepLR', 'MultiplicativeLR',
#  'OneCycleLR', 'PolynomialLR', 'ReduceLROnPlateau', 'StepLR']
```

### CosineAnnealingLR

```python
clf_cos = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(
        max_epochs=60,
        batch_size=128,
        lr=3e-4,
        patience=12,
        optimizer_type="AdamW",
        weight_decay=1e-2,
        scheduler_type="CosineAnnealingLR",
        scheduler_kwargs={"T_max": 60, "eta_min": 1e-6},
        scheduler_interval="epoch",
    ),
    random_state=RANDOM_STATE,
)
clf_cos.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

### ReduceLROnPlateau (default scheduler)

`ReduceLROnPlateau` is the default scheduler. The `monitor` field and the
`mode` field must be consistent — `mode="min"` for loss monitors and
`mode="max"` for metric monitors.

```python
clf_plateau = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(
        max_epochs=60,
        batch_size=128,
        lr=3e-4,
        patience=12,
        optimizer_type="AdamW",
        weight_decay=1e-2,
        scheduler_type="ReduceLROnPlateau",
        scheduler_monitor="val_loss",    # monitor name passed to Lightning
        scheduler_kwargs={
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6,
        },
    ),
    random_state=RANDOM_STATE,
)
clf_plateau.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

```{important}
`scheduler_monitor` and the Lightning `mode` are wired automatically.
DeepTab derives `mode` from whether the monitor name ends in `"loss"` (→
`"min"`) or is a metric name (→ `"max"`).  If your custom monitor does not
follow this convention, pass `scheduler_kwargs={"mode": "min"}` explicitly
to override.
```

### Disabling the scheduler

Set `scheduler_type=None` to use a constant learning rate:

```python
clf_const_lr = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(
        max_epochs=60,
        batch_size=128,
        lr=3e-4,
        patience=12,
        scheduler_type=None,
    ),
    random_state=RANDOM_STATE,
)
clf_const_lr.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

### Step-level scheduler (OneCycleLR)

Some schedulers need to step every batch, not every epoch. Set
`scheduler_interval="step"`:

```python
steps_per_epoch = int(np.ceil(len(X_train) / 128))

clf_onecycle = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(
        max_epochs=40,
        batch_size=128,
        lr=1e-3,
        patience=15,
        optimizer_type="AdamW",
        weight_decay=1e-2,
        scheduler_type="OneCycleLR",
        scheduler_kwargs={
            "max_lr": 1e-3,
            "total_steps": 40 * steps_per_epoch,
            "anneal_strategy": "cos",
        },
        scheduler_interval="step",
    ),
    random_state=RANDOM_STATE,
)
```

```{note}
Some schedulers such as `OneCycleLR` set their own LR curve and work best
with `scheduler_interval="step"`.  Pass all required scheduler arguments
(e.g. `total_steps`) through `scheduler_kwargs`.
```

---

## Part 3 — Custom Optimizer and Scheduler Registration

The registry pattern lets you plug in any optimizer or scheduler that shares
the `torch.optim.Optimizer` / `torch.optim.lr_scheduler.LRScheduler` interface.

### Registering a custom optimizer

```python
class ScaledAdam(torch.optim.Adam):
    """Adam with gradient pre-scaling (toy example)."""

    def __init__(self, params, lr=1e-3, scale=1.0, **kwargs):
        super().__init__(params, lr=lr * scale, **kwargs)


register_optimizer("ScaledAdam", ScaledAdam)

# Verify registration
print("ScaledAdam" in available_optimizers())   # True

# Use it via TrainerConfig
clf_custom_opt = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(
        max_epochs=30,
        batch_size=128,
        lr=3e-4,
        patience=8,
        optimizer_type="ScaledAdam",
        optimizer_kwargs={"scale": 0.8},
    ),
    random_state=RANDOM_STATE,
)
clf_custom_opt.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

### Registering a custom scheduler

```python
class WarmupConstant(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup for `warmup_steps`, then constant LR."""

    def __init__(self, optimizer, warmup_steps: int = 100):
        def _lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            return 1.0

        super().__init__(optimizer, lr_lambda=_lambda)


register_scheduler("WarmupConstant", WarmupConstant)

print("WarmupConstant" in available_schedulers())   # True

clf_warmup = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(
        max_epochs=40,
        batch_size=128,
        lr=3e-4,
        patience=10,
        scheduler_type="WarmupConstant",
        scheduler_kwargs={"warmup_steps": 200},
        scheduler_interval="step",
    ),
    random_state=RANDOM_STATE,
)
clf_warmup.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

---

## Part 4 — Production Inference with `InferenceModel`

`InferenceModel` wraps a fitted estimator and exposes only the prediction
surface. Training methods (`fit`, `optimize_hparams`, etc.) are absent, which
prevents accidental retraining in service code.

### Save a model to disk

```python
clf_wd.save("advanced_clf.pt")
```

### Load via `from_path`

```python
model = InferenceModel.from_path("advanced_clf.pt")
print(model)
# InferenceModel(task='classification', estimator='MambularClassifier',
#                n_features=12, features=['feat_0', ..., 'feat_11'], n_classes=2)
```

### Wrap an already-fitted estimator

If the estimator is already in memory, skip the save/load round-trip:

```python
model_live = InferenceModel.from_estimator(clf_wd)
print(model_live.task)          # classification
print(model_live.n_features)    # 12
```

### Introspection

```python
info = model.describe()
print(info.keys())
# dict_keys(['task', 'estimator_class', 'feature_names', 'n_features',
#            'n_classes', 'classes_', 'task_info'])

rt = model.runtime_info()
print(rt.keys())
# dict_keys(['torch_version', 'device', 'dtype', 'parameter_count'])

params_df = model.parameter_table()
print(params_df.head())
```

### Schema validation

`validate_input` checks that the incoming DataFrame matches the feature schema
seen during training. Call it before every forward pass in production.

```python
# Happy path
X_clean = model.validate_input(X_test)

# Missing column
X_bad = X_test.drop(columns=["feat_0"])
try:
    model.validate_input(X_bad)
except ValueError as exc:
    print(exc)
# ValueError: Input is missing 1 column(s) that were present during training:
# ['feat_0']. Either add the missing columns or retrain the model.

# Extra columns — lenient mode drops them with a warning
X_wide = X_test.copy()
X_wide["audit_id"] = range(len(X_test))
X_clean = model.validate_input(X_wide, allow_extra_columns=True)
# UserWarning: Input has 1 column(s) not seen during training (['audit_id']);
# they will be dropped.
```

### Prediction

```python
# Hard class labels
labels = model.predict(X_clean)
print("Accuracy:", accuracy_score(y_test, labels))

# Class probabilities (classification only)
proba = model.predict_proba(X_clean)
print("AUROC:", roc_auc_score(y_test, proba[:, 1]))
```

`predict_proba` raises `TypeError` for non-classification tasks:

```python
# model.predict_proba(X_clean)
# TypeError: predict_proba() is only available for classification models,
# but this model's task is 'regression'.
```

### Production service pattern

A minimal FastAPI-style handler using `InferenceModel`:

```python
# Module-level: load once at startup
_MODEL = InferenceModel.from_path("advanced_clf.pt")


def score(payload: dict) -> dict:
    X = pd.DataFrame([payload])
    X_clean = _MODEL.validate_input(X, allow_extra_columns=True)
    proba   = _MODEL.predict_proba(X_clean)
    label   = _MODEL.predict(X_clean)
    return {
        "probability_positive": float(proba[0, 1]),
        "label": int(label[0]),
    }
```

---

## Configuration Reference

| `TrainerConfig` field               | Default               | Effect                                                 |
| ----------------------------------- | --------------------- | ------------------------------------------------------ |
| `optimizer_type`                    | `"Adam"`              | Optimizer class name from the registry                 |
| `optimizer_kwargs`                  | `None`                | Extra constructor kwargs (beyond `lr`, `weight_decay`) |
| `weight_decay`                      | `0.0`                 | Passed to optimizer; exempt layers use `0.0`           |
| `no_weight_decay_for_bias_and_norm` | `False`               | Split params into WD/no-WD groups                      |
| `scheduler_type`                    | `"ReduceLROnPlateau"` | Scheduler class name, or `None`                        |
| `scheduler_kwargs`                  | `None`                | Scheduler constructor kwargs                           |
| `scheduler_monitor`                 | `"val_loss"`          | Lightning monitor string for plateau schedulers        |
| `scheduler_interval`                | `"epoch"`             | `"epoch"` or `"step"`                                  |
| `scheduler_frequency`               | `1`                   | Step frequency multiplier                              |

## Next Steps

- [Core concepts: training and evaluation](../core_concepts/training_and_evaluation)
- [Core concepts: inference](../core_concepts/inference)
- [Imbalanced classification tutorial](imbalance_classification)
- [Regression tutorial](regression)
