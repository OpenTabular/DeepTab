# Config System

DeepTab uses a split-config API. Architecture, preprocessing, and training settings are kept in separate dataclasses so experiments can change one layer without mixing concerns.

```{important}
The model constructor accepts `model_config`, `preprocessing_config`, and `trainer_config`. All settings must go through these config objects; the flat constructor arguments from v1 are no longer accepted and raise a `TypeError`.
```

## The Three Config Layers

| Config                | Scope                                     | Examples                                                                             |
| --------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------ |
| `<Model>Config`       | Neural architecture                       | `d_model`, `n_layers`, `dropout`, `n_heads`, `layer_sizes`                           |
| `PreprocessingConfig` | Arguments passed to `pretab.Preprocessor` | `numerical_preprocessing`, `categorical_preprocessing`, `n_bins`, `scaling_strategy` |
| `TrainerConfig`       | Training loop and optimizer               | `max_epochs`, `batch_size`, `lr`, `patience`, `optimizer_type`                       |

All three are optional. If omitted, DeepTab creates default config objects internally.

### Moving from v1

In v1, architecture, preprocessing, and training options were all passed as flat keyword arguments on the estimator. In v2 those same options live in three dedicated config objects. The estimator call is the only thing that changes; `fit`, `predict`, and `evaluate` behave exactly as before.

```python
# v1: every option flat on the estimator
from deeptab.models import MambularClassifier

model = MambularClassifier(
    d_model=128,
    n_layers=4,
    numerical_preprocessing="ple",
    lr=1e-3,
)
```

```python
# v2: options grouped by concern
from deeptab.models import MambularClassifier
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig

model = MambularClassifier(
    model_config=MambularConfig(d_model=128, n_layers=4),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="ple"),
    trainer_config=TrainerConfig(lr=1e-3),
)
```

```{tip}
Each option moves to the config that owns its concern: architecture fields go to the model config, anything passed to `pretab.Preprocessor` goes to `PreprocessingConfig`, and training or optimizer fields go to `TrainerConfig`. The tables further down list which fields belong where.
```

### Where to find every field

Each config has a complete, authoritative field reference. Use the table below as the index.

| Config                | Full field reference                                                                                                                                           |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `<Model>Config`       | Shared fields in `BaseModelConfig`, plus model-specific fields on each [Model Zoo](../model_zoo/stable/index) page and the API reference for that config class |
| `PreprocessingConfig` | The [Preprocessing Config](#preprocessing-config) table below                                                                                                  |
| `TrainerConfig`       | The [Trainer Config](#trainer-config) table below                                                                                                              |

```{tip}
At runtime you can list the fields of any config without leaving Python: `MambularConfig().get_params(deep=False)` returns the field-to-value mapping, and the same call works on `PreprocessingConfig` and `TrainerConfig`.
```

### Keeping each config in the right slot

Each config belongs to a specific constructor argument: a model config goes to `model_config`, a `PreprocessingConfig` to `preprocessing_config`, and a `TrainerConfig` to `trainer_config`. The estimator does not reorder them for you and does not guess intent from the object type.

If you pass a config to the wrong slot, DeepTab now detects it and emits a `ConfigWarning` that names the offending object and the slot it landed in:

```python
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MambularClassifier

# TrainerConfig accidentally passed where the model config belongs
MambularClassifier(model_config=TrainerConfig())
# ConfigWarning: TrainerConfig was passed as 'model_config', but 'model_config'
# expects a BaseModelConfig. Configs are not reordered for you, so this one will
# be misused or silently ignored. Pass it as its matching argument instead.
```

```{warning}
The check warns rather than raises, so construction still succeeds. A misplaced config is then misused or silently ignored: for example a wrong `preprocessing_config` falls back to default preprocessing, and a wrong `trainer_config` falls back to the default optimizer. Treat this warning as an error in your own code and fix the argument it points to.
```

```{note}
The warning only fires for a recognised DeepTab config sitting in the wrong slot. Genuinely custom or duck-typed objects (for example test doubles) are left untouched, so the check never gets in the way of advanced extension code.
```

### Passing a field to the wrong config

A related mistake is putting the right kind of value on the wrong config, for example a model field such as `d_model` on a `TrainerConfig`, or a trainer field such as `lr` on a `PreprocessingConfig`. This case does not need a DeepTab warning because it already fails fast and clearly through the underlying machinery.

Each config is a dataclass, so an unknown field is rejected the moment you build it:

```python
from deeptab.configs import TrainerConfig

TrainerConfig(d_model=64)
# TypeError: TrainerConfig.__init__() got an unexpected keyword argument 'd_model'
```

The same protection applies through `set_params`, where scikit-learn validates the nested field name:

```python
model.set_params(trainer_config__d_model=64)
# ValueError: Invalid parameter 'd_model' for estimator TrainerConfig(...).
```

```{note}
The two mistakes fail in deliberately different ways. A whole config in the wrong **slot** is duck-typed and only triggers an advisory `ConfigWarning`, because a custom object might legitimately stand in for a config. A wrong **field** name has no such ambiguity, so it raises immediately. If you are unsure which config owns a field, check the [field reference index](#where-to-find-every-field) above or call `Config().get_params(deep=False)` to list its valid fields.
```

## Model Configs

Every architecture has a dedicated config class:

```python
from deeptab.configs import MambularConfig
from deeptab.models import MambularClassifier

model = MambularClassifier(
    model_config=MambularConfig(
        d_model=64,
        n_layers=4,
        dropout=0.0,
        pooling_method="avg",
    )
)
```

Model configs inherit shared embedding and architecture fields from `BaseModelConfig`, including `use_embeddings`, `embedding_type`, `d_model`, `batch_norm`, `layer_norm`, `activation`, and `cat_encoding`. Individual models add their own fields; use the model-zoo pages or API reference for model-specific details.

## Preprocessing Config

`PreprocessingConfig` is a thin wrapper around the supported `pretab.Preprocessor` keyword arguments. Fields set to `None` are omitted, leaving the preprocessor default in effect.

```python
from deeptab.configs import PreprocessingConfig

preprocessing_config = PreprocessingConfig(
    numerical_preprocessing="quantile",
    categorical_preprocessing="int",
    n_bins=50,
    scaling_strategy="minmax",
)
```

Valid fields:

| Field                                                                                     | Purpose                                                                                                                                                        |
| ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `numerical_preprocessing`                                                                 | Main numerical transform, e.g. `"standardization"`, `"quantile"`, `"ple"`, `"minmax"`, `"robust"`, `"box-cox"`, `"yeo-johnson"`. Pass `None` for no transform. |
| `categorical_preprocessing`                                                               | Categorical encoding strategy passed to `pretab`, such as `"int"` or `"one-hot"` where supported.                                                              |
| `n_bins`                                                                                  | Number of bins for binned/PLE-style numerical transforms.                                                                                                      |
| `feature_preprocessing`                                                                   | General feature-level preprocessing override.                                                                                                                  |
| `use_decision_tree_bins`, `binning_strategy`                                              | Controls bin edge construction.                                                                                                                                |
| `task`                                                                                    | Optional task hint passed to the preprocessor.                                                                                                                 |
| `cat_cutoff`, `treat_all_integers_as_numerical`                                           | Controls integer-column type inference.                                                                                                                        |
| `degree`, `n_knots`, `use_decision_tree_knots`, `knots_strategy`, `spline_implementation` | Spline/piecewise preprocessing controls.                                                                                                                       |
| `scaling_strategy`                                                                        | Post-transform scaling: `"standardization"`, `"minmax"`, `"robust"`, or `None`.                                                                                |

Embedding width is not a `PreprocessingConfig` field in the current API. It is controlled by model config fields such as `d_model` when an architecture uses `EmbeddingLayer`.

### Running with no numerical preprocessing

Set `numerical_preprocessing=None` (and `categorical_preprocessing=None`) to skip the scaling and encoding transforms and feed near-raw values to the network.

```python
prep = PreprocessingConfig(
    numerical_preprocessing=None,    # no scaling, binning, or PLE on numeric columns
    categorical_preprocessing=None,  # leave categorical encoding at its default
)
model = MambularClassifier(preprocessing_config=prep)
```

```{important}
`None` turns off the numerical transform, not the data layer. DeepTab still detects feature types, turns categorical columns into the integer indices the embedding layers expect, handles missing values, and assembles batched tensors. There is no setting that sends a raw, unconverted DataFrame straight into an `nn.Module`, because the model needs typed, numeric tensors to run.
```

```{note}
Most deep tabular models train better with a numerical transform than without one. `None` is useful when your features are already scaled, or when you want a clean baseline to measure a transform against. For skewed or heavy-tailed inputs, `"quantile"` or `"ple"` are usually stronger starting points.
```

## Trainer Config

`TrainerConfig` controls fit-time defaults used by the estimator.

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

Valid fields:

| Field                               | Meaning                                                                                                                                                               |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_epochs`                        | Maximum Lightning training epochs.                                                                                                                                    |
| `batch_size`                        | Batch size for train/validation/prediction loaders.                                                                                                                   |
| `val_size`                          | Fraction held out when no explicit validation set is passed.                                                                                                          |
| `shuffle`                           | Whether to shuffle the training dataloader.                                                                                                                           |
| `stratify`                          | Whether to stratify the validation split on `y` for classification. Ignored for regression. Default `True`.                                                           |
| `patience`, `monitor`, `mode`       | Early-stopping settings. `monitor` and `mode` also apply to the LR scheduler.                                                                                         |
| `lr`, `lr_patience`, `lr_factor`    | Learning rate and `ReduceLROnPlateau` scheduler defaults.                                                                                                             |
| `weight_decay`                      | Optimizer weight decay (L2 penalty).                                                                                                                                  |
| `optimizer_type`                    | Case-insensitive name of a registered optimizer (e.g. `"Adam"`, `"AdamW"`).                                                                                           |
| `optimizer_kwargs`                  | Extra kwargs forwarded to the optimizer constructor (e.g. `{"betas": (0.9, 0.95)}`).                                                                                  |
| `scheduler_type`                    | Case-insensitive name of a registered LR scheduler, or `None` to disable. Default: `"ReduceLROnPlateau"`.                                                             |
| `scheduler_kwargs`                  | Extra kwargs forwarded to the scheduler constructor. For `ReduceLROnPlateau`, `"factor"` and `"patience"` are synthesised from `lr_factor`/`lr_patience` when absent. |
| `scheduler_monitor`                 | Override the metric watched by the scheduler (defaults to `monitor`).                                                                                                 |
| `scheduler_interval`                | `"epoch"` (default) or `"step"`: Lightning scheduling granularity.                                                                                                    |
| `scheduler_frequency`               | How many intervals to wait between scheduler steps (default `1`).                                                                                                     |
| `no_weight_decay_for_bias_and_norm` | When `True`, bias and normalisation-layer parameters receive zero weight decay. Recommended for transformer-style architectures.                                      |
| `checkpoint_path`                   | Directory for the best-model checkpoint.                                                                                                                              |

Runtime options such as `accelerator`, `devices`, `precision`, `gradient_clip_val`, and logger/callback settings are Lightning trainer keyword arguments, not `TrainerConfig` fields. Pass them to `fit(...)` when needed.

### Optimizer registry

`optimizer_type` resolves through a registry, so any name that is not a built-in `torch.optim` class (or previously registered) raises
`InvalidParamError` immediately with a list of valid options.

```python
from deeptab.training.optimizers import available_optimizers, register_optimizer

print(available_optimizers())
# ['adadelta', 'adagrad', 'adam', 'adamax', 'adamw', 'asgd', ...]

# Register a third-party optimizer
register_optimizer("muon", MyMuonOptimizer)
tc = TrainerConfig(optimizer_type="muon", lr=1e-3)
```

### Scheduler registry

`scheduler_type` resolves through a parallel registry.

```python
from deeptab.training.schedulers import available_schedulers, register_scheduler

print(available_schedulers())
# ['constantlr', 'cosineannealinglr', 'cosineannealingwarmrestarts', ...]

# Switch to cosine annealing
tc = TrainerConfig(
    scheduler_type="CosineAnnealingLR",
    scheduler_kwargs={"T_max": 100, "eta_min": 1e-6},
)

# Disable the scheduler entirely
tc = TrainerConfig(scheduler_type=None)
```

```{important}
`monitor` and `mode` are forwarded to **both** early stopping and the LR
scheduler, so they are always aligned. Previously `ReduceLROnPlateau` always
watched `val_loss` in `min` mode regardless of what early stopping was
configured to use.
```

### Registry lifecycle

The optimizer, scheduler, and loss registries are plain in-memory dictionaries that live for the lifetime of the Python process. DeepTab fills them with its built-in entries at import time, and any name you add joins the same process-global table.

| Stage                 | Optimizer / scheduler                                             | Loss                                        | Metric                                                              |
| --------------------- | ----------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------- |
| Register              | `register_optimizer(name, cls)` / `register_scheduler(name, cls)` | Subclass `BaseLoss` with a `name=` keyword  | No registry API; pass metric instances to `evaluate(metrics={...})` |
| Look up               | `available_optimizers()` / `available_schedulers()`               | `BaseLoss.available()`                      | `METRIC_REGISTRY` holds the per-task defaults                       |
| Re-register same name | Raises `ValueError` unless `override=True`                        | Silently replaces the previous class        | Not applicable                                                      |
| Deregister            | `unregister_optimizer(name)` / `unregister_scheduler(name)`       | No deregister API                           | Not applicable                                                      |
| Process restart       | Built-ins return on import; your entries are gone                 | Built-ins return on import; re-import yours | Defaults rebuilt on import                                          |

**After you register**, the name is usable immediately, everywhere that accepts an `optimizer_type`, `scheduler_type`, or `loss_fct` string, for the rest of that process:

```python
from deeptab.training.optimizers import register_optimizer, available_optimizers

register_optimizer("muon", MyMuonOptimizer)
print("muon" in available_optimizers())          # True
TrainerConfig(optimizer_type="muon", lr=1e-3)    # resolves now
```

**Registering the same name again** is where the registries differ. Optimizers and schedulers refuse to clobber an existing entry unless you opt in:

```python
register_optimizer("muon", MyMuonOptimizer)                 # ValueError: already registered
register_optimizer("muon", MyMuonOptimizer, override=True)  # OK, replaces the entry
```

A loss registers itself the moment its class body runs, so re-importing or redefining a `BaseLoss` subclass with the same `name` silently overwrites the earlier one. There is no `override` flag and no error:

```python
from deeptab.training.losses import BaseLoss

class FocalLoss(BaseLoss, name="focal"):   # replaces the built-in "focal" in this process
    ...
```

**Deregistering** applies only to optimizers and schedulers, and only to names you added. Built-ins are protected:

```python
from deeptab.training.optimizers import unregister_optimizer

unregister_optimizer("muon")                   # removes your entry
unregister_optimizer("muon", missing_ok=True)  # idempotent: no error if already gone
unregister_optimizer("adam")                   # ValueError: built-in, cannot be removed
```

```{important}
Nothing in any registry is persisted to disk. When the interpreter restarts, only DeepTab's built-ins come back automatically at import; every custom optimizer, scheduler, or loss you registered must be registered again. Put your `register_*` calls (and your `BaseLoss` subclass definitions) in a module that is imported at the top of every training script, so they are present in each new process and in each worker when training with multiple processes (DDP).
```

```{note}
Metrics work differently: there is no `register_metric` function. `METRIC_REGISTRY` only holds the per-task default lists. To use a custom metric, subclass `DeepTabMetric` and pass an instance straight to `evaluate(metrics={"my_metric": MyMetric()})`; nothing is registered, so nothing needs cleanup.
```

## Controlling the validation split

When you do not pass an explicit validation set, DeepTab holds one out from the training data. The split is governed by `TrainerConfig` fields, so the split policy lives in the same place as the rest of the training settings.

```python
from deeptab.configs import TrainerConfig

trainer_config = TrainerConfig(
    val_size=0.15,    # fraction held out when no explicit validation set is passed
    shuffle=True,     # shuffle before splitting
    stratify=True,    # keep class proportions in the split (classification only)
)
```

| Field      | Default | Meaning                                                                                                    |
| ---------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `val_size` | `0.2`   | Validation fraction used when no `X_val` is given.                                                         |
| `shuffle`  | `True`  | Shuffle before splitting; `False` keeps the split order-based.                                             |
| `stratify` | `True`  | Stratify the split on `y` so train and validation keep the same class proportions. Ignored for regression. |

The seed for the split comes from the estimator's `random_state` (or the `random_state` you pass to `fit()`), so the same seed always reproduces the same partition.

```{important}
`stratify` applies to classification only. A continuous regression target cannot be stratified, so the flag is ignored there. With `stratify=True` (the default) a classification split keeps the class balance of the full set; set `stratify=False` to draw a purely random split, which is useful for very small or rare-class datasets where stratification would otherwise fail.
```

```{note}
When you provide your own `X_val` and `y_val`, no internal split happens at all, so `val_size`, `shuffle`, and `stratify` do not apply.
```

## Observability Config

The three configs above describe the model and how it trains. A fourth, optional config, `ObservabilityConfig`, controls what gets recorded while training runs: lifecycle events, a per-run artifact directory, and output for experiment trackers such as TensorBoard or MLflow. It is opt-in, so an estimator built without one trains exactly as before and emits nothing.

```python
from deeptab.core.observability import ObservabilityConfig
from deeptab.models import MambularClassifier

model = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=4),
    observability_config=ObservabilityConfig(
        experiment_name="churn_baseline",
        structured_logging=True,
        experiment_trackers=["tensorboard"],
    ),
)
```

```{note}
`ObservabilityConfig` lives in `deeptab.core.observability`, not `deeptab.configs`, because it records training rather than defining the model recipe. Unlike the three configs above it is excluded from `get_params()` and `sklearn.clone`, so it never takes part in hyperparameter search. The [Observability guide](observability) has the full field reference, the run-directory layout, and the verbosity levels.
```

## Using Configs Together

```python
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MambularClassifier

model = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=4),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(max_epochs=100, batch_size=128, lr=3e-4),
    random_state=101,
)

model.fit(X_train, y_train)
```

If `trainer_config` is provided, `fit()` takes its `max_epochs`, `batch_size`, `val_size`, `shuffle`, `stratify`, `patience`, `monitor`, `mode`, and `checkpoint_path`, overriding the matching `fit()` arguments.

## Hyperparameter Search

DeepTab estimators expose nested config fields with scikit-learn's double-underscore syntax.

```python
from sklearn.model_selection import GridSearchCV
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MambularClassifier

estimator = MambularClassifier(
    model_config=MambularConfig(),
    preprocessing_config=PreprocessingConfig(),
    trainer_config=TrainerConfig(max_epochs=30, patience=5),
)

param_grid = {
    "model_config__d_model": [32, 64, 128],
    "model_config__n_layers": [2, 4],
    "trainer_config__lr": [1e-3, 3e-4],
    "preprocessing_config__numerical_preprocessing": ["standardization", "quantile"],
}

search = GridSearchCV(estimator, param_grid=param_grid, cv=3, n_jobs=1)
search.fit(X_train, y_train)
```

Use `n_jobs=1` for GPU experiments unless you intentionally manage multiple processes and devices.

## Inspecting and Updating Parameters

```python
cfg = MambularConfig(d_model=64)
print(cfg.get_params(deep=False))

cfg.set_params(d_model=128, n_layers=6)
```

On estimators:

```python
model = MambularClassifier(
    model_config=MambularConfig(),
    preprocessing_config=PreprocessingConfig(),
    trainer_config=TrainerConfig(),
)

model.set_params(model_config__d_model=128, trainer_config__lr=1e-3)
```

## Practical Guidance

Start with a small model and explicit trainer settings. Add preprocessing and architecture search only after the baseline runs end to end.

1. Use `TrainerConfig(max_epochs=30, patience=5)` for quick checks.
2. Tune `lr` and `batch_size` before deep architecture sweeps.
3. Keep preprocessing choices in `PreprocessingConfig` so experiments are reproducible.
4. Save the three configs with experiment results; they are the primary recipe for reproducing a model.

## Next Steps

- [Training and Evaluation](training_and_evaluation)
- [Observability](observability)
- [Model Zoo](../model_zoo/stable/index)
