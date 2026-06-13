# Config System

DeepTab uses a split-config API. Architecture, preprocessing, and training settings are kept in separate dataclasses so experiments can change one layer without mixing concerns.

```{important}
The model constructor accepts `model_config`, `preprocessing_config`, and `trainer_config`. Flat constructor arguments are legacy compatibility only; new documentation and experiments should use split configs.
```

## The Three Config Layers

| Config                | Scope                                     | Examples                                                                             |
| --------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------ |
| `<Model>Config`       | Neural architecture                       | `d_model`, `n_layers`, `dropout`, `n_heads`, `layer_sizes`                           |
| `PreprocessingConfig` | Arguments passed to `pretab.Preprocessor` | `numerical_preprocessing`, `categorical_preprocessing`, `n_bins`, `scaling_strategy` |
| `TrainerConfig`       | Training loop and optimizer               | `max_epochs`, `batch_size`, `lr`, `patience`, `optimizer_type`                       |

All three are optional. If omitted, DeepTab creates default config objects internally.

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

If `trainer_config` is provided, `fit()` uses its `max_epochs`, `batch_size`, `val_size`, `shuffle`, `patience`, `monitor`, `mode`, and `checkpoint_path` unless overridden by explicit `fit()` arguments in legacy paths.

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
