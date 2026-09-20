# Training and Evaluation

DeepTab estimators train PyTorch models through Lightning while exposing a scikit-learn style API. This page covers everything from preprocessing to training loop configuration, reproducibility, and evaluation.

---

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

Classification splits are stratified automatically. Regression splits are random. You can turn stratification off with `TrainerConfig(stratify=False)`; see the [Config System](config_system) page for the split settings.

---

## Preprocessing

DeepTab delegates tabular preprocessing to `pretab.Preprocessor` and converts the processed output into PyTorch tensors through `TabularDataModule`. For the mechanics of each representation, see [PreTab's own documentation](https://pretab.readthedocs.io/en/latest/index.html).

```{important}
Use pandas DataFrames for mixed tabular data. DataFrames preserve column names and dtypes, which lets the preprocessor separate numerical and categorical features reliably.
```

### Data flow

```text
raw X/y
  -> pretab.Preprocessor.fit(X_train)
  -> pretab.Preprocessor.transform(X_train / X_val / X_test)
  -> feature info dictionaries
  -> TabularDataset
  -> Lightning DataLoader
  -> DeepTab architecture
```

At prediction time the fitted preprocessor is reused, so new data follows exactly the same transformations learned during training.

### PreprocessingConfig

```python
from deeptab.configs import PreprocessingConfig

cfg = PreprocessingConfig(
    numerical_method="quantile",
    categorical_method="int",
    output_dim=50,
    scaling="standardization",
)
```

`numerical_method` sets the main numerical transform (`"standardization"`, `"quantile"`, `"ple"`, `"bspline"`, `"minmax"`, `"robust"`, `"box-cox"`, `"yeo-johnson"`, or `None`), `categorical_method` sets the categorical encoding (`"int"`, `"one-hot"`, `"pretrained"`), and `output_dim` sets the width for bin/knot/expansion-based representations. See the [Config System](config_system) page for the complete field reference, including `target_aware`/`placement_strategy` and the `preset` shortcut.

Practical starting points:

| Data condition                      | Config                                                       |
| ----------------------------------- | ------------------------------------------------------------ |
| Clean continuous features           | `PreprocessingConfig(numerical_method="standardization")`    |
| Skewed / heavy-tailed columns       | `PreprocessingConfig(numerical_method="quantile")`           |
| Nonlinear numeric effects           | `PreprocessingConfig(numerical_method="ple", output_dim=50)` |
| Integer IDs alongside true numerics | Convert ID columns to pandas `category` before fitting.      |

### Validation and leakage

`TabularDataModule.preprocess_data()` fits the preprocessor on the **training split only**. Validation and prediction data are transformed with that fitted state, which avoids leakage from preprocessing statistics.

### Inspecting fitted feature metadata

```python
model.fit(X_train, y_train)

dm = model._data_module
print(dm.num_feature_info)
print(dm.cat_feature_info)

schema = dm.schema
print(schema.total_numerical_dim)
print(schema.num_categorical_features)
```

### External embeddings

```python
model.fit(
    X_train, y_train,
    embeddings=train_text_embeddings,
    embeddings_val=val_text_embeddings,
    X_val=X_val, y_val=y_val,
)
predictions = model.predict(X_test, embeddings=test_text_embeddings)
```

Pass a list of arrays when using multiple embedding sources.

---

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
    optimizer_type="Adam",           # any registered optimizer name
    optimizer_kwargs=None,           # extra kwargs forwarded to the constructor
    scheduler_type="ReduceLROnPlateau",  # any registered scheduler name, or None
    scheduler_kwargs=None,           # extra kwargs for the scheduler
    scheduler_monitor=None,          # defaults to `monitor` when None
    scheduler_interval="epoch",      # "epoch" or "step"
    scheduler_frequency=1,
    no_weight_decay_for_bias_and_norm=False,
    checkpoint_path="model_checkpoints",
)
```

Device, precision, logging, and gradient-clipping are Lightning trainer arguments passed directly to `fit()`:

```python
model.fit(X_train, y_train, accelerator="gpu", devices=1, precision="32-true")
```

### Validation sets

If no validation data is supplied DeepTab creates an internal split. For research prefer explicit splits so every model sees identical data:

```python
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

### Early stopping and checkpointing

Early stopping monitors `TrainerConfig.monitor` (default `"val_loss"`). The best checkpoint is
loaded back after training automatically. Pass `checkpoint_path` directly to `fit()` to save
exactly there (`checkpoint_path/best_model.ckpt`); otherwise (including when `checkpoint_path`
only comes from `TrainerConfig`) each run writes to its own timestamped sub-directory
(`checkpoint_path/<YYYYMMDD_HHMMSS>_<run_id>/best_model.ckpt`, or `<run_dir>/checkpoints/` when
an `ObservabilityConfig` is active) so concurrent or repeated fits never overwrite each other.

### Optimizer and scheduler

The optimizer and LR scheduler are both registry-backed. Any registered name is
accepted; unknown names raise
`InvalidParamError` immediately with a list of
valid options.

**Default behaviour** (backward-compatible):

```python
from deeptab.configs import TrainerConfig

trainer_config = TrainerConfig(
    optimizer_type="Adam",          # default
    scheduler_type="ReduceLROnPlateau",  # default
    lr=1e-4,
    lr_patience=10,
    lr_factor=0.1,
    weight_decay=1e-6,
)
```

**Switch optimizer and pass extra kwargs:**

```python
TrainerConfig(
    optimizer_type="AdamW",
    lr=3e-4,
    weight_decay=1e-2,
    optimizer_kwargs={"betas": (0.9, 0.95)},
)
```

**Selective weight decay** (recommended for transformer models, where bias and `LayerNorm` / `BatchNorm` parameters are excluded):

```python
TrainerConfig(
    optimizer_type="AdamW",
    weight_decay=1e-2,
    no_weight_decay_for_bias_and_norm=True,
)
```

**Switch the scheduler:**

```python
# Cosine annealing
TrainerConfig(
    scheduler_type="CosineAnnealingLR",
    scheduler_kwargs={"T_max": 100, "eta_min": 1e-6},
)

# Disable entirely
TrainerConfig(scheduler_type=None)
```

**Align early stopping and scheduler to the same metric:**

```python
# Both early stopping AND ReduceLROnPlateau now track val_auroc in max mode
TrainerConfig(
    monitor="val_auroc",
    mode="max",
)
```

```{important}
Prior to v2.0 the scheduler always watched `val_loss` in `min` mode
regardless of `monitor` / `mode`. This caused the LR scheduler and early
stopping to track different metrics when using a maximise-mode metric such as
`val_auroc`. Both are now correctly aligned.
```

**Inspect and extend the registries:**

```python
from deeptab.training.optimizers import available_optimizers, register_optimizer
from deeptab.training.schedulers import available_schedulers, register_scheduler

print(available_optimizers())
# ['adadelta', 'adagrad', 'adam', 'adamax', 'adamw', 'asgd', ...]

print(available_schedulers())
# ['constantlr', 'cosineannealinglr', 'cosineannealingwarmrestarts', ...]

# Register a third-party optimizer
register_optimizer("muon", MyMuonOptimizer)
tc = TrainerConfig(optimizer_type="muon", lr=1e-3)

# Register a custom scheduler
register_scheduler("warmup_cosine", MyWarmupCosineScheduler)
tc = TrainerConfig(scheduler_type="warmup_cosine")
```

---

## Fit-time Parameters

`TrainerConfig` sets training defaults at construction time, but `fit()` also
accepts keyword arguments. A value passed to `fit()` overrides the matching
`TrainerConfig` field for that single run, which is convenient for quick
experiments without rebuilding the estimator.

```{note}
Anything you can configure through `TrainerConfig` can also be passed directly
to `fit()`. The `fit()` argument always wins when both are provided.
```

```python
from deeptab.configs import TrainerConfig
from deeptab.models import MLPClassifier

model = MLPClassifier(trainer_config=TrainerConfig(max_epochs=100, lr=1e-3))

# Override training settings just for this run.
model.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    max_epochs=50,          # overrides TrainerConfig(max_epochs=100)
    batch_size=256,
    patience=10,
    monitor="val_auroc",
    mode="max",
    lr=3e-4,
    random_state=42,
)
```

### Available `fit()` arguments

| Argument                       | Default  | Purpose                                                                                                                                                                                                                                                                               |
| ------------------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `X`, `y`                       | required | Training inputs and targets.                                                                                                                                                                                                                                                          |
| `val_size`                     | `None`   | Validation fraction when `X_val` is not given. Falls back to `TrainerConfig.val_size`, then `0.2`. Ignored if `X_val` is provided.                                                                                                                                                    |
| `X_val`, `y_val`               | `None`   | Explicit validation set. Skips the internal split when supplied.                                                                                                                                                                                                                      |
| `embeddings`, `embeddings_val` | `None`   | External feature embeddings for train and validation data.                                                                                                                                                                                                                            |
| `max_epochs`                   | `None`   | Maximum number of training epochs. Falls back to `TrainerConfig.max_epochs`, then `100`.                                                                                                                                                                                              |
| `random_state`                 | `101`    | Seed applied before model build and training for reproducibility.                                                                                                                                                                                                                     |
| `batch_size`                   | `None`   | Samples per gradient update. Falls back to `TrainerConfig.batch_size`, then `128`.                                                                                                                                                                                                    |
| `shuffle`                      | `None`   | Shuffle training data each epoch. Falls back to `TrainerConfig.shuffle`, then `True`.                                                                                                                                                                                                 |
| `patience`                     | `None`   | Early-stopping patience on the monitored metric. Falls back to `TrainerConfig.patience`, then `15`.                                                                                                                                                                                   |
| `monitor`                      | `None`   | Metric watched for early stopping and the LR scheduler. Falls back to `TrainerConfig.monitor`, then `"val_loss"`.                                                                                                                                                                     |
| `mode`                         | `None`   | Whether the monitored metric is minimised (`"min"`) or maximised (`"max"`). Falls back to `TrainerConfig.mode`, then `"min"`.                                                                                                                                                         |
| `lr`                           | `None`   | Learning rate. Falls back to `TrainerConfig.lr` when `None`.                                                                                                                                                                                                                          |
| `lr_patience`, `lr_factor`     | `None`   | LR-scheduler patience and reduction factor.                                                                                                                                                                                                                                           |
| `weight_decay`                 | `None`   | L2 penalty coefficient.                                                                                                                                                                                                                                                               |
| `checkpoint_path`              | `None`   | Directory for best-checkpoint saving and restore. Falls back to `TrainerConfig.checkpoint_path`, then `"model_checkpoints"`. Only an explicit `fit()` value is used as-is; a fallback value gets its own timestamped per-run sub-directory so concurrent/repeated runs never collide. |
| `train_metrics`, `val_metrics` | `None`   | Extra metrics logged during training/validation; accepts `torchmetrics.Metric` objects or `DeepTabMetric` instances (see [Custom metrics during training](#custom-metrics-during-training)).                                                                                          |
| `dataloader_kwargs`            | `{}`     | Extra keyword arguments forwarded to the PyTorch `DataLoader`.                                                                                                                                                                                                                        |
| `rebuild`                      | `None`   | Whether to rebuild the architecture even if one already exists. `None` rebuilds unless the model was warm-started via [`pretrain()`](../api/training/index), in which case it continues training the pretrained model instead. Pass `True`/`False` explicitly to override either way. |
| `**trainer_kwargs`             | -        | Forwarded to Lightning's `Trainer` (`accelerator`, `devices`, `precision`, ...).                                                                                                                                                                                                      |

```{note}
Arguments that fall back to `TrainerConfig` (`val_size`, `max_epochs`, `batch_size`,
`shuffle`, `patience`, `monitor`, `mode`, `checkpoint_path`, and the classifier-only
`stratify`) default to `None` in the `fit()` signature itself. `None` means "not
explicitly passed", so DeepTab can tell the difference between "you asked for the
default" and "you didn't say", and only then falls back to `TrainerConfig` (or the
built-in default when no `TrainerConfig` is set).
```

```{note}
`rebuild`'s `None` default uses that same "not explicitly passed" idea, but resolves
against the model's pretrain state rather than `TrainerConfig`: it rebuilds unless
`pretrain()` was just called. Passing `rebuild=True` after `pretrain()` is still
honored, but discards the pretrained embeddings, so DeepTab raises a `UserWarning`
when that happens.
```

### Classifier-only arguments

| Argument           | Default | Purpose                                                                                                                                                                              |
| ------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `stratify`         | `None`  | Stratify the validation split on `y` so train and validation keep the same class proportions. Falls back to `TrainerConfig.stratify`, then `True`. Ignored when `X_val` is provided. |
| `class_weight`     | `None`  | `"balanced"`, a `{label: weight}` mapping, or an array to reweight the loss for imbalance.                                                                                           |
| `loss_fct`         | `None`  | An `nn.Module` or registered loss name (`"focal"`, `"bce"`, `"cross_entropy"`).                                                                                                      |
| `balanced_sampler` | `False` | Draw class-balanced mini-batches with a `WeightedRandomSampler`.                                                                                                                     |
| `sample_weight`    | `None`  | Explicit per-row sampling weights. Takes precedence over `balanced_sampler`.                                                                                                         |

### LSS-only argument

Distributional (`*LSS`) estimators accept a `family` argument in `fit()` that
selects the output distribution:

```python
from deeptab.models import MLPLSS

model = MLPLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
```

### Lightning Trainer passthrough

Any keyword not listed above flows through `**trainer_kwargs` straight to
Lightning's `Trainer`, so device, precision, and gradient-clipping are set on
`fit()`:

```python
model.fit(
    X_train, y_train,
    accelerator="gpu",
    devices=1,
    precision="32-true",
    gradient_clip_val=1.0,
)
```

---

## Reproducibility

Getting the same result every time is essential for debugging, comparisons, and publication. DeepTab seeds every layer of randomness from data splitting through weight initialisation.

### Platform and device support

| Backend             | Condition                           | What is seeded                             |
| ------------------- | ----------------------------------- | ------------------------------------------ |
| CPU                 | always                              | `torch.manual_seed`                        |
| CUDA                | `torch.cuda.is_available()`         | `torch.cuda.manual_seed_all` + cuDNN flags |
| MPS (Apple Silicon) | `torch.backends.mps.is_available()` | `torch.mps.manual_seed`                    |

### The `random_state` parameter

Pass `random_state` to the estimator constructor. DeepTab calls `set_seed(random_state)` at the start of every `fit()` before `_build_model` and `trainer.fit`:

```python
from deeptab.configs import TrainerConfig
from deeptab.models import MLPRegressor

model = MLPRegressor(
    trainer_config=TrainerConfig(max_epochs=50),
    random_state=42,
)
model.fit(X_train, y_train)
```

Running the same script twice produces bit-identical predictions on the same hardware.

### `set_seed`: standalone utility

```python
from deeptab import set_seed

set_seed(42)
```

| Call                                        | Condition |
| ------------------------------------------- | --------- |
| `random.seed(seed)`                         | always    |
| `os.environ["PYTHONHASHSEED"] = str(seed)`  | always    |
| `numpy.random.seed(seed)`                   | always    |
| `torch.manual_seed(seed)`                   | always    |
| `torch.cuda.manual_seed_all(seed)`          | CUDA only |
| `torch.backends.cudnn.deterministic = True` | CUDA only |
| `torch.backends.cudnn.benchmark = False`    | CUDA only |
| `torch.mps.manual_seed(seed)`               | MPS only  |

For strict reproducibility on any accelerator:

```python
set_seed(42, deterministic=True)  # calls torch.use_deterministic_algorithms(True)
```

### `seed_context`: scoped seeding

```python
from deeptab import seed_context

with seed_context(42):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
```

The seed remains active for the rest of the process after the block exits.

### Recommended workflow

```python
from deeptab import set_seed
from sklearn.model_selection import train_test_split

SEED = 42
set_seed(SEED)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

model = MLPRegressor(
    trainer_config=TrainerConfig(max_epochs=100, lr=1e-3),
    random_state=SEED,
)
model.fit(X_train, y_train)
```

Pass the same integer to both `train_test_split` and `random_state`.

### Known sources of non-determinism

| Source                              | When                          | Mitigation                                               |
| ----------------------------------- | ----------------------------- | -------------------------------------------------------- |
| Non-deterministic CUDA/MPS ops      | GPU/MPS training              | `set_seed(seed, deterministic=True)`                     |
| Multi-worker DataLoaders            | `num_workers > 0`             | Keep `num_workers=0` or supply `worker_init_fn`          |
| Floating-point accumulation order   | Parallel reductions           | `deterministic=True`; accept small numerical differences |
| `PYTHONHASHSEED` in current process | Hash values before `set_seed` | Set in shell before launching Python                     |

---

## Evaluation

Default `evaluate()` outputs are task-specific. With no `metrics` argument the keys are the registry metric short names:

```python
classification_metrics = classifier.evaluate(X_test, y_test)   # {"accuracy": ..., "auroc": ..., "log_loss": ...}
regression_metrics     = regressor.evaluate(X_test, y_test)    # {"rmse": ..., "mae": ..., "r2": ...}
lss_metrics            = lss_model.evaluate(X_test, y_test)    # family-specific
```

Pass explicit metrics for reproducible reports. The dictionary values are callables with the signature `metric(y_true, y_pred)`; the built-in `DeepTabMetric` classes route probability-based metrics to `predict_proba` automatically:

```python
from deeptab.metrics import Accuracy, F1Score, LogLoss

metrics = classifier.evaluate(
    X_test, y_test,
    metrics={
        "accuracy": Accuracy(),
        "f1":       F1Score(),
        "log_loss": LogLoss(),
    },
)
```

### Score method

| Estimator  | Default `score()`       |
| ---------- | ----------------------- |
| Classifier | accuracy                |
| Regressor  | R2                      |
| LSS        | negative log-likelihood |

### Custom metrics during training

`train_metrics`/`val_metrics` accept two kinds of metric objects, each with its own
calling convention:

- **`torchmetrics.Metric` instances** stay on the training device and are called as
  `metric(preds, target)`.
- **`DeepTabMetric` instances** (the same classes used by `evaluate()`) receive
  detached, host-side NumPy arrays and are called as `metric(y_true, y_pred)`.

```python
from torchmetrics.classification import MulticlassAccuracy

model.fit(
    X_train, y_train,
    train_metrics={"train_acc": MulticlassAccuracy(num_classes=3)},
    val_metrics={"val_acc": MulticlassAccuracy(num_classes=3)},
)
```

```python
from deeptab.metrics import MeanAbsoluteError

model.fit(
    X_train, y_train,
    val_metrics={"val_mae": MeanAbsoluteError()},
)
```

---

## Observability

By default a fit is silent. To record what happens while a model trains, its hyperparameters, lifecycle events, and final metrics, attach an `ObservabilityConfig`. Each fit then writes a self-contained run directory, and optional trackers (TensorBoard, MLflow) build on the same configuration.

```python
from deeptab.core.observability import ObservabilityConfig

model = MLPRegressor(
    trainer_config=TrainerConfig(max_epochs=100),
    observability_config=ObservabilityConfig(
        experiment_name="baseline",
        structured_logging=True,
        experiment_trackers=["tensorboard"],
    ),
)
model.fit(X_train, y_train)
```

```{note}
Observability is entirely opt-in. Estimators created without an `ObservabilityConfig` emit nothing, so the training loop above behaves exactly as it did before. The dedicated [Observability](observability) guide covers the configuration reference, the run-directory layout, verbosity levels, and how to plug in your own logger.
```

---

## Troubleshooting

| Symptom                     | First checks                                                              |
| --------------------------- | ------------------------------------------------------------------------- |
| Training is slow            | Reduce `max_epochs`, increase `batch_size`, use GPU via Lightning kwargs. |
| Validation loss unstable    | Lower `lr`, increase `batch_size`, simplify preprocessing.                |
| Overfitting                 | Increase regularization, lower capacity, use explicit validation.         |
| Poor regression scale       | Transform the target manually and inverse-transform predictions.          |
| Unexpected metric names     | Pass explicit `metrics=` to `evaluate()`.                                 |
| Results differ between runs | Set `random_state` and call `set_seed` before data preparation.           |

---

## Next Steps

- [Config System](config_system)
- [Observability](observability)
- [Model Operations](model_operations)
- [sklearn API](sklearn_api)
