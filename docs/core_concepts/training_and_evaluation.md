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

### Retrieval-based models

[TabR](../model_zoo/stable/tabr) and [ModernNCA](../model_zoo/experimental/modernnca) use labeled training rows as retrieval candidates. Their training path must exclude a query's own row identity, even when the DataLoader shuffles or samples with replacement.

`TaskModel.setup("fit")` builds the candidate pool directly from the labeled `TabularDataset`, in dataset order. Candidate-aware training enables indexed batches of the form `(features, labels, row_indices)`; ordinary datasets still return `(features, labels)` by default. Before each training call, the trainer removes all batch IDs from the external candidate features and targets. The architecture then adds the batch as peer context and masks self identities, including repeated IDs within a batch.

The pool is not assembled from the training sampler, so replacement sampling and `drop_last` do not duplicate or omit candidate rows. Identity is independent of feature values: duplicate-valued records with different dataset indices are not automatically discarded.

Validation, test, and prediction use the stored training pool without adding evaluation labels. Custom candidate-aware DataModules must provide a labeled `TabularDataset`; direct architecture calls must provide a candidate pool disjoint from the query batch. See each model's notes for its retrieval and sampling requirements.

```{warning}
Estimator artifacts saved by TabR and ModernNCA include their transformed
training candidate bank and restore it before inference. Weights alone are not
enough: older artifacts without that bank require refitting from the original
training data. Candidate tensors contain training-derived features and labels;
apply the same access controls as for that data. See
{ref}`retrieval artifacts <retrieval-artifacts>` for compatible dependencies,
device limits and the executable round-trip example.
```

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

### Partial batches

Pass PyTorch loader options through `dataloader_kwargs`:

```python
model.fit(X_train, y_train, batch_size=16, dataloader_kwargs={"drop_last": True})
```

`drop_last=True` discards an incomplete **training** batch only. Validation,
test, and prediction loaders always retain partial batches, so checkpoint
selection uses the full validation set and predictions remain aligned with
every input row. A validation set smaller than `batch_size` still yields one
batch. Other loader options, such as `num_workers` and `pin_memory`, remain in
effect.

```{warning}
If the training partition itself is smaller than one batch, `drop_last=True`
still leaves the training loader empty. Reduce `batch_size` or disable
`drop_last` in that case.
```

### Early stopping and checkpointing

Early stopping and checkpoint selection use the effective `monitor` and `mode`:
explicit `fit()` arguments override `TrainerConfig`, whose defaults are
`"val_loss"` and `"min"`. The best checkpoint is
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
from deeptab.configs import TrainerConfig
from deeptab.metrics import AUROC
from deeptab.models import MLPClassifier

model = MLPClassifier(
    trainer_config=TrainerConfig(monitor="val_auroc", mode="max"),
)
model.fit(X_train, y_train, val_metrics={"auroc": AUROC()})
```

Declaring `val_metrics` makes `val_auroc` available to early stopping, the best
checkpoint, and the default plateau scheduler.

(monitoring-precedence)=

### Monitoring precedence

| Consumer                           | Monitored metric                                                                    | Direction                                                                               |
| ---------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Early stopping and best checkpoint | `fit(monitor=...)`, then `TrainerConfig.monitor`                                    | `fit(mode=...)`, then `TrainerConfig.mode`                                              |
| Plateau scheduler                  | Explicit `TrainerConfig.scheduler_monitor`, otherwise the effective general monitor | Explicit `TrainerConfig.scheduler_kwargs["mode"]`, otherwise the effective general mode |

The general defaults are `"val_loss"` and `"min"`, including when no
`TrainerConfig` is supplied. Scheduler-specific overrides affect only the
scheduler. `StepLR` and other non-plateau schedulers do not consume a monitored
metric or direction. Registered `ReduceLROnPlateau` aliases and subclasses follow
the plateau rules.

These rules apply independently on every call, including `rebuild=False` and
continuation of a loaded estimator. A previous call's overrides are not retained
implicitly. Fitting does not mutate the constructor's config or scheduler kwargs.

Declare a custom monitor through `train_metrics` or `val_metrics` on the first
fit or a rebuild. For example, `val_metrics={"auroc": AUROC()}` makes
`"val_auroc"` available. Continuation retains the existing metric objects;
replacing them requires a rebuild. See {ref}`epoch-metric-lifecycle` for input
aggregation and callback timing.

```{important}
Unknown or unavailable monitors and directions other than `"min"`/`"max"`
raise `ValueError` before replacing the fitted task, data module, or trainer.
There is no fallback from an explicitly selected metric to `"val_loss"`.
Validation monitors require validation to be enabled. A step-level plateau
scheduler can monitor `"train_loss"` or `"train_loss_step"`, but not an epoch
metric. A validation plateau scheduler requires a positive integer
`check_val_every_n_epoch` and a `scheduler_frequency` that is a multiple of it.
For validation every two epochs, use `scheduler_frequency=2`. Monitors are
checked against task losses and declared metrics, not arbitrary names logged
by user callbacks.
```

**Runnable scheduler-only override:**

```python
from tempfile import TemporaryDirectory

import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from deeptab.configs import MLPConfig, PreprocessingConfig, TrainerConfig
from deeptab.metrics import MeanSquaredError
from deeptab.models import MLPRegressor

features = np.random.default_rng(17).normal(size=(32, 3))
targets = features[:, 0]

with TemporaryDirectory() as checkpoint_dir:
    config = TrainerConfig(
        max_epochs=1, batch_size=8, checkpoint_path=checkpoint_dir,
        scheduler_monitor="val_mse", scheduler_kwargs={"mode": "min"},
    )
    model = MLPRegressor(
        model_config=MLPConfig(layer_sizes=[8]),
        preprocessing_config=PreprocessingConfig(numerical_method="standardization"),
        trainer_config=config, random_state=17,
    )
    options = dict(
        X_val=features[24:], y_val=targets[24:], accelerator="cpu",
        logger=False, enable_progress_bar=False, enable_model_summary=False,
    )
    model.fit(
        features[:24], targets[:24], val_metrics={"mse": MeanSquaredError()},
        monitor="train_loss", mode="max", **options,
    )
    callbacks = [
        callback for callback in vars(model._trainer)["callbacks"]
        if isinstance(callback, (EarlyStopping, ModelCheckpoint))
    ]
    assert all(callback.monitor == "train_loss" and callback.mode == "max" for callback in callbacks)
    scheduler = model._trainer.lr_scheduler_configs[0]
    assert scheduler.monitor == "val_mse" and scheduler.scheduler.mode == "min"
    assert config.monitor == "val_loss" and config.mode == "min"
    assert config.scheduler_kwargs == {"mode": "min"}

    model.fit(features[:24], targets[:24], rebuild=False, **options)
    callbacks = [
        callback for callback in vars(model._trainer)["callbacks"]
        if isinstance(callback, (EarlyStopping, ModelCheckpoint))
    ]
    assert all(callback.monitor == "val_loss" and callback.mode == "min" for callback in callbacks)
    scheduler = model._trainer.lr_scheduler_configs[0]
    assert scheduler.monitor == "val_mse" and scheduler.scheduler.mode == "min"
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
Only supported per-fit arguments override matching `TrainerConfig` fields.
Set optimizer selection and scheduler configuration, such as `optimizer_type`,
`optimizer_kwargs`, `scheduler_type`, and `scheduler_kwargs`, in `TrainerConfig`,
not directly in `fit()`. Additional Lightning trainer options can be passed
through `fit()` as described below.
```

```python
from deeptab.configs import TrainerConfig
from deeptab.metrics import AUROC
from deeptab.models import MLPClassifier

model = MLPClassifier(trainer_config=TrainerConfig(max_epochs=100, lr=1e-3))

# Override training settings just for this run.
model.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    max_epochs=50,          # overrides TrainerConfig(max_epochs=100)
    batch_size=256,
    patience=10,
    val_metrics={"auroc": AUROC()},
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

```{warning}
Classifier continuation with `rebuild=False` preserves the existing loss and sampler.
Supplying `class_weight`, `loss_fct`, `balanced_sampler=True`, or `sample_weight`
raises `ValueError` instead of silently ignoring the request. Use `rebuild=True`
to apply new settings; this constructs a fresh model and does not preserve its
trained weights. The same rule applies when `rebuild=None` continues a model
after `pretrain()`.
```

### LSS-only arguments

Distributional (`*LSS`) estimators accept a `family` argument in `fit()` that
selects the output distribution:

```python
from deeptab.models import MLPLSS

model = MLPLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
```

Pass `distributional_kwargs` for distribution constructor settings, separately
from Lightning Trainer options:

```python
model.fit(
    X_train, y_train,
    family="quantile",
    distributional_kwargs={"quantiles": [0.1, 0.5, 0.9]},
    max_epochs=50,
)
```

Other examples include `{"p": 1.7}` for `"tweedie"` and
`{"n_components": 3}` for `"mog"`. The resolved settings survive estimator
save/load. Categorical output width is inferred from the training labels;
Dirichlet width is inferred from a target matrix of shape `(n_samples, K)`.
Dirichlet targets must be positive proportions whose rows sum to one.

Targets for `gamma`, `inversegamma`, and `lognormal` must be strictly positive.
Zero or negative targets raise `DataError` before preprocessing or model construction.

LSS `fit()` seeds Python, NumPy, and PyTorch using the resolved `random_state`
before model construction. Equal seeds support repeatable fits on the same
environment, but do not guarantee identical results across devices or versions.

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

## Hyperparameter Search and Final Fitting

`optimize_hparams()` fits a baseline, evaluates the requested trial budget, and
then rebuilds and fits the best successful configuration before returning.
You can call `predict()` immediately; another `fit()` is optional retraining,
not a requirement for using the tuned model. Budget for up to `time + 2` fits.
The final fit retains the supplied validation split, task-specific options,
and LSS family, but does not retain a trial's pruning threshold.

Failed trials are never eligible to win. Model-construction errors, training
errors, and nonfinite validation losses count as failures; all-failed searches
raise `RuntimeError`. Negative validation losses are valid, and pruning adds a
positive tolerance to the best loss instead of multiplying a negative loss
into a stricter threshold.

Pruning uses zero-based training epoch indices. Sanity checks and standalone
validation do not enter training validation history. The search optimizer uses
the estimator's `random_state`; training randomness needs its own controls.
See [Hyperparameter Optimization](../tutorials/hpo) for examples and details.

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

| Estimator  | Default `score()` |
| ---------- | ----------------- |
| Classifier | accuracy          |
| Regressor  | R2                |
| LSS        | negative mean NLL |

(classification-metric-inputs)=

### Custom metrics during training

`train_metrics`/`val_metrics` accept two kinds of metric objects, each with its own
calling convention:

- **`torchmetrics.Metric` instances** stay on the training device and are called as
  `metric(preds, target)`.
- **`DeepTabMetric` instances** (the same classes used by `evaluate()`) receive
  detached, host-side NumPy arrays and are called as `metric(y_true, y_pred)`.

Plain callables use the same NumPy, targets-first convention as `DeepTabMetric`.
Metric dictionary keys determine log names, not input semantics.

Built-in `Accuracy`, `F1Score`, `AUROC`, `AUPRC`, `LogLoss`, `BrierScore`, and
`ExpectedCalibrationError` declare `input_space="probabilities"`. A binary
single-logit head receives sigmoid and produces ordered columns
`[P(class 0), P(class 1)]`; a two-/multi-class head receives softmax. Columns
follow `model.classes_`, while training targets use encoded class indices.
For declared ensembles, axis 1 is the member axis and logits are averaged
**before** activation, matching `predict_proba()`. Averaging member probabilities
is a different operation and is not the contract here.

Scalar TorchMetrics such as `BinaryAccuracy`, `BinaryAUROC`,
`MulticlassAccuracy`, and `MulticlassAUROC` are recognized through their
classification base types. Binary metrics receive a flat positive-class
probability vector; multiclass metrics receive ordered columns. Their targets
are flat integer indices. Other custom TorchMetrics retain their previous
tensor inputs unless explicitly declared. Non-scalar metric outputs are not
supported by Lightning's scalar logging interface.

`AUPRC` and `BrierScore` support binary scores or exactly two probability columns,
not multiclass probability matrices. Use a suitable multiclass metric instead.
Calling built-in metrics directly does not automatically convert logits.

```python
from tempfile import TemporaryDirectory

import numpy as np

from deeptab.configs import MLPConfig, PreprocessingConfig
from deeptab.metrics import Accuracy, BrierScore
from deeptab.models import MLPClassifier

X = np.random.default_rng(17).normal(size=(48, 3))
y = np.arange(48) % 2
model = MLPClassifier(
    model_config=MLPConfig(layer_sizes=[8]),
    preprocessing_config=PreprocessingConfig(numerical_method="standardization"),
    random_state=17,
)
with TemporaryDirectory() as checkpoint_path:
    model.fit(
        X[:32], y[:32], X_val=X[32:], y_val=y[32:],
        val_metrics={"accuracy": Accuracy(), "brier": BrierScore()},
        max_epochs=1, batch_size=16, accelerator="cpu", logger=False,
        checkpoint_path=checkpoint_path, enable_progress_bar=False,
        enable_model_summary=False,
    )
    probabilities = model.predict_proba(X[32:])
    expected_brier = np.mean((probabilities[:, 1] - y[32:]) ** 2)
    assert np.isclose(BrierScore()(y[32:], probabilities), expected_brier)
```

Custom callables with no declaration keep original classifier outputs,
including ensemble dimensions. To opt in, expose `input_space="probabilities"`
or `input_space="labels"` on the callable. Labels are encoded indices, not
original string or numeric class values. `input_space="logits"` preserves raw
outputs. For example:

```python
from deeptab.metrics import DeepTabMetric

class PositivePredictionRate(DeepTabMetric):
    name = "positive_rate"
    input_space = "labels"

    def __call__(self, y_true, y_pred):
        return float(np.mean(y_pred == 1))
```

`needs_raw=True` overrides adaptation, retaining original outputs and axes.
Ordinary regression callables are unchanged. LSS metrics receive family
parameters unless raw inputs are requested; ensemble member logits are reduced
before that parameter transform without collapsing the parameter columns.

(epoch-metric-lifecycle)=

### Epoch metric lifecycle

`train_<name>` and `val_<name>` are computed from a complete training epoch or
validation pass, not an average of nonlinear batch scores. Native TorchMetrics
are updated per batch and computed once at the end. NumPy callables receive
detached, concatenated predictions and targets once at that boundary, using the
same targets-first calling convention and declared input conversion.

Each stage resets before accumulation and after finalization. Sanity validation,
standalone/repeated validation, train/validation, and consecutive epochs do not
share accumulated observations. A repeated native object supplied under several
keys or in both stages is cloned for independent accumulators without changing
the caller's dictionaries. Metric callables should compute a scalar from their
arguments; mutable state hidden inside a custom callable remains its author's
responsibility.

Checkpoint and early-stopping callbacks observe the completed validation metric.
Use its ordinary `val_<name>` key for monitoring. Batch diagnostics must use
separate names; configured custom metrics are no longer invoked once per batch.
Loss logging retains separate step/epoch values and weights batch loss scalars
by the actual row count. This does not redefine the reduction of a custom or
class-weighted loss.

For example, two batches can each have AUROC 1.0 even though their combined
AUROC is 0.75. This standalone CPU example exercises actual Lightning hooks:

```python
import lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, default_collate

from deeptab.configs import MLPConfig
from deeptab.metrics import AUROC
from deeptab.training import TaskModel

class ScoreEcho(nn.Module):
    def __init__(self, config, feature_information, num_classes, lss):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, numerical, categorical, embeddings):
        return numerical[0] + self.anchor * 0

def collate_scores(rows):
    scores, targets = default_collate(rows)
    return ([scores], [], []), targets

dataset = TensorDataset(
    torch.tensor([[0.1], [0.2], [0.3], [0.4]]),
    torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
)
task = TaskModel(
    model_class=ScoreEcho, config=MLPConfig(layer_sizes=[8]),
    feature_information=({}, {}, {}), num_classes=2,
    val_metrics={"auc": AUROC()},
)
trainer = pl.Trainer(
    accelerator="cpu", logger=False, enable_checkpointing=False,
    enable_progress_bar=False, enable_model_summary=False,
)
result = trainer.validate(
    task, dataloaders=DataLoader(dataset, batch_size=2, collate_fn=collate_scores),
    verbose=False,
)
assert result[0]["val_auc"] == 0.75
```

Callable buffers retain at most 64 MiB per stage by default, including targets
and predictions for every callable metric. Copies are counted separately for
each metric; concatenation and distributed gathering need additional temporary
memory. Overflow raises `MemoryError` and releases that stage's buffers rather
than silently approximating the metric. Advanced direct `TaskModel` users can
set `metric_buffer_limit_bytes`; this is not an estimator `fit()` passthrough.
Native TorchMetrics avoid this host-buffer path, but exact AUROC and other
native metrics may retain observations in their own state.

For distributed execution, native metrics use their configured compute-time
synchronization; disabling `sync_on_compute` leaves rank-local results.
Callable inputs are gathered across ranks before computation,
and the retained-byte budget applies to the global input total. All ranks must
participate with the same metric configuration. Validation on padded distributed
samplers includes any repeated rows supplied by that sampler; this layer does
not deduplicate them. Gloo CPU tests cover two real ranks, including partial
final batches. CUDA/MPS and other backends still need their own platform evidence.

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
