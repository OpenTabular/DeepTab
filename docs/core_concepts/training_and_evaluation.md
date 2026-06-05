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

Classification splits are stratified automatically. Regression splits are random.

---

## Preprocessing

DeepTab delegates tabular preprocessing to `pretab.Preprocessor` and converts the processed output into PyTorch tensors through `TabularDataModule`.

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
    numerical_preprocessing="quantile",
    categorical_preprocessing="int",
    n_bins=50,
    scaling_strategy="standard",
)
```

| Field                                        | Purpose                                                       |
| -------------------------------------------- | ------------------------------------------------------------- |
| `numerical_preprocessing`                    | Transform strategy: `"standard"`, `"quantile"`, `"ple"`, etc. |
| `categorical_preprocessing`                  | Encoding strategy: `"int"`, `"one-hot"`, etc.                 |
| `n_bins`                                     | Bins for binned / PLE-style transforms.                       |
| `scaling_strategy`                           | Optional post-transform scaling.                              |
| `binning_strategy`, `use_decision_tree_bins` | How bin edges are built.                                      |
| `n_knots`, `degree`, `spline_implementation` | Spline preprocessing controls.                                |

Practical starting points:

| Data condition                      | Config                                                          |
| ----------------------------------- | --------------------------------------------------------------- |
| Clean continuous features           | `PreprocessingConfig(numerical_preprocessing="standard")`       |
| Skewed / heavy-tailed columns       | `PreprocessingConfig(numerical_preprocessing="quantile")`       |
| Nonlinear numeric effects           | `PreprocessingConfig(numerical_preprocessing="ple", n_bins=50)` |
| Integer IDs alongside true numerics | Convert ID columns to pandas `category` before fitting.         |

### Validation and leakage

`TabularDataModule.preprocess_data()` fits the preprocessor on the **training split only**. Validation and prediction data are transformed with that fitted state — leakage from preprocessing statistics is avoided.

### Inspecting fitted feature metadata

```python
model.fit(X_train, y_train)

dm = model.data_module
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
    optimizer_type="Adam",
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

Early stopping monitors `TrainerConfig.monitor` (default `"val_loss"`). The best checkpoint is saved under `checkpoint_path` and loaded back after training automatically.

### Optimizer and scheduler

The optimizer is selected by name. `TaskModel` automatically attaches a `ReduceLROnPlateau` scheduler:

```python
TrainerConfig(
    optimizer_type="AdamW",
    lr=3e-4,
    weight_decay=1e-4,
    lr_patience=5,
    lr_factor=0.5,
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

### `set_seed` — standalone utility

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

### `seed_context` — scoped seeding

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

Default `evaluate()` outputs are task-specific:

```python
classification_metrics = classifier.evaluate(X_test, y_test)   # {"Accuracy": ...}
regression_metrics     = regressor.evaluate(X_test, y_test)    # {"Mean Squared Error": ...}
lss_metrics            = lss_model.evaluate(X_test, y_test)    # family-specific
```

Pass explicit metrics for reproducible reports:

```python
from sklearn.metrics import accuracy_score, f1_score, log_loss

metrics = classifier.evaluate(
    X_test, y_test,
    metrics={
        "accuracy":  (accuracy_score, False),
        "f1_macro":  (lambda y, p: f1_score(y, p, average="macro"), False),
        "log_loss":  (log_loss, True),
    },
)
```

### Score method

| Estimator  | Default `score()`       |
| ---------- | ----------------------- |
| Classifier | accuracy                |
| Regressor  | mean squared error      |
| LSS        | negative log-likelihood |

### Custom metrics during training

```python
from torchmetrics.classification import MulticlassAccuracy

model.fit(
    X_train, y_train,
    train_metrics={"train_acc": MulticlassAccuracy(num_classes=3)},
    val_metrics={"val_acc": MulticlassAccuracy(num_classes=3)},
)
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
- [Model Operations](model_operations)
- [sklearn API](sklearn_api)
