# FAQ

Frequently asked questions about DeepTab and troubleshooting common issues.

## General

### What's the difference between DeepTab v1 and v2?

v2.0 is a ground-up restructuring of DeepTab. The high-level estimator workflow stays familiar, but the package layout, configuration objects, and import paths have changed. Three things affect existing code:

1. **Import paths** were reorganised under the `deeptab` namespace.
2. **Config classes** dropped their `Default` prefix (`DefaultMambularConfig` is now `MambularConfig`) and settings are split across `MambularConfig` (architecture), `PreprocessingConfig` (feature handling), and `TrainerConfig` (training).
3. **Data modules** were renamed to `TabularDataModule` and `TabularDataset`; the old `Mambular*` aliases are deprecated.

The split-config API is the main thing you reach for day to day. In v1 every option was a flat keyword argument on the estimator; in v2 the same options live in dedicated config objects, while `fit`, `predict`, and `evaluate` behave exactly as before.

```python
# v1: settings passed as flat keyword arguments
model = MambularClassifier(d_model=128, n_layers=4, numerical_preprocessing="ple")
```

```python
# v2: settings grouped into focused config objects
from deeptab.configs import MambularConfig, PreprocessingConfig

model = MambularClassifier(
    model_config=MambularConfig(d_model=128, n_layers=4),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="ple"),
)
```

You only pass the configs you want to change; `MambularClassifier()` uses sensible defaults for all three.

```{important}
v2.0 is not backward compatible with v1, and v1 is no longer maintained. If you need to stay on v1, pin `deeptab<2.0`, but note that the v1 branch receives no bug fixes or security updates.
```

See the [Overview](overview) for the full v2 data API, and the [homepage](../homepage) for the complete list of new features.

### Which model should I use?

```{tip}
When in doubt, start with `MambularClassifier` or `MambularRegressor`.
```

Mambular tends to work well across a variety of tabular problems.

| Goal                            | Try                  |
| ------------------------------- | -------------------- |
| Strong general-purpose baseline | `TabM` or `Mambular` |
| Many categorical features       | `TabTransformer`     |
| Fastest baseline                | `MLP` or `ResNet`    |
| Uncertainty estimates           | any `LSS` variant    |
| Interpretability                | `NODE` or `NDTF`     |

These are starting points, not rules. For the detailed comparison by dataset size, feature mix, and compute budget, see the [Model Comparison](../model_zoo/comparison_tables) page.

### Do I need a GPU?

No, but it helps significantly for larger datasets and more complex architectures. The short answer:

- **MLP, ResNet, TabM, MambaTab**: train comfortably on CPU up to ~100K to 500K rows.
- **Mambular, TabulaRNN, TabTransformer, NODE**: CPU is fine up to ~10K to 20K rows; GPU recommended beyond that.
- **FTTransformer, AutoInt, MambAttention, ENODE, NDTF, TabR**: GPU recommended above ~5K to 10K rows.
- **SAINT**: GPU strongly recommended above ~2K rows (row attention makes every batch expensive).

For a full per-model breakdown including the cost driver for each architecture, see the [Model Zoo Comparison Tables](../model_zoo/comparison_tables) in the Model Zoo.

### How do I know if my GPU is being used?

Print the hardware DeepTab can see:

```python
from deeptab import print_hardware_info

print_hardware_info()
```

The report lists the CPU, any CUDA GPUs, the Apple Silicon MPS backend, and the `accelerator` DeepTab would pick by default. DeepTab uses the first available GPU automatically. If a GPU is listed but you're not seeing speedups, make sure you're training on a reasonably large dataset, since small batches may not benefit from GPU parallelism.

### Can I use DeepTab with PyTorch dataloaders?

```{note}
The high-level API uses `TabularDataModule` internally, but you can access `TabularDataset` directly for custom data loading.
```

Yes. The internal `TabularDataModule` creates PyTorch `DataLoader` instances. If you need custom data loading logic, you can use `TabularDataset` directly:

```python
from deeptab.data import TabularDataset
from torch.utils.data import DataLoader

dataset = TabularDataset(
    cat_features_list=[...],
    num_features_list=[...],
    embeddings_list=None,
    labels=labels,
)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
```

## Data and preprocessing

### What data types are supported?

DeepTab automatically handles:

- **Numerical**: `int`, `float` dtypes
- **Categorical**: `object`, `category`, `bool` dtypes
- **Embeddings**: Pass pre-computed embeddings via the `embeddings` parameter of `fit()`

### How do I handle missing values?

```{tip}
No manual imputation needed! DeepTab handles missing values automatically.
```

DeepTab handles missing values internally during preprocessing:

```python
# DataFrame with missing values
df = pd.DataFrame({
    "age": [25, np.nan, 47, 51],
    "city": ["NYC", "Boston", None, "Chicago"],
})

# Works without manual imputation
model = MambularClassifier()
model.fit(df, y, max_epochs=50)
```

The internal [PreTab](https://github.com/OpenTabular/PreTab) preprocessor imputes missing values as part of fitting, so you do not need a separate imputation step. The exact strategy follows the configured `PreprocessingConfig`; with the defaults it uses PreTab's built-in imputation for numerical and categorical features.

### Can I use NumPy arrays instead of DataFrames?

Yes. DeepTab accepts both:

```python
# NumPy arrays work
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, size=1000)

model = MambularClassifier()
model.fit(X, y, max_epochs=50)
```

However, DataFrames are recommended because they preserve column names and types, which helps with feature type detection and preprocessing.

### How do I tell DeepTab which columns are categorical?

DeepTab infers feature types from DataFrame dtypes:

```python
# Ensure categorical columns have the right dtype
df["city"] = df["city"].astype("category")
df["user_id"] = df["user_id"].astype("category")  # Numeric ID, but categorical

model = MambularClassifier()
model.fit(df, y, max_epochs=50)
```

If you're using NumPy arrays, all features are treated as numerical by default.

### What if I have text or image data?

DeepTab is designed for tabular data. For text or images:

1. Use a pre-trained encoder to generate embeddings
2. Pass embeddings via the `embeddings` parameter of `fit()`

```python
from sentence_transformers import SentenceTransformer

# Encode text to embeddings
text_model = SentenceTransformer("all-MiniLM-L6-v2")
text_embeddings = text_model.encode(df["description"].tolist())

# Pass embeddings alongside tabular features
X_tabular = df.drop(columns=["description", "target"])
model = MambularClassifier()
model.fit(X_tabular, y, embeddings=text_embeddings, max_epochs=50)
```

### Can I customize preprocessing per feature?

Not directly. `PreprocessingConfig` applies the same strategy to all numerical features. If you need per-feature preprocessing, apply it manually before passing to DeepTab:

```python
# Custom preprocessing
df["log_income"] = np.log1p(df["income"])
df["age_binned"] = pd.cut(df["age"], bins=5).astype("category")

# Then fit DeepTab
model = MambularClassifier()
model.fit(df, y, max_epochs=50)
```

## Training and performance

### How do I speed up training?

Start by checking what hardware DeepTab is actually using, then adjust the parts that matter most.

**1. Confirm you are on an accelerator.** Print the detected hardware:

```python
from deeptab import print_hardware_info

print_hardware_info()
```

If the recommended accelerator is `cpu` but you expect a GPU, install a CUDA-enabled PyTorch build (see the [installation guide](installation)). DeepTab uses the first available GPU automatically, but you can also force it explicitly:

```python
model.fit(X_train, y_train, accelerator="gpu", max_epochs=100)
```

**2. Use a batch size that keeps the accelerator busy.** GPUs and MPS only pay off with larger batches. Try 256 or more; on very small datasets (under ~1K rows) the CPU can be faster because of transfer overhead.

**3. Check the learning rate.** Training that crawls for many epochs is often a learning-rate problem, not a hardware one. The default is conservative; a slightly higher rate can converge in far fewer epochs. Raise it carefully and watch the loss.

**4. Lean on early stopping instead of a fixed epoch count**, and speed up data loading with extra workers.

```python
from deeptab.configs import TrainerConfig

model = MambularClassifier(
    trainer_config=TrainerConfig(
        batch_size=512,   # keep the accelerator busy
        lr=1e-3,          # raise from the default if convergence is slow
        patience=10,      # stop once the validation metric plateaus
    )
)

# num_workers is a DataLoader option, so pass it via dataloader_kwargs
model.fit(X_train, y_train, dataloader_kwargs={"num_workers": 4}, max_epochs=100)
```

```{note}
A GPU is not always faster. For small datasets or tiny batches the transfer overhead can outweigh the speedup, and the CPU may win. Benchmark both on your data.
```

```{warning}
Raising the learning rate too far makes training unstable and the loss can diverge. If that happens, lower it again and see [Training is unstable (loss explodes)](#training-is-unstable-loss-explodes).
```

### How do I use multiple GPUs?

Pass Lightning's multi-device arguments straight through `fit()`. Set `devices` to the number of GPUs (or a list of indices) and choose a `strategy` such as `"ddp"`:

```python
model = MambularClassifier()
model.fit(
    X_train, y_train,
    accelerator="gpu",
    devices=2,            # or [0, 1] to pick specific GPUs
    strategy="ddp",       # distributed data parallel
    max_epochs=100,
)
```

For finer control over the distributed setup, drive `TabularDataModule` with your own Lightning module (advanced usage).

### How do I use early stopping?

Early stopping is enabled by default. Adjust patience:

```python
from deeptab.configs import TrainerConfig

model = MambularClassifier(
    trainer_config=TrainerConfig(
        patience=15,  # Stop if no improvement for 15 epochs
    )
)
```

Provide an explicit validation set for better early stopping:

```python
model.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    max_epochs=100,
)
```

### How do I save a trained model?

Use the `.deeptab` extension. DeepTab warns when a different extension is used.

```python
# Save
model.save("my_model.deeptab")

# Load
from deeptab.models import MambularClassifier
loaded = MambularClassifier.load("my_model.deeptab")
predictions = loaded.predict(X_test)
```

The artifact includes weights, fitted preprocessor, feature schema, and task metadata.

### Can I resume training from a checkpoint?

Not directly through the estimator API. If you need this, consider using `TabularDataModule` with PyTorch Lightning's checkpointing directly.

### How do I monitor training metrics?

DeepTab shows a progress bar by default. For richer per-epoch metrics, pass
`train_metrics`/`val_metrics` dicts to `fit()`, or attach an experiment tracker
through `ObservabilityConfig`:

```python
from deeptab.core.observability import ObservabilityConfig

model = MambularClassifier(
    observability_config=ObservabilityConfig(verbosity=2, experiment_trackers=["tensorboard"]),
)
```

For fully custom metrics, use Lightning callbacks (advanced usage, see the Lightning docs).

## Errors and troubleshooting

### CUDA out of memory

```{warning}
GPU memory errors usually indicate batch size is too large for your GPU.
```

Reduce batch size:

```python
from deeptab.configs import TrainerConfig

model = MambularClassifier(
    trainer_config=TrainerConfig(batch_size=64)  # Smaller batch size
)
```

Or force CPU training by passing the Lightning accelerator to `fit()`:

```python
model = MambularClassifier()
model.fit(X_train, y_train, accelerator="cpu")
```

### ValueError: could not convert string to float

```{tip}
This usually means categorical features weren't properly detected. Explicitly set dtypes.
```

This happens when categorical features are not properly encoded. Ensure they have the right dtype:

```python
df["city"] = df["city"].astype("category")
```

Or check for unexpected non-numeric values in numerical columns.

### ImportError: No module named 'deeptab'

Ensure DeepTab is installed in the active environment:

```bash
pip list | grep deeptab
```

If not listed:

```bash
pip install deeptab
```

### AttributeError: 'TabularDataModule' object has no attribute 'embedding_feature_info'

This was a bug in early v2.0 pre-releases. Upgrade to v2.0.0 or later:

```bash
pip install --upgrade deeptab
```

### Training is unstable (loss explodes)

```{warning}
Exploding gradients indicate learning rate may be too high or data has extreme values.
```

Try reducing learning rate:

```python
from deeptab.configs import TrainerConfig

model = MambularClassifier(
    trainer_config=TrainerConfig(lr=1e-4)  # Lower learning rate
)
```

Or enable gradient clipping, which is off by default. Pass it to `fit()` as a Lightning trainer argument:

```python
model = MambularClassifier()
model.fit(X_train, y_train, gradient_clip_val=0.5)
```

### RuntimeError: Expected all tensors to be on the same device

```{note}
The high-level estimator API handles device management automatically. This error typically occurs only with custom training loops.
```

Ensure all tensors are on the same device:

```python
batch = batch.to("cuda")  # Move entire batch
```

The estimator API handles this automatically.

## Choosing a model

### What's the difference between Mambular and MambaTab?

Both use Mamba (State Space Model) blocks, but differ in how they process features:

- **Mambular**: Sequential model. Processes features one at a time in sequence, learning dependencies between features.
- **MambaTab**: Joint model. Applies Mamba to a concatenated representation of all features at once.

Mambular tends to work better for datasets where feature order matters or where you want to learn sequential dependencies.

### When should I use distributional regression (LSS)?

```{tip}
Use LSS models when you need uncertainty estimates, not just point predictions.
```

Use `LSS` models when you need:

- **Uncertainty quantification**: Know when predictions are confident vs uncertain
- **Prediction intervals**: Generate confidence bounds (e.g., 95% intervals)
- **Heteroscedastic noise**: Model varying noise levels across inputs
- **Risk-aware decisions**: Use full distributions for downstream optimization

Example:

```python
from deeptab.models import MambularLSS

model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)

# Get mean and std for each prediction
params = model.predict(X_test)
mean = params[:, 0]
std = params[:, 1]

# 95% prediction interval
lower = mean - 1.96 * std
upper = mean + 1.96 * std
```

### Can I use my own custom architecture?

Yes, but it requires subclassing `BaseTaskModel`. See the source code for examples of how to extend the base classes.

### Do experimental models work the same way as stable models?

Yes, the API is identical. The only difference is that experimental models may change without a deprecation cycle:

```python
from deeptab.models.experimental import TromptClassifier

# Same API as stable models
model = TromptClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

## Integration

### Can I use DeepTab with scikit-learn pipelines?

Yes:

```python
from sklearn.pipeline import Pipeline
from deeptab.models import MambularClassifier

pipeline = Pipeline([
    ("model", MambularClassifier()),
])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

Note: DeepTab does its own preprocessing, so additional preprocessing steps in the pipeline may be redundant.

### Does GridSearchCV work?

Yes:

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    estimator=MambularClassifier(),
    param_grid={
        "model_config__d_model": [64, 128],
        "trainer_config__lr": [1e-3, 5e-4],
    },
    cv=5,
)
search.fit(X_train, y_train)
```

Note: Set `n_jobs=1` in GridSearchCV if using GPU, as each model will try to use the GPU.

### Can I deploy DeepTab models?

Yes. For deployment, use `InferenceModel`. It validates the input schema and exposes only the inference surface, preventing accidental retraining in production:

```python
# Training environment
model.save("model.deeptab")

# Deployment environment
from deeptab import InferenceModel
model = InferenceModel.from_path("model.deeptab")

X_clean = model.validate_input(X_new)  # raises on schema mismatch
predictions = model.predict(X_clean)
```

See the [Inference Model](../core_concepts/inference) guide for the full deployment workflow.

## Advanced usage

### How do I access the underlying PyTorch model?

For most inspection needs, use the public helpers `model.summary()`,
`model.describe()`, and `model.parameter_table()`. They work once the model is
built or fitted and do not require touching internals.

```python
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)

print(model.summary())        # human-readable overview
info = model.describe()       # structured dict (architecture, task, params, ...)
```

If you need direct access for advanced work, the fitted Lightning module lives
in the private `model._task_model` attribute, and the raw `nn.Module`
architecture is `model._task_model.estimator`. These are internal and may change
between releases.

### Can I use custom loss functions?

Yes, for classifiers. Pass `loss_fct` to `fit()`: either an `nn.Module` instance, which is used as-is, or a registered loss name such as `"focal"`, `"bce"`, or `"cross_entropy"`, which is built and combined with any `class_weight` you set.

```python
import torch.nn as nn
from deeptab.models import MambularClassifier

model = MambularClassifier()

# A custom nn.Module loss
model.fit(X_train, y_train, loss_fct=nn.CrossEntropyLoss(label_smoothing=0.1))

# Or a registered loss by name (here combined with class weighting)
model.fit(X_train, y_train, loss_fct="focal", class_weight="balanced")
```

```{note}
When `loss_fct` is an `nn.Module`, it is used as given and `class_weight` is ignored. Regressors use the task default loss; to swap the loss for a regression model, drive `TabularDataModule` with a custom Lightning module.
```

### How do I extract learned features?

Use the public `encode()` method on a fitted model. It runs the backbone and returns dense representations as a tensor of shape `(n_samples, embedding_dim)`, which you can feed into clustering, similarity search, or another downstream model.

```python
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)

embeddings = model.encode(X_test)   # torch.Tensor, shape (n_samples, embedding_dim)
print(embeddings.shape)
```

If you also passed external `embeddings` at fit time, supply them again with `model.encode(X_test, embeddings=...)` so the rows stay aligned.

## Still have questions?

If your question isn't answered here:

1. Check the [Core Concepts](../core_concepts/config_system) guide
2. Browse the [Tutorials](../tutorials/imbalance_classification)
3. Search [GitHub issues](https://github.com/OpenTabular/DeepTab/issues)
4. Open a new issue on GitHub
