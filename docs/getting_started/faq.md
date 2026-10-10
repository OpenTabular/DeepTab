# FAQ

Frequently asked questions about DeepTab and troubleshooting common issues.

## General

### What's the difference between DeepTab v1 and v2?

v2 keeps the `fit` / `predict` workflow but changes import paths and separates architecture, preprocessing, and training settings into dedicated config objects. It is not backward compatible with v1, which is no longer maintained. See [Migrating from v1 to v2](migration) for upgrade examples and the [Config System](../core_concepts/config_system) for current options.

### Which model should I use?

Start with a lightweight baseline, then compare models using the same validation split and metric.

| Goal                            | Try                  |
| ------------------------------- | -------------------- |
| Strong general-purpose baseline | `TabM` or `Mambular` |
| Many categorical features       | `TabTransformer`     |
| Lightweight baseline            | `MLP` or `ResNet`    |
| Uncertainty estimates           | any `LSS` variant    |
| Interpretability                | `NODE` or `NDTF`     |

These are starting points, not rules. For the detailed comparison by dataset size, feature mix, and compute budget, see the [Model Comparison](../model_zoo/comparison_tables) page.

### Do I need a GPU?

No. CPU is often sufficient for small datasets and lightweight models. GPUs can help with wide data, larger batches, and expensive attention or sequence models, but transfer overhead can make small workloads slower. Benchmark the same configuration on both devices rather than relying on a fixed dataset-size threshold. See the [Model Comparison](../model_zoo/comparison_tables) for model-specific guidance.

### How do I know if my GPU is being used?

Use two checks. First, see what hardware DeepTab can detect:

```python
from deeptab import print_hardware_info

print_hardware_info()
```

The report lists the CPU, any CUDA GPUs, the Apple Silicon MPS backend, and the `accelerator` DeepTab would pick by default. This only tells you what is _available_, not what a given run actually used.

To confirm what a fitted estimator is _really_ running on, inspect its runtime info after `fit`:

```python
model.fit(X, y)

info = model.runtime_info()
print(info["accelerator"])   # e.g. "CUDAAccelerator", "MPSAccelerator", "CPUAccelerator"
print(info["root_device"])   # e.g. "cuda:0", "mps:0", "cpu"
print(info["device"])        # device the model parameters live on
print(info["num_devices"])   # number of devices in use
```

`model.summary()` prints the same device, precision, and accelerator fields in a readable block.

```{warning}
By default DeepTab lets Lightning auto-select the best available accelerator, but an explicit `accelerator=` you pass to `fit()` always wins. If you accidentally pass `accelerator="cpu"`, training stays on the CPU even when a GPU is present, and `runtime_info()["accelerator"]` will report `"CPUAccelerator"`. Drop the argument (or set `accelerator="auto"`) to let DeepTab use the GPU.
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

```{note}
A column that is entirely missing (every row is `NaN`/`None`) has nothing for PreTab to learn a strategy from, so DeepTab fills it with a constant instead (`0` for numeric columns, `"missing"` for object/categorical columns) and emits a `DataWarning`. Consider dropping such columns before calling `fit()`.
```

### What if my column names have duplicates?

DeepTab requires unique column names and raises `DuplicateColumnsError` if `X` contains repeated labels:

```python
df.columns = ["age", "age"]  # duplicate name
model.fit(df, y, max_epochs=50)  # raises DuplicateColumnsError
```

Rename or drop the duplicated columns (e.g. `df.columns = [...]`) before calling `fit()`.

### Can I use NumPy arrays instead of DataFrames?

Yes. DeepTab accepts NumPy arrays and plain Python lists of lists, in addition to DataFrames:

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

Check [which device is actually being used](#how-do-i-know-if-my-gpu-is-being-used), then measure time per epoch while changing one setting at a time:

- Increase `batch_size` within the device's memory limit and check validation quality.
- Try a smaller model if the current architecture is unnecessarily expensive.
- Tune `lr` while watching validation loss; a higher rate is not always better.
- Use early stopping to avoid epochs after validation performance stops improving.
- Benchmark additional data-loader workers; they can add overhead for small in-memory datasets.

```python
from deeptab.configs import TrainerConfig

model = MambularClassifier(
    trainer_config=TrainerConfig(
        batch_size=256,
        lr=1e-3,
        patience=10,
    )
)

model.fit(
    X_train, y_train, accelerator="auto", max_epochs=100,
    dataloader_kwargs={"num_workers": 4},
)
```

If the loss diverges, see [Training is unstable](#training-is-unstable-loss-explodes).

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

Install DeepTab in the same Python environment used by your script or notebook:

```bash
python -m pip show deeptab
python -m pip install deeptab
```

If it is already installed, check your editor's selected interpreter or notebook kernel. See [Installation](installation) for environment setup.

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

The model's weights and input tensors must be on the same device. For prediction, you can request CPU inference explicitly:

```python
predictions = model.predict(X_test, device="cpu")
```

In custom loops, move each numerical, categorical, and embedding tensor to the model's device. A DeepTab batch contains nested lists or tuples, so calling `.to()` on the whole batch does not work.

## Choosing a model

### When should I use distributional regression (LSS)?

Use `LSS` models when you need:

- **Uncertainty quantification**: Know when predictions are confident vs uncertain
- **Prediction intervals**: Estimate ranges for future observations (e.g., 95% intervals)
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

For `family="normal"`, the second transformed parameter is the standard deviation, despite its legacy name `variance`. These are prediction intervals under the fitted normal-distribution assumption, not confidence intervals for the mean or guaranteed calibrated uncertainty. See [Training and Evaluation](../core_concepts/training_and_evaluation) for distributional workflows.

### Can I use my own custom architecture?

Yes, but it requires subclassing `BaseModel` (and pairing it with a `BaseModelConfig`
and one of `SklearnBaseClassifier` / `SklearnBaseRegressor` / `SklearnBaseLSS`). See
[Custom Models](../core_concepts/custom_models.md) for a full walkthrough.

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

Yes. Construct the configs explicitly before tuning their nested parameters:

```python
from sklearn.model_selection import GridSearchCV
from deeptab.configs import MambularConfig, TrainerConfig
from deeptab.models import MambularClassifier

search = GridSearchCV(
    estimator=MambularClassifier(
        model_config=MambularConfig(),
        trainer_config=TrainerConfig(max_epochs=50),
        random_state=42,
    ),
    param_grid={
        "model_config__d_model": [64, 128],
        "trainer_config__lr": [1e-3, 5e-4],
    },
    cv=5,
    n_jobs=1,
)
search.fit(X_train, y_train)
```

Nested keys such as `model_config__d_model` require the corresponding config object. Keep `n_jobs=1` when sharing one accelerator to avoid concurrent fits competing for its memory.

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

Yes. Pass `loss_fct` to `fit()`: either an `nn.Module` instance, which is used as-is, or, for classifiers, a registered loss name such as `"focal"`, `"bce"`, or `"cross_entropy"`, which is built and combined with any `class_weight` you set.

```python
import torch.nn as nn
from deeptab.models import MambularClassifier, MambularRegressor

model = MambularClassifier()

# A custom nn.Module loss
model.fit(X_train, y_train, loss_fct=nn.CrossEntropyLoss(label_smoothing=0.1))

# Or a registered loss by name (here combined with class weighting)
model.fit(X_train, y_train, loss_fct="focal", class_weight="balanced")

# Regressors accept an nn.Module loss the same way
regressor = MambularRegressor()
regressor.fit(X_train, y_train, loss_fct=nn.HuberLoss(delta=1.0))
```

```{note}
When `loss_fct` is an `nn.Module`, it is used as given and `class_weight` is ignored. Registered loss names and `class_weight` are classifier-only; regressors take an `nn.Module` loss directly. The fitted loss, including any class weights, is saved with the model and restored on `load()`.
```

### How do I extract learned features?

Use `encode()` on a fitted FTTransformer, TabTransformer, SAINT, Mambular, MambAttention, or TabulaRNN estimator, including their LSS variants. Encoding requires both an embedding layer and a supported contextualizing block; MLP does not support it, even with `use_embeddings=True`.

The method runs the backbone in evaluation mode with gradients disabled and returns the unpooled token sequence as a tensor of shape `(n_samples, n_tokens, hidden_dim)`. The token count includes a CLS token when enabled and any external embedding tokens. The final dimension is the backbone's output width, which can differ from `d_model` for recurrent models. The model stays in evaluation mode after the call.

```python
from deeptab.models import FTTransformerClassifier

model = FTTransformerClassifier()
model.fit(X_train, y_train, max_epochs=50)

tokens = model.encode(X_test)
print(tokens.shape)

features = tokens.mean(dim=1).cpu().numpy()
```

For clustering or another estimator that expects a two-dimensional matrix, choose an explicit pooling method, such as the token mean above. If a CLS token is enabled, you can instead select its configured prepend or append position. `encode()` itself does not pool or apply the prediction head.

Classifier and regressor wrappers accept external embeddings with `model.encode(X_test, embeddings=...)`; supply them again if they were used at fit time, keeping the rows aligned. LSS retains `encode(X_test, batch_size=64)`, with `batch_size` also accepted as the second positional argument.

```{note}
Evaluation mode disables dropout but does not remove cross-row interactions. Models such as SAINT can produce different tokens when batch composition or `batch_size` changes, even in evaluation mode. Exact reproducibility across devices or numerical backends is not guaranteed.
```

## Still have questions?

If your question isn't answered here:

1. Check the [Core Concepts](../core_concepts/config_system) guide
2. Browse the [Tutorials](../tutorials/imbalance_classification)
3. Search [GitHub issues](https://github.com/OpenTabular/DeepTab/issues)
4. Open a new issue on GitHub
