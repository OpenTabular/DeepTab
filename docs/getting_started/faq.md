# FAQ

Frequently asked questions about DeepTab and troubleshooting common issues.

## General

### What's the difference between DeepTab v1 and v2?

Version 2.0 introduces a fully typed data layer (`TabularDataset`, `TabularDataModule`, `FeatureSchema`, `TabularBatch`) that makes it easier to work with tabular data at a lower level. The high-level estimator API remains unchanged and is still the recommended interface for most users.

Key changes in v2.0:

- **Automatic stratification** for classification tasks
- **Typed batch containers** with device management
- **Feature schema tracking** with metadata
- **Consistent label shapes** across tasks
- Deprecated `MambularDataset`/`MambularDataModule` aliases (use `TabularDataset`/`TabularDataModule`)

```{important}
**Note on v1 support**: DeepTab v1 is no longer supported following the v2.0 release. The changes in package structure and API design were substantial enough that maintaining backward compatibility would have compromised the improvements in v2. If you're using v1 in production, we recommend planning a migration to v2. Pin your dependency to `deeptab<2.0` if you need to continue using v1, but be aware that no bug fixes or security updates will be provided for the v1 branch.
```

See the [Overview](overview) for details on the new data API.

### Which model should I use?

```{tip}
When in doubt, start with `MambularClassifier` or `MambularRegressor`.
```

Mambular tends to work well across a variety of tabular problems.

If you want to experiment:

- **Large datasets** → Try `TabM` or `FTTransformer` for efficiency
- **Many categorical features** → Try `TabTransformer` which focuses on categorical embeddings
- **Simple baseline** → Try `MLP` or `ResNet` for comparison
- **Interpretability** → Try `NODE` or `NDTF` (tree-based neural models)

Use [GridSearchCV](quickstart) to compare multiple architectures systematically.

### Do I need a GPU?

No, but it helps. DeepTab works on CPU, but training will be significantly faster on a GPU for larger datasets. For small datasets (< 10K samples), CPU training is usually acceptable.

### How do I know if my GPU is being used?

Check CUDA availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

DeepTab will automatically use the first available GPU. If CUDA is available but you're not seeing speedups, ensure you're training on a reasonably large dataset—small batches may not benefit from GPU parallelism.

### Can I use DeepTab with PyTorch dataloaders?

```{note}
The high-level API uses `TabularDataModule` internally, but you can access `TabularDataset` directly for custom data loading.
```

Yes. The internal `TabularDataModule` creates PyTorch `DataLoader` instances. If you need custom data loading logic, you can use `TabularDataset` directly:

```python
from deeptab.data import TabularDataset
from torch.utils.data import DataLoader

dataset = TabularDataset(
    cat_feature_list=[...],
    num_feature_list=[...],
    embedding_feature_list=None,
    y=labels,
)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
```

## Data and preprocessing

### What data types are supported?

DeepTab automatically handles:

- **Numerical**: `int`, `float` dtypes
- **Categorical**: `object`, `category`, `bool` dtypes
- **Embeddings**: Pass pre-computed embeddings via `X_embedding` parameter

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

The pretab preprocessor (used internally) applies median imputation for numerical features and mode imputation for categoricals by default.

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
2. Pass embeddings via the `X_embedding` parameter

```python
from sentence_transformers import SentenceTransformer

# Encode text to embeddings
text_model = SentenceTransformer("all-MiniLM-L6-v2")
text_embeddings = text_model.encode(df["description"].tolist())

# Pass embeddings alongside tabular features
X_tabular = df.drop(columns=["description", "target"])
model = MambularClassifier()
model.fit(X_tabular, y, X_embedding=text_embeddings, max_epochs=50)
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

```{tip}
Combine GPU acceleration with larger batch sizes and early stopping for fastest training.
```

Several options:

1. **Use a GPU** — Install CUDA-enabled PyTorch
2. **Increase batch size** — Larger batches are more efficient (if memory allows)
3. **Reduce epochs** — Use early stopping instead of fixed epochs
4. **Use multi-worker data loading** — Set `num_workers` in `TrainerConfig`

```python
from deeptab.configs import TrainerConfig

model = MambularClassifier(
    trainer_config=TrainerConfig(
        batch_size=512,      # Larger batch size
        num_workers=4,       # Parallel data loading
        patience=10,         # Early stopping
    )
)
```
```

### Training is slow on GPU

```{note}
GPUs need larger batch sizes to show speedup over CPU. Small batches or datasets may run faster on CPU.
```

Ensure you're using GPU:

```python
import torch
print(torch.cuda.is_available())  # Should be True
```

If True but still slow:

- **Small batches** — GPU efficiency requires larger batches (try 256+)
- **Small dataset** — For < 1K samples, CPU may be faster due to transfer overhead
- **CPU bottleneck** — Increase `num_workers` in `TrainerConfig` for faster data loading

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

```python
# Train and save
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)
model.save("my_model.pkl")

# Load later
from deeptab.models import MambularClassifier
loaded_model = MambularClassifier.load("my_model.pkl")
predictions = loaded_model.predict(X_test)
```

This saves the entire model including weights and preprocessing state.

### Can I resume training from a checkpoint?

Not directly through the estimator API. If you need this, consider using `TabularDataModule` with PyTorch Lightning's checkpointing directly.

### How do I monitor training metrics?

DeepTab shows a progress bar by default. For more detailed logging:

```python
from deeptab.configs import TrainerConfig

model = MambularClassifier(
    trainer_config=TrainerConfig(
        verbose=True,  # Detailed logging
    )
)
```

For custom metrics, use Lightning callbacks (advanced usage—see Lightning docs).

## Errors and troubleshooting

### `CUDA out of memory`

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

Or force CPU training:

```python
from deeptab.configs import TrainerConfig

model = MambularClassifier(
    trainer_config=TrainerConfig(device="cpu")
)
```

### `ValueError: could not convert string to float`

```{tip}
This usually means categorical features weren't properly detected. Explicitly set dtypes.
```

This happens when categorical features are not properly encoded. Ensure they have the right dtype:

```python
df["city"] = df["city"].astype("category")
```

Or check for unexpected non-numeric values in numerical columns.

### `ImportError: No module named 'deeptab'`

Ensure DeepTab is installed in the active environment:

```bash
pip list | grep deeptab
```

If not listed:

```bash
pip install deeptab
```

### `AttributeError: 'TabularDataModule' object has no attribute 'embedding_feature_info'`

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

Or enable stronger gradient clipping (default is already enabled at 1.0):

```python
from deeptab.configs import TrainerConfig

model = MambularClassifier(
    trainer_config=TrainerConfig(gradient_clip_val=0.5)  # Stronger clipping
)
```

### `RuntimeError: Expected all tensors to be on the same device`

```{note}
The high-level estimator API handles device management automatically. This error typically occurs only with custom training loops.
```

Ensure all tensors are on the same device:

```python
batch = batch.to("cuda")  # Move entire batch
```

The estimator API handles this automatically.

## Model-specific

### What's the difference between Mambular and MambaTab?

Both use Mamba (State Space Model) blocks, but differ in how they process features:

- **Mambular** — Sequential model. Processes features one at a time in sequence, learning dependencies between features.
- **MambaTab** — Joint model. Applies Mamba to a concatenated representation of all features at once.

Mambular tends to work better for datasets where feature order matters or where you want to learn sequential dependencies.

### When should I use distributional regression (LSS)?

```{tip}
Use LSS models when you need uncertainty estimates, not just point predictions.
```

Use `LSS` models when you need:

- **Uncertainty quantification** — Know when predictions are confident vs uncertain
- **Prediction intervals** — Generate confidence bounds (e.g., 95% intervals)
- **Heteroscedastic noise** — Model varying noise levels across inputs
- **Risk-aware decisions** — Use full distributions for downstream optimization

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

Yes. Save the model and load it in your deployment environment:

```python
# Training
model.save("model.pkl")

# Deployment
from deeptab.models import MambularClassifier
model = MambularClassifier.load("model.pkl")
predictions = model.predict(X_new)
```

Ensure the deployment environment has the same dependencies (PyTorch, DeepTab, etc.).

## Advanced usage

### How do I access the underlying PyTorch model?

The PyTorch model is stored in `model.model`:

```python
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Access PyTorch model
pytorch_model = model.model
print(pytorch_model)
```

This is a `TaskModel` instance (Lightning module) containing the architecture.

### Can I use custom loss functions?

Not directly through the estimator API. If you need custom losses, use `TabularDataModule` with a custom Lightning module.

### How do I extract learned features?

Access intermediate representations:

```python
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Get feature representations (before final classification layer)
features = model.model.encoder(batch)  # Requires using TabularDataset/DataModule directly
```

This is an advanced use case—see the source code for details.

### Can I use multiple GPUs?

DeepTab uses the first available GPU by default. For multi-GPU training, use Lightning's distributed strategies directly with `TabularDataModule` (advanced usage).

## Contributing and support

### How do I report a bug?

Open an issue on [GitHub](https://github.com/OpenTabular/DeepTab/issues) with:

- DeepTab version (`import deeptab; print(deeptab.__version__)`)
- Python version
- PyTorch version
- Minimal reproducible example
- Full error traceback

### How do I request a feature?

Open a feature request on [GitHub](https://github.com/OpenTabular/DeepTab/issues) describing:

- The use case
- Why existing features don't solve it
- Proposed API (if applicable)

### How do I contribute?

See the [Contributing guide](../developer_guide/contributing) for:

- Setting up the development environment
- Running tests
- Code style guidelines
- Submitting pull requests

### Where can I get help?

- Check this FAQ first
- Search [GitHub issues](https://github.com/OpenTabular/DeepTab/issues)
- Open a new issue for bugs or questions
- Join discussions on the GitHub repo

## Performance comparisons

### How does DeepTab compare to XGBoost?

It depends on the dataset:

- **Small datasets (< 1K samples)** — XGBoost often wins
- **Large datasets (> 10K samples)** — DeepTab competitive or better, especially with complex feature interactions
- **Categorical-heavy data** — XGBoost may be more efficient
- **Need for uncertainty** — DeepTab LSS models provide distributional predictions

Use both and compare on your specific data. DeepTab makes experimentation easy.

### Is DeepTab faster than training PyTorch manually?

No, DeepTab uses PyTorch under the hood. It provides convenience, not speed improvements. However, it does:

- Apply best practices (gradient clipping, early stopping, LR scheduling)
- Handle device management automatically
- Provide efficient data loading

So while not "faster", it helps you get to a working model more quickly.

## Still have questions?

If your question isn't answered here:

1. Check the [Core Concepts](../core_concepts/index) guide
2. Browse the [Tutorials](../tutorials/classification)
3. Search [GitHub issues](https://github.com/OpenTabular/DeepTab/issues)
4. Open a new issue on GitHub
