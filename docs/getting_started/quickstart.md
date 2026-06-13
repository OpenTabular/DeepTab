# Quickstart

This guide shows you how to train your first DeepTab model in less than 5 minutes. By the end, you'll understand the basic workflow and be ready to apply it to your own data.

## Your first model

Let's start with a complete classification example using synthetic data:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from deeptab.models import MambularClassifier

# Generate synthetic data
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_classes=3,
    random_state=42,
)

# Convert to DataFrame (optional, but recommended)
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the model
model = MambularClassifier()

# Train the model
model.fit(X_train, y_train, max_epochs=50)

# Evaluate on test set
metrics = model.evaluate(X_test, y_test)
# Returns e.g. {"accuracy": 0.91, "auroc": 0.96, "log_loss": 0.28}
print(f"Test accuracy: {metrics['accuracy']:.3f}")

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(f"Predictions shape: {predictions.shape}")
print(f"Probabilities shape: {probabilities.shape}")
```

That's it! The model handles preprocessing, batching, device placement, and training automatically.

### What just happened?

1. **Data preparation**: Created a DataFrame with 10 features and 3 classes
2. **Train/test split**: Standard scikit-learn split
3. **Model initialization**: Created a Mambular classifier with default settings
4. **Training**: The `fit` method handles everything, including preprocessing, batching, GPU transfer, and optimization
5. **Evaluation**: The `evaluate` method returns a dict of metrics
6. **Prediction**: Standard `predict` and `predict_proba` methods

## Regression example

Regression follows the same workflow with a different model class:

```python
from sklearn.datasets import make_regression
from deeptab.models import FTTransformerRegressor

# Generate regression data
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    noise=0.1,
    random_state=42,
)

X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use a different architecture
model = FTTransformerRegressor()
model.fit(X_train, y_train, max_epochs=50)

# Evaluate (returns RMSE, MAE, R² for regression)
metrics = model.evaluate(X_test, y_test)
print(f"Test RMSE: {metrics['rmse']:.3f}")

# Predict continuous values
predictions = model.predict(X_test)
```

The only changes are the model class (`*Regressor`) and the interpretation of outputs.

## Using configs for customization

DeepTab separates hyperparameters into three independent config objects. Here's how to customize the model:

```python
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MambularClassifier

model = MambularClassifier(
    # Architecture hyperparameters
    model_config=MambularConfig(
        d_model=128,        # Hidden dimension (default: 64)
        n_layers=8,         # Number of Mamba blocks (default: 4)
        dropout=0.2,        # Dropout rate (default: 0.2)
    ),
    # Preprocessing strategy
    preprocessing_config=PreprocessingConfig(
        numerical_preprocessing="quantile",  # Options: standardization, quantile, minmax, ple
        n_bins=50,                           # For binning strategies
    ),
    # Training loop parameters
    trainer_config=TrainerConfig(
        max_epochs=100,     # Number of epochs (default: 100)
        lr=1e-3,            # Learning rate (default: 1e-4)
        batch_size=256,     # Batch size (default: 128)
        patience=15,        # Early stopping patience (default: 15)
        optimizer_type="AdamW",   # Any torch.optim class name (default: "Adam")
        weight_decay=1e-2,        # L2 regularisation (default: 1e-6)
        scheduler_type="ReduceLROnPlateau",  # LR scheduler (default)
        lr_patience=5,      # Epochs without improvement before LR is reduced
        lr_factor=0.5,      # LR reduction factor (default: 0.1)
    ),
)

model.fit(X_train, y_train)
```

Each config has sensible defaults. You only need to specify the parameters you want to change.

## Working with real data

Here's a more realistic example with mixed feature types:

```python
import pandas as pd
from deeptab.models import TabTransformerClassifier
from sklearn.model_selection import train_test_split

# Load your data (example structure)
data = pd.DataFrame({
    # Numerical features
    "age": [25, 32, 47, 51, 62, 28, 35, 44],
    "income": [35000, 48000, 72000, 55000, 91000, 42000, 58000, 68000],
    "years_experience": [2, 5, 15, 8, 25, 3, 7, 12],

    # Categorical features
    "city": ["NYC", "Boston", "Chicago", "Boston", "NYC", "Chicago", "NYC", "Boston"],
    "education": ["Bachelor", "Master", "PhD", "Master", "Bachelor", "Bachelor", "Master", "PhD"],
    "employment_type": ["full-time", "part-time", "full-time", "full-time", "retired", "full-time", "full-time", "full-time"],

    # Boolean feature (treated as categorical)
    "has_degree": [True, True, True, True, False, True, True, True],

    # Target
    "target": [0, 1, 1, 0, 1, 0, 1, 1],
})

# Separate features and target
X = data.drop(columns=["target"])
y = data["target"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model (handles mixed types automatically)
model = TabTransformerClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(metrics)

# Predict on new data
predictions = model.predict(X_test)
```

DeepTab automatically:

- Detects feature types from DataFrame dtypes
- Standardizes numerical features (`age`, `income`, `years_experience`)
- Encodes and embeds categorical features (`city`, `education`, `employment_type`, `has_degree`)
- Handles missing values if present

## Distributional regression

For uncertainty quantification, use `LSS` models:

```python
from deeptab.models import MambularLSS

# Same data as regression example
X_train, X_test, y_train, y_test = ...

# Initialize LSS model
model = MambularLSS()

# Fit with a parametric family
model.fit(X_train, y_train, family="normal", max_epochs=50)

# Predict distribution parameters
params = model.predict(X_test)

# For family="normal", params has shape (n_samples, 2) with columns [mean, std]
mean_predictions = params[:, 0]
std_predictions = params[:, 1]

# Generate 95% prediction intervals
lower_bound = mean_predictions - 1.96 * std_predictions
upper_bound = mean_predictions + 1.96 * std_predictions

print(f"Prediction intervals: [{lower_bound[0]:.2f}, {upper_bound[0]:.2f}]")
```

### Supported distributions

| Family          | Use case                          | Primary metric   |
| --------------- | --------------------------------- | ---------------- |
| `normal`        | Continuous unbounded values       | CRPS             |
| `lognormal`     | Strictly positive, multiplicative | Log-Normal NLL   |
| `studentt`      | Heavy-tailed continuous values    | CRPS             |
| `gamma`         | Positive continuous values        | Gamma deviance   |
| `beta`          | Values in (0, 1)                  | Beta Brier score |
| `tweedie`       | Zero-inflated positive values     | Tweedie deviance |
| `poisson`       | Count data                        | Poisson deviance |
| `zip`           | Count data with excess zeros      | Poisson deviance |
| `negativebinom` | Overdispersed counts              | NB deviance      |
| `dirichlet`     | Compositional (sum-to-1) vectors  | Dirichlet error  |
| `mog`           | Multimodal continuous values      | CRPS             |
| `quantile`      | Distribution-free percentiles     | Pinball loss     |

Each family automatically selects appropriate evaluation metrics via `model.evaluate()`.
See the [distributions reference](../api/distributions/index) and [metrics reference](../api/metrics/index) for the full API.

## Comparing models

Try different architectures by changing the import:

```python
from deeptab.models import (
    MambularClassifier,
    FTTransformerClassifier,
    TabTransformerClassifier,
    ResNetClassifier,
    MLPClassifier,
)

models = {
    "Mambular": MambularClassifier(),
    "FTTransformer": FTTransformerClassifier(),
    "TabTransformer": TabTransformerClassifier(),
    "ResNet": ResNetClassifier(),
    "MLP": MLPClassifier(),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train, max_epochs=50)
    metrics = model.evaluate(X_test, y_test)
    results[name] = metrics["accuracy"]
    print(f"{name}: {metrics['accuracy']:.3f}")

# Find best model
best_model = max(results, key=results.get)
print(f"\nBest model: {best_model} ({results[best_model]:.3f})")
```

## Using embeddings

If you have pre-computed embeddings (from text, images, etc.), pass them alongside tabular features:

```python
from deeptab.models import MambularClassifier
from sentence_transformers import SentenceTransformer

# Generate text embeddings
df["description"] = ["Product A is great", "Product B is okay", ...]
text_model = SentenceTransformer("all-MiniLM-L6-v2")
text_embeddings = text_model.encode(df["description"].tolist())

# Tabular features (excluding text column)
X = df.drop(columns=["description", "target"])
y = df["target"].values

# Train with both tabular and text embeddings
model = MambularClassifier()
model.fit(
    X_train,
    y_train,
    embeddings=text_embeddings,  # Added alongside tabular features
    max_epochs=50,
)
```

## Hyperparameter tuning

Use scikit-learn's search tools:

```python
from sklearn.model_selection import GridSearchCV
from deeptab.models import MambularClassifier

# Define search space
param_grid = {
    "model_config__d_model": [64, 128],
    "model_config__n_layers": [4, 6, 8],
    "trainer_config__lr": [1e-3, 5e-4],
}

# Grid search with cross-validation
search = GridSearchCV(
    estimator=MambularClassifier(),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=1,  # Each model uses GPU, so avoid parallel jobs
)

search.fit(X_train, y_train)

print(f"Best params: {search.best_params_}")
print(f"Best CV score: {search.best_score_:.3f}")

# Use best model
best_model = search.best_estimator_
test_metrics = best_model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_metrics['accuracy']:.3f}")
```

## Using experimental models

Experimental models may change without a deprecation cycle. Import them explicitly:

```python
from deeptab.models.experimental import TromptClassifier

# Same API as stable models
model = TromptClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)
```

See [Using experimental models](../tutorials/experimental) for more details.

## Saving and loading models

Save trained models for later use:

```python
# Train a model
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Save to disk
model.save("my_model.deeptab")

# Load later
from deeptab.models import MambularClassifier
loaded_model = MambularClassifier.load("my_model.deeptab")

# Use loaded model
predictions = loaded_model.predict(X_test)
```

Use the `.deeptab` extension for saved models. DeepTab accepts any extension but warns when a different one is used, so sticking to `.deeptab` keeps artifacts easy to recognise.

Note: `save()` writes a fitted estimator artifact, not just neural-network weights. The artifact includes the architecture/config, trained weights, fitted preprocessing state, feature schema and column order, task metadata such as classifier `classes_`, and package versions for debugging reloads across environments.

## Common patterns

### Stratified K-fold cross-validation

```python
from sklearn.model_selection import StratifiedKFold
from deeptab.models import MambularClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for train_idx, val_idx in skf.split(X, y):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    model = MambularClassifier()
    model.fit(X_train_fold, y_train_fold, max_epochs=50)
    metrics = model.evaluate(X_val_fold, y_val_fold)
    scores.append(metrics["accuracy"])

print(f"Mean accuracy: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
```

### Early stopping on validation set

```python
from deeptab.configs import TrainerConfig
from deeptab.models import MambularClassifier

# Provide explicit validation set
model = MambularClassifier(
    trainer_config=TrainerConfig(
        patience=10,    # Stop if monitored metric doesn't improve for 10 epochs
        monitor="val_loss",  # Metric to watch (default: "val_loss")
        mode="min",          # "min" to minimise, "max" to maximise
    )
)

model.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,  # Explicit validation set
    max_epochs=100,
)
```

```{tip}
`monitor` and `mode` apply to **both** early stopping and the LR scheduler.
Setting `monitor="val_auroc"` and `mode="max"` keeps them perfectly aligned,
so the scheduler reduces the learning rate in the same direction it is optimised.
```

### Optimizer and LR scheduler

Switch to a different optimizer or scheduler without subclassing anything:

```python
from deeptab.configs import TrainerConfig
from deeptab.models import FTTransformerClassifier

# AdamW with custom betas, a good default for transformer models
model = FTTransformerClassifier(
    trainer_config=TrainerConfig(
        optimizer_type="AdamW",
        lr=3e-4,
        weight_decay=1e-2,
        optimizer_kwargs={"betas": (0.9, 0.95)},
        # Bias and LayerNorm parameters get weight_decay=0
        no_weight_decay_for_bias_and_norm=True,
    )
)
```

Switch the LR schedule independently:

```python
# Cosine annealing, no plateau needed
model = FTTransformerClassifier(
    trainer_config=TrainerConfig(
        optimizer_type="AdamW",
        lr=3e-4,
        scheduler_type="CosineAnnealingLR",
        scheduler_kwargs={"T_max": 100, "eta_min": 1e-6},
    )
)

# Disable the scheduler entirely
model = FTTransformerClassifier(
    trainer_config=TrainerConfig(scheduler_type=None)
)
```

Inspect all available optimizers and schedulers:

```python
from deeptab.training.optimizers import available_optimizers
from deeptab.training.schedulers import available_schedulers

print(available_optimizers())
# ['adadelta', 'adagrad', 'adam', 'adamax', 'adamw', 'asgd', ...]

print(available_schedulers())
# ['constantlr', 'cosineannealinglr', 'cosineannealingwarmrestarts', ...]
```

Register a custom optimizer from a third-party library:

```python
from deeptab.training.optimizers import register_optimizer
from deeptab.configs import TrainerConfig

register_optimizer("muon", MyMuonOptimizer)

model = FTTransformerClassifier(
    trainer_config=TrainerConfig(optimizer_type="muon", lr=1e-3)
)
```

### Custom preprocessing for specific features

```python
from deeptab.configs import PreprocessingConfig

# Override defaults
config = PreprocessingConfig(
    numerical_preprocessing="quantile",  # Use quantile transform
    n_bins=50,                           # For binning strategies
    scaling_strategy="minmax",           # MinMax scaling after encoding
)

model = MambularClassifier(preprocessing_config=config)
model.fit(X_train, y_train, max_epochs=50)
```

## Debugging tips

### Check GPU usage

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Monitor training progress

DeepTab shows a progress bar by default. For richer per-epoch metrics, pass
`train_metrics`/`val_metrics` to `fit()`, or attach an experiment tracker through
`ObservabilityConfig` (MLflow or TensorBoard):

```python
from deeptab.core.observability import ObservabilityConfig

model = MambularClassifier(
    observability_config=ObservabilityConfig(verbosity=2, experiment_trackers=["tensorboard"]),
)
model.fit(X_train, y_train, max_epochs=50)
```

### Reduce batch size for memory errors

```python
from deeptab.configs import TrainerConfig

model = MambularClassifier(
    trainer_config=TrainerConfig(
        batch_size=64,  # Smaller batch size
    )
)
```

### Force CPU training

```python
model = MambularClassifier()
model.fit(X_train, y_train, accelerator="cpu")
```

## Next steps

Now that you've run your first models, explore:

- **[Core Concepts](../core_concepts/config_system)**: Deep dive into the config system, preprocessing, and distributional regression
- **[Tutorials](../tutorials/imbalance_classification)**: Complete end-to-end workflows for different tasks
- **[API Reference](../api/models/index)**: Full documentation of all models and configs
- **[FAQ](faq)**: Answers to common questions

For questions or issues, check the [FAQ](faq) or open an issue on [GitHub](https://github.com/OpenTabular/DeepTab/issues).
