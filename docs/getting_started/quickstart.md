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
print(f"Test accuracy: {metrics['accuracy']:.3f}")

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(f"Predictions shape: {predictions.shape}")
print(f"Probabilities shape: {probabilities.shape}")
```

That's it! The model handles preprocessing, batching, device placement, and training automatically.

### What just happened?

1. **Data preparation** — Created a DataFrame with 10 features and 3 classes
2. **Train/test split** — Standard scikit-learn split
3. **Model initialization** — Created a Mambular classifier with default settings
4. **Training** — The `fit` method handles everything: preprocessing, batching, GPU transfer, and optimization
5. **Evaluation** — The `evaluate` method returns a dict of metrics
6. **Prediction** — Standard `predict` and `predict_proba` methods

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

# Evaluate (returns RMSE, MAE, etc. for regression)
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
        numerical_preprocessing="quantile",  # Options: standard, quantile, minmax, ple
        n_bins=50,                           # For binning strategies
    ),
    # Training loop parameters
    trainer_config=TrainerConfig(
        max_epochs=100,     # Number of epochs (default: 100)
        lr=1e-3,            # Learning rate (default: 1e-4)
        batch_size=256,     # Batch size (default: 128)
        patience=15,        # Early stopping patience (default: 10)
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

| Family      | Use case                       |
| ----------- | ------------------------------ |
| `normal`    | Continuous unbounded values    |
| `poisson`   | Count data                     |
| `gamma`     | Positive continuous values     |
| `beta`      | Values in (0, 1)               |
| `student_t` | Heavy-tailed continuous values |

See the [API reference](../../api/models/index) for the complete list.

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
    X_embedding=text_embeddings,  # Added alongside tabular features
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
model.save("my_model.pkl")

# Load later
from deeptab.models import MambularClassifier
loaded_model = MambularClassifier.load("my_model.pkl")

# Use loaded model
predictions = loaded_model.predict(X_test)
```

Note: This saves the entire model including architecture, weights, and preprocessing state.

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
        patience=10,  # Stop if no improvement for 10 epochs
    )
)

model.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,  # Explicit validation set
    max_epochs=100,
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

DeepTab shows a progress bar by default. To see more detailed logging:

```python
from deeptab.configs import TrainerConfig

model = MambularClassifier(
    trainer_config=TrainerConfig(
        verbose=True,  # More detailed output
    )
)
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
from deeptab.configs import TrainerConfig

model = MambularClassifier(
    trainer_config=TrainerConfig(
        device="cpu",  # Explicitly use CPU
    )
)
```

## Next steps

Now that you've run your first models, explore:

- **[Core Concepts](../core_concepts/index)** — Deep dive into the config system, preprocessing, and distributional regression
- **[Tutorials](../tutorials/classification)** — Complete end-to-end workflows for different tasks
- **[API Reference](../../api/models/index)** — Full documentation of all models and configs
- **[FAQ](faq)** — Answers to common questions

For questions or issues, check the [FAQ](faq) or open an issue on [GitHub](https://github.com/OpenTabular/DeepTab/issues).
