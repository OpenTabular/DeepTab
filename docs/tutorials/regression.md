# Regression Tutorial

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <a href="https://colab.research.google.com/github/basf/DeepTab/blob/main/docs/tutorials/notebooks/regression.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/basf/DeepTab/blob/main/docs/tutorials/notebooks/regression.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on-GitHub-181717?logo=github&logoColor=white" alt="View on GitHub"/>
  </a>
</div>

This tutorial demonstrates how to train regression models with DeepTab using the sklearn-compatible API.

```{tip}
Click the badges above to run this tutorial in Google Colab or view the notebook on GitHub!
```

## Basic workflow

### Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deeptab.models import MambularRegressor
```

### Generate data

We create a synthetic dataset with 1,000 samples and 5 numeric features. The target is a continuous value derived from a linear combination of features plus noise.

```python
np.random.seed(42)

n_samples, n_features = 1000, 5
X = np.random.randn(n_samples, n_features)
y = np.dot(X, np.random.randn(n_features)) + np.random.randn(n_samples)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
df["target"] = y
```

### Split data

```python
X = df.drop(columns=["target"])
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Train

Instantiate `MambularRegressor` with default settings and fit on the training data.

```python
model = MambularRegressor()
model.fit(X_train, y_train, max_epochs=50)
```

DeepTab automatically:

- Detects numerical vs categorical features
- Creates a validation split (20% by default)
- Enables early stopping
- Uses GPU if available

### Predict

Get continuous predictions:

```python
predictions = model.predict(X_test)
print(predictions[:10])
# [ 1.23 -0.45  2.11 -1.67  0.89 ...]
```

### Evaluate

```python
metrics = model.evaluate(X_test, y_test)
print(metrics)
# {'rmse': 1.234, 'mae': 0.987, 'loss': 1.523}
```

For sklearn compatibility, use `score()` to get R² score:

```python
r2 = model.score(X_test, y_test)
print(f"Test R²: {r2:.3f}")
```

### Save and load

```python
# Save trained model
model.save("my_regressor.pkl")

# Load later
from deeptab.models import MambularRegressor
loaded_model = MambularRegressor.load("my_regressor.pkl")
predictions = loaded_model.predict(X_test)
```

## Customization with configs

### Model architecture

```python
from deeptab.configs import MambularConfig

model_cfg = MambularConfig(
    d_model=256,          # Embedding dimension
    n_layers=8,           # Number of Mamba layers
    dropout=0.2,          # Dropout rate
    layer_norm_eps=1e-5,  # Layer norm epsilon
)

model = MambularRegressor(model_config=model_cfg)
model.fit(X_train, y_train, max_epochs=50)
```

### Preprocessing

```python
from deeptab.configs import PreprocessingConfig

prep_cfg = PreprocessingConfig(
    numerical_preprocessing="quantile",  # Transform to uniform distribution
    use_ple=True,                         # Piecewise Linear Encoding
    n_bins=50,                            # Number of bins for PLE
    categorical_preprocessing="ordinal",  # Ordinal encoding for cats
)

model = MambularRegressor(preprocessing_config=prep_cfg)
model.fit(X_train, y_train, max_epochs=50)
```

### Training loop

```python
from deeptab.configs import TrainerConfig

trainer_cfg = TrainerConfig(
    lr=5e-4,                          # Learning rate
    batch_size=256,                   # Batch size
    max_epochs=150,                   # Max epochs
    patience=20,                      # Early stopping patience
    lr_scheduler="cosine",            # Cosine annealing
    optimizer="adamw",                # AdamW optimizer
    weight_decay=1e-4,                # L2 regularization
    gradient_clip_val=1.0,            # Gradient clipping
)

model = MambularRegressor(trainer_config=trainer_cfg)
model.fit(X_train, y_train, max_epochs=trainer_cfg.max_epochs)
```

### Combine all configs

```python
model = MambularRegressor(
    model_config=model_cfg,
    preprocessing_config=prep_cfg,
    trainer_config=trainer_cfg,
)
model.fit(X_train, y_train, max_epochs=150)
```

## Target preprocessing

### Log transform for skewed targets

```python
# For strictly positive targets with right skew
y_log = np.log1p(y)  # log(1 + y) to handle zeros

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

model = MambularRegressor()
model.fit(X_train, y_train_log, max_epochs=50)

# Transform predictions back
predictions_log = model.predict(X_test)
predictions = np.expm1(predictions_log)  # exp(y) - 1

# Evaluate on original scale
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
print(f"RMSE: {rmse:.3f}, R²: {r2:.3f}")
```

### Standardize targets

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(
    X, y_scaled, test_size=0.2, random_state=42
)

model = MambularRegressor()
model.fit(X_train, y_train_scaled, max_epochs=50)

# Transform predictions back
predictions_scaled = model.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
```

### Clip outliers

```python
# Clip target to reasonable range
lower, upper = np.percentile(y, [1, 99])
y_clipped = np.clip(y, lower, upper)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_clipped, test_size=0.2, random_state=42
)

model = MambularRegressor()
model.fit(X_train, y_train, max_epochs=50)
```

## Integration with scikit-learn

### Cross-validation

```python
from sklearn.model_selection import cross_val_score

model = MambularRegressor()

# Negative MSE (sklearn convention)
scores = cross_val_score(
    model, X_train, y_train,
    cv=5,
    scoring="neg_mean_squared_error",
)

rmse_scores = np.sqrt(-scores)
print(f"CV RMSE: {rmse_scores.mean():.3f} (+/- {rmse_scores.std():.3f})")
```

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model_config__d_model": [128, 256],
    "model_config__n_layers": [4, 6, 8],
    "trainer_config__lr": [1e-4, 5e-4, 1e-3],
    "preprocessing_config__numerical_preprocessing": ["standard", "quantile", "minmax"],
}

model = MambularRegressor()

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=1,  # Use 1 for GPU models
    verbose=2,
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.3f}")

# Use best model
best_model = grid_search.best_estimator_
test_r2 = best_model.score(X_test, y_test)
print(f"Test R²: {test_r2:.3f}")
```

### RandomizedSearchCV

For faster hyperparameter search:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform

param_distributions = {
    "model_config__d_model": [64, 128, 256, 512],
    "model_config__n_layers": [2, 4, 6, 8],
    "model_config__dropout": uniform(0.0, 0.5),
    "trainer_config__lr": loguniform(1e-5, 1e-2),
    "trainer_config__batch_size": [64, 128, 256, 512],
}

model = MambularRegressor()

random_search = RandomizedSearchCV(
    model,
    param_distributions,
    n_iter=20,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=1,
    verbose=2,
    random_state=42,
)

random_search.fit(X_train, y_train)
```

## Advanced patterns

### Residual analysis

```python
import matplotlib.pyplot as plt

# Get predictions
predictions = model.predict(X_test)
residuals = y_test - predictions

# Plot residuals
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Residual plot
axes[0].scatter(predictions, residuals, alpha=0.5)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residual Plot")

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot")

plt.tight_layout()
plt.show()

# Check for patterns
print(f"Mean residual: {residuals.mean():.4f}")
print(f"Std residual: {residuals.std():.4f}")
```

### Feature importance with permutation

```python
from sklearn.inspection import permutation_importance

# Compute permutation importance
result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_squared_error",
)

# Sort by importance
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": result.importances_mean,
    "std": result.importances_std,
}).sort_values("importance", ascending=False)

print(importance_df)

# Plot
plt.figure(figsize=(8, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Permutation Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
```

### Multivariate regression

For multiple targets:

```python
# Create dataset with multiple targets
y_multi = np.column_stack([
    y,
    y + np.random.randn(len(y)),  # Correlated second target
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_multi, test_size=0.2, random_state=42
)

model = MambularRegressor()
model.fit(X_train, y_train, max_epochs=50)

predictions = model.predict(X_test)
print(predictions.shape)  # (n_samples, 2)

# Evaluate each target
for i in range(y_multi.shape[1]):
    r2 = r2_score(y_test[:, i], predictions[:, i])
    print(f"Target {i} R²: {r2:.3f}")
```

### Ensemble predictions

```python
# Train multiple models
models = []
for i in range(5):
    model = MambularRegressor()
    # Use different random seeds via train/val splits
    model.fit(X_train, y_train, max_epochs=50)
    models.append(model)

# Average predictions
predictions_list = [m.predict(X_test) for m in models]
ensemble_predictions = np.mean(predictions_list, axis=0)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test, ensemble_predictions))
r2 = r2_score(y_test, ensemble_predictions)
print(f"Ensemble RMSE: {rmse:.3f}, R²: {r2:.3f}")
```

### Time series splits

For temporal data:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

scores = []
for train_idx, val_idx in tscv.split(X):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    model = MambularRegressor()
    model.fit(X_train_fold, y_train_fold, max_epochs=50)

    score = model.score(X_val_fold, y_val_fold)
    scores.append(score)

print(f"Time series CV R²: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
```

### Mixed data types

```python
# Dataset with numerical and categorical features
df = pd.DataFrame({
    "age": np.random.randint(18, 80, size=1000),
    "income": np.random.randint(20000, 200000, size=1000),
    "city": np.random.choice(["NYC", "LA", "Chicago"], size=1000),
    "education": np.random.choice(["HS", "BS", "MS", "PhD"], size=1000),
    "experience_years": np.random.randint(0, 40, size=1000),
    "target": np.random.randn(1000) * 10000 + 50000,
})

X = df.drop(columns=["target"])
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Automatically handles both types
model = MambularRegressor()
model.fit(X_train, y_train, max_epochs=50)

metrics = model.evaluate(X_test, y_test)
print(metrics)
```

### With pre-computed embeddings

```python
# Add external embeddings (e.g., from text or images)
text_embeddings_train = np.random.randn(len(X_train), 128)
text_embeddings_test = np.random.randn(len(X_test), 128)

model = MambularRegressor()
model.fit(
    X_train, y_train,
    X_embedding=text_embeddings_train,
    max_epochs=50,
)

predictions = model.predict(X_test, X_embedding=text_embeddings_test)
```

## Using your own data

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from deeptab.models import MambularRegressor

# Load data
df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = MambularRegressor()
model.fit(X_train, y_train, max_epochs=100)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"MAE: {metrics['mae']:.3f}")
print(f"R²: {model.score(X_test, y_test):.3f}")

# Predict
predictions = model.predict(X_test)
```

## All stable regressors

Swap `MambularRegressor` for any class below — no other code changes needed:

| Class                     | Architecture                          | Best for                         |
| ------------------------- | ------------------------------------- | -------------------------------- |
| `MLPRegressor`            | Feedforward MLP                       | Fastest baseline                 |
| `ResNetRegressor`         | Residual MLP                          | Deeper networks                  |
| `FTTransformerRegressor`  | Feature-Tokenizer Transformer         | General-purpose strong baseline  |
| `TabTransformerRegressor` | Transformer on categorical embeddings | Categorical-heavy data           |
| `SAINTRegressor`          | Self + intersample attention          | Semi-supervised settings         |
| `TabMRegressor`           | Batch-ensembling MLP                  | Ensemble accuracy at low cost    |
| `TabRRegressor`           | Retrieval-augmented                   | Local similarity patterns        |
| `NODERegressor`           | Differentiable decision trees         | Gradient-boosting inductive bias |
| `NDTFRegressor`           | Neural decision tree forest           | Tree ensemble benefits           |
| `TabulaRNNRegressor`      | RNN / LSTM / GRU                      | Sequential feature interactions  |
| `MambularRegressor`       | Stacked Mamba SSM                     | Efficient sequence modeling      |
| `MambaTabRegressor`       | Single Mamba block                    | Lightweight Mamba variant        |
| `MambAttentionRegressor`  | Mamba + attention hybrid              | Local + global patterns          |
| `ENODERegressor`          | Extended NODE                         | NODE with feature embeddings     |
| `AutoIntRegressor`        | Attention-based interaction           | Explicit feature crossing        |

Example:

```python
from deeptab.models import (
    FTTransformerRegressor,
    ResNetRegressor,
    NODERegressor,
    MambularRegressor,
)

# Compare architectures
for ModelClass in [FTTransformerRegressor, ResNetRegressor, NODERegressor, MambularRegressor]:
    model = ModelClass()
    model.fit(X_train, y_train, max_epochs=50)
    r2 = model.score(X_test, y_test)
    print(f"{ModelClass.__name__}: R² = {r2:.3f}")
```

```{note}
All stable regressors share the same API. Import, instantiate, fit, predict — done.
```

## Next steps

- **Understand metrics** → Read [Regression](../core_concepts/regression) for evaluation details
- **Quantify uncertainty** → Try [Distributional Regression Tutorial](distributional) for prediction intervals
- **Optimize training** → See [Training and Evaluation](../core_concepts/training_and_evaluation)
- **Try classification** → Check out the [Classification Tutorial](classification)
- **Full config reference** → Browse [API docs](../api/configs/index)
