# Regression

This page covers regression-specific concepts, including continuous predictions, evaluation metrics, and handling different target distributions.

## Creating a regressor

Import any model with the `Regressor` suffix:

```python
from deeptab.models import MambularRegressor

model = MambularRegressor()
model.fit(X_train, y_train, max_epochs=100)
predictions = model.predict(X_test)
```

All stable models are available as regressors. See [Model Tiers](model_tiers) for the full list.

## Basic example

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from deeptab.models import FTTransformerRegressor

# Generate regression data
X, y = make_regression(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    noise=10,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = FTTransformerRegressor()
model.fit(X_train, y_train, max_epochs=50)

# Predict
predictions = model.predict(X_test)
# [12.34, 45.67, -23.45, ...]

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"MAE: {metrics['mae']:.3f}")
```

## Target preprocessing

Regression targets don't need special preprocessing, but you may want to apply transformations for better performance.

### Log transform for skewed targets

```python
import numpy as np

# Skewed target (e.g., income, prices)
y_train_log = np.log1p(y_train)  # log(1 + y) handles zeros

# Train on log-transformed target
model = MambularRegressor()
model.fit(X_train, y_train_log, max_epochs=50)

# Predict and inverse transform
predictions_log = model.predict(X_test)
predictions = np.expm1(predictions_log)  # Inverse: exp(y) - 1
```

### Standardize target

For very large or very small targets:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

model = MambularRegressor()
model.fit(X_train, y_train_scaled, max_epochs=50)

# Predict and inverse transform
predictions_scaled = model.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
```

### Clip outliers

For targets with extreme outliers:

```python
# Clip to reasonable range
y_train_clipped = np.clip(y_train, -100, 100)

model = MambularRegressor()
model.fit(X_train, y_train_clipped, max_epochs=50)
```

## Evaluation metrics

### Default: RMSE and MAE

```python
metrics = model.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"MAE: {metrics['mae']:.3f}")
print(f"Loss: {metrics['loss']:.3f}")  # MSE loss
```

### R² score

```python
score = model.score(X_test, y_test)
print(f"R² score: {score:.3f}")
```

### Custom metrics

Use `TrainerConfig`:

```python
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError
from deeptab.configs import TrainerConfig

cfg = TrainerConfig(
    metrics=[
        MeanSquaredError(),
        MeanAbsolutePercentageError(),
    ]
)

model = MambularRegressor(trainer_config=cfg)
model.fit(X_train, y_train, max_epochs=50)

metrics = model.evaluate(X_test, y_test)
# Includes all specified metrics
```

### scikit-learn metrics

Use after prediction:

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

predictions = model.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, predictions):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.3f}")
print(f"MAE: {mean_absolute_error(y_test, predictions):.3f}")
print(f"R²: {r2_score(y_test, predictions):.3f}")
```

## Output format

### predict()

Returns continuous values as floats:

```python
predictions = model.predict(X_test)
# [12.34, 45.67, -23.45, 78.90, ...]
print(predictions.dtype)  # float32
print(predictions.shape)  # (n_samples,)
```

### evaluate()

Returns dict of metrics:

```python
metrics = model.evaluate(X_test, y_test)
# {'rmse': 12.34, 'mae': 8.56, 'loss': 152.3}
print(type(metrics))  # dict
```

## Label shapes (v2.0)

DeepTab v2.0 enforces shape `(n_samples, 1)` for regression targets internally:

```python
# Your input (either shape works)
y_train = np.array([1.2, 3.4, 5.6, 7.8])  # Shape: (4,)
# Or
y_train = np.array([[1.2], [3.4], [5.6], [7.8]])  # Shape: (4, 1)

# Both work, handled automatically by estimator API
model = MambularRegressor()
model.fit(X_train, y_train, max_epochs=50)
```

## Handling different target distributions

### Normally distributed targets

Use default settings:

```python
model = MambularRegressor()
model.fit(X_train, y_train, max_epochs=50)
```

### Positive targets (prices, counts, durations)

Consider log transform:

```python
y_train_log = np.log1p(y_train)
model = MambularRegressor()
model.fit(X_train, y_train_log, max_epochs=50)

predictions_log = model.predict(X_test)
predictions = np.expm1(predictions_log)
```

Or use distributional regression with gamma family (see [Distributional Regression](distributional_regression)).

### Bounded targets (percentages, probabilities)

Transform to unbounded range:

```python
# Logit transform for (0, 1) range
from scipy.special import logit, expit

y_train_logit = logit(np.clip(y_train, 1e-6, 1-1e-6))
model = MambularRegressor()
model.fit(X_train, y_train_logit, max_epochs=50)

predictions_logit = model.predict(X_test)
predictions = expit(predictions_logit)
```

Or use distributional regression with beta family.

### Targets with outliers

Use quantile preprocessing:

```python
from deeptab.configs import PreprocessingConfig

cfg = PreprocessingConfig(numerical_preprocessing="quantile")
model = MambularRegressor(preprocessing_config=cfg)
model.fit(X_train, y_train, max_epochs=50)
```

Or clip targets:

```python
y_train_clipped = np.clip(y_train,
    np.percentile(y_train, 1),   # 1st percentile
    np.percentile(y_train, 99)   # 99th percentile
)
```

## Multivariate regression

For multiple continuous targets, train separate models:

```python
# Multi-output data
y1_train = ...  # Target 1
y2_train = ...  # Target 2

# Separate models
model1 = MambularRegressor()
model1.fit(X_train, y1_train, max_epochs=50)

model2 = MambularRegressor()
model2.fit(X_train, y2_train, max_epochs=50)

# Predict
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
```

## Residual analysis

Check model fit by analyzing residuals:

```python
predictions = model.predict(X_test)
residuals = y_test - predictions

# Plot residuals
import matplotlib.pyplot as plt

plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

# Check for patterns
# - Random scatter → good fit
# - Patterns → model misspecification
# - Funnel shape → heteroscedasticity (use distributional regression)
```

## Cross-validation

K-fold cross-validation for regression:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = []
for train_idx, val_idx in kf.split(X):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    model = MambularRegressor()
    model.fit(X_train_fold, y_train_fold, max_epochs=50)
    metrics = model.evaluate(X_val_fold, y_val_fold)
    rmse_scores.append(metrics["rmse"])

print(f"CV RMSE: {np.mean(rmse_scores):.3f} (+/- {np.std(rmse_scores):.3f})")
```

## Hyperparameter tuning

Regression-specific tuning:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    "model_config__d_model": randint(32, 256),
    "model_config__n_layers": randint(2, 10),
    "trainer_config__lr": uniform(1e-5, 1e-2),
}

search = RandomizedSearchCV(
    estimator=MambularRegressor(),
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring="neg_root_mean_squared_error",  # Or "r2", "neg_mean_absolute_error"
    random_state=42,
)

search.fit(X_train, y_train)
print(f"Best RMSE: {-search.best_score_:.3f}")
print(f"Best params: {search.best_params_}")
```

## Comparing architectures

```python
from deeptab.models import (
    MambularRegressor,
    FTTransformerRegressor,
    ResNetRegressor,
    MLPRegressor,
)

models = {
    "Mambular": MambularRegressor(),
    "FTTransformer": FTTransformerRegressor(),
    "ResNet": ResNetRegressor(),
    "MLP": MLPRegressor(),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train, max_epochs=50)
    metrics = model.evaluate(X_test, y_test)
    results[name] = metrics["rmse"]

# Best model
best = min(results, key=results.get)
print(f"Best: {best} (RMSE: {results[best]:.3f})")
```

## Feature importance

DeepTab models don't provide built-in feature importance. Use permutation importance:

```python
from sklearn.inspection import permutation_importance

# Wrap predict in a scorer
def scorer(X, y):
    preds = model.predict(X)
    return -mean_squared_error(y, preds)  # Negative for "higher is better"

# Compute importance
result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring=scorer,
)

# Plot
feature_names = [f"Feature {i}" for i in range(X.shape[1])]
indices = np.argsort(result.importances_mean)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(indices)), result.importances_mean[indices])
plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
```

## Prediction intervals

For uncertainty quantification, use distributional regression instead of standard regression:

```python
from deeptab.models import MambularLSS

# Train LSS model
model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)

# Get mean and std
params = model.predict(X_test)
mean = params[:, 0]
std = params[:, 1]

# 95% prediction intervals
lower = mean - 1.96 * std
upper = mean + 1.96 * std
```

See [Distributional Regression](distributional_regression) for details.

## Common patterns

### Ensemble predictions

Average predictions from multiple models:

```python
models = [
    MambularRegressor(),
    FTTransformerRegressor(),
    ResNetRegressor(),
]

# Train all
for model in models:
    model.fit(X_train, y_train, max_epochs=50)

# Predict and average
predictions = np.mean([
    model.predict(X_test) for model in models
], axis=0)
```

### Time series regression

For time series, ensure no data leakage:

```python
# Time-based split (no shuffle)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train
model = MambularRegressor()
model.fit(X_train, y_train, max_epochs=50)
```

Add lag features manually before passing to DeepTab:

```python
# Create lag features
df["lag_1"] = df["target"].shift(1)
df["lag_7"] = df["target"].shift(7)
df = df.dropna()

X = df.drop(columns=["target"])
y = df["target"].values
```

### Handling missing targets

Remove samples with missing targets:

```python
mask = ~np.isnan(y)
X_clean = X[mask]
y_clean = y[mask]

model = MambularRegressor()
model.fit(X_clean, y_clean, max_epochs=50)
```

## Best practices

1. **Check target distribution** before training
2. **Transform skewed targets** (log, sqrt) if needed
3. **Standardize very large targets** for stable training
4. **Use multiple metrics** (RMSE, MAE, R²)
5. **Analyze residuals** to check model fit
6. **Consider distributional regression** for uncertainty
7. **Use cross-validation** for reliable performance estimates

## Troubleshooting

### Poor R² score

- Check for outliers in target
- Try different preprocessing (quantile transform)
- Increase model capacity (larger d_model, more layers)
- Train longer (more epochs)

### Predictions all similar

- Model is predicting the mean (underfitting)
- Increase model capacity
- Decrease regularization (lower dropout)
- Check if features are informative

### Large residuals for some samples

- May indicate heteroscedasticity (varying noise)
- Use distributional regression to model varying uncertainty
- Check for subgroups with different relationships

### Training is unstable

- Standardize target values
- Reduce learning rate
- Enable gradient clipping (default)
- Check for NaN/Inf values in data

## Next steps

- **[Distributional Regression](distributional_regression)** — Predict full distributions for uncertainty
- **[Classification](classification)** — Classification-specific concepts
- **[Training and Evaluation](training_and_evaluation)** — Training loop details
- **[Examples: Regression](../../examples/regression)** — Complete workflows
