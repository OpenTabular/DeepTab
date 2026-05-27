# Distributional Regression Tutorial

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <a href="https://colab.research.google.com/github/basf/DeepTab/blob/main/docs/tutorials/notebooks/distributional.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/basf/DeepTab/blob/main/docs/tutorials/notebooks/distributional.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on-GitHub-181717?logo=github&logoColor=white" alt="View on GitHub"/>
  </a>
</div>

Distributional regression (LSS models) predicts the full conditional distribution of the target rather than a single point estimate. This enables uncertainty quantification, prediction intervals, and handling of asymmetric or heavy-tailed distributions.

```{tip}
Click the badges above to run this tutorial in Google Colab or view the notebook on GitHub!
```

## What is distributional regression?

Traditional regression predicts a single value (point estimate):

```
y_pred = model.predict(X)  → Single number per sample
```

Distributional regression predicts **distribution parameters**:

```
params = lss_model.predict(X)  → Multiple parameters per sample
```

These parameters define a full probability distribution, allowing you to:

- Generate prediction intervals (e.g., 90% confidence)
- Extract specific quantiles (e.g., median, 5th percentile)
- Quantify aleatoric uncertainty
- Handle asymmetric target distributions

## Basic workflow

### Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deeptab.models import MambularLSS
```

### Generate data

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

Pass `family` to specify the output distribution:

```python
model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
```

### Predict distribution parameters

```python
# Get distribution parameters
params = model.predict(X_test)
print(params.shape)
# (200, 2)  → (n_samples, n_parameters)
# For normal: column 0 = mean, column 1 = log(std)

# Extract mean and std
mean = params[:, 0]
log_std = params[:, 1]
std = np.exp(log_std)

print(f"Sample 0: mean={mean[0]:.3f}, std={std[0]:.3f}")
```

### Generate prediction intervals

```python
from scipy import stats

# 90% prediction interval
alpha = 0.10
z = stats.norm.ppf(1 - alpha / 2)

lower = mean - z * std
upper = mean + z * std

# Check coverage
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"90% interval coverage: {coverage:.3f}")
# Should be close to 0.90
```

### Evaluate

```python
metrics = model.evaluate(X_test, y_test)
print(metrics)
# {'loss': -234.5}  → Negative log-likelihood (higher is better)
```

### Save and load

```python
model.save("my_lss_model.pkl")

from deeptab.models import MambularLSS
loaded_model = MambularLSS.load("my_lss_model.pkl")
params = loaded_model.predict(X_test)
```

## Distribution families

Choose the family based on your target distribution:

### Normal (continuous, symmetric)

For unbounded continuous targets with symmetric noise:

```python
model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)

params = model.predict(X_test)
mean = params[:, 0]
log_std = params[:, 1]
std = np.exp(log_std)
```

**Use when:** Temperature, financial returns, measurement errors

### Poisson (count data)

For non-negative integer counts:

```python
model = MambularLSS()
model.fit(X_train, y_train, family="poisson", max_epochs=50)

params = model.predict(X_test)
log_lambda = params[:, 0]
lambda_rate = np.exp(log_lambda)

# Mean and variance both equal lambda
print(f"Sample 0: lambda={lambda_rate[0]:.3f}")
```

**Use when:** Number of events, customer counts, click counts

### Gamma (positive continuous)

For strictly positive continuous targets with right skew:

```python
model = MambularLSS()
model.fit(X_train, y_train, family="gamma", max_epochs=50)

params = model.predict(X_test)
log_alpha = params[:, 0]  # Shape
log_beta = params[:, 1]   # Rate

alpha = np.exp(log_alpha)
beta = np.exp(log_beta)

mean = alpha / beta
variance = alpha / (beta ** 2)
```

**Use when:** Waiting times, insurance claims, income

### Beta (bounded [0, 1])

For targets constrained to the unit interval:

```python
# Rescale target to (0, 1)
y_scaled = (y - y.min()) / (y.max() - y.min())
y_scaled = y_scaled * 0.98 + 0.01  # Avoid exactly 0 and 1

model = MambularLSS()
model.fit(X_train, y_scaled_train, family="beta", max_epochs=50)

params = model.predict(X_test)
log_alpha = params[:, 0]
log_beta = params[:, 1]

alpha = np.exp(log_alpha)
beta = np.exp(log_beta)

mean = alpha / (alpha + beta)
```

**Use when:** Proportions, probabilities, percentages

### Negative Binomial (overdispersed counts)

For count data with variance > mean:

```python
model = MambularLSS()
model.fit(X_train, y_train, family="negative_binomial", max_epochs=50)

params = model.predict(X_test)
log_mu = params[:, 0]      # Mean
log_alpha = params[:, 1]   # Dispersion

mu = np.exp(log_mu)
alpha = np.exp(log_alpha)

variance = mu + alpha * (mu ** 2)
```

**Use when:** Count data with extra variance (over-dispersed)

### Student's t (heavy tails)

For continuous targets with outliers:

```python
model = MambularLSS()
model.fit(X_train, y_train, family="student_t", max_epochs=50)

params = model.predict(X_test)
loc = params[:, 0]         # Location
log_scale = params[:, 1]   # Scale
log_df = params[:, 2]      # Degrees of freedom

scale = np.exp(log_scale)
df = np.exp(log_df)
```

**Use when:** Data with outliers, robust regression

## Customization with configs

### Model architecture

```python
from deeptab.configs import MambularConfig

model_cfg = MambularConfig(
    d_model=256,
    n_layers=8,
    dropout=0.2,
)

model = MambularLSS(model_config=model_cfg)
model.fit(X_train, y_train, family="normal", max_epochs=50)
```

### Preprocessing

```python
from deeptab.configs import PreprocessingConfig

prep_cfg = PreprocessingConfig(
    numerical_preprocessing="quantile",
    use_ple=True,
    n_bins=50,
)

model = MambularLSS(preprocessing_config=prep_cfg)
model.fit(X_train, y_train, family="normal", max_epochs=50)
```

### Training loop

```python
from deeptab.configs import TrainerConfig

trainer_cfg = TrainerConfig(
    lr=1e-3,
    batch_size=256,
    max_epochs=150,
    patience=20,
    lr_scheduler="reduce_on_plateau",
)

model = MambularLSS(trainer_config=trainer_cfg)
model.fit(X_train, y_train, family="normal", max_epochs=150)
```

## Advanced patterns

### Prediction intervals (symmetric)

For normal distribution:

```python
from scipy import stats

params = model.predict(X_test)
mean = params[:, 0]
std = np.exp(params[:, 1])

# Generate multiple interval levels
for confidence in [0.50, 0.68, 0.90, 0.95]:
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    lower = mean - z * std
    upper = mean + z * std

    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    print(f"{confidence*100:.0f}% interval: coverage = {coverage:.3f}")
```

### Prediction intervals (asymmetric)

For asymmetric distributions like gamma:

```python
from scipy.stats import gamma as gamma_dist

params = model.predict(X_test)
alpha = np.exp(params[:, 0])
beta = np.exp(params[:, 1])

# 90% prediction interval
lower = gamma_dist.ppf(0.05, alpha, scale=1/beta)
upper = gamma_dist.ppf(0.95, alpha, scale=1/beta)

coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"90% interval coverage: {coverage:.3f}")
```

### Quantile predictions

Extract specific quantiles:

```python
from scipy import stats

params = model.predict(X_test)
mean = params[:, 0]
std = np.exp(params[:, 1])

# Get median (50th percentile)
median = stats.norm.ppf(0.5, loc=mean, scale=std)

# Get 5th and 95th percentiles
q05 = stats.norm.ppf(0.05, loc=mean, scale=std)
q95 = stats.norm.ppf(0.95, loc=mean, scale=std)

print(f"Sample 0: P5={q05[0]:.2f}, P50={median[0]:.2f}, P95={q95[0]:.2f}")
```

### Visualizing predictions

```python
import matplotlib.pyplot as plt
from scipy import stats

# Get predictions for first 50 test samples
params = model.predict(X_test[:50])
mean = params[:, 0]
std = np.exp(params[:, 1])

# Plot point predictions with intervals
fig, ax = plt.subplots(figsize=(12, 6))

indices = np.arange(50)
ax.scatter(indices, y_test[:50], color="black", label="Actual", alpha=0.6)
ax.scatter(indices, mean, color="blue", label="Predicted mean", alpha=0.6)

# 90% intervals
z = stats.norm.ppf(0.95)
lower = mean - z * std
upper = mean + z * std

ax.fill_between(indices, lower, upper, alpha=0.3, color="blue", label="90% interval")

ax.set_xlabel("Sample")
ax.set_ylabel("Target")
ax.set_title("LSS Predictions with Uncertainty")
ax.legend()
plt.tight_layout()
plt.show()
```

### Visualizing distributions

```python
# Plot predicted distributions for 5 samples
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i in range(5):
    mean_i = mean[i]
    std_i = std[i]

    x_range = np.linspace(mean_i - 3*std_i, mean_i + 3*std_i, 100)
    pdf = stats.norm.pdf(x_range, loc=mean_i, scale=std_i)

    axes[i].plot(x_range, pdf)
    axes[i].axvline(y_test[i], color="red", linestyle="--", label="Actual")
    axes[i].axvline(mean_i, color="blue", linestyle="--", label="Mean")
    axes[i].set_title(f"Sample {i}")
    axes[i].legend()

plt.tight_layout()
plt.show()
```

### Coverage validation

Check if intervals are well-calibrated:

```python
from scipy import stats

params = model.predict(X_test)
mean = params[:, 0]
std = np.exp(params[:, 1])

# Test multiple confidence levels
confidence_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

results = []
for confidence in confidence_levels:
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    lower = mean - z * std
    upper = mean + z * std

    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    results.append((confidence, coverage))

# Plot calibration
plt.figure(figsize=(8, 6))
plt.plot([r[0] for r in results], [r[1] for r in results], marker="o", label="Observed")
plt.plot([0.5, 1.0], [0.5, 1.0], "r--", label="Perfect calibration")
plt.xlabel("Nominal coverage")
plt.ylabel("Empirical coverage")
plt.title("Prediction Interval Calibration")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Comparing with point-estimate regressor

```python
from deeptab.models import MambularRegressor

# Train point-estimate regressor
regressor = MambularRegressor()
regressor.fit(X_train, y_train, max_epochs=50)

# Train LSS model
lss_model = MambularLSS()
lss_model.fit(X_train, y_train, family="normal", max_epochs=50)

# Compare predictions
point_pred = regressor.predict(X_test)
lss_params = lss_model.predict(X_test)
lss_mean = lss_params[:, 0]

# Both should be similar
from sklearn.metrics import mean_squared_error, r2_score

print(f"Point regressor RMSE: {np.sqrt(mean_squared_error(y_test, point_pred)):.3f}")
print(f"LSS mean RMSE: {np.sqrt(mean_squared_error(y_test, lss_mean)):.3f}")

# But LSS also provides uncertainty
lss_std = np.exp(lss_params[:, 1])
print(f"Mean predicted std: {lss_std.mean():.3f}")
```

### Hyperparameter tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model_config__d_model": [128, 256],
    "model_config__n_layers": [4, 6],
    "trainer_config__lr": [5e-4, 1e-3],
}

# Note: family is fixed during fit, not a hyperparameter
model = MambularLSS()

# Define custom scorer (negative log-likelihood)
def neg_log_likelihood_scorer(estimator, X, y):
    metrics = estimator.evaluate(X, y)
    return -metrics["loss"]  # Higher is better

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring=neg_log_likelihood_scorer,
    n_jobs=1,
)

# fit requires family argument
class LSS_Wrapper:
    def __init__(self, family="normal", **kwargs):
        self.family = family
        self.model = MambularLSS(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y, family=self.family, max_epochs=50)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def get_params(self, deep=True):
        params = self.model.get_params(deep=deep)
        params["family"] = self.family
        return params

    def set_params(self, **params):
        if "family" in params:
            self.family = params.pop("family")
        self.model.set_params(**params)
        return self

wrapper = LSS_Wrapper(family="normal")
grid_search = GridSearchCV(wrapper, param_grid, cv=3, scoring=neg_log_likelihood_scorer)
grid_search.fit(X_train, y_train)
```

## Using your own data

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from deeptab.models import MambularLSS

# Load data
df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Choose appropriate family based on target distribution
# - Continuous symmetric → "normal"
# - Counts → "poisson" or "negative_binomial"
# - Positive continuous → "gamma"
# - Bounded [0,1] → "beta"

model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=100)

# Get distribution parameters
params = model.predict(X_test)

# Generate prediction intervals
from scipy import stats
mean = params[:, 0]
std = np.exp(params[:, 1])

lower = stats.norm.ppf(0.05, loc=mean, scale=std)
upper = stats.norm.ppf(0.95, loc=mean, scale=std)

coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"90% interval coverage: {coverage:.3f}")
```

## All stable LSS models

Swap `MambularLSS` for any class below — pass `family=` to `.fit()`:

| Class               | Architecture                          | Best for                         |
| ------------------- | ------------------------------------- | -------------------------------- |
| `MLPLSS`            | Feedforward MLP                       | Fastest baseline                 |
| `ResNetLSS`         | Residual MLP                          | Deeper networks                  |
| `FTTransformerLSS`  | Feature-Tokenizer Transformer         | General-purpose strong baseline  |
| `TabTransformerLSS` | Transformer on categorical embeddings | Categorical-heavy data           |
| `SAINTLSS`          | Self + intersample attention          | Semi-supervised settings         |
| `TabMLSS`           | Batch-ensembling MLP                  | Ensemble accuracy at low cost    |
| `TabRLSS`           | Retrieval-augmented                   | Local similarity patterns        |
| `NODELSS`           | Differentiable decision trees         | Gradient-boosting inductive bias |
| `NDTFLSS`           | Neural decision tree forest           | Tree ensemble benefits           |
| `TabulaRNNLSS`      | RNN / LSTM / GRU                      | Sequential feature interactions  |
| `MambularLSS`       | Stacked Mamba SSM                     | Efficient sequence modeling      |
| `MambaTabLSS`       | Single Mamba block                    | Lightweight Mamba variant        |
| `MambAttentionLSS`  | Mamba + attention hybrid              | Local + global patterns          |
| `ENODELSS`          | Extended NODE                         | NODE with feature embeddings     |
| `AutoIntLSS`        | Attention-based interaction           | Explicit feature crossing        |

Example:

```python
from deeptab.models import FTTransformerLSS, ResNetLSS, NODELSS

for ModelClass in [FTTransformerLSS, ResNetLSS, NODELSS]:
    model = ModelClass()
    model.fit(X_train, y_train, family="normal", max_epochs=50)
    metrics = model.evaluate(X_test, y_test)
    print(f"{ModelClass.__name__}: NLL = {metrics['loss']:.3f}")
```

```{note}
All stable LSS models share the same API and support all distribution families.
```

## Next steps

- **Understand distributions** → Read [Distributional Regression](../core_concepts/distributional_regression) for all distribution families
- **Try point estimates** → See [Regression Tutorial](regression) for standard regressors
- **Optimize training** → Check [Training and Evaluation](../core_concepts/training_and_evaluation)
- **Full config reference** → Browse [API docs](../api/configs/index)
