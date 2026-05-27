# Distributional Regression

Distributional regression (Location, Scale, and Shape modeling, or LSS) predicts the parameters of a full probability distribution rather than a single point estimate. This enables uncertainty quantification, prediction intervals, and better modeling of heteroscedastic noise.

## Why distributional regression?

Standard regression predicts a single value:

```python
# Point prediction
prediction = model.predict(X_test)[0]  # → 42.5
```

Distributional regression predicts a full distribution:

```python
# Distribution parameters
params = lss_model.predict(X_test)[0]  # → [mean=42.5, std=5.2]
```

This tells you both the expected value and the uncertainty.

### Use cases

- **Uncertainty quantification** — Know when predictions are confident vs uncertain
- **Prediction intervals** — Generate confidence bounds (e.g., 95% intervals)
- **Heteroscedastic noise** — Model varying noise levels across the input space
- **Risk-aware decisions** — Use full distribution for downstream optimization
- **Quantile predictions** — Extract specific percentiles for business requirements

## Creating an LSS model

Import any model with the `LSS` suffix:

```python
from deeptab.models import MambularLSS

model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=100)
params = model.predict(X_test)
```

All stable models are available as LSS variants.

## Basic example

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from deeptab.models import MambularLSS
import numpy as np

# Generate regression data
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    noise=10,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train LSS model
model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)

# Predict distribution parameters
params = model.predict(X_test)
# Shape: (n_samples, n_params)
# For family="normal": (n_samples, 2) with columns [mean, std]

mean_predictions = params[:, 0]
std_predictions = params[:, 1]

# Generate 95% prediction intervals
lower_bound = mean_predictions - 1.96 * std_predictions
upper_bound = mean_predictions + 1.96 * std_predictions

print(f"Prediction: {mean_predictions[0]:.2f}")
print(f"95% interval: [{lower_bound[0]:.2f}, {upper_bound[0]:.2f}]")
```

## Distribution families

LSS models support various parametric families. Choose based on your target's characteristics.

### Normal distribution

**Parameters:** mean (μ), standard deviation (σ)

**When to use:**

- Unbounded continuous targets
- Symmetric noise
- General-purpose default

```python
model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)

params = model.predict(X_test)
mean = params[:, 0]
std = params[:, 1]

# 95% prediction interval
lower = mean - 1.96 * std
upper = mean + 1.96 * std
```

### Poisson distribution

**Parameters:** rate (λ)

**When to use:**

- Count data (non-negative integers)
- Events per time period
- Low mean counts

```python
model = MambularLSS()
model.fit(X_train, y_train, family="poisson", max_epochs=50)

params = model.predict(X_test)
rate = params[:, 0]  # Expected count

# Variance equals mean in Poisson
std = np.sqrt(rate)
```

### Gamma distribution

**Parameters:** shape (α), rate (β)

**When to use:**

- Positive continuous values
- Right-skewed data
- Waiting times, durations

```python
model = MambularLSS()
model.fit(X_train, y_train, family="gamma", max_epochs=50)

params = model.predict(X_test)
shape = params[:, 0]
rate = params[:, 1]

# Mean and variance
mean = shape / rate
variance = shape / (rate ** 2)
```

### Beta distribution

**Parameters:** α, β

**When to use:**

- Values bounded in (0, 1)
- Probabilities, proportions, percentages
- Rates

```python
# Targets must be in (0, 1)
y_scaled = (y - y.min()) / (y.max() - y.min())
y_scaled = np.clip(y_scaled, 1e-6, 1 - 1e-6)  # Avoid exact 0 or 1

model = MambularLSS()
model.fit(X_train, y_scaled, family="beta", max_epochs=50)

params = model.predict(X_test)
alpha = params[:, 0]
beta = params[:, 1]

# Mean and variance
mean = alpha / (alpha + beta)
variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
```

### Negative binomial distribution

**Parameters:** n (dispersion), p (probability)

**When to use:**

- Overdispersed count data
- Counts with variance > mean
- Poisson doesn't fit well

```python
model = MambularLSS()
model.fit(X_train, y_train, family="negative_binomial", max_epochs=50)

params = model.predict(X_test)
n = params[:, 0]
p = params[:, 1]

# Mean and variance
mean = n * (1 - p) / p
variance = n * (1 - p) / (p ** 2)  # Variance > mean
```

### Student's t distribution

**Parameters:** degrees of freedom (df), location (μ), scale (σ)

**When to use:**

- Heavy-tailed distributions
- Outliers in target
- Robustness to extreme values

```python
model = MambularLSS()
model.fit(X_train, y_train, family="student_t", max_epochs=50)

params = model.predict(X_test)
df = params[:, 0]
loc = params[:, 1]
scale = params[:, 2]

# Mean (for df > 1)
mean = loc

# Variance (for df > 2)
variance = scale**2 * df / (df - 2)
```

### Full list of families

Check the API reference for the complete list, including:

- `"normal"`, `"lognormal"`
- `"poisson"`, `"negative_binomial"`, `"zero_inflated_poisson"`
- `"gamma"`, `"exponential"`, `"weibull"`
- `"beta"`, `"beta_binomial"`
- `"student_t"`, `"cauchy"`, `"laplace"`

## Output format

### predict()

Returns distribution parameters as a 2D array:

```python
params = model.predict(X_test)
# Shape: (n_samples, n_params)
# For family="normal": (200, 2) → [mean, std]
# For family="gamma": (200, 2) → [shape, rate]
# For family="student_t": (200, 3) → [df, loc, scale]

print(params.shape)  # (n_samples, n_params)
print(params.dtype)  # float32
```

### Parameter extraction

```python
# Normal distribution
params = model.predict(X_test)
mean = params[:, 0]
std = params[:, 1]

# Gamma distribution
params = model.predict(X_test)
shape = params[:, 0]
rate = params[:, 1]
```

## Prediction intervals

Generate confidence intervals for predictions:

### Symmetric distributions (Normal)

```python
model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)

params = model.predict(X_test)
mean = params[:, 0]
std = params[:, 1]

# 68% interval (±1σ)
lower_68 = mean - std
upper_68 = mean + std

# 95% interval (±1.96σ)
lower_95 = mean - 1.96 * std
upper_95 = mean + 1.96 * std

# 99% interval (±2.58σ)
lower_99 = mean - 2.58 * std
upper_99 = mean + 2.58 * std
```

### Asymmetric distributions

Use the inverse CDF (quantile function):

```python
from scipy import stats

model = MambularLSS()
model.fit(X_train, y_train, family="gamma", max_epochs=50)

params = model.predict(X_test)
shape = params[:, 0]
rate = params[:, 1]

# 95% interval for each sample
lower = np.array([stats.gamma.ppf(0.025, a=s, scale=1/r) for s, r in zip(shape, rate)])
upper = np.array([stats.gamma.ppf(0.975, a=s, scale=1/r) for s, r in zip(shape, rate)])
```

## Quantile predictions

Extract specific percentiles:

```python
# Normal distribution
mean = params[:, 0]
std = params[:, 1]

# Median (50th percentile)
median = mean  # For symmetric distributions

# 90th percentile
p90 = mean + 1.28 * std  # z-score for 90th percentile

# 10th percentile
p10 = mean - 1.28 * std
```

Or use scipy:

```python
from scipy import stats

# 25th, 50th, 75th percentiles
quantiles = [0.25, 0.50, 0.75]
results = np.array([
    [stats.norm.ppf(q, loc=m, scale=s) for q in quantiles]
    for m, s in zip(mean, std)
])
# Shape: (n_samples, 3)
```

## Evaluation

LSS models are evaluated using negative log-likelihood:

```python
metrics = model.evaluate(X_test, y_test)
print(f"Negative log-likelihood: {metrics['loss']:.3f}")
```

Lower is better (higher likelihood).

You can also evaluate point predictions (mean):

```python
params = model.predict(X_test)
mean_predictions = params[:, 0]

from sklearn.metrics import mean_squared_error, mean_absolute_error

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, mean_predictions)):.3f}")
print(f"MAE: {mean_absolute_error(y_test, mean_predictions):.3f}")
```

## Comparing with standard regression

```python
from deeptab.models import MambularRegressor, MambularLSS

# Standard regression
reg_model = MambularRegressor()
reg_model.fit(X_train, y_train, max_epochs=50)
reg_pred = reg_model.predict(X_test)

# Distributional regression
lss_model = MambularLSS()
lss_model.fit(X_train, y_train, family="normal", max_epochs=50)
lss_params = lss_model.predict(X_test)
lss_mean = lss_params[:, 0]
lss_std = lss_params[:, 1]

# Compare point predictions
print(f"Regressor RMSE: {np.sqrt(mean_squared_error(y_test, reg_pred)):.3f}")
print(f"LSS mean RMSE: {np.sqrt(mean_squared_error(y_test, lss_mean)):.3f}")

# LSS provides additional uncertainty info
print(f"Mean uncertainty (std): {lss_std.mean():.3f}")
```

## Visualizing predictions

### Prediction intervals

```python
import matplotlib.pyplot as plt

# Sort by true values for better visualization
indices = np.argsort(y_test)
y_sorted = y_test[indices]
mean_sorted = mean[indices]
lower_sorted = lower_95[indices]
upper_sorted = upper_95[indices]

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_sorted)), y_sorted, label="True", alpha=0.5, s=10)
plt.plot(mean_sorted, label="Predicted mean", color="red")
plt.fill_between(
    range(len(y_sorted)),
    lower_sorted,
    upper_sorted,
    alpha=0.3,
    label="95% interval",
)
plt.xlabel("Sample (sorted)")
plt.ylabel("Target")
plt.legend()
plt.show()
```

### Predicted distributions

```python
# Plot distributions for a few samples
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

for i, idx in enumerate(np.random.choice(len(X_test), 6, replace=False)):
    x = np.linspace(
        mean[idx] - 3*std[idx],
        mean[idx] + 3*std[idx],
        100,
    )
    y_dist = stats.norm.pdf(x, loc=mean[idx], scale=std[idx])

    axes[i].plot(x, y_dist, label="Predicted")
    axes[i].axvline(y_test[idx], color="red", linestyle="--", label="True")
    axes[i].axvline(mean[idx], color="green", linestyle="--", label="Mean")
    axes[i].fill_between(
        x,
        0,
        y_dist,
        where=((x >= lower_95[idx]) & (x <= upper_95[idx])),
        alpha=0.3,
        label="95% CI",
    )
    axes[i].set_title(f"Sample {idx}")
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.show()
```

## Uncertainty decomposition

LSS models can reveal different types of uncertainty:

### Aleatoric uncertainty (data noise)

Captured by the predicted standard deviation:

```python
# High aleatoric uncertainty → inherently noisy region
high_noise_mask = std > np.percentile(std, 90)
print(f"Samples with high aleatoric uncertainty: {high_noise_mask.sum()}")
```

### Heteroscedastic noise

Check if uncertainty varies with input:

```python
# Plot uncertainty vs. predicted mean
plt.scatter(mean, std, alpha=0.5)
plt.xlabel("Predicted mean")
plt.ylabel("Predicted std")
plt.title("Heteroscedasticity check")
plt.show()

# If scatter shows pattern → heteroscedastic
# If scatter is flat → homoscedastic
```

## Ensemble of LSS models

Average parameters from multiple models:

```python
models = [MambularLSS(), FTTransformerLSS(), ResNetLSS()]

# Train all
for model in models:
    model.fit(X_train, y_train, family="normal", max_epochs=50)

# Average parameters
all_params = np.array([model.predict(X_test) for model in models])
mean_params = all_params.mean(axis=0)

# Use averaged parameters
ensemble_mean = mean_params[:, 0]
ensemble_std = mean_params[:, 1]
```

## Choosing the right family

Decision tree:

1. **Target range**
   - Unbounded → Normal, Student's t
   - Positive only → Gamma, Lognormal, Exponential
   - In (0, 1) → Beta
   - Non-negative integers → Poisson, Negative binomial

2. **Target distribution**
   - Symmetric → Normal
   - Right-skewed → Gamma, Lognormal
   - Heavy-tailed → Student's t

3. **Noise characteristics**
   - Constant variance → Normal
   - Variance increases with mean → Poisson, Gamma
   - Overdispersion (variance > mean) → Negative binomial

4. **Try and compare**

```python
families = ["normal", "gamma", "student_t"]
results = {}

for family in families:
    model = MambularLSS()
    model.fit(X_train, y_train, family=family, max_epochs=50)
    metrics = model.evaluate(X_test, y_test)
    results[family] = metrics["loss"]

# Best family (lowest negative log-likelihood)
best_family = min(results, key=results.get)
print(f"Best family: {best_family} (NLL: {results[best_family]:.3f})")
```

## Best practices

1. **Choose family based on target characteristics**
2. **Validate intervals** — check coverage (% of true values in predicted intervals)
3. **Visualize predictions** — plot distributions for a few samples
4. **Compare with standard regression** — LSS should have similar or better point predictions
5. **Use uncertainty for downstream decisions** — don't just predict, act on uncertainty
6. **Check calibration** — predicted intervals should match empirical coverage

## Coverage validation

Check if prediction intervals have correct coverage:

```python
# 95% interval
coverage = ((y_test >= lower_95) & (y_test <= upper_95)).mean()
print(f"95% interval coverage: {coverage:.2%}")  # Should be ~95%

# 68% interval
coverage_68 = ((y_test >= lower_68) & (y_test <= upper_68)).mean()
print(f"68% interval coverage: {coverage_68:.2%}")  # Should be ~68%
```

If coverage is too low → model is overconfident (predicted std too small)
If coverage is too high → model is underconfident (predicted std too large)

## Next steps

- **[Regression](regression)** — Standard point prediction regression
- **[Training and Evaluation](training_and_evaluation)** — Training loop details
- **[Tutorials: Distributional](../../tutorials/distributional)** — Complete workflows
- **[API Reference](../../api/models/index)** — Full parameter documentation
