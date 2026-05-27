# Distributional Regression

Distributional regression (LSS - Location, Scale, and Shape) predicts **full probability distributions** rather than point estimates, enabling uncertainty quantification and prediction intervals.

```{tip}
For hands-on examples and complete workflows, see the [Distributional Tutorial](../tutorials/distributional).
```

## Why Distributional Regression?

**Standard regression** predicts a single value:

```python
prediction = model.predict(X_test)[0]  # → 42.5
```

**Distributional regression** predicts distribution parameters:

```python
params = lss_model.predict(X_test)[0]  # → [mean=42.5, std=5.2]
```

This provides both **expected value** and **uncertainty**.

```{important}
**Key use cases:**

- Uncertainty quantification (know when predictions are confident)
- Prediction intervals (95% confidence bounds)
- Heteroscedastic noise (varying noise levels across input space)
- Risk-aware decisions (use full distribution for optimization)
- Quantile predictions (specific percentiles for business needs)
```

## Getting Started

All models support LSS via the `*LSS` suffix:

```python
from deeptab.models import MambularLSS

model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=100)
params = model.predict(X_test)  # Returns distribution parameters
```

## Distribution Families

Choose based on your target's characteristics:

| Family              | Parameters          | Support | Use case                                |
| ------------------- | ------------------- | ------- | --------------------------------------- |
| `normal`            | μ (mean), σ (std)   | ℝ       | Unbounded continuous (default)          |
| `poisson`           | λ (rate)            | ℕ₀      | Count data                              |
| `gamma`             | α (shape), β (rate) | ℝ₊      | Positive continuous (prices, durations) |
| `beta`              | α, β                | (0, 1)  | Proportions, probabilities              |
| `negative_binomial` | n, p                | ℕ₀      | Overdispersed count data                |
| `student_t`         | df, μ, σ            | ℝ       | Heavy-tailed distributions              |
| `exponential`       | λ (rate)            | ℝ₊      | Waiting times, lifetimes                |
| `laplace`           | μ, b                | ℝ       | L1 loss equivalent                      |
| `lognormal`         | μ, σ                | ℝ₊      | Multiplicative processes                |

```{note}
See the [API reference](../api/distributions/index) for the complete list of supported families.
```

### Example: Normal Distribution

```python
from deeptab.models import SAINTLSS

model = SAINTLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)

# Returns (n_samples, 2): [mean, std] for each sample
params = model.predict(X_test)

mean_predictions = params[:, 0]
std_predictions = params[:, 1]

# 95% prediction intervals
lower = mean_predictions - 1.96 * std_predictions
upper = mean_predictions + 1.96 * std_predictions
```

### Example: Poisson for Count Data

```python
model = FTTransformerLSS()
model.fit(X_train, y_train_counts, family="poisson", max_epochs=50)

# Returns (n_samples, 1): [rate] for each sample
params = model.predict(X_test)
rate = params[:, 0]

# Expected count
expected_counts = rate
```

## When to Use Which Family

| Target characteristics | Recommended family             |
| ---------------------- | ------------------------------ |
| Continuous, unbounded  | `normal`                       |
| Positive continuous    | `gamma`, `lognormal`           |
| Counts (0, 1, 2, ...)  | `poisson`, `negative_binomial` |
| Proportions (0 to 1)   | `beta`                         |
| Heavy outliers         | `student_t`, `laplace`         |
| Waiting times          | `exponential`                  |

```{warning}
Choosing the wrong family can lead to poor fits. Match the family's support to your target's range (e.g., don't use `gamma` for negative values).
```

## Heteroscedastic Noise

A key advantage of LSS: modeling **varying uncertainty**:

```python
# Standard regression assumes constant noise
# LSS learns input-dependent noise

params = model.predict(X_test)
uncertainty = params[:, 1]  # Standard deviation varies by input

# Find high-uncertainty predictions
high_uncertainty_idx = uncertainty > uncertainty.mean() + 2 * uncertainty.std()
```

## Evaluation

LSS models are evaluated using **negative log-likelihood** (lower is better):

```python
metrics = model.evaluate(X_test, y_test)
print(f"Negative log-likelihood: {metrics['loss']:.3f}")
```

**Compare to point predictions:**

```python
# Extract point predictions (e.g., mean for normal)
mean_predictions = model.predict(X_test)[:, 0]

# Use standard regression metrics
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, mean_predictions))
```

## Output Format

| Method       | Returns                 | Shape                   | Dtype   |
| ------------ | ----------------------- | ----------------------- | ------- |
| `predict()`  | Distribution parameters | `(n_samples, n_params)` | `float` |
| `evaluate()` | Negative log-likelihood | -                       | -       |

## Next Steps

- [Distributional Tutorial](../tutorials/distributional) — Complete examples with all families
- [API: Distributions](../api/distributions/index) — Full list of families and parameters
- [Regression](regression) — For standard point predictions
