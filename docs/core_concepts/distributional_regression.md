# Distributional Regression

Distributional regression estimates parameters of a conditional probability distribution instead of a single point prediction. In DeepTab, these estimators use the `*LSS` suffix.

```python
from deeptab.models import MambularLSS

model = MambularLSS()
model.fit(X_train, y_train, family="normal")
params = model.predict(X_test)
```

## Why Use It?

Use LSS models when the target has meaningful uncertainty:

| Need | Why distributional regression helps |
| --- | --- |
| Prediction intervals | Parameters define full predictive distributions. |
| Heteroscedastic noise | Scale/shape can change with input features. |
| Risk-aware decisions | Downstream systems can use quantiles or tail probabilities. |
| Non-Gaussian targets | Choose a family matching target support. |

## Families

Choose a family whose support matches the target:

| Family | Typical target |
| --- | --- |
| `"normal"` | Continuous unbounded values. |
| `"poisson"` | Count data. |
| `"gamma"` | Positive continuous values. |
| `"beta"` | Values in `(0, 1)`. |
| `"studentt"` | Heavy-tailed continuous values. |
| `"negativebinom"` | Overdispersed counts. |
| `"inversegamma"` | Positive heavy-tailed values. |
| `"categorical"` | Distributional classification-style outputs. |

The exact parameterization is defined by the distribution classes in `deeptab.distributions`.

## Prediction Intervals

For a normal-family model:

```python
import numpy as np
from scipy import stats

params = model.predict(X_test)
mean = params[:, 0]
variance_or_scale = params[:, 1]
std = np.sqrt(np.maximum(variance_or_scale, 1e-12))

lower = stats.norm.ppf(0.05, loc=mean, scale=std)
upper = stats.norm.ppf(0.95, loc=mean, scale=std)
```

Verify the parameter convention for the chosen family before computing intervals. Some distribution implementations return transformed or constrained parameters.

## Evaluation

`evaluate()` uses family-specific default metrics:

```python
metrics = model.evaluate(X_test, y_test, distribution_family="normal")
```

For normal, the current defaults include MSE on the mean and CRPS. `score()` computes negative log-likelihood through the fitted family.

```python
nll = model.score(X_test, y_test)
```

For papers and benchmarks, report both point quality and distribution quality when relevant:

1. RMSE/MAE/R2 on the predictive mean.
2. NLL or CRPS.
3. Empirical coverage for prediction intervals.
4. Calibration curves across multiple interval levels.

## Practical Guidance

1. Start with `family="normal"` for unbounded continuous targets.
2. Use `gamma` or `lognormal`-style modeling only for strictly positive targets where the family is available and parameterized as expected.
3. Clip or rescale targets to valid support for `beta` and count families.
4. Always validate interval coverage on held-out data.

## Next Steps

- [Distributional Tutorial](../tutorials/distributional)
- [Regression](regression)
- [API: Distributions](../api/distributions/index)
