# Distributional Regression Tutorial

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <a href="https://colab.research.google.com/github/basf/DeepTab/blob/main/docs/tutorials/notebooks/distributional.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/basf/DeepTab/blob/main/docs/tutorials/notebooks/distributional.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on-GitHub-181717?logo=github&logoColor=white" alt="View on GitHub"/>
  </a>
</div>

Distributional regression predicts distribution parameters instead of only point estimates. In DeepTab, these estimators use the `*LSS` suffix.

```{note}
The notebook linked above is generated from this same tutorial content. It includes the same explanation and code cells as this markdown page.
```

## What You Will Learn

- How to train an LSS model with `family="normal"`.
- How to turn predicted distribution parameters into intervals.
- How to evaluate both point accuracy and distribution quality.
- Why family choice and parameter conventions matter.

## Setup

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MambularLSS, MambularRegressor
```

## Data

Create a regression problem with input-dependent noise.

```python
X_num, base_y = make_regression(
    n_samples=1500,
    n_features=6,
    n_informative=5,
    noise=5.0,
    random_state=101,
)

rng = np.random.default_rng(101)
noise_scale = 0.5 + 2.0 / (1.0 + np.exp(-X_num[:, 0]))
y = base_y + rng.normal(0.0, noise_scale * 10.0)

X = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(X_num.shape[1])])

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=101)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=101)
```

## Train an LSS Model

```python
lss_model = MambularLSS(
    model_config=MambularConfig(d_model=64, n_layers=4),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="standard"),
    trainer_config=TrainerConfig(max_epochs=60, batch_size=128, lr=3e-4, patience=10),
    random_state=101,
)

lss_model.fit(X_train, y_train, family="normal", X_val=X_val, y_val=y_val)
```

## Predict Distribution Parameters

```python
params = lss_model.predict(X_test)
print(params.shape)

mean = params[:, 0]
scale_param = params[:, 1]
std = np.sqrt(np.maximum(scale_param, 1e-12))
```

For the current normal-family metrics, DeepTab treats the second parameter as a variance-like scale in CRPS calculations. Always verify parameter conventions when using a different family.

```{important}
Distribution parameters are model outputs, not universal statistics. Before computing intervals for a family, check whether the implementation returns means, variances, rates, logits, or transformed positive parameters.
```

## Prediction Intervals

```python
lower = stats.norm.ppf(0.05, loc=mean, scale=std)
upper = stats.norm.ppf(0.95, loc=mean, scale=std)

coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"90% interval coverage: {coverage:.3f}")
```

## Evaluate

```python
lss_metrics = lss_model.evaluate(X_test, y_test, distribution_family="normal")
print(lss_metrics)

point_rmse = np.sqrt(mean_squared_error(y_test, mean))
point_r2 = r2_score(y_test, mean)
print({"rmse_on_mean": point_rmse, "r2_on_mean": point_r2})
```

## Compare With Point Regression

```python
point_model = MambularRegressor(
    trainer_config=TrainerConfig(max_epochs=60, batch_size=128, lr=3e-4, patience=10),
    random_state=101,
)
point_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

point_pred = point_model.predict(X_test)
print({
    "point_rmse": np.sqrt(mean_squared_error(y_test, point_pred)),
    "lss_mean_rmse": np.sqrt(mean_squared_error(y_test, mean)),
})
```

## Other Families

Match the family to the target support:

```{tip}
Wrong support is a modeling error, not just a tuning issue. Do not use a positive-only family for negative targets or a count family for continuous targets.
```

| Target                  | Candidate family                 |
| ----------------------- | -------------------------------- |
| Continuous unbounded    | `"normal"`                       |
| Count data              | `"poisson"` or `"negativebinom"` |
| Positive continuous     | `"gamma"`                        |
| Proportions in `(0, 1)` | `"beta"`                         |
| Heavy-tailed continuous | `"studentt"`                     |

Example for counts:

```python
count_y = np.random.default_rng(101).poisson(lam=np.exp(0.2 * X_num[:, 0]))
model = MambularLSS(trainer_config=TrainerConfig(max_epochs=30, patience=5))
model.fit(X, count_y, family="poisson")
```

## Save and Load

```python
lss_model.save("lss_model.pt")
loaded = MambularLSS.load("lss_model.pt")
loaded_params = loaded.predict(X_test)
```

## Next Steps

- [Regression tutorial](regression)
- [Distribution API](../api/distributions/index)
