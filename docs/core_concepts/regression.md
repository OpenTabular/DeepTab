# Regression

DeepTab regressors predict continuous targets with the `*Regressor` estimator variants.

```python
from deeptab.models import ResNetRegressor

model = ResNetRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Target Handling

DeepTab preprocesses features, not targets. Transform targets manually when their scale or distribution makes optimization difficult.

| Target condition | Common strategy |
| --- | --- |
| Strictly positive and skewed | Train on `np.log1p(y)`, inverse with `np.expm1`. |
| Very large or small magnitude | Standardize target with `StandardScaler`. |
| Severe outliers | Clip/winsorize target or use robust metrics. |
| Input-dependent noise | Consider LSS distributional regression. |

Example:

```python
import numpy as np

y_train_log = np.log1p(y_train)
model.fit(X_train, y_train_log)

pred_log = model.predict(X_test)
pred = np.expm1(pred_log)
```

## Metrics

The current default `evaluate()` metric for regressors is mean squared error:

```python
model.evaluate(X_test, y_test)
# {"Mean Squared Error": ...}
```

For reporting, pass explicit metrics:

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

metrics = model.evaluate(
    X_test,
    y_test,
    metrics={
        "rmse": lambda y, pred: np.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error,
        "r2": r2_score,
    },
)
```

The current default `score()` is also mean squared error. Use `r2_score` explicitly when you want R2:

```python
from sklearn.metrics import r2_score

r2 = model.score(X_test, y_test, metric=r2_score)
```

## Residual Diagnostics

After fitting:

```python
pred = model.predict(X_test)
residuals = y_test - pred
```

Useful checks:

| Check | Why |
| --- | --- |
| Residuals vs predictions | Detect nonlinearity or heteroscedasticity. |
| Residual distribution | Detect skew/heavy tails. |
| Error by subgroup | Detect feature-dependent failure modes. |
| Prediction scale | Detect target transform mistakes. |

## Model Choice

| Goal | Models |
| --- | --- |
| Fast baseline | `MLPRegressor`, `ResNetRegressor` |
| Strong neural baseline | `TabMRegressor`, `FTTransformerRegressor` |
| Retrieval/local similarity | `TabRRegressor` |
| Differentiable tree bias | `NODERegressor`, `ENODERegressor`, `NDTFRegressor` |
| Feature-sequence experiments | `MambularRegressor`, `TabulaRNNRegressor` |

## Next Steps

- [Regression Tutorial](../tutorials/regression)
- [Distributional Regression](distributional_regression)
- [Training and Evaluation](training_and_evaluation)
