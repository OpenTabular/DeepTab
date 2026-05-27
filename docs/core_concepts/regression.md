# Regression

Key concepts for regression tasks: continuous predictions, target preprocessing, and evaluation metrics.

```{tip}
For hands-on examples and complete workflows, see the [Regression Tutorial](../tutorials/regression).
```

## Continuous Predictions

Regression models predict continuous numerical values:

```python
from deeptab.models import ResNetRegressor

model = ResNetRegressor()
model.fit(X_train, y_train, max_epochs=100)
predictions = model.predict(X_test)  # [12.34, 45.67, -23.45, ...]
```

**All stable models are available as regressors** — just use the `*Regressor` suffix.

## Target Preprocessing

```{important}
Unlike features, targets are **not** automatically preprocessed. Apply transformations manually when needed for better performance.
```

**Common transformations:**

| Transform       | Use case                         | Example                 |
| --------------- | -------------------------------- | ----------------------- |
| Log transform   | Skewed/positive targets (prices) | `np.log1p(y)`           |
| Standardization | Very large/small magnitudes      | `StandardScaler()`      |
| Clip outliers   | Extreme values                   | `np.clip(y, -100, 100)` |

**Log example:**

```python
import numpy as np

# Transform target
y_train_log = np.log1p(y_train)  # log(1 + y)
model.fit(X_train, y_train_log, max_epochs=50)

# Inverse transform predictions
predictions_log = model.predict(X_test)
predictions = np.expm1(predictions_log)  # exp(y) - 1
```

```{warning}
Remember to **inverse transform** predictions to get values in the original scale!
```

## Evaluation Metrics

**Default metrics:**

```python
metrics = model.evaluate(X_test, y_test)
# Returns: {'rmse': 12.34, 'mae': 8.56, 'loss': 152.3}
```

| Metric | Description                    | When to use                             |
| ------ | ------------------------------ | --------------------------------------- |
| RMSE   | Root Mean Squared Error        | General-purpose, penalizes large errors |
| MAE    | Mean Absolute Error            | Less sensitive to outliers              |
| R²     | Coefficient of determination   | Proportion of variance explained        |
| MAPE   | Mean Absolute Percentage Error | When relative errors matter             |

**Custom metrics via TrainerConfig:**

```python
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError

cfg = TrainerConfig(
    metrics=[MeanSquaredError(), MeanAbsolutePercentageError()]
)
model = TabRRegressor(trainer_config=cfg)
```

## Different Target Distributions

| Target type          | Strategy               | Alternative                   |
| -------------------- | ---------------------- | ----------------------------- |
| Normally distributed | Default (no transform) | -                             |
| Positive (prices)    | Log transform          | LSS with gamma family         |
| Bounded (0 to 1)     | Logit transform        | LSS with beta family          |
| Count data           | Log transform          | LSS with poisson family       |
| Heavy outliers       | Quantile preprocessing | Clip outliers                 |
| Heteroscedastic      | -                      | **Use LSS** for varying noise |

```{tip}
For targets with **varying uncertainty** (heteroscedastic noise), use [Distributional Regression](distributional_regression) instead of standard regression.
```

## Output Format

| Method       | Returns            | Shape          | Dtype   |
| ------------ | ------------------ | -------------- | ------- |
| `predict()`  | Continuous values  | `(n_samples,)` | `float` |
| `evaluate()` | Metrics dictionary | -              | -       |

## Next Steps

- [Regression Tutorial](../tutorials/regression) — Complete examples
- [Distributional Regression](distributional_regression) — For uncertainty quantification
- [Training and Evaluation](training_and_evaluation) — Training loop details
