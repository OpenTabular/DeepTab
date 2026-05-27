# Regression Tutorial

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <a href="https://colab.research.google.com/github/basf/DeepTab/blob/main/docs/tutorials/notebooks/regression.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/basf/DeepTab/blob/main/docs/tutorials/notebooks/regression.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on-GitHub-181717?logo=github&logoColor=white" alt="View on GitHub"/>
  </a>
</div>

This tutorial trains a DeepTab regressor end to end and reports explicit regression metrics.

```{note}
The notebook linked above is generated from this same tutorial content. The markdown page is the readable lesson; the notebook is the executable copy.
```

## What You Will Learn

- How to train a standard `*Regressor` model.
- Why target scale matters for neural tabular regression.
- How to pass explicit regression metrics instead of relying on implementation defaults.
- How to compare several architectures under the same split.

## Setup

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MLPRegressor, MambularRegressor, ResNetRegressor
```

## Data

```python
X_num, y = make_regression(
    n_samples=1200,
    n_features=8,
    n_informative=6,
    noise=15.0,
    random_state=101,
)

X = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(X_num.shape[1])])
X["segment"] = pd.qcut(X["num_0"], q=4, labels=["A", "B", "C", "D"]).astype("category")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=101)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=101)
```

## Configure and Train

```python
model = MambularRegressor(
    model_config=MambularConfig(d_model=64, n_layers=4, pooling_method="avg"),
    preprocessing_config=PreprocessingConfig(
        numerical_preprocessing="standard",
        categorical_preprocessing="int",
    ),
    trainer_config=TrainerConfig(max_epochs=60, batch_size=128, lr=3e-4, patience=10),
    random_state=101,
)

model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

## Evaluate

```python
metrics = model.evaluate(
    X_test,
    y_test,
    metrics={
        "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error,
        "r2": r2_score,
    },
)

print(metrics)
```

The default regressor `evaluate()` metric is `"Mean Squared Error"`, so explicit metrics are better for tutorials and papers.

```{important}
Regression metrics answer different questions. RMSE emphasizes large errors, MAE is more robust to outliers, and R2 is scale-normalized but can hide subgroup failures.
```

## Target Scaling

Targets are not automatically transformed. For large-magnitude targets, scale `y` manually:

```{tip}
If you transform the target before training, always inverse-transform predictions before reporting metrics in the original unit.
```

```python
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).ravel()

scaled_model = MambularRegressor(
    trainer_config=TrainerConfig(max_epochs=60, patience=10, lr=3e-4),
    random_state=101,
)
scaled_model.fit(X_train, y_train_scaled, X_val=X_val, y_val=y_val_scaled)

pred_scaled = scaled_model.predict(X_test)
pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
print(r2_score(y_test, pred))
```

## Compare Architectures

```python
models = {
    "MLP": MLPRegressor(trainer_config=TrainerConfig(max_epochs=30, patience=5, lr=1e-3), random_state=101),
    "ResNet": ResNetRegressor(trainer_config=TrainerConfig(max_epochs=30, patience=5, lr=1e-3), random_state=101),
    "Mambular": MambularRegressor(trainer_config=TrainerConfig(max_epochs=30, patience=5, lr=3e-4), random_state=101),
}

results = {}
for name, estimator in models.items():
    estimator.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    pred = estimator.predict(X_test)
    results[name] = {
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "r2": r2_score(y_test, pred),
    }

print(results)
```

## Save and Load

```python
model.save("regression_model.pt")

loaded = MambularRegressor.load("regression_model.pt")
loaded_pred = loaded.predict(X_test)
print(r2_score(y_test, loaded_pred))
```

## Next Steps

- [Regression concept](../core_concepts/regression)
- [Distributional regression](distributional)
- [Recommended configs](../model_zoo/recommended_configs)
