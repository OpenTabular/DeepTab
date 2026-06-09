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
        numerical_preprocessing="standardization",
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

## Production Inference with `InferenceModel`

Once a model is trained and saved, use `InferenceModel` to load it in service
code. It provides a narrow, read-only surface — training methods such as `fit`
are absent, so they cannot be called accidentally.

```python
from deeptab import InferenceModel
import pandas as pd

# Load once at service startup
model = InferenceModel.from_path("regression_model.pt")

print(model)
# InferenceModel(task='regression', estimator='MambularRegressor',
#                n_features=9, features=['num_0', ..., 'segment'])

# Validate schema before prediction
X_clean = model.validate_input(X_test)

# Predict
predictions = model.predict(X_clean)
print(r2_score(y_test, predictions))
```

Schema validation catches common deployment mistakes before they reach the
neural network:

```python
# Drop a column by accident
X_bad = X_test.drop(columns=["num_0"])
model.validate_input(X_bad)
# ValueError: Input is missing 1 column(s) that were present during training: ['num_0'].

# Extra columns from a wider upstream pipeline
X_wide = X_test.copy()
X_wide["debug_id"] = range(len(X_test))

# Lenient mode: drop extras with a warning
X_clean = model.validate_input(X_wide, allow_extra_columns=True)
# UserWarning: Input has 1 column(s) not seen during training (['debug_id']); they will be dropped.
```

See [Inference Model](../core_concepts/inference) for the full production API.

## Next Steps

- [Distributional regression](distributional)
- [Advanced training](advanced_training)
- [Recommended configs](../model_zoo/recommended_configs)
