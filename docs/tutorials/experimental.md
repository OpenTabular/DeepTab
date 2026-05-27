# Using Experimental Models

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <a href="https://colab.research.google.com/github/basf/DeepTab/blob/main/docs/tutorials/notebooks/experimental.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/basf/DeepTab/blob/main/docs/tutorials/notebooks/experimental.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on-GitHub-181717?logo=github&logoColor=white" alt="View on GitHub"/>
  </a>
</div>

Experimental models live in `deeptab.models.experimental`. They use the same estimator workflow as stable models, but their APIs and defaults may change between releases.

```{note}
The notebook linked above is generated from this same tutorial content, so the runnable version and the documentation version stay aligned.
```

## What You Will Learn

- How to import experimental models explicitly.
- How to use the correct experimental config class for each model.
- How to compare an experimental model against a stable baseline.
- How to keep experimental results reproducible with version pinning.

```{warning}
Pin the exact DeepTab version when using experimental models in research artifacts or production-like pipelines.
```

## Setup

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from deeptab.configs import ModernNCAConfig, PreprocessingConfig, TangosConfig, TrainerConfig, TromptConfig
from deeptab.models import MambularClassifier
from deeptab.models.experimental import ModernNCARegressor, TangosClassifier, TromptClassifier
```

## Classification With Trompt

```python
Xc_num, yc = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=5,
    n_classes=3,
    random_state=101,
)
Xc = pd.DataFrame(Xc_num, columns=[f"num_{i}" for i in range(Xc_num.shape[1])])

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, stratify=yc, random_state=101
)

model = TromptClassifier(
    model_config=TromptConfig(d_model=128, n_cycles=6, n_cells=4, P=128),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(max_epochs=50, batch_size=128, lr=3e-4, patience=10),
    random_state=101,
)
model.fit(Xc_train, yc_train)

pred = model.predict(Xc_test)
print(accuracy_score(yc_test, pred))
```

```{important}
Trompt uses `TromptConfig`, not a stable model config such as `MambularConfig`. Experimental pages should always use the config class that belongs to the model being demonstrated.
```

## Regression With ModernNCA

```python
Xr_num, yr = make_regression(
    n_samples=1000,
    n_features=8,
    n_informative=6,
    noise=10.0,
    random_state=101,
)
Xr = pd.DataFrame(Xr_num, columns=[f"num_{i}" for i in range(Xr_num.shape[1])])

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=101)

regressor = ModernNCARegressor(
    model_config=ModernNCAConfig(dim=128, n_blocks=4, temperature=0.75),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(max_epochs=50, batch_size=128, lr=3e-4, patience=10),
    random_state=101,
)
regressor.fit(Xr_train, yr_train)

pred = regressor.predict(Xr_test)
print(np.sqrt(mean_squared_error(yr_test, pred)))
```

## TANGOS Classification

```python
tangos = TangosClassifier(
    model_config=TangosConfig(layer_sizes=[256, 128, 32], lamda1=0.5, lamda2=0.1),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="standard"),
    trainer_config=TrainerConfig(max_epochs=50, batch_size=128, lr=1e-3, patience=10),
    random_state=101,
)
tangos.fit(Xc_train, yc_train)
```

## Compare Experimental and Stable

```python
stable = MambularClassifier(
    trainer_config=TrainerConfig(max_epochs=30, patience=5),
    random_state=101,
)
experimental = TromptClassifier(
    model_config=TromptConfig(d_model=128, n_cycles=4, n_cells=4, P=128),
    trainer_config=TrainerConfig(max_epochs=30, patience=5),
    random_state=101,
)

for name, estimator in {"Mambular": stable, "Trompt": experimental}.items():
    estimator.fit(Xc_train, yc_train)
    pred = estimator.predict(Xc_test)
    print(name, accuracy_score(yc_test, pred))
```

## Save and Load

```python
model.save("trompt_model.pt")

loaded = TromptClassifier.load("trompt_model.pt")
loaded_pred = loaded.predict(Xc_test)
```

## Practical Rules

1. Use explicit experimental imports.
2. Use the matching experimental config class (`TromptConfig`, `ModernNCAConfig`, `TangosConfig`).
3. Pin the exact DeepTab version in experiments.
4. Compare against stable baselines before drawing conclusions.
5. Read the experimental model page for implementation caveats.

```{tip}
Treat experimental results as hypotheses. Always compare against at least one simple stable baseline, such as MLP, ResNet, TabM, or Mambular.
```

## Next Steps

- [Experimental model zoo](../model_zoo/experimental/index)
- [Model tiers](../core_concepts/model_tiers)
- [Stable model zoo](../model_zoo/stable/index)
