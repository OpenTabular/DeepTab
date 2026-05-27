# Classification Tutorial

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <a href="https://colab.research.google.com/github/basf/DeepTab/blob/main/docs/tutorials/notebooks/classification.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/basf/DeepTab/blob/main/docs/tutorials/notebooks/classification.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on-GitHub-181717?logo=github&logoColor=white" alt="View on GitHub"/>
  </a>
</div>

This tutorial is an end-to-end classification workflow: generate mixed tabular data, split it, configure DeepTab, train a model, evaluate it, compare architectures, and save the fitted estimator.

```{note}
The notebook linked above is generated from this same tutorial content. Use the markdown page to read the workflow in the docs, and use the notebook when you want to run or modify the cells.
```

## What You Will Learn

- How DeepTab treats classification labels and class probabilities.
- How to keep train, validation, and test splits explicit for research comparisons.
- How `ModelConfig`, `PreprocessingConfig`, and `TrainerConfig` work together.
- How to report metrics that are more informative than raw accuracy.

## Setup

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MLPClassifier, MambularClassifier, ResNetClassifier
```

## Data

```python
X_num, y = make_classification(
    n_samples=1200,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    n_classes=3,
    random_state=101,
)

X = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(X_num.shape[1])])
X["segment"] = pd.qcut(X["num_0"], q=4, labels=["A", "B", "C", "D"]).astype("category")
X["region"] = pd.Series(np.where(X["num_1"] > 0, "north", "south"), dtype="category")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=101
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=101
)
```

Explicit validation data keeps the comparison reproducible across models.

```{important}
For classification, preserve class proportions in every split. DeepTab can stratify its internal validation split, but explicit splits make model comparisons easier to audit.
```

## Configure and Train

```python
model = MambularClassifier(
    model_config=MambularConfig(
        d_model=64,
        n_layers=4,
        dropout=0.0,
        pooling_method="avg",
    ),
    preprocessing_config=PreprocessingConfig(
        numerical_preprocessing="quantile",
        categorical_preprocessing="int",
    ),
    trainer_config=TrainerConfig(
        max_epochs=50,
        batch_size=128,
        lr=3e-4,
        patience=10,
        optimizer_type="Adam",
    ),
    random_state=101,
)

model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

## Predict and Evaluate

```python
pred = model.predict(X_test)
proba = model.predict_proba(X_test)

metrics = model.evaluate(
    X_test,
    y_test,
    metrics={
        "accuracy": (accuracy_score, False),
        "f1_macro": (lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), False),
        "log_loss": (log_loss, True),
    },
)

print(metrics)
print(proba[:3])
```

The boolean in each metric tuple tells DeepTab whether the metric needs probabilities (`True`) or hard labels (`False`).

```{tip}
Use probability-based metrics such as log loss or AUROC when confidence matters. Use label-based metrics such as macro F1 when class balance matters.
```

## Compare Architectures

```python
models = {
    "MLP": MLPClassifier(
        trainer_config=TrainerConfig(max_epochs=30, patience=5, lr=1e-3),
        random_state=101,
    ),
    "ResNet": ResNetClassifier(
        trainer_config=TrainerConfig(max_epochs=30, patience=5, lr=1e-3),
        random_state=101,
    ),
    "Mambular": MambularClassifier(
        model_config=MambularConfig(d_model=64, n_layers=4),
        trainer_config=TrainerConfig(max_epochs=30, patience=5, lr=3e-4),
        random_state=101,
    ),
}

results = {}
for name, estimator in models.items():
    estimator.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    pred = estimator.predict(X_test)
    results[name] = accuracy_score(y_test, pred)

print(results)
```

## Save and Load

```python
model.save("classification_model.pt")

loaded = MambularClassifier.load("classification_model.pt")
loaded_pred = loaded.predict(X_test)
print(accuracy_score(y_test, loaded_pred))
```

## Using Your Own Data

```python
df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=101
)

model = MambularClassifier(
    trainer_config=TrainerConfig(max_epochs=100, patience=15),
    random_state=101,
)
model.fit(X_train, y_train)
```

## Next Steps

- [Classification concept](../core_concepts/classification)
- [Config system](../core_concepts/config_system)
- [Stable model zoo](../model_zoo/stable/index)
