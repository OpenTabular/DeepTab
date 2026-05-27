# Classification

DeepTab classifiers handle binary and multiclass tabular classification with the same estimator API.

## Label Requirements

Labels should be encoded as integers:

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_labels)
```

Binary classification labels may be shaped `(n_samples,)` or `(n_samples, 1)`. Multiclass labels are handled as a one-dimensional integer vector.

## Outputs

```python
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

| Method | Output |
| --- | --- |
| `predict()` | Hard class labels. |
| `predict_proba()` | Class probabilities. |
| `evaluate()` | Metric dictionary. Default is `{"Accuracy": ...}`. |

For custom thresholds in binary classification:

```python
probs = model.predict_proba(X_test)
positive_class = probs[:, 1]
predictions = (positive_class >= 0.7).astype(int)
```

## Validation Splits

When DeepTab creates the validation split internally, classification tasks use stratification:

```python
model.fit(X_train, y_train)
```

For research, explicit splits are preferable:

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=101,
)

model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

## Metrics

Use explicit metrics when reporting results:

```python
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

metrics = model.evaluate(
    X_test,
    y_test,
    metrics={
        "accuracy": (accuracy_score, False),
        "f1_macro": (lambda y, pred: f1_score(y, pred, average="macro"), False),
        "log_loss": (log_loss, True),
    },
)
```

For binary AUROC:

```python
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
```

## Class Imbalance

DeepTab does not currently expose `class_weights` as a `TrainerConfig` field. Use external strategies:

1. Stratified train/validation/test splits.
2. Resampling before fitting.
3. Threshold tuning on validation probabilities.
4. Metrics such as balanced accuracy, macro F1, AUROC, and average precision.

Example with validation threshold tuning:

```python
from sklearn.metrics import f1_score

probs = model.predict_proba(X_val)[:, 1]
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
best_threshold = max(
    thresholds,
    key=lambda t: f1_score(y_val, (probs >= t).astype(int)),
)
```

## Model Choice

Good starting points:

| Data condition | Models |
| --- | --- |
| Need a fast baseline | `MLPClassifier`, `ResNetClassifier`, `TabMClassifier` |
| Many numerical columns | `FTTransformerClassifier`, `MambularClassifier` |
| Categorical-heavy data | `TabTransformerClassifier`, `SAINTClassifier` |
| Local-neighbor signal | `TabRClassifier` |
| Tree-like structure | `NODEClassifier`, `ENODEClassifier`, `NDTFClassifier` |

## Next Steps

- [Classification Tutorial](../tutorials/classification)
- [Training and Evaluation](training_and_evaluation)
- [Model Zoo](../model_zoo/stable/index)
