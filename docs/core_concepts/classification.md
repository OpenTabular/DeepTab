# Classification

Key concepts for classification tasks: binary vs multiclass, class imbalance, stratification, and probability outputs.

```{tip}
For hands-on examples and complete workflows, see the [Classification Tutorial](../tutorials/classification).
```

## Binary vs Multiclass

| Type       | Classes | Output shape (predict_proba) | Use case                  |
| ---------- | ------- | ---------------------------- | ------------------------- |
| Binary     | 2       | `(n_samples, 2)`             | Yes/No, True/False        |
| Multiclass | N > 2   | `(n_samples, N)`             | Multiple exclusive labels |

**Label requirements:**

- Must be integers starting from 0: `[0, 1, 2, ...]`
- Use `sklearn.preprocessing.LabelEncoder` if needed

```{warning}
String labels like `["cat", "dog", "bird"]` must be converted to integers `[0, 1, 2]` first.
```

````{note}
**Label shape:** Binary labels are automatically reshaped internally if needed:
```python
# Both work - automatically handled
y_train = np.array([0, 1, 0, 1])  # Shape: (4,) → internally (4, 1)
y_train = np.array([[0], [1], [0], [1]])  # Shape: (4, 1)
````

This is only relevant if using `TabularDataModule` directly—the high-level estimator API handles it automatically.

````

## Probability Outputs

All classifiers support both hard predictions and probability estimates:

```python
predictions = model.predict(X_test)       # Class labels: [0, 1, 0, ...]
probabilities = model.predict_proba(X_test)  # [[0.9, 0.1], [0.3, 0.7], ...]
````

**Custom decision thresholds:**

```python
probs = model.predict_proba(X_test)
predictions = (probs[:, 1] > 0.7).astype(int)  # 70% threshold instead of 50%
```

## Automatic Stratification (v2.0+)

```{important}
Classification tasks automatically use **stratified train/val splits** to preserve class distributions. This is especially critical for imbalanced datasets.
```

```python
# Imbalanced data: 90% class 0, 10% class 1
model.fit(X_train, y_train, max_epochs=50)
# Validation set automatically maintains 90/10 ratio
```

**Override with explicit validation:**

```python
model.fit(X_train, y_train, X_val=X_val, y_val=y_val, max_epochs=50)
```

## Handling Class Imbalance

Beyond stratification, use these techniques for severe imbalance:

**Class weights:**

```python
from sklearn.utils.class_weight import compute_class_weight
from deeptab.configs import TrainerConfig

weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
model = FTTransformerClassifier(
    trainer_config=TrainerConfig(class_weights=weights)
)
```

**Resampling (before DeepTab):**

```python
from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
model.fit(X_resampled, y_resampled, max_epochs=50)
```

## Evaluation Metrics

**Default metrics:**

```python
metrics = model.evaluate(X_test, y_test)
# Returns: {'accuracy': 0.85, 'loss': 0.42}
```

**Custom metrics via TrainerConfig:**

```python
from torchmetrics import F1Score, Precision, Recall

cfg = TrainerConfig(
    metrics=[F1Score(task="binary"), Precision(task="binary")]
)
model = SAINTClassifier(trainer_config=cfg)
```

```{tip}
For imbalanced data, use balanced metrics (F1, balanced accuracy, ROC-AUC) instead of raw accuracy.
```

## Output Formats

| Method            | Returns             | Shape                | Dtype   |
| ----------------- | ------------------- | -------------------- | ------- |
| `predict()`       | Class labels        | `(n_samples,)`       | `int64` |
| `predict_proba()` | Class probabilities | `(n_samples, n_cls)` | `float` |
| `evaluate()`      | Metrics dictionary  | -                    | -       |

## Next Steps

- [Classification Tutorial](../tutorials/classification) — Complete examples
- [Training and Evaluation](training_and_evaluation) — Training loop details
- [sklearn API](sklearn_api) — Method signatures and integration

## Comparing architectures

Try different models on the same data:

```python
from deeptab.models import (
    MambularClassifier,
    FTTransformerClassifier,
    TabTransformerClassifier,
    ResNetClassifier,
)

models = {
    "Mambular": MambularClassifier(),
    "FTTransformer": FTTransformerClassifier(),
    "TabTransformer": TabTransformerClassifier(),
    "ResNet": ResNetClassifier(),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train, max_epochs=50)
    metrics = model.evaluate(X_test, y_test)
    results[name] = metrics["accuracy"]

# Best model
best = max(results, key=results.get)
print(f"Best: {best} ({results[best]:.3f})")
```

## Hyperparameter tuning

Classification-specific tuning with GridSearchCV:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model_config__d_model": [64, 128, 256],
    "model_config__n_layers": [4, 6, 8],
    "trainer_config__lr": [1e-3, 5e-4, 1e-4],
    "trainer_config__batch_size": [128, 256],
}

search = GridSearchCV(
    estimator=MambularClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",  # Or "accuracy", "roc_auc", etc.
)

search.fit(X_train, y_train)
print(f"Best score: {search.best_score_:.3f}")
print(f"Best params: {search.best_params_}")
```

## Common patterns

### Stratified K-fold

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for train_idx, val_idx in skf.split(X, y):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    model = MambularClassifier()
    model.fit(X_train_fold, y_train_fold, max_epochs=50)
    metrics = model.evaluate(X_val_fold, y_val_fold)
    scores.append(metrics["accuracy"])

print(f"CV accuracy: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
```

### Probability calibration

Calibrate probabilities for better confidence estimates:

```python
from sklearn.calibration import CalibratedClassifierCV

# Wrap DeepTab model
model = MambularClassifier()
calibrated = CalibratedClassifierCV(model, cv=3, method="sigmoid")
calibrated.fit(X_train, y_train)

# Calibrated probabilities
cal_probs = calibrated.predict_proba(X_test)
```

### Handling string labels

Convert string labels to integers:

```python
from sklearn.preprocessing import LabelEncoder

# String labels
y_str = ["cat", "dog", "cat", "bird", "dog"]

# Encode
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_str)  # [0, 1, 0, 2, 1]

# Train
model = MambularClassifier()
model.fit(X_train, y_encoded, max_epochs=50)

# Predict and decode
predictions = model.predict(X_test)
predicted_labels = encoder.inverse_transform(predictions)  # ["cat", "dog", ...]
```

## Best practices

1. **Check class distribution** before training
2. **Use stratified splits** for imbalanced data (automatic in v2.0)
3. **Monitor multiple metrics** not just accuracy
4. **Calibrate probabilities** if using them for decisions
5. **Consider class weights** for severe imbalance
6. **Use cross-validation** for small datasets
7. **Save best models** during training (automatic with early stopping)

## Troubleshooting

### Low accuracy on imbalanced data

- Check class distribution: `np.bincount(y_train)`
- Use class weights or resampling
- Evaluate with balanced metrics (F1, balanced accuracy)

### Overconfident probabilities

- Use probability calibration
- Increase dropout in model config
- Use label smoothing (advanced)

### Different test performance

- Ensure test data has same preprocessing
- Check for data leakage
- Verify class distributions are similar

## Next steps

- **[Regression](regression)** — Regression-specific concepts
- **[Distributional Regression](distributional_regression)** — Beyond point predictions
- **[Training and Evaluation](training_and_evaluation)** — Training loop details
- **[Tutorials: Classification](../../tutorials/classification)** — Complete workflows
