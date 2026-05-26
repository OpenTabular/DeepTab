# Classification

This page covers classification-specific concepts, including binary vs multiclass, class imbalance, stratification, and output formats.

## Creating a classifier

Import any model with the `Classifier` suffix:

```python
from deeptab.models import Mambul

arClassifier

model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=100)
predictions = model.predict(X_test)
```

All stable models are available as classifiers. See [Model Tiers](model_tiers) for the full list.

## Binary classification

Binary classification predicts one of two classes (0 or 1).

### Labels

Labels should be integers (0 or 1) or boolean:

```python
y = [0, 1, 0, 1, 1, 0]  # ✓ integers
y = [False, True, False, True]  # ✓ boolean
y = ["no", "yes", "no", "yes"]  # ✗ strings (convert first)
```

### Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from deeptab.models import MambularClassifier

# Binary classification data
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Predict class labels
predictions = model.predict(X_test)  # [0, 1, 1, 0, ...]

# Predict probabilities
probabilities = model.predict_proba(X_test)
# [[0.9, 0.1],   # 90% class 0
#  [0.3, 0.7],   # 70% class 1
#  ...]
```

### Probability outputs

`predict_proba` returns a 2D array with shape `(n_samples, 2)`:

```python
probs = model.predict_proba(X_test)

# Class 0 probabilities
p_class_0 = probs[:, 0]

# Class 1 probabilities
p_class_1 = probs[:, 1]

# They sum to 1
assert np.allclose(p_class_0 + p_class_1, 1.0)
```

### Decision threshold

By default, predictions use threshold 0.5. For custom thresholds:

```python
probs = model.predict_proba(X_test)
custom_predictions = (probs[:, 1] > 0.7).astype(int)  # 70% threshold
```

## Multiclass classification

Multiclass predicts one of N classes (N > 2).

### Labels

Labels should be integers from 0 to N-1:

```python
y = [0, 1, 2, 0, 2, 1]  # ✓ 3 classes (0, 1, 2)
y = [1, 2, 3, 1, 3, 2]  # ✗ Must start from 0 (convert with LabelEncoder)
```

### Example

```python
from sklearn.datasets import make_classification
from deeptab.models import FTTransformerClassifier

# 5-class problem
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=5,
    n_informative=15,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = FTTransformerClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Predict
predictions = model.predict(X_test)  # [0, 2, 4, 1, ...]

# Probabilities
probabilities = model.predict_proba(X_test)
# Shape: (n_samples, 5)
# Each row sums to 1
```

### Probability outputs

For N classes, `predict_proba` returns shape `(n_samples, N)`:

```python
probs = model.predict_proba(X_test)  # (200, 5) for 5 classes

# Probability of class 2 for all samples
p_class_2 = probs[:, 2]

# Most likely class (same as model.predict)
predicted_classes = np.argmax(probs, axis=1)
```

### Confidence scores

Get the confidence (max probability) for each prediction:

```python
probs = model.predict_proba(X_test)
confidence = np.max(probs, axis=1)

# Samples with low confidence (< 50%)
uncertain = confidence < 0.5
print(f"Uncertain predictions: {uncertain.sum()}")
```

## Class imbalance

Imbalanced datasets have unequal class distributions (e.g., 95% class 0, 5% class 1).

### Stratified splits

Starting in v2.0, DeepTab automatically uses stratified train/val splits for classification, preserving class distributions:

```python
# Imbalanced data: 90% class 0, 10% class 1
X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    weights=[0.9, 0.1],
    flip_y=0,
    random_state=42,
)

# Automatic stratification during fit
model = MambularClassifier()
model.fit(X, y, max_epochs=50)
# Validation set will also have 90/10 split
```

### Class weights

For severe imbalance, use class weights in the loss function:

```python
from deeptab.configs import TrainerConfig

# Compute class weights (inversely proportional to frequency)
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(y_train),
    y=y_train,
)

# Pass to trainer config
cfg = TrainerConfig(class_weights=class_weights)
model = MambularClassifier(trainer_config=cfg)
model.fit(X_train, y_train, max_epochs=50)
```

### Oversampling/undersampling

Apply before passing to DeepTab:

```python
from imblearn.over_sampling import SMOTE

# Oversample minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train on resampled data
model = MambularClassifier()
model.fit(X_resampled, y_resampled, max_epochs=50)
```

### Evaluation metrics for imbalanced data

Accuracy can be misleading for imbalanced data. Use other metrics:

```python
from sklearn.metrics import classification_report, balanced_accuracy_score

predictions = model.predict(X_test)

# Balanced accuracy
balanced_acc = balanced_accuracy_score(y_test, predictions)

# Full report
print(classification_report(y_test, predictions))
```

## Evaluation metrics

### Default: accuracy

```python
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Loss: {metrics['loss']:.3f}")
```

### Custom metrics

Specify metrics in `TrainerConfig`:

```python
from torchmetrics import F1Score, Precision, Recall
from deeptab.configs import TrainerConfig

cfg = TrainerConfig(
    metrics=[
        F1Score(task="binary", average="macro"),
        Precision(task="binary", average="macro"),
        Recall(task="binary", average="macro"),
    ]
)

model = MambularClassifier(trainer_config=cfg)
model.fit(X_train, y_train, max_epochs=50)

# Evaluate with all metrics
metrics = model.evaluate(X_test, y_test)
print(metrics)  # Includes accuracy, F1, precision, recall
```

### scikit-learn metrics

Use after prediction:

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"F1: {f1_score(y_test, predictions, average='macro'):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, probabilities[:, 1]):.3f}")  # Binary
```

## Multioutput classification

For multiple binary classification tasks, use separate models:

```python
# Multi-label data
y1 = [0, 1, 0, 1]  # Label 1
y2 = [1, 1, 0, 0]  # Label 2

# Train separate models
model1 = MambularClassifier()
model1.fit(X_train, y1_train, max_epochs=50)

model2 = MambularClassifier()
model2.fit(X_train, y2_train, max_epochs=50)

# Predict
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
```

Or stack predictions:

```python
preds = np.column_stack([pred1, pred2])
```

## Output formats

### predict()

Returns class labels as integers:

```python
predictions = model.predict(X_test)
# [0, 1, 2, 0, 1, ...]
print(predictions.dtype)  # int64
print(predictions.shape)  # (n_samples,)
```

### predict_proba()

Returns probabilities as floats:

```python
probabilities = model.predict_proba(X_test)
# [[0.8, 0.1, 0.1],
#  [0.2, 0.7, 0.1],
#  ...]
print(probabilities.dtype)  # float32
print(probabilities.shape)  # (n_samples, n_classes)
```

### evaluate()

Returns dict of metrics:

```python
metrics = model.evaluate(X_test, y_test)
# {'accuracy': 0.85, 'loss': 0.42, ...}
print(type(metrics))  # dict
```

## Label shapes (v2.0)

DeepTab v2.0 enforces consistent label shapes:

### During training

- **Multiclass**: Shape `(n_samples,)`, dtype `int64`
- **Binary**: Shape `(n_samples, 1)`, dtype `float32`

```python
# Multiclass
y_train = np.array([0, 1, 2, 0, 1])  # Shape: (5,)

# Binary (automatically reshaped internally if needed)
y_train = np.array([0, 1, 0, 1])  # Shape: (4,) → internally (4, 1)
```

The high-level estimator API handles this automatically. Only relevant if using `TabularDataModule` directly.

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
- **[Examples: Classification](../../examples/classification)** — Complete workflows
