# Classification Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/basf/DeepTab/blob/main/docs/tutorials/notebooks/classification.ipynb)

This tutorial demonstrates how to train classification models with DeepTab using the sklearn-compatible API.

```{tip}
Click the badge above to run this tutorial interactively in Google Colab with free GPU access!
```

## Basic workflow

### Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deeptab.models import MambularClassifier
```

### Generate data

We create a synthetic dataset with 1,000 samples and 5 numeric features. The continuous target is bucketed into four quartile classes.

```python
np.random.seed(42)

n_samples, n_features = 1000, 5
X = np.random.randn(n_samples, n_features)
y_continuous = np.dot(X, np.random.randn(n_features)) + np.random.randn(n_samples)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
df["target"] = pd.qcut(y_continuous, q=4, labels=False)
```

### Split data

```python
X = df.drop(columns=["target"])
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Train

Instantiate `MambularClassifier` with default settings and fit on the training data.

```python
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

DeepTab automatically:

- Detects numerical vs categorical features
- Creates a validation split (20% by default)
- Applies stratified sampling for classification
- Enables early stopping
- Uses GPU if available

### Predict

Get class predictions:

```python
predictions = model.predict(X_test)
print(predictions[:10])
# [2 1 3 0 1 2 3 1 0 2]
```

Get class probabilities:

```python
probabilities = model.predict_proba(X_test)
print(probabilities[:3])
# [[0.05 0.15 0.70 0.10]
#  [0.10 0.65 0.20 0.05]
#  [0.02 0.08 0.15 0.75]]
```

### Evaluate

```python
metrics = model.evaluate(X_test, y_test)
print(metrics)
# {'accuracy': 0.85, 'loss': 0.42}
```

For sklearn compatibility, use `score()`:

```python
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.3f}")
```

### Save and load

```python
# Save trained model
model.save("my_classifier.pkl")

# Load later
from deeptab.models import MambularClassifier
loaded_model = MambularClassifier.load("my_classifier.pkl")
predictions = loaded_model.predict(X_test)
```

## Customization with configs

DeepTab uses three independent config classes for fine-grained control:

### Model architecture

```python
from deeptab.configs import MambularConfig

model_cfg = MambularConfig(
    d_model=128,          # Embedding dimension
    n_layers=6,           # Number of Mamba layers
    dropout=0.3,          # Dropout rate
    use_cls_token=True,   # Classification token
)

model = MambularClassifier(model_config=model_cfg)
model.fit(X_train, y_train, max_epochs=50)
```

### Preprocessing

```python
from deeptab.configs import PreprocessingConfig

prep_cfg = PreprocessingConfig(
    numerical_preprocessing="quantile",  # or "standard", "minmax", "ple", "binning"
    use_ple=True,                         # Piecewise Linear Encoding
    n_bins=50,                            # For binning/PLE
    categorical_preprocessing="ordinal",  # or "onehot"
    embedding_dim=16,                     # Categorical embedding dimension
)

model = MambularClassifier(preprocessing_config=prep_cfg)
model.fit(X_train, y_train, max_epochs=50)
```

### Training loop

```python
from deeptab.configs import TrainerConfig

trainer_cfg = TrainerConfig(
    lr=1e-3,                          # Learning rate
    batch_size=256,                   # Batch size
    max_epochs=100,                   # Max epochs
    patience=15,                      # Early stopping patience
    lr_scheduler="reduce_on_plateau", # LR scheduling
    optimizer="adamw",                # Optimizer
    weight_decay=1e-4,                # L2 regularization
)

model = MambularClassifier(trainer_config=trainer_cfg)
model.fit(X_train, y_train, max_epochs=trainer_cfg.max_epochs)
```

### Combine all configs

```python
model = MambularClassifier(
    model_config=model_cfg,
    preprocessing_config=prep_cfg,
    trainer_config=trainer_cfg,
)
model.fit(X_train, y_train, max_epochs=100)
```

## Handling class imbalance

### Stratified splits (automatic in v2.0)

DeepTab automatically uses stratified sampling for train/validation splits in classification:

```python
# Validation split is stratified by default
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)  # Creates stratified 80/20 train/val split
```

Provide explicit validation set for custom splits:

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model.fit(X_train, y_train, X_val=X_val, y_val=y_val, max_epochs=50)
```

### Class weights

Balance classes with automatic weighting:

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train
)

# Convert to dict for loss function
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Pass to trainer config (requires custom loss - advanced usage)
# For most cases, stratified sampling is sufficient
```

## Integration with scikit-learn

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model_config__d_model": [64, 128, 256],
    "model_config__n_layers": [4, 6, 8],
    "trainer_config__lr": [1e-4, 5e-4, 1e-3],
    "preprocessing_config__numerical_preprocessing": ["standard", "quantile"],
}

model = MambularClassifier()

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=1,  # Use 1 for GPU models
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Use best model
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
```

### Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Note: DeepTab handles preprocessing internally, but you can still use pipelines
pipeline = Pipeline([
    ("classifier", MambularClassifier()),
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Cross-validation

```python
from sklearn.model_selection import cross_val_score

model = MambularClassifier()

scores = cross_val_score(
    model, X_train, y_train,
    cv=5,
    scoring="accuracy",
)

print(f"CV scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

## Advanced patterns

### Binary classification

```python
# Binary classification (2 classes)
y_binary = (y > 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Probability outputs
proba = model.predict_proba(X_test)
print(proba[:3])
# [[0.85 0.15]
#  [0.23 0.77]
#  [0.92 0.08]]

# Get probability for positive class
positive_proba = proba[:, 1]
```

### Mixed data types

DeepTab automatically handles mixed numerical and categorical features:

```python
df = pd.DataFrame({
    "age": np.random.randint(18, 80, size=1000),
    "income": np.random.randint(20000, 200000, size=1000),
    "city": np.random.choice(["NYC", "LA", "Chicago"], size=1000),
    "education": np.random.choice(["HS", "BS", "MS", "PhD"], size=1000),
    "target": np.random.randint(0, 2, size=1000),
})

X = df.drop(columns=["target"])
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Automatically detects numerical (age, income) and categorical (city, education)
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

### With pre-computed embeddings

Add external embeddings (e.g., from text or images):

```python
# Assume we have text descriptions encoded to embeddings
text_embeddings_train = np.random.randn(len(X_train), 128)  # 128-dim embeddings
text_embeddings_test = np.random.randn(len(X_test), 128)

model = MambularClassifier()
model.fit(
    X_train, y_train,
    X_embedding=text_embeddings_train,
    max_epochs=50,
)

predictions = model.predict(X_test, X_embedding=text_embeddings_test)
```

### Ensemble predictions

```python
# Train multiple models
models = []
for seed in [42, 123, 456]:
    np.random.seed(seed)
    model = MambularClassifier()
    model.fit(X_train, y_train, max_epochs=50)
    models.append(model)

# Average predictions
all_proba = np.array([m.predict_proba(X_test) for m in models])
ensemble_proba = all_proba.mean(axis=0)
ensemble_pred = ensemble_proba.argmax(axis=1)

from sklearn.metrics import accuracy_score
print(f"Ensemble accuracy: {accuracy_score(y_test, ensemble_pred):.3f}")
```

## Using your own data

Replace the synthetic data with your CSV:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from deeptab.models import MambularClassifier

# Load data
df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=100)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(metrics)

# Get predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## All stable classifiers

Swap `MambularClassifier` for any class below — no other code changes needed:

| Class                      | Architecture                          | Best for                         |
| -------------------------- | ------------------------------------- | -------------------------------- |
| `MLPClassifier`            | Feedforward MLP                       | Fastest baseline                 |
| `ResNetClassifier`         | Residual MLP                          | Deeper networks                  |
| `FTTransformerClassifier`  | Feature-Tokenizer Transformer         | General-purpose strong baseline  |
| `TabTransformerClassifier` | Transformer on categorical embeddings | Categorical-heavy data           |
| `SAINTClassifier`          | Self + intersample attention          | Semi-supervised settings         |
| `TabMClassifier`           | Batch-ensembling MLP                  | Ensemble accuracy at low cost    |
| `TabRClassifier`           | Retrieval-augmented                   | Local similarity patterns        |
| `NODEClassifier`           | Differentiable decision trees         | Gradient-boosting inductive bias |
| `NDTFClassifier`           | Neural decision tree forest           | Tree ensemble benefits           |
| `TabulaRNNClassifier`      | RNN / LSTM / GRU                      | Sequential feature interactions  |
| `MambularClassifier`       | Stacked Mamba SSM                     | Efficient sequence modeling      |
| `MambaTabClassifier`       | Single Mamba block                    | Lightweight Mamba variant        |
| `MambAttentionClassifier`  | Mamba + attention hybrid              | Local + global patterns          |

Example:

```python
from deeptab.models import FTTransformerClassifier, ResNetClassifier, NODEClassifier

# Try different architectures with identical API
for ModelClass in [FTTransformerClassifier, ResNetClassifier, NODEClassifier]:
    model = ModelClass()
    model.fit(X_train, y_train, max_epochs=50)
    accuracy = model.score(X_test, y_test)
    print(f"{ModelClass.__name__}: {accuracy:.3f}")
```

```{note}
All stable classifiers share the same API. Import, instantiate, fit, predict — done.
```

## Next steps

- **Understand training** → Read [Training and Evaluation](../core_concepts/training_and_evaluation) to learn what happens during `fit()`
- **Handle imbalance** → See [Classification](../core_concepts/classification) for class imbalance strategies
- **Try regression** → Check out the [Regression Tutorial](regression)
- **Quantify uncertainty** → Explore [Distributional Regression Tutorial](distributional)
- **Full config reference** → Browse [API docs](../api/configs/index)
