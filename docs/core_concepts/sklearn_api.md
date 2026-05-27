# scikit-learn Compatible API

DeepTab models implement the scikit-learn `BaseEstimator` interface, making them drop-in replacements for traditional machine learning models.

```{tip}
If you've used scikit-learn before, you already know how to use DeepTab. The API is identical.
```

## The four-step workflow

Every DeepTab model follows the same pattern:

```python
from deeptab.models import MambularClassifier

# 1. Instantiate
model = MambularClassifier()

# 2. Fit
model.fit(X_train, y_train, max_epochs=100)

# 3. Predict
predictions = model.predict(X_test)

# 4. Evaluate
metrics = model.evaluate(X_test, y_test)
```

This consistency means you can swap models without changing your workflow.

## Accepted input formats

DeepTab accepts the same data formats as scikit-learn:

```{important}
**Recommended:** Use **pandas DataFrames** for automatic feature type detection (numerical vs categorical). NumPy arrays treat all features as numerical.
```

### DataFrames (recommended)

```python
import pandas as pd

df = pd.DataFrame({
    "age": [25, 32, 47],
    "city": ["NYC", "Boston", "Chicago"],
    "income": [50000, 75000, 90000],
})

model = MambularClassifier()
model.fit(df, y, max_epochs=50)
```

DataFrames preserve column names and types, which helps with feature type detection and preprocessing.

### NumPy arrays

```python
import numpy as np

X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, size=1000)

model = MambularClassifier()
model.fit(X, y, max_epochs=50)
```

When using NumPy arrays, all features are treated as numerical by default.

### Mixed types

DeepTab automatically handles mixed numerical and categorical features in DataFrames:

```python
data = pd.DataFrame({
    "age": [25, 32, 47],               # numerical
    "city": ["NYC", "Boston", "Chicago"],  # categorical
    "has_degree": [True, False, True],     # categorical
})

model.fit(data, y, max_epochs=50)  # Handles types automatically
```

## Core methods

### fit()

Train the model on data:

```python
model.fit(X_train, y_train, max_epochs=100)
```

**Parameters:**

- `X_train`: Features (DataFrame or array)
- `y_train`: Labels (array-like)
- `max_epochs`: Maximum training epochs
- `X_val`, `y_val`: Optional validation set
- `X_embedding`: Optional pre-computed embeddings

**Behavior:**

```{note}
**Automatic during `fit()`:**
- ✅ Preprocessing (feature detection, encoding, scaling)
- ✅ Train/validation split (if no `X_val` provided)
- ✅ Stratification (for classification)
- ✅ Early stopping (based on validation loss)
- ✅ Best model checkpointing
```

- Returns `self` for method chaining

**Example with validation set:**

```python
model.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    max_epochs=100,
)
```

### predict()

Generate predictions on new data:

```python
predictions = model.predict(X_test)
```

**Returns:**

- **Classification**: Class labels (integers)
- **Regression**: Continuous values (floats)
- **LSS**: Distribution parameters (2D array)

**Example:**

```python
# Classification
predictions = model.predict(X_test)  # [0, 1, 0, 1, ...]

# Regression
predictions = model.predict(X_test)  # [1.23, 4.56, 7.89, ...]

# LSS (distributional)
params = model.predict(X_test)  # [[mean1, std1], [mean2, std2], ...]
```

### predict_proba()

Get class probabilities (classification only):

```python
probabilities = model.predict_proba(X_test)
```

**Returns:**

- 2D array with shape `(n_samples, n_classes)`
- Each row sums to 1.0

**Example:**

```python
probs = model.predict_proba(X_test)
# [[0.8, 0.1, 0.1],   # Sample 1: 80% class 0
#  [0.2, 0.7, 0.1],   # Sample 2: 70% class 1
#  [0.1, 0.1, 0.8]]   # Sample 3: 80% class 2
```

### evaluate()

Compute metrics on a test set:

```python
metrics = model.evaluate(X_test, y_test)
```

**Returns:**

- Dictionary of metrics appropriate for the task

**Classification metrics:**

- `accuracy`: Overall accuracy
- `loss`: Cross-entropy loss
- Additional metrics if specified in `TrainerConfig`

**Regression metrics:**

- `rmse`: Root mean squared error
- `mae`: Mean absolute error
- `loss`: MSE loss

**Example:**

```python
metrics = model.evaluate(X_test, y_test)
print(f"Test accuracy: {metrics['accuracy']:.3f}")
print(f"Test loss: {metrics['loss']:.3f}")
```

### score()

Get the default scoring metric (scikit-learn compatibility):

```python
score = model.score(X_test, y_test)
```

**Returns:**

- **Classification**: Accuracy
- **Regression**: R² score

This is useful for compatibility with scikit-learn tools like `GridSearchCV`.

### save() and load()

Persist trained models to disk:

```python
# Save
model.fit(X_train, y_train, max_epochs=50)
model.save("my_model.pkl")

# Load
from deeptab.models import MambularClassifier
loaded_model = MambularClassifier.load("my_model.pkl")
predictions = loaded_model.predict(X_test)
```

The saved file includes:

- Model architecture and weights
- Preprocessing state (fitted transformers)
- Configuration objects
- Training history

## Integration with scikit-learn tools

Because DeepTab implements `BaseEstimator`, it works seamlessly with the entire scikit-learn ecosystem.

### Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from deeptab.models import MambularClassifier

pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Optional: DeepTab does its own scaling
    ("model", MambularClassifier()),
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

Note: DeepTab applies its own preprocessing, so adding additional preprocessing steps may be redundant.

### Cross-validation

```python
from sklearn.model_selection import cross_val_score
from deeptab.models import FTTransformerClassifier

model = FTTransformerClassifier()
scores = cross_val_score(
    model, X, y,
    cv=5,
    scoring="accuracy",
)
print(f"CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from deeptab.models import MambularClassifier

param_grid = {
    "model_config__d_model": [64, 128, 256],
    "model_config__n_layers": [4, 6, 8],
    "trainer_config__lr": [1e-3, 5e-4, 1e-4],
}

search = GridSearchCV(
    estimator=MambularClassifier(),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=1,  # Each model uses GPU, avoid parallel
)

search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(f"Best score: {search.best_score_:.3f}")

# Use best model
best_model = search.best_estimator_
```

### RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    "model_config__d_model": randint(32, 256),
    "model_config__n_layers": randint(2, 10),
    "trainer_config__lr": uniform(1e-4, 1e-2),
}

search = RandomizedSearchCV(
    estimator=MambularClassifier(),
    param_distributions=param_distributions,
    n_iter=20,
    cv=3,
    random_state=42,
)

search.fit(X_train, y_train)
```

## Parameter access via get_params / set_params

DeepTab configs implement the scikit-learn parameter protocol:

### Inspecting parameters

```python
from deeptab.configs import MambularConfig

cfg = MambularConfig(d_model=128, n_layers=6)
params = cfg.get_params()
print(params)
# {'d_model': 128, 'n_layers': 6, 'dropout': 0.2, ...}
```

### Updating parameters

```python
cfg.set_params(d_model=256, dropout=0.3)
print(cfg.d_model)  # 256
print(cfg.dropout)  # 0.3
```

### Model-level parameters

The estimator delegates to its configs using double-underscore notation:

```python
model = MambularClassifier()

# Get all parameters
all_params = model.get_params()

# Update via double-underscore
model.set_params(
    model_config__d_model=128,
    trainer_config__lr=1e-3,
)
```

This enables GridSearchCV to work seamlessly:

```python
param_grid = {
    "model_config__d_model": [64, 128],     # Searches MambularConfig.d_model
    "trainer_config__lr": [1e-3, 5e-4],     # Searches TrainerConfig.lr
}
```

## Differences from standard scikit-learn

While DeepTab follows scikit-learn conventions, there are a few differences:

### 1. Training happens during fit

Unlike scikit-learn models that fit instantly, DeepTab models train neural networks, which takes time:

```python
# This runs multiple epochs of gradient descent
model.fit(X_train, y_train, max_epochs=100)
```

You can monitor progress via the progress bar or enable verbose logging.

### 2. GPU usage

DeepTab automatically uses GPU if available. You don't need to specify this:

```python
import torch
print(torch.cuda.is_available())  # True

# Automatically uses GPU
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

To force CPU:

```python
from deeptab.configs import TrainerConfig

model = MambularClassifier(
    trainer_config=TrainerConfig(device="cpu")
)
```

### 3. Validation sets are encouraged

DeepTab benefits from explicit validation sets for early stopping:

```python
model.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,  # Recommended
    max_epochs=100,
)
```

If you don't provide one, DeepTab creates it automatically via train/val split.

### 4. Additional fit parameters

DeepTab's `fit` method accepts extra parameters:

- `max_epochs`: Number of training epochs
- `X_val`, `y_val`: Validation set
- `X_embedding`: Pre-computed embeddings
- `family`: Distribution family (LSS models only)

### 5. Task-specific outputs

The output format of `predict` varies by task:

```python
# Classifier returns integers
clf_pred = classifier.predict(X)  # [0, 1, 2, ...]

# Regressor returns floats
reg_pred = regressor.predict(X)  # [1.23, 4.56, ...]

# LSS returns 2D array of parameters
lss_pred = lss_model.predict(X)  # [[mean, std], ...]
```

## Method chaining

`fit` returns `self`, enabling method chaining:

```python
predictions = (
    MambularClassifier()
    .fit(X_train, y_train, max_epochs=50)
    .predict(X_test)
)
```

This is idiomatic for quick experiments but less common in production code.

## Reproducibility

For reproducible results, set random seeds:

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

For full reproducibility (at the cost of performance):

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Next steps

- **[Model Tiers](model_tiers)** — Understand stable vs experimental models
- **[Config System](config_system)** — Learn the split-config API
- **[Classification](classification)** — Classification-specific details
- **[Regression](regression)** — Regression-specific details
