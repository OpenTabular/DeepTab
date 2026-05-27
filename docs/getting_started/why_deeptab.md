# Why DeepTab

This page explains the specific advantages of using DeepTab and when it's the right tool for your project.

## You already know the API

If you've used scikit-learn, you already know how to use DeepTab. Every model follows the same pattern:

```python
from deeptab.models import MambularClassifier

model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=100)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
metrics = model.evaluate(X_test, y_test)
```

This means you can:

- **Drop in replacements** — Swap a `RandomForestClassifier` with `MambularClassifier` without changing other code
- **Use existing tools** — GridSearchCV, cross-validation, pipelines all work out of the box
- **Minimal learning curve** — If you know `fit` / `predict` / `evaluate`, you're ready to start

### Example: Grid search

```python
from sklearn.model_selection import GridSearchCV
from deeptab.models import FTTransformerClassifier

search = GridSearchCV(
    estimator=FTTransformerClassifier(),
    param_grid={
        "model_config__d_model": [64, 128],
        "model_config__n_layers": [4, 6, 8],
        "trainer_config__lr": [1e-3, 5e-4],
    },
    cv=5,
    scoring="accuracy",
)
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
```

No special handling needed—it just works.

## One model class, three tasks

Every architecture ships in three variants identified by the suffix:

| Suffix       | Task                      | Output                         |
| ------------ | ------------------------- | ------------------------------ |
| `Classifier` | Classification            | Class labels and probabilities |
| `Regressor`  | Regression                | Continuous point estimates     |
| `LSS`        | Distributional regression | Full distribution parameters   |

Switching between tasks is as simple as changing the import:

```python
from deeptab.models import MambularClassifier  # classification
from deeptab.models import MambularRegressor   # regression
from deeptab.models import MambularLSS         # distributional regression
```

The `fit` / `predict` / `evaluate` workflow stays identical across all three. This means:

- **Consistent API** — Learn it once, use it everywhere
- **Easy experimentation** — Try different task formulations without rewriting code
- **Unified codebase** — No separate implementations for each task type

### Example: Switching tasks

```python
# Same data, different tasks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Classification
clf = MambularClassifier()
clf.fit(X_train, y_train, max_epochs=50)
print(clf.evaluate(X_test, y_test))

# Regression (for continuous y)
reg = MambularRegressor()
reg.fit(X_train, y_train, max_epochs=50)
print(reg.evaluate(X_test, y_test))

# Distributional regression (for uncertainty)
lss = MambularLSS()
lss.fit(X_train, y_train, family="normal", max_epochs=50)
dist_params = lss.predict(X_test)  # Returns (mean, std) for each sample
```

## Automatic preprocessing

DeepTab inspects your DataFrame and applies sensible defaults without manual intervention:

### What's automatic

- **Type detection** — Identifies numerical vs categorical columns from dtypes
- **Categorical encoding** — Ordinal encoding + learned embeddings
- **Numerical scaling** — Standardization or quantile transform based on config
- **Missing values** — Handled internally during preprocessing
- **Batching** — Efficient data loading with PyTorch DataLoader

### Example: Mixed data types

```python
import pandas as pd
from deeptab.models import TabTransformerClassifier

# DataFrame with mixed types
data = pd.DataFrame({
    "age": [25, 32, 47, 51, 62],
    "income": [35000, 48000, 72000, 55000, 91000],
    "city": ["New York", "Boston", "Chicago", "Boston", "New York"],
    "has_degree": [True, True, False, True, False],
    "employment_status": ["full-time", "part-time", "full-time", "full-time", "retired"],
})

X = data  # No preprocessing needed
y = [0, 1, 1, 0, 1]

model = TabTransformerClassifier()
model.fit(X, y, max_epochs=50)  # Handles everything automatically
```

Numerical columns (`age`, `income`) are scaled. Categorical columns (`city`, `has_degree`, `employment_status`) are encoded and embedded. You don't need to manually split features or apply transformers.

### Configurable when needed

Override defaults through `PreprocessingConfig`:

```python
from deeptab.configs import PreprocessingConfig
from deeptab.models import MambularClassifier

model = MambularClassifier(
    preprocessing_config=PreprocessingConfig(
        numerical_preprocessing="quantile",  # Quantile transform instead of standard scaling
        n_bins=50,                           # For binning-based encodings
        scaling_strategy="minmax",           # MinMax scaling
    )
)
```

## Integrates with your workflow

Because DeepTab implements scikit-learn's `BaseEstimator` interface, it works seamlessly with the ecosystem you already use:

### Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from deeptab.models import MambularRegressor

pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Optional: DeepTab does its own scaling
    ("model", MambularRegressor()),
])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Cross-validation

```python
from sklearn.model_selection import cross_val_score
from deeptab.models import FTTransformerClassifier

model = FTTransformerClassifier()
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Hyperparameter search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from deeptab.models import MambularClassifier

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

## Designed for real data

Tabular datasets come with messy realities. DeepTab is built to handle them:

### Stratified splits for classification

Starting in v2.0, classification tasks automatically use stratified train/val splits to preserve class distributions. This is especially important for imbalanced datasets:

```python
from deeptab.models import MambularClassifier

# Imbalanced data: 80% class 0, 20% class 1
X, y = make_classification(n_samples=1000, weights=[0.8, 0.2])

model = MambularClassifier()
model.fit(X, y, max_epochs=50)  # Validation set preserves 80/20 ratio
```

For regression tasks, splits are random without stratification. If you pass an explicit `X_val` and `y_val`, those are used directly without further splitting.

### Flexible preprocessing strategies

Choose from multiple approaches based on your data:

| Strategy   | Use case                               |
| ---------- | -------------------------------------- |
| `standard` | Normally distributed features          |
| `quantile` | Features with outliers or skewed dists |
| `minmax`   | Bounded features (e.g., percentages)   |
| `ple`      | Piecewise linear encoding              |
| `binning`  | Convert to categorical bins            |

```python
from deeptab.configs import PreprocessingConfig

# For data with heavy outliers
cfg = PreprocessingConfig(numerical_preprocessing="quantile")
```

### Embeddings as inputs

Pass pre-computed embeddings (from text encoders, images, or any other source) alongside your tabular features:

```python
from deeptab.models import MambularClassifier

# Text embeddings from a sentence encoder
text_embeddings = sentence_model.encode(df["description"])  # shape: (n, 768)

model = MambularClassifier()
model.fit(
    X_train,
    y_train,
    X_embedding=text_embeddings,  # Concatenated with tabular features
    max_epochs=50,
)
```

### Custom metrics

Define your own evaluation metrics using PyTorch or Lightning conventions:

```python
from torchmetrics import F1Score
from deeptab.configs import TrainerConfig
from deeptab.models import MambularClassifier

model = MambularClassifier(
    trainer_config=TrainerConfig(
        metrics=[F1Score(task="binary")],
    )
)
```

## More than point predictions

Distributional regression (`LSS` models) goes beyond predicting a single number. Instead, you predict the parameters of a full probability distribution:

```python
from deeptab.models import MambularLSS

model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)

# Returns distribution parameters for each sample
# For family="normal", this is (mean, std)
params = model.predict(X_test)

mean_predictions = params[:, 0]
std_predictions = params[:, 1]

# Generate prediction intervals
lower_bound = mean_predictions - 1.96 * std_predictions
upper_bound = mean_predictions + 1.96 * std_predictions
```

### Why this matters

- **Uncertainty quantification** — Know when the model is confident vs uncertain
- **Risk-aware decisions** — Use full distribution for downstream optimization
- **Heteroscedastic noise** — Model varying noise levels across the input space
- **Quantile predictions** — Extract specific percentiles for business requirements

### Supported distributions

DeepTab supports a range of parametric families:

| Family              | Parameters     | Use case                       |
| ------------------- | -------------- | ------------------------------ |
| `normal`            | mean, std      | Continuous unbounded values    |
| `poisson`           | rate           | Count data                     |
| `gamma`             | shape, rate    | Positive continuous values     |
| `beta`              | alpha, beta    | Values in (0, 1)               |
| `negative_binomial` | n, p           | Overdispersed count data       |
| `student_t`         | df, loc, scale | Heavy-tailed continuous values |

See the API reference for the complete list.

## Performance at scale

DeepTab is designed to handle real-world dataset sizes efficiently:

### Batching and data loading

- Uses PyTorch `DataLoader` for efficient batching
- Supports multi-worker data loading (set `num_workers` in `TrainerConfig`)
- Automatic device placement (CPU or GPU)
- Pin memory for faster GPU transfers

### Memory efficiency

- Processes data in batches, not all at once
- Gradient accumulation for large effective batch sizes
- Automatic mixed precision training (AMP) available via Lightning

### Example: Large dataset

```python
from deeptab.configs import TrainerConfig
from deeptab.models import MambularClassifier

# Dataset with 1M samples
X_train, y_train = ...  # shape: (1_000_000, 50)

model = MambularClassifier(
    trainer_config=TrainerConfig(
        batch_size=512,      # Process 512 samples at a time
        num_workers=4,       # Parallel data loading
        max_epochs=50,
    )
)

model.fit(X_train, y_train)  # Handles batching automatically
```

## Experiment faster

DeepTab reduces the iteration time for modeling experiments:

### Quick baselines

Get a competitive baseline in 5 lines of code:

```python
from deeptab.models import MambularClassifier

model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)
print(model.evaluate(X_test, y_test))
```

### Easy architecture comparisons

Try different models by changing one import:

```python
from deeptab.models import (
    MambularClassifier,
    FTTransformerClassifier,
    TabTransformerClassifier,
    ResNetClassifier,
)

models = [
    MambularClassifier(),
    FTTransformerClassifier(),
    TabTransformerClassifier(),
    ResNetClassifier(),
]

for model in models:
    model.fit(X_train, y_train, max_epochs=50)
    metrics = model.evaluate(X_test, y_test)
    print(f"{model.__class__.__name__}: {metrics['accuracy']:.3f}")
```

### Hyperparameter search

Leverage scikit-learn's search tools without custom training code:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model_config__d_model": [64, 128, 256],
    "trainer_config__lr": [1e-3, 5e-4, 1e-4],
}

search = GridSearchCV(
    MambularClassifier(),
    param_grid,
    cv=5,
    n_jobs=-1,  # Parallel across folds
)
search.fit(X_train, y_train)
```

## When to choose DeepTab

DeepTab is a strong choice when you have:

✅ **Tabular data** with mixed feature types (numerical and categorical)  
✅ **Moderate to large datasets** (1K+ samples) where deep learning can excel  
✅ **Complex feature interactions** that benefit from learned representations  
✅ **Need for uncertainty** via distributional regression  
✅ **Integration requirements** with scikit-learn pipelines  
✅ **Time constraints** and need a quick competitive baseline

DeepTab may not be the best choice for:

❌ **Very small datasets** (< 1000 samples) — simpler models often work better  
❌ **Extremely large datasets** that don't fit in memory — consider XGBoost with out-of-core training  
❌ **Pure categorical data** — tree-based methods may be more efficient  
❌ **Strict latency requirements** — neural networks are slower than tree ensembles at inference

## Next steps

- **[Installation](installation)** — Set up DeepTab in your environment
- **[Quickstart](quickstart)** — Run your first model in 5 minutes
- **[Core Concepts](../core_concepts/index)** — Deep dive into the config system and API patterns
- **[Tutorials](../tutorials/classification)** — Complete end-to-end workflows
