# Preprocessing

DeepTab automatically detects feature types and applies appropriate preprocessing. This page explains how preprocessing works, available strategies, and how to customize them.

## Automatic feature type detection

DeepTab infers feature types from DataFrame dtypes:

| DataFrame dtype              | DeepTab type | Default preprocessing        |
| ---------------------------- | ------------ | ---------------------------- |
| `int`, `float`               | Numerical    | Standardization              |
| `object`, `category`, `bool` | Categorical  | Ordinal encoding + embedding |

### Example

```python
import pandas as pd

df = pd.DataFrame({
    "age": [25, 32, 47],              # int → numerical
    "income": [50000.0, 75000.0, 90000.0],  # float → numerical
    "city": ["NYC", "Boston", "Chicago"],    # object → categorical
    "employed": [True, False, True],         # bool → categorical
})

model = MambularClassifier()
model.fit(df, y, max_epochs=50)  # Automatic type detection
```

### Forcing categorical treatment

If you have numerical IDs that should be categorical:

```python
df["user_id"] = df["user_id"].astype("category")
df["zip_code"] = df["zip_code"].astype("str")  # or "object"
```

### NumPy arrays

When using NumPy arrays, all features are treated as numerical:

```python
X = np.random.randn(1000, 10)  # All 10 features are numerical
```

## Numerical preprocessing

Numerical features go through three stages:

1. **Imputation** — Fill missing values
2. **Encoding** — Transform values (optional)
3. **Scaling** — Standardize ranges

### Preprocessing strategies

Configure via `PreprocessingConfig`:

```python
from deeptab.configs import PreprocessingConfig

cfg = PreprocessingConfig(
    numerical_preprocessing="quantile",  # The main strategy
    scaling_strategy="standard",         # Post-encoding scaling
)
```

### Available strategies

#### standard (default)

Z-score standardization: $x_{scaled} = \frac{x - \mu}{\sigma}$

```python
cfg = PreprocessingConfig(numerical_preprocessing="standard")
```

**When to use:**

- Features are approximately normally distributed
- No extreme outliers
- General-purpose default

**Example:**

```python
# Before: [1, 2, 3, 4, 5]
# After:  [-1.41, -0.71, 0, 0.71, 1.41]
```

#### quantile

Maps features to a uniform distribution using quantile transformation:

```python
cfg = PreprocessingConfig(numerical_preprocessing="quantile")
```

**When to use:**

- Features have outliers
- Skewed distributions
- Mixed scales across features

**Advantages:**

- Robust to outliers
- Makes distributions more uniform
- Improves neural network training

**Example:**

```python
# Before: [1, 2, 3, 100]  # Outlier
# After:  [0.25, 0.50, 0.75, 1.0]  # Uniform
```

#### minmax

Scales to [0, 1] range: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$

```python
cfg = PreprocessingConfig(numerical_preprocessing="minmax")
```

**When to use:**

- Features are already bounded
- Need output in specific range
- Interpretability matters

**Disadvantages:**

- Sensitive to outliers
- Can compress most values if outliers exist

#### ple (Piecewise Linear Encoding)

Approximates non-linear transformations with piecewise linear functions:

```python
cfg = PreprocessingConfig(
    numerical_preprocessing="ple",
    n_bins=50,  # Number of segments
)
```

**When to use:**

- Non-linear relationships with target
- Want to capture complex patterns
- Have sufficient data

**How it works:**

- Divides range into bins
- Learns linear transformation per bin
- Can capture monotonic non-linearities

#### binning

Converts numerical to categorical by creating bins:

```python
cfg = PreprocessingConfig(
    numerical_preprocessing="binning",
    n_bins=10,
)
```

**When to use:**

- Want to treat numerical as categorical
- Have very few unique values
- Interpretability is important

**Example:**

```python
# age: [25, 32, 47, 51, 62]
# bins: [0-30), [30-40), [40-50), [50-60), [60+]
# encoded: [0, 1, 2, 2, 3]
```

### Scaling strategy

Applied after the main preprocessing:

```python
cfg = PreprocessingConfig(
    numerical_preprocessing="ple",
    scaling_strategy="standard",  # Options: "standard", "minmax", "robust", "none"
)
```

| Strategy     | Description             | When to use      |
| ------------ | ----------------------- | ---------------- |
| `"standard"` | Z-score standardization | General purpose  |
| `"minmax"`   | Scale to [0, 1]         | Bounded features |
| `"robust"`   | Median and IQR based    | With outliers    |
| `"none"`     | No scaling              | Already scaled   |

### Missing value handling

DeepTab handles missing values automatically:

```python
cfg = PreprocessingConfig(
    numerical_imputation_strategy="median",  # Options: "median", "mean", "zero"
)
```

| Strategy   | Behavior                   | When to use          |
| ---------- | -------------------------- | -------------------- |
| `"median"` | Fill with median (default) | Robust to outliers   |
| `"mean"`   | Fill with mean             | Normally distributed |
| `"zero"`   | Fill with 0                | Sparse data          |

## Categorical preprocessing

Categorical features are encoded then embedded:

1. **Ordinal encoding** — Map categories to integers
2. **Embedding** — Learn dense representations

### Basic configuration

```python
cfg = PreprocessingConfig(
    cat_encoding_strategy="ordinal",  # Currently only option
    embedding_dim=None,                # Auto-computed by default
)
```

### Embedding dimensions

By default, DeepTab uses: $d_{embed} = \min(50, \lceil n_{categories}^{0.5} \rceil)$

Override for all categoricals:

```python
cfg = PreprocessingConfig(embedding_dim=64)
```

Or let DeepTab compute per-feature automatically:

```python
# Auto: city (5 categories) → embed_dim = 3
# Auto: country (200 categories) → embed_dim = 14
cfg = PreprocessingConfig(embedding_dim=None)  # Default
```

### High-cardinality categories

For features with many categories (e.g., user IDs with 100K+ values):

```python
cfg = PreprocessingConfig(
    embedding_dim=128,  # Larger embeddings for high cardinality
)
```

Or consider target encoding / feature hashing (requires manual preprocessing before DeepTab).

### Boolean features

Treated as categorical with 2 categories:

```python
df["is_member"] = [True, False, True, False]
# Encoded as: [1, 0, 1, 0]
# Embedded to: learnable 2-way embedding
```

### Missing categorical values

```python
cfg = PreprocessingConfig(
    categorical_imputation_strategy="mode",  # Options: "mode", "constant"
)
```

| Strategy     | Behavior                          | When to use           |
| ------------ | --------------------------------- | --------------------- |
| `"mode"`     | Fill with most frequent (default) | General purpose       |
| `"constant"` | Fill with a special category      | Missingness is signal |

## Pre-computed embeddings

If you have embeddings from external models (text encoders, image models), pass them via `X_embedding`:

```python
from sentence_transformers import SentenceTransformer

# Generate text embeddings
text_model = SentenceTransformer("all-MiniLM-L6-v2")
text_embeddings = text_model.encode(df["description"].tolist())
# Shape: (n_samples, 384)

# Tabular features
X_tabular = df.drop(columns=["description", "target"])

# Fit with both
model = MambularClassifier()
model.fit(
    X_tabular,
    y,
    X_embedding=text_embeddings,  # Concatenated with tabular features
    max_epochs=50,
)
```

### Multiple embedding sources

Concatenate them before passing:

```python
import numpy as np

text_embeds = text_model.encode(df["text"])       # (n, 384)
image_embeds = image_model.encode(df["image"])    # (n, 512)

combined_embeds = np.concatenate([text_embeds, image_embeds], axis=1)  # (n, 896)

model.fit(X_tabular, y, X_embedding=combined_embeds, max_epochs=50)
```

## Preprocessing pipeline

The full preprocessing pipeline:

```
1. Feature type detection
   ├─ DataFrame dtypes → numerical vs categorical
   └─ NumPy arrays → all numerical

2. Missing value imputation
   ├─ Numerical: median/mean/zero
   └─ Categorical: mode/constant

3. Numerical encoding
   ├─ standard / quantile / minmax / ple / binning
   └─ Transform values

4. Numerical scaling
   └─ standard / minmax / robust / none

5. Categorical encoding
   ├─ Ordinal encoding (categories → integers)
   └─ Embedding layer (integers → dense vectors)

6. Concatenation
   └─ [numerical_encoded, categorical_embedded, external_embeddings]

7. Feed to neural network
```

## Validation set preprocessing

When you provide a validation set, it uses the same transformations fitted on the training set:

```python
model.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,  # Uses train-fitted transformers
    max_epochs=100,
)
```

**Important:** Validation and test sets must have the same feature names and types as training data.

## Handling new categories at inference

If test data has categories not seen during training, they're mapped to a special "unknown" category:

```python
# Training: city in ["NYC", "Boston", "Chicago"]
model.fit(X_train, y_train, max_epochs=50)

# Test: city includes "Miami" (unseen)
predictions = model.predict(X_test)  # "Miami" → unknown category
```

## Custom preprocessing

If you need custom preprocessing, apply it before passing to DeepTab:

```python
# Custom log transform
df["log_income"] = np.log1p(df["income"])

# Custom binning
df["age_group"] = pd.cut(df["age"], bins=[0, 18, 35, 50, 100]).astype("category")

# Then use DeepTab
model = MambularClassifier()
model.fit(df, y, max_epochs=50)
```

DeepTab will still apply its own preprocessing on top, so consider:

```python
# Disable DeepTab's preprocessing if you've already done it
cfg = PreprocessingConfig(
    numerical_preprocessing="standard",  # Minimal: just standardize
    scaling_strategy="none",             # No additional scaling
)
```

## Inspecting preprocessing

After fitting, inspect the preprocessing state:

```python
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Access the data module
datamodule = model.model.datamodule

# Numerical feature names
print(datamodule.num_feature_info)

# Categorical feature names and cardinalities
print(datamodule.cat_feature_info)

# Embedding feature info (if provided)
print(datamodule.embedding_feature_info)

# Feature schema
schema = datamodule.schema
print(f"Numerical features: {schema.num_numerical_features}")
print(f"Categorical features: {schema.num_categorical_features}")
print(f"Total dimensions: {schema.total_numerical_dims + schema.total_embedding_dims}")
```

## Preprocessing for different tasks

Preprocessing is the same across classification, regression, and LSS:

```python
# Same preprocessing config works for all tasks
cfg = PreprocessingConfig(numerical_preprocessing="quantile")

classifier = MambularClassifier(preprocessing_config=cfg)
regressor = MambularRegressor(preprocessing_config=cfg)
lss_model = MambularLSS(preprocessing_config=cfg)
```

## Performance considerations

### Speed

- `"standard"` and `"minmax"` are fastest
- `"quantile"` is slower but more robust
- `"ple"` has moderate overhead

For large datasets (1M+ samples), prefer `"standard"` or `"minmax"`.

### Memory

Preprocessing is done in memory. For very large datasets:

1. Use smaller batch sizes
2. Consider subsampling for preprocessing (fit on subset, transform all)
3. Or use out-of-core preprocessing with Dask/Vaex before DeepTab

## Common recipes

### Default (recommended starting point)

```python
# Let DeepTab handle everything
model = MambularClassifier()
```

### Data with outliers

```python
cfg = PreprocessingConfig(numerical_preprocessing="quantile")
model = MambularClassifier(preprocessing_config=cfg)
```

### Interpretable bins

```python
cfg = PreprocessingConfig(
    numerical_preprocessing="binning",
    n_bins=10,
)
model = MambularClassifier(preprocessing_config=cfg)
```

### High-cardinality categoricals

```python
cfg = PreprocessingConfig(embedding_dim=128)
model = MambularClassifier(preprocessing_config=cfg)
```

### Minimal preprocessing (you've done most of it)

```python
cfg = PreprocessingConfig(
    numerical_preprocessing="standard",
    scaling_strategy="none",
)
model = MambularClassifier(preprocessing_config=cfg)
```

## Troubleshooting

### "ValueError: Unknown category"

**Cause:** Test set has a category not in training set.

**Solution:** DeepTab handles this automatically by mapping to unknown. If you want to avoid it, ensure train set has all categories:

```python
# Include all categories in training
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=X["category_column"],  # Ensures all categories in both splits
)
```

### "Memory error during preprocessing"

**Solution:** Reduce batch size or use a subset for fitting transformers:

```python
# Fit preprocessing on a subset
sample_indices = np.random.choice(len(X_train), size=10000, replace=False)
X_sample = X_train.iloc[sample_indices]
y_sample = y_train[sample_indices]

model.fit(X_sample, y_sample, max_epochs=50)
```

### Preprocessing is slow

**Solution:** Use simpler strategies:

```python
cfg = PreprocessingConfig(
    numerical_preprocessing="standard",  # Faster than quantile
)
```

## Next steps

- **[Classification](classification)** — Classification-specific preprocessing notes
- **[Regression](regression)** — Regression-specific preprocessing notes
- **[Config System](config_system)** — Full PreprocessingConfig reference
- **[Training and Evaluation](training_and_evaluation)** — What happens after preprocessing
