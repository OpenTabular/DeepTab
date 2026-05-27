# Preprocessing

DeepTab delegates tabular preprocessing to `pretab.Preprocessor` and then converts the processed output into PyTorch tensors through `TabularDataModule` and `TabularDataset`.

```{important}
Use pandas DataFrames for mixed tabular data. DataFrames preserve column names and dtypes, which lets the preprocessor separate numerical and categorical features more reliably than NumPy arrays.
```

## Data Flow

The high-level `fit()` call builds this pipeline:

```text
raw X/y
  -> pretab.Preprocessor.fit(...)
  -> pretab.Preprocessor.transform(...)
  -> feature info dictionaries
  -> TabularDataset
  -> Lightning DataLoader
  -> DeepTab architecture
```

At prediction time, the fitted preprocessor is reused so new data follows the same transformations learned during training.

## Feature Type Handling

For pandas inputs, dtype information influences whether a feature is treated as numerical or categorical. For NumPy inputs, DeepTab first wraps the array as a DataFrame, so all columns typically behave as numerical unless configured otherwise.

```python
import pandas as pd

X = pd.DataFrame({
    "age": [25, 32, 47],
    "income": [50000.0, 75000.0, 90000.0],
    "city": pd.Series(["NYC", "Boston", "Chicago"], dtype="category"),
    "is_member": [True, False, True],
})
```

For identifier-like integer columns, convert them before fitting:

```python
X["zip_code"] = X["zip_code"].astype("category")
X["product_id"] = X["product_id"].astype("string")
```

Alternatively, use `PreprocessingConfig(cat_cutoff=..., treat_all_integers_as_numerical=...)` when the preprocessor should infer integer-column behavior.

## PreprocessingConfig

`PreprocessingConfig` contains only fields accepted by the current DeepTab wrapper:

```python
from deeptab.configs import PreprocessingConfig

cfg = PreprocessingConfig(
    numerical_preprocessing="quantile",
    categorical_preprocessing="int",
    n_bins=50,
    scaling_strategy="standard",
)
```

Common fields:

| Field | Use |
| --- | --- |
| `numerical_preprocessing` | Choose the numerical transform, such as `"standard"`, `"quantile"`, or `"ple"` where supported by `pretab`. |
| `categorical_preprocessing` | Choose categorical encoding, such as `"int"` or `"one-hot"` where supported. |
| `n_bins` | Number of bins for binned/PLE-style transforms. |
| `scaling_strategy` | Optional scaling after the main numerical transform. |
| `binning_strategy`, `use_decision_tree_bins` | How bin edges are built. |
| `n_knots`, `knots_strategy`, `degree`, `spline_implementation` | Spline-style preprocessing controls. |

Do not use `embedding_dim`, `cat_encoding_strategy`, `numerical_imputation_strategy`, or `categorical_imputation_strategy` in current `PreprocessingConfig`; those are not fields in the DeepTab 2.x config dataclass.

## Numerical Features

A practical starting point:

```python
PreprocessingConfig(numerical_preprocessing="standard")
```

For skewed or heavy-tailed numerical columns:

```python
PreprocessingConfig(numerical_preprocessing="quantile")
```

For piecewise encodings:

```python
PreprocessingConfig(
    numerical_preprocessing="ple",
    n_bins=50,
)
```

The exact available strategy names come from `pretab.Preprocessor`. DeepTab passes non-`None` config values directly to that preprocessor.

## Categorical Features

Categorical preprocessing happens before the neural architecture. DeepTab's neural models then consume either categorical tensors or embedded feature tokens depending on the architecture and model config.

```python
PreprocessingConfig(categorical_preprocessing="int")
```

Model-side embedding behavior is controlled by model config fields, for example:

```python
from deeptab.configs import MLPConfig

model_config = MLPConfig(
    use_embeddings=True,
    d_model=32,
    embedding_type="linear",
)
```

## External Embeddings

Some estimator methods accept precomputed embeddings through the `embeddings` and `embeddings_val` arguments.

```python
model.fit(
    X_train,
    y_train,
    embeddings=train_text_embeddings,
    embeddings_val=val_text_embeddings,
    X_val=X_val,
    y_val=y_val,
)

predictions = model.predict(X_test, embeddings=test_text_embeddings)
```

For multiple embedding sources, pass a list of arrays. Each array should have the same number of rows as the corresponding tabular input.

## Validation and Leakage

`TabularDataModule.preprocess_data()` currently fits the preprocessor on the combined training and validation features after the split is created. This means validation data can influence preprocessing statistics. For benchmark-grade research, prefer explicit preprocessing outside DeepTab when strict train-only preprocessing is required, or document this behavior in the protocol.

## Inspecting Fitted Feature Metadata

After fitting:

```python
model.fit(X_train, y_train)

datamodule = model.data_module
print(datamodule.num_feature_info)
print(datamodule.cat_feature_info)
print(datamodule.embedding_feature_info)

schema = datamodule.schema
print(schema.num_numerical_features)
print(schema.num_categorical_features)
print(schema.total_numerical_dim)
```

The schema is useful when debugging model input shapes and understanding how preprocessing changed the original table.

## Practical Recipes

| Data condition | Starting config |
| --- | --- |
| Mostly clean continuous features | `PreprocessingConfig(numerical_preprocessing="standard")` |
| Outliers or skewed marginals | `PreprocessingConfig(numerical_preprocessing="quantile")` |
| Nonlinear numeric effects | `PreprocessingConfig(numerical_preprocessing="ple", n_bins=50)` |
| Integer IDs mixed with true numerics | Convert ID columns to pandas `category` or tune `cat_cutoff`. |
| Already preprocessed outside DeepTab | Use minimal DeepTab preprocessing and document the external pipeline. |

## Next Steps

- [Config System](config_system)
- [Training and Evaluation](training_and_evaluation)
- [sklearn API](sklearn_api)
