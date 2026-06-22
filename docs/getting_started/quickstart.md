# Quickstart

This guide shows you how to train your first DeepTab model in less than 5 minutes. By the end, you'll understand the basic workflow and be ready to apply it to your own data.

## Your first model

Let's start with a complete classification example using synthetic data:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from deeptab.models import MambularClassifier

# Generate synthetic data
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_classes=3,
    random_state=42,
)

# Convert to DataFrame (optional, but recommended)
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the model
model = MambularClassifier()

# Train the model
model.fit(X_train, y_train, max_epochs=50)

# Evaluate on test set
metrics = model.evaluate(X_test, y_test)
print(f"Test accuracy: {metrics['accuracy']:.3f}")

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(f"Predictions shape: {predictions.shape}")
print(f"Probabilities shape: {probabilities.shape}")
```

That's it! The model handles preprocessing, batching, device placement, and training automatically.

### What just happened?

1. **Data preparation**: Created a DataFrame with 10 features and 3 classes
2. **Train/test split**: Standard scikit-learn split
3. **Model initialization**: Created a Mambular classifier with default settings
4. **Training**: The `fit` method handles everything, including preprocessing, batching, GPU transfer, and optimization
5. **Evaluation**: The `evaluate` method returns a dict of metrics
6. **Prediction**: Standard `predict` and `predict_proba` methods

## Regression example

Regression follows the same workflow with a different model class:

```python
from sklearn.datasets import make_regression
from deeptab.models import FTTransformerRegressor

# Generate regression data
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    noise=0.1,
    random_state=42,
)

X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use a different architecture
model = FTTransformerRegressor()
model.fit(X_train, y_train, max_epochs=50)

metrics = model.evaluate(X_test, y_test)
print(f"Test RMSE: {metrics['rmse']:.3f}")

# Predict continuous values
predictions = model.predict(X_test)
```

The only changes are the model class (`*Regressor`) and the interpretation of outputs.

## Using configs for customization

DeepTab separates hyperparameters into three independent config objects. Here's how to customize the model:

```python
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MambularClassifier

model = MambularClassifier(
    # Architecture hyperparameters
    model_config=MambularConfig(
        d_model=128,        # Hidden dimension (default: 64)
        n_layers=8,         # Number of Mamba blocks (default: 4)
        dropout=0.2,        # Dropout rate (default: 0.2)
    ),
    # Preprocessing strategy
    preprocessing_config=PreprocessingConfig(
        numerical_preprocessing="quantile",  # Options: standardization, quantile, minmax, ple
        n_bins=50,                           # For binning strategies
    ),
    # Training loop parameters
    trainer_config=TrainerConfig(
        max_epochs=100,     # Number of epochs (default: 100)
        lr=1e-3,            # Learning rate (default: 1e-4)
        batch_size=256,     # Batch size (default: 128)
        patience=15,        # Early stopping patience (default: 15)
        optimizer_type="AdamW",   # Any torch.optim class name (default: "Adam")
        weight_decay=1e-2,        # L2 regularisation (default: 1e-6)
        scheduler_type="ReduceLROnPlateau",  # LR scheduler (default)
        lr_patience=5,      # Epochs without improvement before LR is reduced
        lr_factor=0.5,      # LR reduction factor (default: 0.1)
    ),
)

model.fit(X_train, y_train)
```

Each config has sensible defaults. You only need to specify the parameters you want to change.

## Working with real data

Here's a more realistic example with mixed feature types:

```python
import pandas as pd
from deeptab.models import TabTransformerClassifier
from sklearn.model_selection import train_test_split

# Load your data (example structure)
data = pd.DataFrame({
    # Numerical features
    "age": [25, 32, 47, 51, 62, 28, 35, 44],
    "income": [35000, 48000, 72000, 55000, 91000, 42000, 58000, 68000],
    "years_experience": [2, 5, 15, 8, 25, 3, 7, 12],

    # Categorical features
    "city": ["NYC", "Boston", "Chicago", "Boston", "NYC", "Chicago", "NYC", "Boston"],
    "education": ["Bachelor", "Master", "PhD", "Master", "Bachelor", "Bachelor", "Master", "PhD"],
    "employment_type": ["full-time", "part-time", "full-time", "full-time", "retired", "full-time", "full-time", "full-time"],

    # Boolean feature (treated as categorical)
    "has_degree": [True, True, True, True, False, True, True, True],

    # Target
    "target": [0, 1, 1, 0, 1, 0, 1, 1],
})

# Separate features and target
X = data.drop(columns=["target"])
y = data["target"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model (handles mixed types automatically)
model = TabTransformerClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(metrics)

# Predict on new data
predictions = model.predict(X_test)
```

DeepTab automatically:

- Detects feature types from DataFrame dtypes
- Standardizes numerical features (`age`, `income`, `years_experience`)
- Encodes and embeds categorical features (`city`, `education`, `employment_type`, `has_degree`)
- Handles missing values if present

## Distributional regression

For uncertainty quantification, use `LSS` models:

```python
from deeptab.models import MambularLSS

# Same data as regression example
X_train, X_test, y_train, y_test = ...

# Initialize LSS model
model = MambularLSS()

# Fit with a parametric family
model.fit(X_train, y_train, family="normal", max_epochs=50)

# Predict distribution parameters
params = model.predict(X_test)

# For family="normal", params has shape (n_samples, 2) with columns [mean, std]
mean_predictions = params[:, 0]
std_predictions = params[:, 1]

# Generate 95% prediction intervals
lower_bound = mean_predictions - 1.96 * std_predictions
upper_bound = mean_predictions + 1.96 * std_predictions

print(f"Prediction intervals: [{lower_bound[0]:.2f}, {upper_bound[0]:.2f}]")
```

### Supported distributions

DeepTab supports a wide range of families, including `normal`, `studentt`, `gamma`, `beta`, `poisson`, `negativebinom`, and `quantile`. Each family automatically selects appropriate evaluation metrics through `model.evaluate()`. See the [distributions reference](../api/distributions/index) and the [Uncertainty Quantification tutorial](../tutorials/uncertainty_quantification) for the full list and worked examples.

## Saving and loading models

Save trained models for later use:

```python
# Train a model
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Save to disk
model.save("my_model.deeptab")

# Load later
from deeptab.models import MambularClassifier
loaded_model = MambularClassifier.load("my_model.deeptab")

# Use loaded model
predictions = loaded_model.predict(X_test)
```

Use the `.deeptab` extension for saved models. DeepTab accepts any extension but warns when a different one is used, so sticking to `.deeptab` keeps artifacts easy to recognise.

Note: `save()` writes a fitted estimator artifact, not just neural-network weights. The artifact includes the architecture/config, trained weights, fitted preprocessing state, feature schema and column order, task metadata such as classifier `classes_`, and package versions for debugging reloads across environments.

## Inference guide

For serving a fitted model, DeepTab provides `InferenceModel`, a read-only wrapper built for production. It loads an artifact, validates incoming data against the training schema, and predicts, while deliberately hiding `fit` and other training methods so deployment code cannot retrain a model by accident.

```python
from deeptab import InferenceModel

# Load a saved artifact (one type, regardless of the architecture inside)
model = InferenceModel.from_path("my_model.deeptab")

# Validate new data against the training schema, then predict
X_valid = model.validate_input(X_new)
predictions = model.predict(X_valid)

# Task-specific outputs
probabilities = model.predict_proba(X_valid)   # classifiers only
params = model.predict_params(X_valid)         # LSS models only
```

`validate_input` checks that the expected columns are present, reorders them to the training order, and reports missing or unexpected columns with clear messages. You can also wrap an already-fitted estimator without going through disk using `InferenceModel.from_estimator(estimator)`.

```{note}
`InferenceModel` exposes only the prediction method that matches the task: `predict_proba` for classifiers and `predict_params` for `LSS` models. Calling the wrong one raises a clear error, so serving code never branches on the concrete model class.
```

If you only need a quick prediction during experimentation, calling `predict` on the fitted estimator directly works too:

```python
predictions = model.predict(X_new)         # estimator or InferenceModel
metrics = trained_estimator.evaluate(X_new, y_new)
```

See [Inference and deployment](../core_concepts/inference) for the full production contract, schema-validation options, and introspection helpers.

## Next steps

Now that you've run your first models, explore:

- **[Core Concepts](../core_concepts/config_system)**: Deep dive into the config system, preprocessing, and distributional regression
- **[Tutorials](../tutorials/imbalance_classification)**: Complete end-to-end workflows for different tasks
- **[API Reference](../api/models/index)**: Full documentation of all models and configs
- **[FAQ](faq)**: Answers to common questions

For questions or issues, open an issue on [GitHub](https://github.com/OpenTabular/DeepTab/issues).
