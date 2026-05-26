"""Regression Example with DeepTab v2.0.

Demonstrates:
- Basic regression workflow
- Automatic feature detection
- Model evaluation with RMSE, MAE, R²
- Using configs for preprocessing and training
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from deeptab.configs import PreprocessingConfig, TrainerConfig
from deeptab.models import MambularRegressor

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("DeepTab v2.0 Regression Example")
print("=" * 60)

# Generate synthetic data
print("\n[1/5] Generating synthetic data...")
n_samples, n_features = 1000, 5
X = np.random.randn(n_samples, n_features)
coefficients = np.random.randn(n_features)
y = np.dot(X, coefficients) + np.random.randn(n_samples)

# Create DataFrame
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
df["target"] = y

print(f"  - Samples: {n_samples}")
print(f"  - Features: {n_features}")
print(f"  - Target mean: {y.mean():.3f}, std: {y.std():.3f}")

# Split data
print("\n[2/5] Splitting data (80/20)...")
X = df.drop(columns=["target"])
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"  - Training samples: {len(X_train)}")
print(f"  - Test samples: {len(X_test)}")

# Train model with default settings
print("\n[3/5] Training model with default settings...")
model = MambularRegressor()
model.fit(X_train, y_train, max_epochs=50)

# Evaluate
print("\n[4/5] Evaluating on test set...")
metrics = model.evaluate(X_test, y_test)
print(f"  - RMSE: {metrics['rmse']:.3f}")
print(f"  - MAE: {metrics['mae']:.3f}")
print(f"  - R² score: {model.score(X_test, y_test):.3f}")

# Get predictions
print("\n[5/5] Making predictions...")
predictions = model.predict(X_test)
print(f"  - Predictions shape: {predictions.shape}")
print(f"  - Sample predictions: {predictions[:5]}")
print(f"  - Prediction mean: {predictions.mean():.3f}")

# Example with custom configs
print("\n" + "=" * 60)
print("Training with Custom Configs (v2.0 Feature)")
print("=" * 60)

prep_cfg = PreprocessingConfig(
    numerical_preprocessing="quantile",  # Transform to uniform distribution
    use_ple=True,  # Piecewise Linear Encoding
    n_bins=50,
)

trainer_cfg = TrainerConfig(
    lr=5e-4,
    batch_size=256,
    patience=15,
    lr_scheduler="cosine",
)

model_custom = MambularRegressor(
    preprocessing_config=prep_cfg,
    trainer_config=trainer_cfg,
)

print("\nTraining with quantile preprocessing and cosine LR schedule...")
model_custom.fit(X_train, y_train, max_epochs=50)

metrics_custom = model_custom.evaluate(X_test, y_test)
print(f"  - Custom model RMSE: {metrics_custom['rmse']:.3f}")
print(f"  - Custom model R²: {model_custom.score(X_test, y_test):.3f}")

print("\n" + "=" * 60)
print("Example complete! See docs/tutorials/ for more examples.")
print("=" * 60)
