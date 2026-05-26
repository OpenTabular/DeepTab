"""Distributional Regression (LSS) Example with DeepTab v2.0.

Demonstrates:
- Training LSS models for uncertainty quantification
- Predicting distribution parameters (mean and std)
- Generating prediction intervals
- Validating interval coverage
- Using different distribution families
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from deeptab.configs import TrainerConfig
from deeptab.models import MambularLSS

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("DeepTab v2.0 Distributional Regression (LSS) Example")
print("=" * 60)

# Generate synthetic data
print("\n[1/6] Generating synthetic data...")
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
print("\n[2/6] Splitting data (80/20)...")
X = df.drop(columns=["target"])
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"  - Training samples: {len(X_train)}")
print(f"  - Test samples: {len(X_test)}")

# Train LSS model
print("\n[3/6] Training LSS model with 'normal' family...")
model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)

# Evaluate
print("\n[4/6] Evaluating on test set...")
metrics = model.evaluate(X_test, y_test)
print(f"  - Negative log-likelihood: {metrics['loss']:.3f}")

# Get distribution parameters
print("\n[5/6] Predicting distribution parameters...")
params = model.predict(X_test)
print(f"  - Parameters shape: {params.shape}")
print("  - Column 0: mean, Column 1: log(std)")

# Extract mean and std
mean = params[:, 0]
log_std = params[:, 1]
std = np.exp(log_std)

print(f"  - Mean of predicted means: {mean.mean():.3f}")
print(f"  - Mean of predicted stds: {std.mean():.3f}")

# Generate prediction intervals
print("\n[6/6] Generating prediction intervals...")

for confidence in [0.50, 0.68, 0.90, 0.95]:
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    lower = mean - z * std
    upper = mean + z * std

    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    print(f"  - {confidence * 100:.0f}% interval: empirical coverage = {coverage:.3f}")

# Show sample predictions with intervals
print("\n" + "=" * 60)
print("Sample Predictions with 90% Intervals")
print("=" * 60)

z_90 = stats.norm.ppf(0.95)
for i in range(5):
    actual = y_test[i]
    pred_mean = mean[i]
    pred_std = std[i]
    lower_90 = pred_mean - z_90 * pred_std
    upper_90 = pred_mean + z_90 * pred_std

    in_interval = "✓" if lower_90 <= actual <= upper_90 else "✗"

    print(
        f"Sample {i}: actual={actual:6.3f}, "
        f"pred={pred_mean:6.3f} ± {pred_std:.3f}, "
        f"90%=[{lower_90:6.3f}, {upper_90:6.3f}] {in_interval}"
    )

# Example with different family
print("\n" + "=" * 60)
print("Training with Different Distribution Family")
print("=" * 60)

# For positive targets, use gamma distribution
y_positive = np.abs(y) + 1.0
y_train_pos = y_positive[X_train.index]
y_test_pos = y_positive[X_test.index]

print("\nTraining with 'gamma' family for positive targets...")
model_gamma = MambularLSS()
model_gamma.fit(X_train, y_train_pos, family="gamma", max_epochs=50)

metrics_gamma = model_gamma.evaluate(X_test, y_test_pos)
print(f"  - Gamma model NLL: {metrics_gamma['loss']:.3f}")

params_gamma = model_gamma.predict(X_test)
log_alpha = params_gamma[:, 0]
log_beta = params_gamma[:, 1]

alpha = np.exp(log_alpha)
beta = np.exp(log_beta)

mean_gamma = alpha / beta
print(f"  - Mean of gamma means: {mean_gamma.mean():.3f}")
print(f"  - Actual mean: {y_test_pos.mean():.3f}")

print("\n" + "=" * 60)
print("Example complete! See docs/tutorials/ for more examples.")
print("  - Available families: normal, poisson, gamma, beta, negative_binomial, student_t")
print("  - See distributional tutorial for interval generation and visualization")
print("=" * 60)
