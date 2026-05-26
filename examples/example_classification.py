"""Classification Example with DeepTab v2.0.

Demonstrates:
- Basic classification workflow
- Automatic feature detection
- Stratified train/validation splits
- Model evaluation and predictions
- Using configs for customization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deeptab.configs import MambularConfig, TrainerConfig
from deeptab.models import MambularClassifier

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("DeepTab v2.0 Classification Example")
print("=" * 60)

# Generate synthetic data
print("\n[1/5] Generating synthetic data...")
n_samples, n_features = 1000, 5
X = np.random.randn(n_samples, n_features)
y_continuous = np.dot(X, np.random.randn(n_features)) + np.random.randn(n_samples)

# Create DataFrame with numerical features
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])

# Convert to multiclass classification (4 classes)
df["target"] = pd.qcut(y_continuous, q=4, labels=False)

print(f"  - Samples: {n_samples}")
print(f"  - Features: {n_features}")
print(f"  - Classes: {df['target'].nunique()}")

# Split data
print("\n[2/5] Splitting data (80/20)...")
X = df.drop(columns=["target"])
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"  - Training samples: {len(X_train)}")
print(f"  - Test samples: {len(X_test)}")

# Train model with default settings
print("\n[3/5] Training model with default settings...")
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Evaluate
print("\n[4/5] Evaluating on test set...")
metrics = model.evaluate(X_test, y_test)
print(f"  - Accuracy: {metrics['accuracy']:.3f}")
print(f"  - Loss: {metrics['loss']:.3f}")

# Get predictions
print("\n[5/5] Making predictions...")
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(f"  - Predictions shape: {predictions.shape}")
print(f"  - Probabilities shape: {probabilities.shape}")
print(f"  - Sample predictions: {predictions[:5]}")

# Example with custom configs
print("\n" + "=" * 60)
print("Training with Custom Configs (v2.0 Feature)")
print("=" * 60)

model_cfg = MambularConfig(
    d_model=128,
    n_layers=6,
    dropout=0.2,
)

trainer_cfg = TrainerConfig(
    lr=1e-3,
    batch_size=256,
    patience=10,
)

model_custom = MambularClassifier(
    model_config=model_cfg,
    trainer_config=trainer_cfg,
)

print("\nTraining with custom architecture and training settings...")
model_custom.fit(X_train, y_train, max_epochs=50)

metrics_custom = model_custom.evaluate(X_test, y_test)
print(f"  - Custom model accuracy: {metrics_custom['accuracy']:.3f}")

print("\n" + "=" * 60)
print("Example complete! See docs/tutorials/ for more examples.")
print("=" * 60)
