# Recommended Configurations

Battle-tested hyperparameter configurations for all DeepTab models across different scenarios.

## Quick Start Recipes

### For Quick Experimentation

```python
from deeptab.models import MambularClassifier
from deeptab.configs import TrainerConfig

trainer_cfg = TrainerConfig(
    max_epochs=20,
    patience=5,
    batch_size=512,
)

model = MambularClassifier(trainer_config=trainer_cfg)
model.fit(X_train, y_train, max_epochs=20)
```

### For Production

```python
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig

model_cfg = MambularConfig(
    d_model=256,
    n_layers=8,
    dropout=0.1,
)

prep_cfg = PreprocessingConfig(
    numerical_preprocessing="quantile",
    use_ple=True,
)

trainer_cfg = TrainerConfig(
    lr=5e-4,
    max_epochs=200,
    patience=20,
    lr_scheduler="reduce_on_plateau",
    weight_decay=1e-4,
)

model = MambularClassifier(
    model_config=model_cfg,
    preprocessing_config=prep_cfg,
    trainer_config=trainer_cfg,
)
```

## Model-Specific Recommendations

### Mambular

**Small datasets (<5K samples):**

```python
from deeptab.configs import MambularConfig, TrainerConfig

model_cfg = MambularConfig(
    d_model=64,
    n_layers=4,
    dropout=0.2,
)

trainer_cfg = TrainerConfig(
    lr=1e-3,
    batch_size=128,
    max_epochs=100,
    patience=15,
)
```

**Medium datasets (5K-50K samples):**

```python
model_cfg = MambularConfig(
    d_model=128,
    n_layers=6,
    dropout=0.1,
)

trainer_cfg = TrainerConfig(
    lr=5e-4,
    batch_size=256,
    max_epochs=150,
    patience=20,
)
```

**Large datasets (>50K samples):**

```python
model_cfg = MambularConfig(
    d_model=256,
    n_layers=8,
    dropout=0.0,
)

trainer_cfg = TrainerConfig(
    lr=1e-4,
    batch_size=512,
    max_epochs=200,
    patience=25,
)
```

### FTTransformer

**Balanced setup:**

```python
from deeptab.configs import FTTransformerConfig

model_cfg = FTTransformerConfig(
    d_model=128,
    n_heads=8,
    n_layers=6,
    attn_dropout=0.1,
    ffn_dropout=0.1,
)

trainer_cfg = TrainerConfig(
    lr=1e-4,
    batch_size=256,
    max_epochs=150,
)
```

**High capacity:**

```python
model_cfg = FTTransformerConfig(
    d_model=256,
    n_heads=16,
    n_layers=8,
    attn_dropout=0.1,
    ffn_dropout=0.2,
)
```

### ResNet

**Fast baseline:**

```python
from deeptab.configs import ResNetConfig

model_cfg = ResNetConfig(
    d_model=128,
    n_layers=8,
    dropout=0.1,
)

trainer_cfg = TrainerConfig(
    lr=1e-3,
    batch_size=512,
    max_epochs=100,
)
```

### TabTransformer

**For categorical-heavy data:**

```python
from deeptab.configs import TabTransformerConfig

model_cfg = TabTransformerConfig(
    d_model=128,
    n_heads=8,
    n_layers=6,
    attn_dropout=0.1,
)

trainer_cfg = TrainerConfig(
    lr=1e-4,
    batch_size=256,
)
```

### NODE

**Tree-based setup:**

```python
from deeptab.configs import NODEConfig

model_cfg = NODEConfig(
    n_layers=8,
    depth=6,
    n_trees=2048,
)

trainer_cfg = TrainerConfig(
    lr=1e-3,
    batch_size=512,
    max_epochs=150,
)
```

## Preprocessing Configurations

### Standard Scaling (default)

```python
from deeptab.configs import PreprocessingConfig

prep_cfg = PreprocessingConfig(
    numerical_preprocessing="standard",
    categorical_preprocessing="ordinal",
)
```

### Quantile Transformation

Best for skewed numerical features:

```python
prep_cfg = PreprocessingConfig(
    numerical_preprocessing="quantile",
    n_bins=100,  # More bins for large datasets
)
```

### Piecewise Linear Encoding (PLE)

Advanced numerical encoding:

```python
prep_cfg = PreprocessingConfig(
    numerical_preprocessing="standard",
    use_ple=True,
    n_bins=50,
)
```

### For Categorical-Heavy Data

```python
prep_cfg = PreprocessingConfig(
    numerical_preprocessing="quantile",
    categorical_preprocessing="ordinal",
    embedding_dim=32,  # Larger embeddings for rich categoricals
)
```

## Training Configurations

### Conservative (prevent overfitting)

```python
trainer_cfg = TrainerConfig(
    lr=1e-4,
    batch_size=128,
    max_epochs=100,
    patience=15,
    dropout=0.3,  # Model config
    weight_decay=1e-3,
)
```

### Aggressive (maximize performance)

```python
trainer_cfg = TrainerConfig(
    lr=1e-3,
    batch_size=512,
    max_epochs=200,
    patience=25,
    dropout=0.0,
    weight_decay=0.0,
)
```

### With Learning Rate Scheduling

**Reduce on plateau:**

```python
trainer_cfg = TrainerConfig(
    lr=1e-3,
    lr_scheduler="reduce_on_plateau",
    lr_scheduler_patience=10,
    lr_scheduler_factor=0.5,
)
```

**Cosine annealing:**

```python
trainer_cfg = TrainerConfig(
    lr=1e-3,
    lr_scheduler="cosine",
    lr_scheduler_t_max=50,
)
```

## Task-Specific Recommendations

### Classification

**Binary classification:**

```python
# More conservative to avoid overfitting
model_cfg = MambularConfig(
    d_model=128,
    n_layers=6,
    dropout=0.2,
)

trainer_cfg = TrainerConfig(
    lr=5e-4,
    batch_size=256,
    patience=15,
)
```

**Multiclass (many classes):**

```python
# Higher capacity for complex decision boundaries
model_cfg = MambularConfig(
    d_model=256,
    n_layers=8,
    dropout=0.1,
)
```

### Regression

**Standard regression:**

```python
model_cfg = MambularConfig(
    d_model=128,
    n_layers=6,
)

trainer_cfg = TrainerConfig(
    lr=1e-3,
    batch_size=512,
)
```

**With target normalization:**

```python
# Standardize targets for stable training
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

model.fit(X_train, y_train_scaled, max_epochs=100)

# Transform predictions back
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
```

### LSS (Distributional Regression)

**Normal family:**

```python
model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=100)
```

**Gamma family (positive targets):**

```python
# Ensure positive targets
y_train_pos = np.abs(y_train) + 1e-6

model = MambularLSS()
model.fit(X_train, y_train_pos, family="gamma", max_epochs=100)
```

## Dataset Size Guidelines

### Very Small (<1K samples)

```python
# Minimal model, high regularization
model_cfg = MambularConfig(
    d_model=32,
    n_layers=2,
    dropout=0.3,
)

trainer_cfg = TrainerConfig(
    lr=1e-3,
    batch_size=64,
    max_epochs=50,
    patience=10,
)
```

### Small (1K-5K samples)

```python
model_cfg = MambularConfig(
    d_model=64,
    n_layers=4,
    dropout=0.2,
)

trainer_cfg = TrainerConfig(
    lr=1e-3,
    batch_size=128,
    max_epochs=100,
    patience=15,
)
```

### Medium (5K-50K samples)

```python
model_cfg = MambularConfig(
    d_model=128,
    n_layers=6,
    dropout=0.1,
)

trainer_cfg = TrainerConfig(
    lr=5e-4,
    batch_size=256,
    max_epochs=150,
    patience=20,
)
```

### Large (50K-500K samples)

```python
model_cfg = MambularConfig(
    d_model=256,
    n_layers=8,
    dropout=0.0,
)

trainer_cfg = TrainerConfig(
    lr=1e-4,
    batch_size=512,
    max_epochs=200,
    patience=25,
)
```

### Very Large (>500K samples)

```python
model_cfg = MambularConfig(
    d_model=512,
    n_layers=10,
    dropout=0.0,
)

trainer_cfg = TrainerConfig(
    lr=5e-5,
    batch_size=1024,
    max_epochs=300,
    patience=30,
)
```

## Hyperparameter Tuning

### Quick Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model_config__d_model": [64, 128],
    "model_config__n_layers": [4, 6],
    "trainer_config__lr": [1e-4, 5e-4, 1e-3],
}

model = MambularClassifier()
grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=1)
grid_search.fit(X_train, y_train)
```

### Comprehensive Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform

param_distributions = {
    "model_config__d_model": [64, 128, 256],
    "model_config__n_layers": [4, 6, 8, 10],
    "model_config__dropout": uniform(0.0, 0.5),
    "trainer_config__lr": loguniform(1e-5, 1e-2),
    "trainer_config__batch_size": [128, 256, 512],
    "preprocessing_config__numerical_preprocessing": ["standard", "quantile", "minmax"],
}

random_search = RandomizedSearchCV(
    MambularClassifier(),
    param_distributions,
    n_iter=30,
    cv=3,
    n_jobs=1,
)
```

## Common Pitfalls and Solutions

### Overfitting

**Symptoms**: Great train performance, poor validation
**Solutions**:

- Increase dropout (0.1 → 0.3)
- Add weight decay (1e-4)
- Reduce model size
- Use early stopping (patience=15)

### Underfitting

**Symptoms**: Poor train and validation performance
**Solutions**:

- Increase model size (d_model, n_layers)
- Train longer (more epochs)
- Increase learning rate
- Reduce regularization

### Unstable Training

**Symptoms**: Loss spikes, NaN values
**Solutions**:

- Reduce learning rate (1e-3 → 1e-4)
- Enable gradient clipping (default=1.0)
- Use smaller batch sizes
- Check for outliers in data

### Slow Convergence

**Symptoms**: Loss decreases very slowly
**Solutions**:

- Increase learning rate
- Use learning rate scheduling
- Better preprocessing (quantile transform)
- Larger batch sizes

## GPU Memory Optimization

### Out of Memory Errors

```python
# Reduce batch size
trainer_cfg = TrainerConfig(batch_size=64)

# Reduce model size
model_cfg = MambularConfig(d_model=64, n_layers=4)

# Use mixed precision
trainer_cfg = TrainerConfig(precision="16")
```

### Maximize GPU Utilization

```python
# Larger batches if memory allows
trainer_cfg = TrainerConfig(
    batch_size=1024,
    num_workers=4,  # Parallel data loading
)
```

## See Also

- [Comparison Tables](comparison_tables) — Model performance comparison
- [Core Concepts: Training](../core_concepts/training_and_evaluation) — Training details
- [Core Concepts: Config System](../core_concepts/config_system) — Config reference
- [Tutorials](../tutorials/index) — Hands-on examples
