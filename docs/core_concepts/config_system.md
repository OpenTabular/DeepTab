# Config System

DeepTab separates hyperparameters into three independent config dataclasses. This split-config design makes it easy to tune different aspects of your model independently and enables clean integration with hyperparameter search tools.

```{important}
**Split-config design:** Architecture, preprocessing, and training are **independently configurable**. This makes hyperparameter tuning more systematic and sharable across models.
```

## The three configs

| Config                | Controls            | Example parameters                  |
| --------------------- | ------------------- | ----------------------------------- |
| `<Model>Config`       | Neural architecture | `d_model`, `n_layers`, `dropout`    |
| `PreprocessingConfig` | Feature engineering | `numerical_preprocessing`, `n_bins` |
| `TrainerConfig`       | Training loop       | `lr`, `max_epochs`, `batch_size`    |

```{tip}
All three configs are **optional**. Omitting a config applies sensible defaults.
```

## Model config

Each architecture has its own config class defining the neural network structure.

### Example: MambularConfig

```python
from deeptab.configs import MambularConfig
from deeptab.models import MambularClassifier

model = MambularClassifier(
    model_config=MambularConfig(
        d_model=128,         # Hidden dimension
        n_layers=8,          # Number of Mamba blocks
        dropout=0.2,         # Dropout rate
        use_learnable_interaction=True,  # Feature interaction
    )
)
```

### Common architecture parameters

While each model has specific parameters, many share common patterns:

| Parameter  | Type  | Description                              | Typical range |
| ---------- | ----- | ---------------------------------------- | ------------- |
| `d_model`  | int   | Hidden dimension / embedding size        | 32-512        |
| `n_layers` | int   | Number of blocks/layers                  | 2-12          |
| `dropout`  | float | Dropout rate for regularization          | 0.0-0.5       |
| `d_ff`     | int   | Feedforward dimension (Transformers)     | 128-2048      |
| `n_heads`  | int   | Number of attention heads (Transformers) | 4-16          |

### Available model configs

- `MambularConfig` — Sequential Mamba blocks
- `FTTransformerConfig` — Feature tokenization + Transformer
- `TabTransformerConfig` — Transformer with categorical embeddings
- `ResNetConfig` — Residual blocks
- `MLPConfig` — Multi-layer perceptron
- `NODEConfig` — Oblivious decision trees
- `TabMConfig` — Batch ensembling
- And more (see [API reference](../../api/configs/index))

### Viewing all parameters

```python
from deeptab.configs import MambularConfig

cfg = MambularConfig()
print(cfg.get_params())
# {'d_model': 64, 'n_layers': 4, 'dropout': 0.2, ...}
```

### Updating parameters

```python
# At initialization
cfg = MambularConfig(d_model=128, n_layers=6)

# After initialization
cfg.set_params(d_model=256, dropout=0.3)
```

## Preprocessing config

`PreprocessingConfig` controls how features are encoded and scaled before entering the neural network.

### Basic usage

```python
from deeptab.configs import PreprocessingConfig

cfg = PreprocessingConfig(
    numerical_preprocessing="quantile",  # Quantile transform
    n_bins=50,                           # For binning strategies
    scaling_strategy="standard",         # Standardization
)
```

### Numerical preprocessing strategies

```{note}
**Choose based on your data characteristics:**
- **Standard:** Normal distributions, no outliers
- **Quantile:** Heavy outliers, skewed distributions
- **MinMax:** Bounded features (percentages, ratings)
- **PLE:** Complex non-linear relationships
- **Binning:** Convert continuous to categorical
```

| Strategy     | Description                                | When to use                        |
| ------------ | ------------------------------------------ | ---------------------------------- |
| `"standard"` | Z-score standardization (mean=0, std=1)    | Normally distributed features      |
| `"quantile"` | Quantile transform to uniform distribution | Features with outliers             |
| `"minmax"`   | Scale to [0, 1] range                      | Bounded features                   |
| `"ple"`      | Piecewise linear encoding                  | Capturing non-linear relationships |
| `"binning"`  | Convert to categorical bins                | When you want discrete buckets     |

```python
# For data with heavy outliers
cfg = PreprocessingConfig(numerical_preprocessing="quantile")

# For features already in reasonable ranges
cfg = PreprocessingConfig(numerical_preprocessing="standard")
```

### Categorical encoding

DeepTab uses ordinal encoding + learned embeddings by default. You can configure embedding dimensions:

```python
cfg = PreprocessingConfig(
    cat_encoding_strategy="ordinal",  # Default
    embedding_dim=32,                 # Embedding size (auto by default)
)
```

### Scaling strategy

Applied after numerical preprocessing:

```python
cfg = PreprocessingConfig(
    numerical_preprocessing="ple",
    scaling_strategy="standard",  # Options: "standard", "minmax", "robust"
)
```

### Missing value handling

Missing values are handled automatically with median (numerical) and mode (categorical) imputation. You can configure this:

```python
cfg = PreprocessingConfig(
    numerical_imputation_strategy="median",  # Or "mean", "zero"
    categorical_imputation_strategy="mode",  # Or "constant"
)
```

### Full parameter list

```python
cfg = PreprocessingConfig(
    # Numerical features
    numerical_preprocessing="quantile",
    n_bins=50,
    scaling_strategy="standard",
    numerical_imputation_strategy="median",

    # Categorical features
    cat_encoding_strategy="ordinal",
    embedding_dim=None,  # Auto-computed by default
    categorical_imputation_strategy="mode",

    # Advanced
    use_pretrained_embeddings=False,
    embedding_activation="linear",
)
```

See [Preprocessing](preprocessing) for detailed explanations.

## Trainer config

`TrainerConfig` controls the training loop, optimization, and device management.

### Basic usage

```python
from deeptab.configs import TrainerConfig

cfg = TrainerConfig(
    max_epochs=100,      # Maximum training epochs
    lr=1e-3,             # Learning rate
    batch_size=256,      # Batch size
    patience=15,         # Early stopping patience
)
```

### Training parameters

| Parameter    | Type  | Description                         | Default |
| ------------ | ----- | ----------------------------------- | ------- |
| `max_epochs` | int   | Maximum training epochs             | 100     |
| `lr`         | float | Learning rate                       | 1e-4    |
| `batch_size` | int   | Batch size                          | 128     |
| `patience`   | int   | Early stopping patience             | 10      |
| `val_split`  | float | Validation split if no val provided | 0.2     |

```python
# Conservative training
cfg = TrainerConfig(
    max_epochs=200,
    lr=1e-4,
    patience=20,
)

# Fast experimentation
cfg = TrainerConfig(
    max_epochs=50,
    lr=1e-3,
    patience=5,
)
```

### Optimization settings

```python
cfg = TrainerConfig(
    lr=1e-3,
    optimizer="adam",              # Options: "adam", "adamw", "sgd"
    weight_decay=1e-4,             # L2 regularization
    gradient_clip_val=1.0,         # Gradient clipping
    lr_scheduler="reduce_on_plateau",  # Learning rate scheduling
)
```

### Device and parallelism

```python
cfg = TrainerConfig(
    device="cuda",        # "cuda", "cpu", or "cuda:0"
    num_workers=4,        # Parallel data loading
    persistent_workers=True,  # Keep workers alive between epochs
)
```

### Monitoring and logging

```python
cfg = TrainerConfig(
    verbose=True,         # Detailed logging
    progress_bar=True,    # Show progress bar
    log_every_n_steps=10, # Logging frequency
)
```

### Full parameter list

```python
cfg = TrainerConfig(
    # Training
    max_epochs=100,
    lr=1e-4,
    batch_size=128,
    patience=10,
    val_split=0.2,

    # Optimization
    optimizer="adam",
    weight_decay=0.0,
    gradient_clip_val=1.0,
    lr_scheduler=None,

    # Device
    device="cuda",
    num_workers=0,
    persistent_workers=False,

    # Monitoring
    verbose=False,
    progress_bar=True,
    log_every_n_steps=50,

    # Advanced
    accumulate_grad_batches=1,
    precision="32",  # Or "16" for mixed precision
    deterministic=False,
)
```

## Using configs together

All three configs are passed to the model constructor:

```python
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MambularClassifier

model = MambularClassifier(
    model_config=MambularConfig(
        d_model=128,
        n_layers=8,
        dropout=0.2,
    ),
    preprocessing_config=PreprocessingConfig(
        numerical_preprocessing="quantile",
        n_bins=50,
    ),
    trainer_config=TrainerConfig(
        max_epochs=100,
        lr=1e-3,
        batch_size=256,
        patience=15,
    ),
)

model.fit(X_train, y_train)
```

## Default configs

Omit any config to use defaults:

```python
# All defaults
model = MambularClassifier()

# Some custom, some default
model = MambularClassifier(
    model_config=MambularConfig(d_model=128),
    # preprocessing_config uses defaults
    # trainer_config uses defaults
)
```

## Accessing configs from a fitted model

After fitting, you can inspect the configs:

```python
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)

print(model.model_config.d_model)       # 64 (default)
print(model.trainer_config.lr)          # 1e-4 (default)
```

## Integration with hyperparameter search

The split-config design works seamlessly with scikit-learn's search tools via double-underscore notation:

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    # Architecture
    "model_config__d_model": [64, 128, 256],
    "model_config__n_layers": [4, 6, 8],

    # Training
    "trainer_config__lr": [1e-3, 5e-4, 1e-4],
    "trainer_config__batch_size": [128, 256],
}

search = GridSearchCV(
    estimator=MambularClassifier(),
    param_grid=param_grid,
    cv=3,
)
search.fit(X_train, y_train)
```

### RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    "model_config__d_model": randint(32, 256),
    "model_config__dropout": uniform(0.1, 0.4),
    "trainer_config__lr": uniform(1e-4, 1e-2),
}

search = RandomizedSearchCV(
    estimator=MambularClassifier(),
    param_distributions=param_distributions,
    n_iter=20,
    cv=3,
)
search.fit(X_train, y_train)
```

## Config validation

Configs validate parameters at initialization:

```python
# This raises ValueError
cfg = MambularConfig(d_model=-128)  # ValueError: d_model must be positive

# This raises ValueError
cfg = TrainerConfig(lr=10.0)  # ValueError: lr too high
```

## Serialization

Configs can be saved and loaded:

```python
from deeptab.configs import MambularConfig
import pickle

# Save
cfg = MambularConfig(d_model=128, n_layers=8)
with open("config.pkl", "wb") as f:
    pickle.dump(cfg, f)

# Load
with open("config.pkl", "rb") as f:
    loaded_cfg = pickle.load(f)

model = MambularClassifier(model_config=loaded_cfg)
```

Or use JSON:

```python
import json

cfg = MambularConfig(d_model=128)
config_dict = cfg.get_params()

# Save
with open("config.json", "w") as f:
    json.dump(config_dict, f)

# Load
with open("config.json", "r") as f:
    params = json.load(f)

cfg = MambularConfig(**params)
```

## Task-specific configs

Some models have task-specific config variants:

```python
from deeptab.configs import MambularConfig

# Same config works for all tasks
cfg = MambularConfig(d_model=128)

classifier = MambularClassifier(model_config=cfg)
regressor = MambularRegressor(model_config=cfg)
lss_model = MambularLSS(model_config=cfg)
```

The config is task-agnostic; the model class determines the task.

## Advanced: Custom configs

You can create custom configs by subclassing:

```python
from dataclasses import dataclass
from deeptab.configs import MambularConfig

@dataclass
class MyMambularConfig(MambularConfig):
    custom_param: int = 42

    def __post_init__(self):
        super().__post_init__()
        # Custom validation
        if self.custom_param < 0:
            raise ValueError("custom_param must be non-negative")

cfg = MyMambularConfig(d_model=128, custom_param=100)
```

## Best practices

1. **Start with defaults**: Only customize when you have a reason
2. **Tune architecture first**: Model capacity matters most
3. **Then tune training**: Learning rate and batch size
4. **Preprocessing last**: Usually defaults work well
5. **Use hyperparameter search**: Don't hand-tune excessively
6. **Version your configs**: Save them alongside trained models

## Common config recipes

### Quick experimentation

```python
# Fast iterations
model = MambularClassifier(
    trainer_config=TrainerConfig(
        max_epochs=20,
        patience=5,
        batch_size=512,
    )
)
```

### Production training

```python
# Thorough training
model = MambularClassifier(
    model_config=MambularConfig(d_model=256, n_layers=8),
    trainer_config=TrainerConfig(
        max_epochs=200,
        patience=20,
        lr=5e-4,
        batch_size=256,
    )
)
```

### Data with outliers

```python
# Robust to outliers
model = MambularClassifier(
    preprocessing_config=PreprocessingConfig(
        numerical_preprocessing="quantile",
    )
)
```

### Large dataset

```python
# Efficient for large data
model = MambularClassifier(
    trainer_config=TrainerConfig(
        batch_size=1024,
        num_workers=4,
        persistent_workers=True,
    )
)
```

## Next steps

- **[Preprocessing](preprocessing)** — Deep dive into preprocessing strategies
- **[Training and Evaluation](training_and_evaluation)** — Training loop details
- **[Classification](classification)** — Classification-specific usage
- **[Regression](regression)** — Regression-specific usage
