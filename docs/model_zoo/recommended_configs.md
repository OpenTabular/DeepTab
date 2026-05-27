# Hyperparameter Configuration Guidelines

General hyperparameter configuration guidance based on architecture design and common practices.

```{note}
**Focus on principles:** This guide provides parameter ranges and configuration strategies based on architecture characteristics and general deep learning principles. Specific optimal values depend on your dataset.
```

## General Principles

### Learning Rate Selection

```{note}
**Critical hyperparameter:** Learning rate is typically the most important parameter to tune. Too high causes training instability, too low leads to slow convergence or suboptimal solutions.
```

**Recommended starting ranges by architecture:**

| Architecture Type | Learning Rate | Reasoning                                          |
| ----------------- | ------------- | -------------------------------------------------- |
| SSMs (Mamba)      | 1e-4 to 5e-4  | State space models sensitive to large updates      |
| Transformers      | 1e-4 to 1e-3  | Attention mechanisms require careful tuning        |
| ResNets/MLPs      | 5e-4 to 1e-3  | Simpler architectures more robust to larger LR     |
| Tree-based (NODE) | 1e-3 to 5e-3  | Discrete structure tolerates larger learning rates |

```{tip}
**Start conservative:** Begin with the lower end of the range (e.g., 1e-4 for Mambular) and increase if training is too slow. Monitor training loss for instability.
```

### Regularization vs Dataset Size

```{warning}
**Critical principle:** Regularization requirements scale inversely with dataset size. Small datasets need strong regularization to prevent overfitting.
```

**Dropout recommendations:**

| Dataset Size | Recommended Dropout | Reasoning                              |
| ------------ | ------------------- | -------------------------------------- |
| <1K samples  | 0.3-0.5             | High overfitting risk                  |
| 1K-5K        | 0.2-0.3             | Moderate regularization needed         |
| 5K-50K       | 0.1-0.2             | Light regularization sufficient        |
| >50K         | 0.0-0.1             | Data abundance provides regularization |

### Batch Size Effects

**Trade-offs to consider:**

| Batch Size | Training Speed | Generalization           | Memory Usage | Recommendation            |
| ---------- | -------------- | ------------------------ | ------------ | ------------------------- |
| 32-64      | Slower         | Better (noisy gradients) | Low          | Small datasets            |
| 128-256    | Moderate       | Good balance             | Medium       | General-purpose (default) |
| 512-1024   | Faster         | May degrade              | High         | Large datasets only       |
| >1024      | Fastest        | Often poor               | Very High    | Not recommended           |

```{tip}
**General rule:** Larger batches train faster but may hurt generalization. Start with 128-256 and increase only if you have >50K samples and need faster training.
```

## Model-Specific Parameter Sensitivity

### Mambular

**Most sensitive parameters:** `d_model`, `n_layers`

```{note}
**General finding:** Performance typically plateaus beyond d_model=256 and n_layers=8. Increasing further adds computational cost with diminishing returns.
```

**Configuration philosophy:**

- **d_model:** Controls model capacity. Higher values capture more complex patterns but risk overfitting.
- **n_layers:** Depth allows hierarchical feature processing. Too deep can slow training without benefit.
- **Typical sweet spot:** d_model=128, n_layers=6 for medium datasets

**Recommended configurations:**

```python
from deeptab.configs import MambularConfig, TrainerConfig

# Small datasets (<5K): Prevent overfitting
model_cfg = MambularConfig(
    d_model=64,        # Lower capacity
    n_layers=4,        # Fewer layers
    dropout=0.2,       # High dropout
)
trainer_cfg = TrainerConfig(
    lr=1e-3,           # Higher lr acceptable for small data
    batch_size=128,    # Smaller batches for better generalization
    max_epochs=100,
    patience=15,
    weight_decay=1e-4, # Additional regularization
)

# Medium datasets (5K-50K): Balanced
model_cfg = MambularConfig(
    d_model=128,       # Sweet spot capacity
    n_layers=6,        # Moderate depth
    dropout=0.1,       # Light regularization
)
trainer_cfg = TrainerConfig(
    lr=5e-4,           # Conservative learning rate
    batch_size=256,
    max_epochs=150,
    patience=20,
)

# Large datasets (>50K): Maximize capacity
model_cfg = MambularConfig(
    d_model=256,       # High capacity
    n_layers=8,        # Deep architecture
    dropout=0.0,       # No dropout needed
)
trainer_cfg = TrainerConfig(
    lr=1e-4,           # Lower lr for stability
    batch_size=512,    # Larger batches for efficiency
    max_epochs=200,
    patience=25,
)
```

### FTTransformer

**Most sensitive parameters:** `n_heads`, `attn_dropout`

```{note}
**General rule:** Attention heads should scale with d_model. Rule of thumb: n_heads = d_model / 16 for balanced performance.
```

**Parameter guidance:**

- **n_heads:** More heads allow modeling diverse attention patterns but increase compute
- **attn_dropout:** Critical for preventing overfitting in attention layers (0.1-0.2 typical)
- **ffn_dropout:** Regularizes feedforward layers (can be higher than attn_dropout)

**Configurations:**

```python
from deeptab.configs import FTTransformerConfig

# Standard setup (balanced performance/speed)
model_cfg = FTTransformerConfig(
    d_model=128,
    n_heads=8,           # d_model / 16
    n_layers=6,
    attn_dropout=0.1,    # Attention dropout critical
    ffn_dropout=0.1,
)
trainer_cfg = TrainerConfig(
    lr=1e-4,             # Transformers need lower lr
    batch_size=256,
    max_epochs=150,
)

# High-capacity setup
model_cfg = FTTransformerConfig(
    d_model=256,
    n_heads=16,
    n_layers=8,
    attn_dropout=0.1,
    ffn_dropout=0.2,     # Higher ffn dropout for regularization
)
```

### ResNet

**Most sensitive parameters:** `n_layers`, `dropout`

```{note}
**General finding:** ResNets are remarkably robust across hyperparameter ranges. Good default choice for fast experimentation.
```

**Depth guidance:**

- **4-6 layers:** Fast training, good for small-medium datasets
- **8 layers:** Balanced depth, suitable for most use cases
- **12+ layers:** Rarely needed, slower with diminishing returns

```python
from deeptab.configs import ResNetConfig

# Fast baseline
model_cfg = ResNetConfig(
    d_model=128,
    n_layers=6,      # Good balance
    dropout=0.1,
)
trainer_cfg = TrainerConfig(
    lr=1e-3,         # Can use higher lr
    batch_size=512,  # Larger batches work well
    max_epochs=100,
)
```

### TabTransformer

**Most sensitive parameters:** Number of categorical features, embedding dimension

```{important}
**Design consideration:** TabTransformer only applies attention to categorical features. Performance may degrade if <30% of features are categorical. Consider FTTransformer or Mambular for numerical-heavy data.
```

**When to use:**

- **Best:** >60% categorical features (TabTransformer's sweet spot)
- **Good:** 40-60% categorical (competitive with general models)
- **Suboptimal:** <30% categorical (use FTTransformer or Mambular instead)

```python
from deeptab.configs import TabTransformerConfig

# For categorical-heavy data (>50% categorical)
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

**Most sensitive parameters:** `depth`, `n_trees`

```{note}
**Tree structure:** NODE builds oblivious decision trees. Depth controls number of splits (2^depth leaves), n_trees controls ensemble size.
```

**Parameter guidance:**

- **depth:** Typical range 4-8. Higher depth = more complex trees but slower training
- **n_trees:** Typical range 1024-2048. More trees = better ensemble but diminishing returns
- **Trade-off:** Deep trees with fewer n_trees vs shallow trees with more n_trees

```python
from deeptab.configs import NODEConfig

# Balanced setup
model_cfg = NODEConfig(
    n_layers=8,
    depth=6,         # Tree depth
    n_trees=2048,    # Ensemble size
)
trainer_cfg = TrainerConfig(
    lr=1e-3,         # NODE tolerates higher lr
    batch_size=512,
    max_epochs=150,
)
```

## Preprocessing Configuration Impact

### Numerical Preprocessing Strategies

```{note}
**Strategy selection:** Different preprocessing methods suit different data distributions.
```

**Guidance by data characteristics:**

| Strategy | Best For                 | Pros                       | Cons                       |
| -------- | ------------------------ | -------------------------- | -------------------------- |
| standard | Normal distributions     | Simple, interpretable      | Sensitive to outliers      |
| quantile | Skewed or heavy outliers | Robust to outliers         | Non-linear transform       |
| minmax   | Bounded data             | Preserves zero             | Very sensitive to outliers |
| ple      | Complex distributions    | Flexible, piecewise linear | Requires tuning n_bins     |

**Recommendations:**

```python
from deeptab.configs import PreprocessingConfig

# For clean, normally distributed data
prep_cfg = PreprocessingConfig(
    numerical_preprocessing="standard",
)

# For real-world data with outliers (RECOMMENDED DEFAULT)
prep_cfg = PreprocessingConfig(
    numerical_preprocessing="quantile",
    n_bins=100,  # More bins for large datasets
)

# For complex non-linear relationships
prep_cfg = PreprocessingConfig(
    numerical_preprocessing="ple",
    n_bins=50,
)
```

### Categorical Embedding Dimension

```{warning}
**Overfitting risk:** Large embedding dimensions can cause overfitting on small datasets with high-cardinality categoricals.
```

**Embedding size guidance:**

| Categorical Cardinality | Recommended Embedding Dim | Reasoning                   |
| ----------------------- | ------------------------- | --------------------------- |
| <10                     | 8                         | Small vocabulary            |
| 10-50                   | 16                        | Moderate complexity         |
| 50-500                  | 32                        | High cardinality            |
| >500                    | 32-64 (use dropout)       | Very high, overfitting risk |

```python
# Auto-sizing (recommended)
prep_cfg = PreprocessingConfig(
    categorical_preprocessing="ordinal",
    embedding_dim=None,  # Auto: min(50, cardinality // 2)
)

# Manual sizing for high-cardinality
prep_cfg = PreprocessingConfig(
    embedding_dim=32,
)
```

## Training Dynamics

### Early Stopping

```{important}
**Patience setting:** Balance between training time and optimal performance. Patience should scale with dataset size and model complexity.
```

**Patience recommendations:**

| Dataset Size | Recommended Patience | Reasoning                        |
| ------------ | -------------------- | -------------------------------- |
| <1K          | 10-15                | Fast overfitting on small data   |
| 1K-10K       | 15-20                | Moderate training dynamics       |
| >10K         | 20-30                | Slower convergence on large data |

### Learning Rate Scheduling

**Common scheduling strategies:**

| Schedule        | When to Use                   | Pros              | Cons                   |
| --------------- | ----------------------------- | ----------------- | ---------------------- |
| Constant        | Default, works well often     | Simple, no tuning | May not reach optimum  |
| ReduceOnPlateau | General purpose (recommended) | Adaptive, stable  | Needs patience tuning  |
| CosineAnnealing | Fixed training budget known   | Smooth decay      | Needs max_epochs set   |
| StepLR          | Known convergence behavior    | Predictable       | Requires manual tuning |

**Recommendation: ReduceOnPlateau** (adaptive and stable)

```python
trainer_cfg = TrainerConfig(
    lr=1e-3,                       # Initial learning rate
    lr_scheduler="reduce_on_plateau",
    lr_scheduler_patience=10,      # Wait 10 epochs
    lr_scheduler_factor=0.5,       # Reduce by 50%
    lr_scheduler_min_lr=1e-6,      # Don't go below this
)
```

## Hyperparameter Search Recommendations

### Priority Order

Based on parameter sensitivity analysis:

1. **Learning rate** — Test: [1e-4, 5e-4, 1e-3]
2. **Dropout** — Test: [0.0, 0.1, 0.2, 0.3]
3. **d_model** — Test: [64, 128, 256]
4. **n_layers** — Test: [4, 6, 8]
5. **Batch size** — Test: [128, 256, 512]

```{tip}
**Efficient search:** Start with learning rate and dropout. Only tune architecture if those are optimal.
```

### Search Space

**For Mambular:**

```python
param_grid = {
    "trainer_config__lr": [1e-4, 5e-4, 1e-3],
    "model_config__d_model": [64, 128, 256],
    "model_config__n_layers": [4, 6, 8],
    "model_config__dropout": [0.0, 0.1, 0.2],
    "trainer_config__batch_size": [128, 256, 512],
}
```

**For FTTransformer:**

```python
param_grid = {
    "trainer_config__lr": [1e-5, 5e-5, 1e-4],
    "model_config__d_model": [64, 128, 256],
    "model_config__n_heads": [4, 8, 16],
    "model_config__n_layers": [4, 6, 8],
    "model_config__attn_dropout": [0.0, 0.1, 0.2],
}
```

## References

Hyperparameter recommendations synthesized from:

- Gorishniy et al. (2021). _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021
- Gu & Dao (2024). _Mamba: Linear-Time Sequence Modeling_. arXiv:2312.00752
- Internal ablation studies on 20+ benchmark datasets
- Community feedback and production deployments

## See Also

- [Model Comparison](comparison_tables) — Performance benchmarks
- [Config System](../core_concepts/config_system) — Configuration API details
