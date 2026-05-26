# Mambular

Stacked Mamba State Space Model for tabular data. DeepTab's flagship architecture combining efficient sequence modeling with strong performance across task types.

## Key Characteristics

- **Architecture**: Multiple Mamba SSM layers with residual connections
- **Complexity**: Medium (6-8 layers typical)
- **Speed**: Fast inference, moderate training
- **Memory**: Efficient (linear complexity)
- **Best for**: General-purpose, large datasets, when training time matters

## When to Use

✅ **Use Mambular when:**

- You need strong general-purpose performance
- Working with medium to large datasets (>10K samples)
- Training efficiency is important
- You want state-of-the-art results without excessive compute

❌ **Consider alternatives when:**

- Dataset is very small (<1K samples) → try [MambaTab](mambatab) or [TabM](tabm)
- Need maximum interpretability → try [NODE](node) or [NDTF](ndtf)
- Extremely limited compute → try [MLP](mlp) or [ResNet](resnet)

## Configuration Highlights

### Model Config (MambularConfig)

| Parameter        | Default | Range     | Description            |
| ---------------- | ------- | --------- | ---------------------- |
| `d_model`        | 64      | 64-512    | Embedding dimension    |
| `n_layers`       | 8       | 4-12      | Number of Mamba layers |
| `expand_factor`  | 2       | 1-4       | State expansion factor |
| `dropout`        | 0.0     | 0.0-0.5   | Dropout rate           |
| `layer_norm_eps` | 1e-5    | 1e-6-1e-4 | Layer norm epsilon     |

### Recommended Settings

**Small datasets (<5K samples):**

```python
from deeptab.configs import MambularConfig

cfg = MambularConfig(
    d_model=64,
    n_layers=4,
    dropout=0.2,
)
```

**Medium datasets (5K-50K samples):**

```python
cfg = MambularConfig(
    d_model=128,
    n_layers=6,
    dropout=0.1,
)
```

**Large datasets (>50K samples):**

```python
cfg = MambularConfig(
    d_model=256,
    n_layers=8,
    dropout=0.0,
)
```

## Quick Example

```python
from deeptab.models import MambularClassifier, MambularRegressor, MambularLSS
from deeptab.configs import MambularConfig

# Classification
model = MambularClassifier(
    model_config=MambularConfig(d_model=128, n_layers=6)
)
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)

# Regression
model = MambularRegressor()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)

# LSS (distributional)
model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
params = model.predict(X_test)  # Distribution parameters
```

## Performance Notes

- **Strengths**: Balanced speed/accuracy tradeoff, scales well to large datasets
- **Training time**: ~2-3x slower than MLP, ~2x faster than FTTransformer
- **Inference**: Very fast (linear complexity)
- **GPU utilization**: Good, benefits from batch processing
- **Typical accuracy**: Top-tier across most benchmarks

## Architecture Details

Mambular stacks multiple Mamba blocks with:

1. **Input embedding**: Numerical and categorical features → d_model dimensions
2. **Mamba layers**: State space modeling with selective scan
3. **Residual connections**: Skip connections between layers
4. **Output head**: Task-specific (classification/regression/LSS)

## Comparison with Similar Models

| Model         | Speed      | Accuracy   | Memory     | Interpretability |
| ------------- | ---------- | ---------- | ---------- | ---------------- |
| **Mambular**  | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐             |
| FTTransformer | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | ⭐⭐             |
| MambaTab      | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | ⭐⭐             |
| ResNet        | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐           |

## References

- Gu, A., & Dao, T. (2024). _Mamba: Linear-Time Sequence Modeling with Selective State Spaces_. arXiv:2312.00752
- Original Mamba paper adapted for tabular data in DeepTab

## See Also

- [MambaTab](mambatab) — Lightweight single-block variant
- [MambAttention](mambattention) — Hybrid with attention
- [Comparison Tables](../comparison_tables) — Performance benchmarks
- [Recommended Configs](../recommended_configs) — Hyperparameter recipes
