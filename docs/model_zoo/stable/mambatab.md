# MambaTab

Single Mamba block architecture. Lightweight variant of Mambular for faster training with competitive accuracy.

## Key Characteristics

- **Architecture**: Single Mamba SSM block
- **Complexity**: Low
- **Speed**: Very fast training and inference
- **Memory**: Very efficient
- **Best for**: Small datasets, fast experimentation, resource-constrained settings

## When to Use

✅ **Use MambaTab when:**
- Dataset is small (<5K samples)
- Need fast training times
- Limited computational resources
- Quick prototyping

❌ **Consider alternatives when:**
- Large datasets → try [Mambular](mambular)
- Need maximum accuracy → try [Mambular](mambular) or [FTTransformer](fttransformer)

## Configuration Highlights

### Model Config (MambaTabConfig)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `d_model` | 64 | 32-256 | Embedding dimension |
| `expand_factor` | 2 | 1-4 | State expansion |
| `dropout` | 0.0 | 0.0-0.3 | Dropout rate |

### Recommended Settings

```python
from deeptab.configs import MambaTabConfig

cfg = MambaTabConfig(
    d_model=128,
    expand_factor=2,
    dropout=0.1,
)
```

## Quick Example

```python
from deeptab.models import MambaTabClassifier, MambaTabRegressor

model = MambaTabClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)
```

## Performance Notes

- **Training time**: 2-3x faster than Mambular
- **Accuracy**: ~95% of Mambular's performance
- **Sweet spot**: Small to medium datasets where speed matters

## See Also

- [Mambular](mambular) — Multi-layer version for better accuracy
- [MambAttention](mambattention) — Hybrid with attention
