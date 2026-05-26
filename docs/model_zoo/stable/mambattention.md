# MambAttention

Hybrid architecture combining Mamba state space modeling with attention mechanisms for both local and global feature interactions.

## Key Characteristics

- **Architecture**: Mamba layers + attention layers
- **Complexity**: Medium-high
- **Speed**: Moderate (slower than pure Mamba)
- **Memory**: Medium
- **Best for**: Complex feature interactions, when both local and global patterns matter

## When to Use

✅ **Use MambAttention when:**

- Need both local (Mamba) and global (attention) modeling
- Complex interdependent features
- Have sufficient compute budget

❌ **Consider alternatives when:**

- Limited compute → try [Mambular](mambular) or [MambaTab](mambatab)
- Pure attention sufficient → try [FTTransformer](fttransformer)

## Configuration Highlights

### Model Config (MambAttentionConfig)

| Parameter  | Default | Range  | Description             |
| ---------- | ------- | ------ | ----------------------- |
| `d_model`  | 64      | 64-256 | Embedding dimension     |
| `n_layers` | 6       | 4-10   | Number of hybrid blocks |
| `n_heads`  | 8       | 4-16   | Attention heads         |

### Recommended Settings

```python
from deeptab.configs import MambAttentionConfig

cfg = MambAttentionConfig(
    d_model=128,
    n_layers=6,
    n_heads=8,
)
```

## Quick Example

```python
from deeptab.models import MambAttentionClassifier

model = MambAttentionClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

## Performance Notes

- **Strengths**: Captures both local and global patterns
- **Training time**: Between Mambular and FTTransformer
- **Best for**: Tasks requiring multi-scale feature interactions

## See Also

- [Mambular](mambular) — Pure Mamba (faster)
- [FTTransformer](fttransformer) — Pure attention
