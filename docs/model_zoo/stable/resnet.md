# ResNet

Residual MLP with skip connections. Simple, fast, and effective baseline for tabular data.

## Key Characteristics

- **Architecture**: Feedforward MLP with residual connections
- **Complexity**: Low-medium
- **Speed**: Very fast training and inference
- **Memory**: Very efficient
- **Best for**: Baselines, fast iteration, limited compute

## When to Use

✅ **Use ResNet when:**

- You need a fast baseline
- Limited computational resources
- Quick experimentation
- Simple feature relationships

❌ **Consider alternatives when:**

- Need maximum accuracy → try [Mambular](mambular) or [FTTransformer](fttransformer)
- Complex feature interactions → try transformers
- Want interpretability → try [NODE](node)

## Configuration Highlights

### Model Config (ResNetConfig)

| Parameter  | Default | Range   | Description               |
| ---------- | ------- | ------- | ------------------------- |
| `d_model`  | 64      | 32-256  | Hidden dimension          |
| `n_layers` | 8       | 4-16    | Number of residual blocks |
| `dropout`  | 0.0     | 0.0-0.5 | Dropout rate              |

### Recommended Settings

```python
from deeptab.configs import ResNetConfig

cfg = ResNetConfig(
    d_model=128,
    n_layers=8,
    dropout=0.1,
)
```

## Quick Example

```python
from deeptab.models import ResNetClassifier, ResNetRegressor

model = ResNetClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)
```

## Performance Notes

- **Strengths**: Fast, simple, competitive on many tasks
- **Training time**: Fastest among complex models
- **Typically**: 80-90% of best model accuracy with 2-3x speed

## References

- He, K., et al. (2016). _Deep Residual Learning for Image Recognition_. CVPR 2016
- Adapted for tabular data

## See Also

- [MLP](mlp) — Even simpler baseline
- [Mambular](mambular) — Better accuracy, slower
- [Comparison Tables](../comparison_tables)
