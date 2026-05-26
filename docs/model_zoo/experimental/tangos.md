# Tangos

Tangent-based optimization for tabular learning. Experimental architecture with novel optimization approach.

## Key Characteristics

- **Architecture**: Neural network with tangent-based updates
- **Complexity**: Medium
- **Speed**: Moderate
- **Best for**: When standard optimization plateaus
- **Status**: ⚠️ Experimental - API may change

## When to Use

✅ **Use Tangos when:**

- Standard optimization struggles on your data
- Exploring novel optimization methods
- Willing to experiment (pin versions!)

❌ **Consider stable alternatives:**

- [Mambular](../stable/mambular) — Proven optimization
- [ResNet](../stable/resnet) — Simple and effective

## Configuration

```python
from deeptab.configs import TangosConfig

cfg = TangosConfig(
    d_model=128,
    n_layers=6,
)
```

## Quick Example

```python
from deeptab.models.experimental import TangosRegressor

# Always pin version for experimental models!
# pip install deeptab==2.0.0

model = TangosRegressor()
model.fit(X_train, y_train, max_epochs=50)
```

## Important Notes

- ⚠️ **Experimental API** — May change without deprecation
- 📌 **Version pinning required** — Use exact version
- 🔄 **Check release notes** — Before upgrading
- ✅ **Potential promotion** — If validation succeeds

## See Also

- [Experimental Models Tutorial](../../tutorials/experimental)
- [Using Experimental Models](../../core_concepts/model_tiers)
