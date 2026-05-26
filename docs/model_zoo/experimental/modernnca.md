# ModernNCA

Modern Neighborhood Component Analysis for tabular learning. Experimental metric learning approach.

## Key Characteristics

- **Architecture**: Metric learning with neural embeddings
- **Complexity**: Medium
- **Speed**: Moderate
- **Best for**: When local structure and distances matter
- **Status**: ⚠️ Experimental - API may change

## When to Use

✅ **Use ModernNCA when:**

- Willing to experiment with cutting-edge methods
- Metric learning approach seems promising
- Can handle potential API changes (pin versions!)

❌ **Consider stable alternatives:**

- [Mambular](../stable/mambular) — Stable, proven performance
- [FTTransformer](../stable/fttransformer) — Stable baseline

## Configuration

```python
from deeptab.configs import ModernNCAConfig

cfg = ModernNCAConfig(
    d_model=128,
    n_layers=6,
)
```

## Quick Example

```python
from deeptab.models.experimental import ModernNCAClassifier

# Always pin version for experimental models!
# pip install deeptab==2.0.0

model = ModernNCAClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

## Important Notes

- ⚠️ **Not semantically versioned** — API may change in minor releases
- 📌 **Pin DeepTab version** — Use `deeptab==x.y.z` in requirements
- 🔄 **Check release notes** — Monitor for API changes
- ✅ **Will migrate to stable** — If promotion criteria are met

## See Also

- [Experimental Models Tutorial](../../tutorials/experimental) — Best practices
- [Model Tiers](../../core_concepts/model_tiers) — Understanding experimental vs stable
