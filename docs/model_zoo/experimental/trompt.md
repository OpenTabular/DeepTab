# Trompt

Transformer with prompting for tabular data. Experimental architecture using prompt-based learning.

## Key Characteristics

- **Architecture**: Transformer with learnable prompts
- **Complexity**: Medium-high
- **Speed**: Moderate
- **Best for**: When prompt-based learning helps
- **Status**: ⚠️ Experimental - API may change

## When to Use

✅ **Use Trompt when:**

- Exploring prompt-based methods
- Willing to experiment and provide feedback
- Can handle API changes (pin versions!)

❌ **Consider stable alternatives:**

- [FTTransformer](../stable/fttransformer) — Stable transformer
- [Mambular](../stable/mambular) — Stable general-purpose

## Configuration

```python
from deeptab.configs import TromptConfig

cfg = TromptConfig(
    d_model=128,
    n_heads=8,
    n_layers=6,
)
```

## Quick Example

```python
from deeptab.models.experimental import TromptClassifier

# Always pin version for experimental models!
# pip install deeptab==2.0.0

model = TromptClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

## Important Notes

- ⚠️ **Not semantically versioned** — API may change
- 📌 **Pin DeepTab version** — Use `deeptab==x.y.z`
- 🔄 **Monitor releases** — Check for changes before upgrading
- ✅ **Promotion path** — May become stable if criteria met

## See Also

- [Experimental Models Tutorial](../../tutorials/experimental)
- [Model Promotion Policy](../../developer_guide/model_promotion_policy)
