# TabTransformer

Transformer architecture applied to categorical feature embeddings. Excellent for categorical-heavy tabular data.

## Key Characteristics

- **Architecture**: Attention on categorical embeddings only
- **Complexity**: Medium
- **Speed**: Fast (attention only on categorical features)
- **Best for**: Categorical-heavy datasets

## When to Use

✅ **Use TabTransformer when:**

- Dataset has many categorical features
- Categorical interactions are important
- Fewer numerical features

❌ **Consider alternatives when:**

- Mostly numerical features → try [FTTransformer](fttransformer) or [Mambular](mambular)
- No categorical features → try other models

## Configuration

```python
from deeptab.configs import TabTransformerConfig

cfg = TabTransformerConfig(
    d_model=128,
    n_heads=8,
    n_layers=6,
)
```

## Quick Example

```python
from deeptab.models import TabTransformerClassifier

model = TabTransformerClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

## Performance Notes

- **Best performance**: 5+ categorical features
- **Memory efficient**: Attention only on categoricals
- **Training**: Faster than FTTransformer

## References

- Huang, X., et al. (2020). _TabTransformer: Tabular Data Modeling Using Contextual Embeddings_
