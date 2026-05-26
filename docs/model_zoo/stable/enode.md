# ENODE

Extended NODE with feature embeddings. Enhanced version of NODE with better feature representation.

## Key Characteristics

- **Architecture**: NODE + learned feature embeddings
- **Complexity**: Medium-high
- **Speed**: Moderate
- **Best for**: When NODE works but needs better feature handling

## When to Use

✅ **Use ENODE when:**

- NODE performs well but you want better accuracy
- Need tree inductive bias with rich features
- Mix of numerical and categorical features

❌ **Consider alternatives when:**

- NODE doesn't help → try other architectures
- Need speed → try [NODE](node) or simpler models

## Configuration

```python
from deeptab.configs import ENODEConfig

cfg = ENODEConfig(
    d_model=128,
    n_layers=8,
    depth=6,
)
```

## Quick Example

```python
from deeptab.models import ENODERegressor

model = ENODERegressor()
model.fit(X_train, y_train, max_epochs=50)
```

## Performance Notes

- **Improvement over NODE**: +1-3% accuracy typically
- **Cost**: Slower than NODE due to embeddings
- **Best**: When NODE is promising but not quite enough
