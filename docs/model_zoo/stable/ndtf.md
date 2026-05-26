# NDTF

Neural Decision Tree Forest. Ensemble of differentiable decision trees.

## Key Characteristics

- **Architecture**: Forest of neural decision trees
- **Complexity**: Medium
- **Speed**: Moderate
- **Best for**: Tree ensemble benefits in neural form

## When to Use

✅ **Use NDTF when:**

- Random forest works well on your data
- Want neural network + tree ensemble benefits
- Need interpretability

❌ **Consider alternatives when:**

- Trees don't help → try other architectures
- Need maximum accuracy → try [Mambular](mambular)

## Configuration

```python
from deeptab.configs import NDTFConfig

cfg = NDTFConfig(
    n_ensembles=8,  # Number of trees
    max_depth=6,  # Tree depth
    d_model=64,
)
```

## Quick Example

```python
from deeptab.models import NDTFClassifier

model = NDTFClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

## Performance Notes

- **Strengths**: Combines neural nets with forest ensembling
- **Interpretability**: Better than black-box models
- **Training**: Moderate speed

## See Also

- [NODE](node) — Related tree-based architecture
- [ENODE](enode) — Extended NODE variant
