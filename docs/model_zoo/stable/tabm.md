# TabM

Batch-ensembling MLP. Efficient ensemble method providing ensemble accuracy at near-single-model cost.

## Key Characteristics

- **Architecture**: Batch-ensembled feedforward network
- **Complexity**: Low-medium
- **Speed**: Fast (similar to single MLP)
- **Best for**: Getting ensemble benefits without ensemble cost

## When to Use

✅ **Use TabM when:**

- Want ensemble accuracy without training multiple models
- Limited resources but need robustness
- Small to medium datasets

❌ **Consider alternatives when:**

- Can afford true ensembles → train multiple models
- Need maximum single-model accuracy → try [Mambular](mambular)

## Configuration

```python
from deeptab.configs import TabMConfig

cfg = TabMConfig(
    d_model=128,
    n_layers=8,
    n_ensembles=4,  # Number of ensemble members
)
```

## Quick Example

```python
from deeptab.models import TabMClassifier

model = TabMClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

## Performance Notes

- **Training time**: Similar to single MLP
- **Accuracy**: Between single model and full ensemble
- **Memory**: ~1.5x single model

## References

- Gorishniy, Y., et al. (2022). _On Embeddings for Numerical Features in Tabular Deep Learning_
