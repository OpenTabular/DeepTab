# SAINT

Self-attention and intersample attention network. Combines row-wise and column-wise attention for tabular data.

## Key Characteristics

- **Architecture**: Dual attention (self + intersample)
- **Complexity**: High
- **Speed**: Slower (two attention mechanisms)
- **Best for**: Semi-supervised learning, complex dependencies

## When to Use

✅ **Use SAINT when:**

- Have unlabeled data for semi-supervised learning
- Need to model both feature and sample relationships
- Sufficient computational budget

❌ **Consider alternatives when:**

- Fully supervised only → try [FTTransformer](fttransformer)
- Limited compute → try [Mambular](mambular)
- Need speed → try [ResNet](resnet)

## Configuration

```python
from deeptab.configs import SAINTConfig

cfg = SAINTConfig(
    d_model=128,
    n_heads=8,
    n_layers=6,
)
```

## Quick Example

```python
from deeptab.models import SAINTClassifier

model = SAINTClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

## Performance Notes

- **Strengths**: Excellent for semi-supervised tasks
- **Training time**: Slower than most models
- **Best suited**: When you have unlabeled data to leverage

## References

- Somepalli, G., et al. (2021). _SAINT: Improved Neural Networks for Tabular Data_
