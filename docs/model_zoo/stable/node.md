# NODE

Neural Oblivious Decision Ensembles. Differentiable decision trees with gradient boosting inductive bias.

## Key Characteristics

- **Architecture**: Ensemble of oblivious decision trees
- **Complexity**: Medium
- **Speed**: Moderate
- **Best for**: When tree inductive bias helps, some interpretability

## When to Use

✅ **Use NODE when:**

- Tree-based inductive bias is beneficial
- Need some interpretability
- Gradient boosting performs well on your data

❌ **Consider alternatives when:**

- Need maximum accuracy → try [Mambular](mambular)
- Full interpretability required → use XGBoost/LightGBM
- Very large datasets → may be slow

## Configuration

```python
from deeptab.configs import NODEConfig

cfg = NODEConfig(
    n_layers=8,
    depth=6,  # Tree depth
    n_trees=2048,  # Number of trees per layer
)
```

## Quick Example

```python
from deeptab.models import NODEClassifier, NODERegressor

model = NODEClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

## Performance Notes

- **Strengths**: Good on data where GBDTs excel
- **Interpretability**: Partial (tree structure visible)
- **Training**: Moderate speed

## References

- Popov, S., et al. (2020). _Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data_
