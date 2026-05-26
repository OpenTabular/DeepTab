# AutoInt

Automatic feature interaction learning via multi-head self-attention. Explicitly models feature crosses.

## Key Characteristics

- **Architecture**: Multi-head attention for feature interactions
- **Complexity**: Medium
- **Speed**: Moderate
- **Best for**: When feature interactions are crucial

## When to Use

✅ **Use AutoInt when:**

- Feature interactions are key to performance
- Need explicit interaction modeling
- Moderate number of features

❌ **Consider alternatives when:**

- Too many features → attention becomes expensive
- Simple patterns → simpler models work

## Configuration

```python
from deeptab.configs import AutoIntConfig

cfg = AutoIntConfig(
    d_model=128,
    n_heads=8,
    n_layers=4,
)
```

## Quick Example

```python
from deeptab.models import AutoIntRegressor

model = AutoIntRegressor()
model.fit(X_train, y_train, max_epochs=50)
```

## Performance Notes

- **Strengths**: Excellent at learning feature interactions
- **Memory**: Scales with number of features
- **Best**: Mid-size feature sets with rich interactions

## References

- Song, W., et al. (2019). _AutoInt: Automatic Feature Interaction Learning_
