# MLP

Simple feedforward neural network. The fastest baseline for tabular data.

## Key Characteristics

- **Architecture**: Plain feedforward layers
- **Complexity**: Low
- **Speed**: Fastest training and inference
- **Best for**: Quick baselines, simple patterns

## When to Use

✅ **Use MLP when:**

- Need fastest possible training
- Quick baseline for comparison
- Simple feature relationships
- Extremely limited resources

❌ **Consider alternatives when:**

- Need best accuracy → try [Mambular](mambular) or [FTTransformer](fttransformer)
- Complex interactions → try transformers or SSMs

## Configuration

```python
from deeptab.configs import MLPConfig

cfg = MLPConfig(
    d_model=128,
    n_layers=8,
    dropout=0.1,
)
```

## Quick Example

```python
from deeptab.models import MLPClassifier, MLPRegressor

model = MLPClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

## Performance Notes

- **Training**: Fastest among all models
- **Accuracy**: Solid baseline, ~80-90% of best models
- **Use case**: When speed > accuracy or as baseline
