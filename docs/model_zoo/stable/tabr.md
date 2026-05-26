# TabR

Retrieval-augmented tabular learning. Uses k-nearest neighbors for context-aware predictions.

## Key Characteristics

- **Architecture**: Neural network + kNN retrieval
- **Complexity**: Medium
- **Speed**: Moderate (kNN search overhead)
- **Best for**: Local similarity matters, large datasets

## When to Use

✅ **Use TabR when:**

- Local patterns/similarity is important
- Large training datasets (>50K samples)
- Non-parametric behavior is beneficial

❌ **Consider alternatives when:**

- Small datasets (<10K) → retrieval less effective
- Need fast inference → kNN adds overhead

## Configuration

```python
from deeptab.configs import TabRConfig

cfg = TabRConfig(
    d_model=128,
    n_layers=4,
    k_neighbors=32,  # Number of neighbors to retrieve
)
```

## Quick Example

```python
from deeptab.models import TabRClassifier

model = TabRClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

## Performance Notes

- **Strengths**: Excellent on large datasets with local structure
- **Inference**: Slower due to retrieval step
- **Memory**: Stores training data for retrieval

## References

- Rubachev, I., et al. (2023). _Retrieval-Augmented Deep Tabular Learning_
