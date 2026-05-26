# TabulaRNN

Recurrent neural network for tabular data. Uses RNN/LSTM/GRU cells for sequential feature processing.

## Key Characteristics

- **Architecture**: RNN/LSTM/GRU on features
- **Complexity**: Medium
- **Speed**: Moderate to slow (sequential processing)
- **Best for**: When feature order matters, temporal data

## When to Use

✅ **Use TabulaRNN when:**

- Features have natural sequential order
- Temporal dependencies in features
- Working with time series as features

❌ **Consider alternatives when:**

- Features are unordered → try other models
- Need speed → RNNs are inherently sequential

## Configuration

```python
from deeptab.configs import TabulaRNNConfig

cfg = TabulaRNNConfig(
    d_model=128,
    n_layers=4,
    model_type="lstm",  # or "gru", "rnn"
)
```

## Quick Example

```python
from deeptab.models import TabulaRNNClassifier

model = TabulaRNNClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

## Performance Notes

- **Strengths**: Good for sequential/temporal features
- **Training**: Slower due to sequential nature
- **Best**: When feature ordering is meaningful
