# FTTransformer

Feature Tokenizer Transformer for tabular data. A strong general-purpose model using attention mechanisms on feature tokens.

## Key Characteristics

- **Architecture**: Transformer with feature-level tokenization
- **Complexity**: Medium-high (attention on all features)
- **Speed**: Moderate training and inference
- **Memory**: High (quadratic attention complexity)
- **Best for**: General-purpose, feature interactions, high-capacity needs

## When to Use

✅ **Use FTTransformer when:**

- You need strong baseline performance
- Feature interactions are important
- Have sufficient compute and memory
- Dataset has many features (>20)

❌ **Consider alternatives when:**

- Limited memory/compute → try [Mambular](mambular) or [ResNet](resnet)
- Very large datasets → try [Mambular](mambular) or [TabR](tabr)
- Need fastest training → try [MLP](mlp) or [ResNet](resnet)

## Configuration Highlights

### Model Config (FTTransformerConfig)

| Parameter      | Default | Range   | Description                  |
| -------------- | ------- | ------- | ---------------------------- |
| `d_model`      | 64      | 64-512  | Token embedding dimension    |
| `n_heads`      | 8       | 4-16    | Number of attention heads    |
| `n_layers`     | 6       | 3-12    | Number of transformer blocks |
| `attn_dropout` | 0.0     | 0.0-0.3 | Attention dropout            |
| `ffn_dropout`  | 0.0     | 0.0-0.5 | Feedforward dropout          |

### Recommended Settings

```python
from deeptab.configs import FTTransformerConfig

# Balanced setup
cfg = FTTransformerConfig(
    d_model=128,
    n_heads=8,
    n_layers=6,
    attn_dropout=0.1,
    ffn_dropout=0.1,
)
```

## Quick Example

```python
from deeptab.models import FTTransformerClassifier, FTTransformerRegressor

model = FTTransformerClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)
```

## Performance Notes

- **Strengths**: Excellent accuracy, handles feature interactions well
- **Training time**: Slower than SSMs, faster than SAINT
- **Memory**: Scales quadratically with number of features
- **Best suited**: Medium-sized datasets with meaningful feature interactions

## References

- Gorishniy, Y., et al. (2021). _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021

## See Also

- [TabTransformer](tabtransformer) — Transformer on categorical features only
- [Mambular](mambular) — More efficient alternative
- [Comparison Tables](../comparison_tables)
