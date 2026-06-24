# FTTransformer

**Available as:** `FTTransformerClassifier`, `FTTransformerRegressor`, and `FTTransformerLSS` in `deeptab.models`.

## Overview

FTTransformer is a feature-token Transformer for tabular data. It represents each column as a token, applies Transformer encoder layers over the feature sequence, pools the sequence, and predicts with an MLP head.

Use it when feature interactions are expected to be high-order and nonlocal, especially on medium-to-large datasets where attention layers can be trained reliably.

## Architectural Details

DeepTab's `FTTransformer` implementation follows the RTDL-style feature-token design:

1. `EmbeddingLayer` tokenizes numerical, categorical, and embedding features into `(batch, n_features, d_model)`.
2. `CustomTransformerEncoderLayer` is stacked with `nn.TransformerEncoder`.
3. `pool_sequence` converts the token sequence to one vector using `pooling_method`.
4. Optional final normalization is applied.
5. `MLPhead` maps the pooled vector to the task output.

```text
feature tokens -> TransformerEncoder x n_layers -> pooling -> optional norm -> MLPhead
```

## Main Building Blocks

| Component     | DeepTab implementation          | Role                                                   |
| ------------- | ------------------------------- | ------------------------------------------------------ |
| Tokenizer     | `EmbeddingLayer`                | Creates one vector per input feature.                  |
| Encoder block | `CustomTransformerEncoderLayer` | Multi-head attention plus feed-forward transformation. |
| Encoder stack | `nn.TransformerEncoder`         | Repeats the block `n_layers` times.                    |
| Pooling       | `pooling_method`, `use_cls`     | Reduces feature tokens to one representation.          |
| Head          | `MLPhead`                       | Task-specific prediction head.                         |

## Implementation Notes

Unlike `TabTransformer`, FTTransformer embeds all supported feature types before attention. This makes it a better default Transformer when the dataset has many numerical features or a balanced mix of numerical and categorical columns.

The default configuration uses `d_model=128`, `n_layers=4`, `n_heads=8`, `attn_dropout=0.2`, and `ff_dropout=0.1`.

## Practical Config

```python
from deeptab.configs import FTTransformerConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import FTTransformerClassifier

model = FTTransformerClassifier(
    model_config=FTTransformerConfig(
        d_model=128,
        n_layers=4,
        n_heads=8,
        attn_dropout=0.2,
        ff_dropout=0.1,
        pooling_method="avg",
    ),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(lr=3e-4, batch_size=128, max_epochs=100),
    random_state=101,
)
```

Key settings:

| Setting                       | Typical range        | Effect                                  |
| ----------------------------- | -------------------- | --------------------------------------- |
| `d_model`                     | `64` to `256`        | Token width and main capacity driver.   |
| `n_layers`                    | `2` to `6`           | Transformer depth.                      |
| `n_heads`                     | `4` to `8`           | Attention heads; must divide `d_model`. |
| `transformer_dim_feedforward` | `2x` to `4x d_model` | Feed-forward capacity.                  |
| `pooling_method`              | `"avg"` or `"cls"`   | Sequence aggregation strategy.          |

## When To Use

Use FTTransformer for research comparisons involving attention over feature tokens. It is usually a more general Transformer baseline than TabTransformer because it handles numerical tokens directly.

## References

- Gorishniy et al., [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959).
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
