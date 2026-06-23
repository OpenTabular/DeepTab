# TabTransformer

**Available as:** `TabTransformerClassifier`, `TabTransformerRegressor`, `TabTransformerLSS` — import from `deeptab.models`.

## Overview

TabTransformer uses self-attention to contextualize categorical feature embeddings. DeepTab's implementation follows that core idea: categorical and external embedding features pass through a Transformer encoder, while numerical features are normalized and concatenated afterward before the prediction head.

Use it when categorical interactions are central to the task. If the dataset has no categorical features, use FTTransformer, MLP, ResNet, or TabM instead.

## Architectural Details

DeepTab's `TabTransformer` pipeline is:

1. Validate that categorical feature information is present.
2. Embed categorical and external embedding features with `EmbeddingLayer`.
3. Apply a Transformer encoder to the categorical token sequence.
4. Pool the contextualized categorical tokens.
5. Concatenate the pooled categorical representation with layer-normalized numerical features.
6. Predict with `MLPhead`.

```text
categorical tokens -> TransformerEncoder -> pooling
numerical features -> LayerNorm
[pooled categorical, normalized numerical] -> MLPhead
```

## Main Building Blocks

| Component             | DeepTab implementation                                      | Role                                                |
| --------------------- | ----------------------------------------------------------- | --------------------------------------------------- |
| Categorical tokenizer | `EmbeddingLayer(*({}, cat_feature_info, emb_feature_info))` | Embeds categorical columns only.                    |
| Transformer           | `CustomTransformerEncoderLayer` in `nn.TransformerEncoder`  | Contextualizes categorical tokens.                  |
| Numerical path        | `nn.LayerNorm(num_input_dim)`                               | Normalizes raw numerical vector.                    |
| Pooling               | `pool_sequence`                                             | Reduces categorical tokens.                         |
| Head                  | `MLPhead`                                                   | Combines categorical and numerical representations. |

## Implementation Notes

DeepTab raises a `ValueError` if no categorical features are available. This is intentional for this implementation, because the Transformer body is applied only to categorical tokens.

The default config uses `d_model=128`, `n_layers=4`, `n_heads=8`, `transformer_activation=ReGLU()`, and `transformer_dim_feedforward=512`.

## Practical Config

```python
from deeptab.configs import PreprocessingConfig, TabTransformerConfig, TrainerConfig
from deeptab.models import TabTransformerClassifier

model = TabTransformerClassifier(
    model_config=TabTransformerConfig(
        d_model=128,
        n_layers=4,
        n_heads=8,
        attn_dropout=0.2,
        ff_dropout=0.1,
        pooling_method="avg",
    ),
    preprocessing_config=PreprocessingConfig(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
    ),
    trainer_config=TrainerConfig(lr=3e-4, batch_size=128, max_epochs=100),
    random_state=101,
)
```

Key settings:

| Setting            | Typical range       | Effect                              |
| ------------------ | ------------------- | ----------------------------------- |
| `d_model`          | `64` to `256`       | Categorical token width.            |
| `n_layers`         | `2` to `6`          | Contextualization depth.            |
| `n_heads`          | `4` to `8`          | Attention heads.                    |
| `pooling_method`   | `"avg"` or `"cls"`  | How categorical tokens are reduced. |
| `head_layer_sizes` | `[]` to `[128, 64]` | Extra capacity after concatenation. |

## When To Use

Use TabTransformer for categorical-heavy datasets where context-dependent categorical embeddings are likely to matter. Prefer FTTransformer for numerical-heavy datasets.

## References

- Huang et al., [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678).
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
