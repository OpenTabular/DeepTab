# SAINT

**Available as:** `SAINTClassifier`, `SAINTRegressor`, and `SAINTLSS` in `deeptab.models`.

## Overview

SAINT is an attention architecture for tabular data that combines feature-wise attention with row-wise attention. In DeepTab, SAINT embeds all supported feature types, applies a row/column Transformer block, pools the resulting sequence, and predicts with an MLP head.

Use it when you want a Transformer-style model that can mix information across both columns and samples, especially for research comparisons with FTTransformer and TabTransformer.

## Architectural Details

DeepTab's `SAINT` implementation uses:

1. `EmbeddingLayer` to build feature tokens.
2. Optional class token support through `use_cls`.
3. `RowColTransformer`, which alternates column-wise attention over feature tokens and row-wise attention after reshaping the batch/feature representation.
4. `pool_sequence` to aggregate tokens.
5. Optional final normalization and `MLPhead`.

```text
feature tokens -> RowColTransformer -> pooling -> optional norm -> MLPhead
```

## Main Building Blocks

| Component           | DeepTab implementation                                  | Role                                       |
| ------------------- | ------------------------------------------------------- | ------------------------------------------ |
| Tokenizer           | `EmbeddingLayer`                                        | Converts each input feature to a token.    |
| Column attention    | `nn.MultiheadAttention` inside `RowColTransformer`      | Models feature interactions within a row.  |
| Row attention       | Flattened row representation inside `RowColTransformer` | Mixes sample-level context within a batch. |
| Feed-forward blocks | LayerNorm + Linear + activation + dropout               | Adds nonlinear token updates.              |
| Prediction head     | `MLPhead`                                               | Produces final outputs.                    |

## Implementation Notes

The original SAINT paper also emphasizes contrastive pretraining and data augmentation. DeepTab's stable model page documents the supervised architecture path implemented in `deeptab.architectures.saint`; do not assume contrastive pretraining is active unless added explicitly in the training workflow.

The default config uses `d_model=128`, `n_layers=1`, `n_heads=2`, `pooling_method="cls"`, and `use_cls=True`.

## Practical Config

```python
from deeptab.configs import PreprocessingConfig, SAINTConfig, TrainerConfig
from deeptab.models import SAINTClassifier

model = SAINTClassifier(
    model_config=SAINTConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        attn_dropout=0.1,
        ff_dropout=0.1,
        pooling_method="cls",
        use_cls=True,
    ),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(lr=3e-4, batch_size=128, max_epochs=100),
    random_state=101,
)
```

Key settings:

| Setting                      | Typical range                     | Effect                      |
| ---------------------------- | --------------------------------- | --------------------------- |
| `d_model`                    | `64` to `192`                     | Token width.                |
| `n_layers`                   | `1` to `4`                        | Row/column attention depth. |
| `n_heads`                    | `2` to `8`                        | Number of attention heads.  |
| `attn_dropout`, `ff_dropout` | `0.0` to `0.3`                    | Regularization.             |
| `pooling_method`, `use_cls`  | `"cls"`/`True` or `"avg"`/`False` | Token aggregation behavior. |

## When To Use

Use SAINT when modeling interactions across both features and samples is part of the experimental question. It can be more expensive and batch-sensitive than FTTransformer because row attention depends on the batch representation.

## References

- Somepalli et al., [SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training](https://arxiv.org/abs/2106.01342).
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
