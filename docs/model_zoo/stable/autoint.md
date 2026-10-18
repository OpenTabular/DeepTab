# AutoInt

**Available as:** `AutoIntClassifier`, `AutoIntRegressor`, and `AutoIntLSS` in `deeptab.models`.

## Overview

AutoInt learns feature interactions with stacked multi-head self-attention layers. It treats tabular columns as feature tokens, repeatedly attends across tokens, then either flattens the sequence or selects a CLS token before predicting with a linear head.

Use AutoInt when the main research question is automatic feature interaction learning rather than full Transformer encoder modeling.

## Architectural Details

DeepTab's `AutoInt` implementation uses:

1. `EmbeddingLayer` to create a `(batch, n_features, d_model)` token sequence.
2. A stack of `n_layers` attention interaction layers.
3. Each layer applies attention and a linear projection with residual connections, then a ReLU; `fprenorm` selects normalization before attention or after its residual addition.
4. With `fprenorm=True`, a final `LayerNorm` is applied to the token sequence.
5. Select the CLS token when `use_cls=True`; otherwise flatten the sequence. Pass the result to a linear output head.

```text
feature tokens -> attention interaction layers -> optional final norm -> CLS selection or flatten -> Linear
```

## Main Building Blocks

| Component             | DeepTab implementation                   | Role                                                 |
| --------------------- | ---------------------------------------- | ---------------------------------------------------- |
| Tokenizer             | `EmbeddingLayer`                         | Builds feature tokens.                               |
| Interaction layer     | `nn.MultiheadAttention`                  | Learns pairwise and higher-order token interactions. |
| Residual projection   | `nn.Linear(d_model, d_model)`            | Updates each attended token.                         |
| Key/value compression | Optional `nn.Linear` over the token axis | Shortens keys and values when compression is set.    |
| Output head           | `nn.Linear(head_input_dim, num_classes)` | Uses CLS width or flattened token width.             |

## Implementation Notes

`kv_compression` is `None` by default, so every layer attends over all tokens. A ratio in `(0, 1]` adds learned projections along the token axis that shorten keys and values to `max(1, int(n_tokens * ratio))` tokens, where `n_tokens` includes the CLS token when `use_cls=True`. Queries keep their full length, so every token still receives an update. With eight feature tokens and `kv_compression=0.5`, each query attends over four compressed tokens.

`kv_compression_sharing` decides how compression projections are shared. `"layerwise"` uses one projection for keys and values in every layer, `"key-value"` uses one projection per layer for both keys and values, and `"headwise"` uses separate key and value projections in each layer. Other ratios or sharing values raise a `ValueError`.

> **Configuration note:** Previously, AutoInt built compression layers but never called them, so the old default `kv_compression=0.5` had no effect on predictions and its parameters never trained. Compression is now applied whenever a ratio is set, and the default is `None` so that default models keep full attention. Each interaction layer also ends with the ReLU from the original AutoInt design. Models saved by earlier versions load with this added ReLU, and those that stored the old default `kv_compression=0.5` also load with compression active. Retrain them rather than reusing old predictions.

`fprenorm=True` normalizes the attention input and enables a final `LayerNorm` before CLS selection or flattening. `fprenorm=False` normalizes after the attention residual addition and creates no final normalization layer. It does not move `last_norm` after pooling.

> **Configuration note:** `use_cls=True` adds a summary token and selects that token for prediction. With two feature tokens of width 16, flattening would give 32 values without CLS and 48 with it. CLS selection instead gives 16 values, so the head must be `Linear(16, num_classes)`. Previously, the extra token was flattened into a head sized for the original feature count. The implementation now keeps token counting, selection, and head sizing consistent.

## Practical Config

```python
from deeptab.configs import AutoIntConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import AutoIntClassifier

model = AutoIntClassifier(
    model_config=AutoIntConfig(
        d_model=128,
        n_layers=4,
        n_heads=8,
        attn_dropout=0.2,
    ),
    preprocessing_config=PreprocessingConfig(numerical_method="quantile"),
    trainer_config=TrainerConfig(lr=3e-4, batch_size=128, max_epochs=100),
    random_state=101,
)
```

Key settings:

| Setting                       | Typical range     | Effect                                          |
| ----------------------------- | ----------------- | ----------------------------------------------- |
| `d_model`                     | `64` to `256`     | Token width.                                    |
| `n_layers`                    | `2` to `6`        | Number of interaction layers.                   |
| `n_heads`                     | `4` to `8`        | Attention heads; must divide `d_model`.         |
| `attn_dropout`                | `0.0` to `0.3`    | Attention regularization.                       |
| `kv_compression`              | `None` or `0.5`   | Key/value compression ratio.                    |
| `transformer_dim_feedforward` | Present in config | Not used by the current `AutoInt` architecture. |

## When To Use

Use AutoInt for attention-based feature interaction studies and as a lighter alternative to full Transformer encoders. Prefer FTTransformer when you need a feed-forward Transformer block and sequence pooling.

## References

- Song et al., [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921).
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
