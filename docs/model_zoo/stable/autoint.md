# AutoInt

**Available as:** `AutoIntClassifier`, `AutoIntRegressor`, `AutoIntLSS` — import from `deeptab.models`.

## Overview

AutoInt learns feature interactions with stacked multi-head self-attention layers. It treats tabular columns as feature tokens, repeatedly attends across tokens, flattens the final token sequence, and predicts with a linear head.

Use AutoInt when the main research question is automatic feature interaction learning rather than full Transformer encoder modeling.

## Architectural Details

DeepTab's `AutoInt` implementation uses:

1. `EmbeddingLayer` to create a `(batch, n_features, d_model)` token sequence.
2. A stack of `n_layers` attention interaction layers.
3. Each layer applies `LayerNorm`, `nn.MultiheadAttention`, a residual connection, a linear projection, and a second residual connection.
4. The final token sequence is flattened and passed to a linear output head.

```text
feature tokens -> [LayerNorm -> MultiheadAttention -> residual -> Linear -> residual] x n_layers -> flatten -> Linear
```

## Main Building Blocks

| Component | DeepTab implementation | Role |
| --- | --- | --- |
| Tokenizer | `EmbeddingLayer` | Builds feature tokens. |
| Interaction layer | `nn.MultiheadAttention` | Learns pairwise and higher-order token interactions. |
| Residual projection | `nn.Linear(d_model, d_model)` | Updates each attended token. |
| Output head | `nn.Linear(d_model * n_inputs, num_classes)` | Uses all token states for prediction. |

## Implementation Notes

`AutoIntConfig` exposes `kv_compression` and `kv_compression_sharing`, and the architecture constructs compression layers. In the current DeepTab forward path, those compression layers are not applied to the attention call; the runtime behavior is standard multi-head self-attention over all feature tokens.

The config field is named `fprenorm`, while the architecture checks `prenorm` for `last_norm`. Unless this is aligned in code, the final optional normalization path is effectively inactive with the default config field name.

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
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(lr=3e-4, batch_size=128, max_epochs=100),
    random_state=101,
)
```

Key settings:

| Setting | Typical range | Effect |
| --- | --- | --- |
| `d_model` | `64` to `256` | Token width. |
| `n_layers` | `2` to `6` | Number of interaction layers. |
| `n_heads` | `4` to `8` | Attention heads; must divide `d_model`. |
| `attn_dropout` | `0.0` to `0.3` | Attention regularization. |
| `transformer_dim_feedforward` | Present in config | Not used by the current `AutoInt` architecture. |

## When To Use

Use AutoInt for attention-based feature interaction studies and as a lighter alternative to full Transformer encoders. Prefer FTTransformer when you need a feed-forward Transformer block and sequence pooling.

## References

- Song et al., [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921).
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
