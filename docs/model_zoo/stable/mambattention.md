# MambAttention

## Overview

MambAttention is a hybrid model that alternates Mamba-style sequence processing with multi-head attention over feature tokens. It is useful for testing whether state-space layers and explicit attention provide complementary inductive biases.

Use it when Mambular is too restrictive but a full Transformer is not the desired baseline.

## Architectural Details

DeepTab's `MambAttention` pipeline is:

1. `EmbeddingLayer` creates feature tokens.
2. Optional feature-token shuffling is applied.
3. `MambAttn` builds a sequence of Mamba residual blocks and `nn.MultiheadAttention` layers according to the config.
4. The feature sequence is pooled.
5. Final normalization and `MLPhead` produce predictions.

```text
feature tokens -> optional shuffle -> Mamba/Attention hybrid stack -> pooling -> norm -> MLPhead
```

## Main Building Blocks

| Component | DeepTab implementation | Role |
| --- | --- | --- |
| Tokenizer | `EmbeddingLayer` | Builds one token per input feature. |
| Mamba blocks | `ResidualBlock` inside `MambAttn` | Local/selective state-space sequence processing. |
| Attention blocks | `nn.MultiheadAttention` | Explicit global token mixing. |
| Hybrid schedule | `n_mamba_per_attention`, `n_attention_layers`, `last_layer` | Controls where attention is inserted. |
| Head | `MLPhead` | Final task prediction. |

## Implementation Notes

`MambAttn` creates `config.n_layers + config.n_attention_layers` blocks, inserts an attention layer after every `n_mamba_per_attention` Mamba blocks, and then enforces the requested `last_layer` type.

The default config uses `d_model=64`, `n_layers=4`, `n_heads=8`, `n_attention_layers=1`, `n_mamba_per_attention=1`, and `last_layer="attn"`.

## Practical Config

```python
from deeptab.configs import MambAttentionConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MambAttentionClassifier

model = MambAttentionClassifier(
    model_config=MambAttentionConfig(
        d_model=64,
        n_layers=4,
        n_attention_layers=1,
        n_mamba_per_attention=1,
        n_heads=8,
        last_layer="attn",
    ),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(lr=3e-4, batch_size=128, max_epochs=100),
    random_state=101,
)
```

Key settings:

| Setting | Typical range | Effect |
| --- | --- | --- |
| `n_layers` | `2` to `6` | Mamba-block budget. |
| `n_attention_layers` | `1` to `3` | Number of explicit attention insertions. |
| `n_mamba_per_attention` | `1` to `3` | Frequency of attention layers. |
| `last_layer` | `"attn"` or `"mamba"` | Final mixing type. |
| `attn_dropout` | `0.0` to `0.3` | Attention regularization. |

## When To Use

Use MambAttention for ablations that compare pure Mamba, pure attention, and hybrid token mixers. It is more complex than Mambular, so tune it after establishing MLP/ResNet/FTTransformer baselines.

## References

- Gu and Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Thielmann et al., [Mambular: A Sequential Model for Tabular Deep Learning](https://arxiv.org/abs/2408.06291).
