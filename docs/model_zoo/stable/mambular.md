# Mambular

**Available as:** `MambularClassifier`, `MambularRegressor`, `MambularLSS` — import from `deeptab.models`.

## Overview

Mambular treats tabular columns as a sequence of feature tokens and processes that sequence with Mamba-style state-space blocks. It is DeepTab's main stable state-space model for tabular data.

Use Mambular when you want to compare sequence modeling over columns against attention models such as FTTransformer and SAINT.

## Architectural Details

DeepTab's `Mambular` pipeline is:

1. `EmbeddingLayer` tokenizes numerical, categorical, and embedding features.
2. Optional feature-token shuffling is applied when `shuffle_embeddings=True`.
3. A Mamba block stack processes the token sequence.
4. `pool_sequence` aggregates the sequence.
5. `MLPhead` predicts the target.

```text
feature tokens -> optional shuffle -> Mamba/MambaOriginal -> pooling -> MLPhead
```

## Main Building Blocks

| Component | DeepTab implementation | Role |
| --- | --- | --- |
| Tokenizer | `EmbeddingLayer` | Converts columns to a token sequence. |
| Sequence block | `Mamba` or `MambaOriginal` | Applies selective state-space sequence processing. |
| Pooling | `pooling_method` | Reduces tokens to a row representation. |
| Head | `MLPhead` | Task-specific prediction. |

## Implementation Notes

The default config uses `d_model=64`, `n_layers=4`, `d_state=128`, `d_conv=4`, `expand_factor=2`, `norm="RMSNorm"`, and `pooling_method="avg"`.

`mamba_version="mamba-torch"` selects DeepTab's local Mamba block; other values select `MambaOriginal`. `bidirectional`, `use_learnable_interaction`, and `use_pscan` expose implementation variants for research comparisons.

## Practical Config

```python
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MambularClassifier

model = MambularClassifier(
    model_config=MambularConfig(
        d_model=64,
        n_layers=4,
        d_state=128,
        d_conv=4,
        pooling_method="avg",
    ),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(lr=3e-4, batch_size=128, max_epochs=100),
    random_state=101,
)
```

Key settings:

| Setting | Typical range | Effect |
| --- | --- | --- |
| `d_model` | `32` to `128` | Token width. |
| `n_layers` | `2` to `6` | Number of Mamba blocks. |
| `d_state` | `64` to `256` | State-space memory size. |
| `d_conv` | `2` to `8` | Local convolution width inside Mamba. |
| `bidirectional` | `False` or `True` | Whether to process feature order in both directions. |

## When To Use

Use Mambular when feature order or sequential token mixing is part of the model hypothesis. Because tabular columns do not have a natural order, compare against shuffled-token variants and attention baselines.

## References

- Gu and Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).
- Thielmann et al., [Mambular: A Sequential Model for Tabular Deep Learning](https://arxiv.org/abs/2408.06291).
