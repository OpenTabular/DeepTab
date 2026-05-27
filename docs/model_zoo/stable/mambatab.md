# MambaTab

## Overview

MambaTab is exposed as a stable Mamba-family model, but the current DeepTab forward path behaves as a lightweight projected-feature network: it concatenates input features, projects them to `d_model`, normalizes and activates the representation, then predicts with `MLPhead`.

Use it as a compact baseline in the current release. For an active Mamba sequence model over feature tokens, prefer [Mambular](mambular) or [MambAttention](mambattention).

## Architectural Details

The current `MambaTab` forward path is:

1. Concatenate all input tensors.
2. Apply `initial_layer` from input dimension to `d_model`.
3. Temporarily unsqueeze along `axis`, apply `LayerNorm`, and apply `embedding_activation`.
4. Squeeze back to a row representation.
5. Predict with `MLPhead`.

```text
features -> concat -> Linear(input_dim, d_model) -> LayerNorm -> activation -> MLPhead
```

## Main Building Blocks

| Component | DeepTab implementation | Role |
| --- | --- | --- |
| Input path | `torch.cat(...)` | Uses raw/preprocessed feature tensors directly. |
| Projection | `initial_layer` | Maps input vector to `d_model`. |
| Normalization | `LayerNorm` | Stabilizes projected representation. |
| Head | `MLPhead` | Produces predictions. |
| Mamba block | `self.mamba = Mamba(...)` or `MambaOriginal(...)` | Instantiated in `__init__`, but not called in the current `forward`. |

## Implementation Notes

The presence of Mamba-related config fields (`d_state`, `d_conv`, `expand_factor`, `mamba_version`, `bidirectional`) does not mean they affect the current forward pass. They configure the instantiated `self.mamba` module, but that module is not applied before the head.

This distinction matters for research comparisons: document the DeepTab version and verify the forward path if you report MambaTab as a state-space model.

## Practical Config

```python
from deeptab.configs import MambaTabConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MambaTabRegressor

model = MambaTabRegressor(
    model_config=MambaTabConfig(
        d_model=64,
        dropout=0.05,
        head_layer_sizes=[128],
        head_dropout=0.1,
    ),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="standard"),
    trainer_config=TrainerConfig(lr=1e-3, batch_size=256, max_epochs=100),
    random_state=101,
)
```

Key settings in the current forward path:

| Setting | Typical range | Effect |
| --- | --- | --- |
| `d_model` | `32` to `128` | Width of the projected representation. |
| `embedding_activation` | `Identity`, `ReLU`, `SiLU` | Activation after projection/norm. |
| `head_layer_sizes` | `[]` to `[256, 128]` | Extra MLPhead capacity. |
| `head_dropout` | `0.0` to `0.3` | Head regularization. |
| `axis` | `1` or `0` | Temporary unsqueeze axis before normalization. |

## When To Use

Use MambaTab when you want a lightweight projection baseline from the Mamba-family API. Use Mambular for sequence modeling experiments where the Mamba block must be active.

## References

- Gu and Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).
- Thielmann et al., [Mambular: A Sequential Model for Tabular Deep Learning](https://arxiv.org/abs/2408.06291).
