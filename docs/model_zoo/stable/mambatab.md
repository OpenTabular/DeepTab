# MambaTab

**Available as:** `MambaTabClassifier`, `MambaTabRegressor`, and `MambaTabLSS` in `deeptab.models`.

## Overview

MambaTab is a compact Mamba-family model. It concatenates the input features into one vector, projects it to `d_model`, normalizes and activates it, runs the Mamba block over the resulting length-one sequence (or over the batch, see `axis` below), and predicts with `MLPhead`.

Because the whole row becomes a single token, MambaTab doesn't model interactions between feature tokens. For a Mamba sequence model over per-feature tokens, prefer [Mambular](mambular) or [MambAttention](mambattention).

## Architectural Details

The `MambaTab` forward path is:

1. Concatenate all input tensors.
2. Apply `initial_layer` from input dimension to `d_model`.
3. Unsqueeze along `axis`, apply `LayerNorm`, and apply `embedding_activation`.
4. Apply the Mamba block (`self.mamba`).
5. Squeeze back to a row representation.
6. Predict with `MLPhead`.

```text
features -> concat -> Linear(input_dim, d_model) -> LayerNorm -> activation -> Mamba -> MLPhead
```

With the default `axis=1`, each row is a sequence of length one. With `axis=0`, the batch dimension becomes the sequence axis, so the rows of a batch are scanned as one sequence.

## Main Building Blocks

| Component     | DeepTab implementation                            | Role                                                     |
| ------------- | ------------------------------------------------- | -------------------------------------------------------- |
| Input path    | `torch.cat(...)`                                  | Uses raw/preprocessed feature tensors directly.          |
| Projection    | `initial_layer`                                   | Maps input vector to `d_model`.                          |
| Normalization | `LayerNorm`                                       | Stabilizes projected representation.                     |
| Head          | `MLPhead`                                         | Produces predictions.                                    |
| Mamba block   | `self.mamba = Mamba(...)` or `MambaOriginal(...)` | Applied to the projected representation before the head. |

## Implementation Notes

The Mamba-related config fields (`d_state`, `d_conv`, `expand_factor`, `mamba_version`, `bidirectional`) configure `self.mamba`, which `forward` applies before the head.

`mamba_version="mamba-torch"` (the default) selects DeepTab's local `Mamba` block; any other value selects `MambaOriginal`.

DeepTab 2.0.x skipped the Mamba block in `forward`, so MambaTab results from those versions describe a projected-feature network rather than a state-space model. Report the DeepTab version with research comparisons.

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
    preprocessing_config=PreprocessingConfig(numerical_method="standardization"),
    trainer_config=TrainerConfig(lr=1e-3, batch_size=256, max_epochs=100),
    random_state=101,
)
```

Key settings in the current forward path:

| Setting                | Typical range              | Effect                                         |
| ---------------------- | -------------------------- | ---------------------------------------------- |
| `d_model`              | `32` to `128`              | Width of the projected representation.         |
| `embedding_activation` | `Identity`, `ReLU`, `SiLU` | Activation after projection/norm.              |
| `head_layer_sizes`     | `[]` to `[256, 128]`       | Extra MLPhead capacity.                        |
| `head_dropout`         | `0.0` to `0.3`             | Head regularization.                           |
| `axis`                 | `1` or `0`                 | Temporary unsqueeze axis before normalization. |

## When To Use

Use MambaTab when you want a lightweight projection baseline from the Mamba-family API. Use Mambular for sequence modeling experiments where the Mamba block must be active.

## References

- Gu and Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).
- Thielmann et al., [Mambular: A Sequential Model for Tabular Deep Learning](https://arxiv.org/abs/2408.06291).
