# ResNet

**Available as:** `ResNetClassifier`, `ResNetRegressor`, `ResNetLSS` — import from `deeptab.models`.

## Overview

ResNet is DeepTab's residual feed-forward architecture for tabular data. It keeps the simplicity and speed of an MLP while adding residual blocks that make deeper nonlinear transformations easier to optimize.

Use ResNet when an MLP underfits, when you want a stronger classical neural baseline, or when you need a model that is still much cheaper than attention or retrieval-based methods.

## Architectural Details

DeepTab's `ResNet` pipeline is:

1. Concatenate preprocessed features, or embed features with `EmbeddingLayer` and flatten tokens.
2. Project the input vector with `initial_layer`.
3. Apply `num_blocks` residual blocks.
4. Use a final linear output layer for the target task.

The residual blocks are implemented with `deeptab.nn.blocks.resnet.ResidualBlock` and use the configured activation, dropout, and optional normalization.

```text
features -> optional embeddings -> initial Linear -> ResidualBlock x num_blocks -> output
```

## Main Building Blocks

| Component            | DeepTab implementation                    | Role                                        |
| -------------------- | ----------------------------------------- | ------------------------------------------- |
| Input representation | Raw concatenation or `EmbeddingLayer`     | Converts heterogeneous columns to a tensor. |
| Initial projection   | `nn.Linear(input_dim, layer_sizes[0])`    | Sets hidden width.                          |
| Residual body        | `ResidualBlock`                           | Learns transformations with skip paths.     |
| Output layer         | `nn.Linear(layer_sizes[-1], num_classes)` | Produces task outputs.                      |

## Implementation Notes

`num_blocks` controls how many residual blocks are instantiated. Each block uses `layer_sizes[i]` as input width and `layer_sizes[i + 1]` when available, otherwise the last width is reused. Keep `num_blocks` aligned with the length of `layer_sizes`; if `num_blocks` exceeds the number of transitions, later blocks stay at the final width.

## Practical Config

```python
from deeptab.configs import PreprocessingConfig, ResNetConfig, TrainerConfig
from deeptab.models import ResNetRegressor

model = ResNetRegressor(
    model_config=ResNetConfig(
        layer_sizes=[256, 128, 64],
        num_blocks=3,
        dropout=0.2,
        norm=True,
    ),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="standardization"),
    trainer_config=TrainerConfig(lr=1e-3, batch_size=256, max_epochs=100),
    random_state=101,
)
```

Key settings:

| Setting          | Typical range                    | Effect                                        |
| ---------------- | -------------------------------- | --------------------------------------------- |
| `layer_sizes`    | `[128, 64]` to `[512, 256, 128]` | Width schedule.                               |
| `num_blocks`     | `2` to `5`                       | Depth of residual processing.                 |
| `dropout`        | `0.0` to `0.5`                   | Regularization.                               |
| `norm`           | `False` or `True`                | Enables normalization inside residual blocks. |
| `use_embeddings` | `False` or `True`                | Useful for categorical-heavy data.            |

## When To Use

Use ResNet as a default stable baseline beside MLP and TabM. It is a good choice when you want a stronger inductive bias than a plain MLP but do not want the memory and tuning cost of Transformer models.

## References

- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- Gorishniy et al., [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959).
