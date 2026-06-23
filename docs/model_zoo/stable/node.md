# NODE

**Available as:** `NODEClassifier`, `NODERegressor`, `NODELSS` — import from `deeptab.models`.

## Overview

NODE implements Neural Oblivious Decision Ensembles: differentiable oblivious decision trees trained inside a neural network. It is a useful bridge between tree-based inductive bias and gradient-based deep learning.

Use NODE when you want soft tree-like feature partitioning while keeping the sklearn-style DeepTab training interface.

## Architectural Details

DeepTab's `NODE` pipeline is:

1. Use raw/preprocessed concatenated features, or optionally embed features and flatten them.
2. Pass the vector through a `DenseBlock` of differentiable oblivious trees.
3. Flatten the dense block output.
4. Predict with `MLPhead`.

```text
features -> optional embeddings -> DenseBlock(num_layers, layer_dim, depth, tree_dim) -> MLPhead
```

## Main Building Blocks

| Component            | DeepTab implementation                | Role                                      |
| -------------------- | ------------------------------------- | ----------------------------------------- |
| Input representation | raw concatenation or `EmbeddingLayer` | Builds the vector consumed by trees.      |
| Differentiable trees | `deeptab.nn.blocks.node.DenseBlock`   | Stacks NODE-style tree layers.            |
| Tree depth           | `depth`                               | Controls number of soft splits per tree.  |
| Layer width          | `layer_dim`                           | Number of trees/features per dense layer. |
| Head                 | `MLPhead`                             | Maps tree representation to task output.  |

## Implementation Notes

`num_layers * layer_dim` determines the input dimension to the prediction head. Larger values increase capacity and memory use. `tree_dim` controls the output dimension per tree.

## Practical Config

```python
from deeptab.configs import NODEConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import NODEClassifier

model = NODEClassifier(
    model_config=NODEConfig(
        num_layers=4,
        layer_dim=128,
        depth=6,
        tree_dim=1,
        head_dropout=0.3,
    ),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(lr=1e-3, batch_size=256, max_epochs=100),
    random_state=101,
)
```

Key settings:

| Setting            | Typical range   | Effect                          |
| ------------------ | --------------- | ------------------------------- |
| `num_layers`       | `2` to `6`      | Number of dense tree layers.    |
| `layer_dim`        | `64` to `256`   | Width of each tree layer.       |
| `depth`            | `4` to `8`      | Soft decision depth.            |
| `tree_dim`         | `1` to `3`      | Output dimension per tree.      |
| `head_layer_sizes` | `[]` to `[128]` | Extra prediction-head capacity. |

## When To Use

Use NODE when you want a differentiable tree ensemble baseline. Compare it with gradient-boosted trees and neural MLP/ResNet baselines because tree-like inductive bias can dominate or underperform depending on preprocessing and dataset size.

## References

- Popov et al., [Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312).
