# ENODE

**Available as:** `ENODEClassifier`, `ENODERegressor`, `ENODELSS` — import from `deeptab.models`.

## Overview

ENODE is DeepTab's enhanced NODE variant. It keeps differentiable oblivious tree layers but operates on embedded feature tokens and aggregates the learned tree representation before a compact prediction head.

Use ENODE when you want NODE-style inductive bias with feature embeddings rather than a purely flattened raw input vector.

## Architectural Details

DeepTab's `ENODE` pipeline is:

1. `EmbeddingLayer` creates feature tokens.
2. `ENODEDenseBlock` processes the token sequence with differentiable tree layers.
3. The block output is squeezed and averaged across the feature axis.
4. A two-layer MLP head maps the embedding representation to the task output.

```text
feature tokens -> ENODEDenseBlock -> mean over feature axis -> Linear/ReLU/Dropout/Linear
```

## Main Building Blocks

| Component   | DeepTab implementation                      | Role                                                  |
| ----------- | ------------------------------------------- | ----------------------------------------------------- |
| Tokenizer   | `EmbeddingLayer`                            | Builds embedded feature tokens.                       |
| Tree block  | `ENODEDenseBlock`                           | Applies enhanced differentiable tree transformations. |
| Aggregation | `x.mean(axis=1)`                            | Produces one row representation.                      |
| Head        | `nn.Linear -> ReLU -> Dropout -> nn.Linear` | Task output.                                          |

## Implementation Notes

The model always constructs an `EmbeddingLayer`. Unlike `NODE`, it does not branch to a raw concatenated input path. The architecture computes `input_dim` as the number of feature tokens and uses `d_model` as the embedding dimension inside the tree block.

## Practical Config

```python
from deeptab.configs import ENODEConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import ENODERegressor

model = ENODERegressor(
    model_config=ENODEConfig(
        d_model=8,
        num_layers=4,
        layer_dim=64,
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

| Setting        | Typical range  | Effect                          |
| -------------- | -------------- | ------------------------------- |
| `d_model`      | `4` to `32`    | Embedded feature width.         |
| `num_layers`   | `2` to `6`     | Number of tree layers.          |
| `layer_dim`    | `32` to `128`  | Tree-layer width.               |
| `depth`        | `4` to `8`     | Soft decision depth.            |
| `head_dropout` | `0.0` to `0.5` | Prediction-head regularization. |

## When To Use

Use ENODE when you want to compare raw-vector NODE against an embedding-based neural tree variant. It is especially relevant when categorical embeddings or learned numerical embeddings may improve tree-style partitions.

## References

- Popov et al., [Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312).
