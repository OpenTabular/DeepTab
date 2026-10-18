# MLP

**Available as:** `MLPClassifier`, `MLPRegressor`, and `MLPLSS` in `deeptab.models`.

## Overview

MLP is DeepTab's plain feed-forward baseline for tabular data. It is the first model to include in most studies because it is fast, easy to tune, and makes very few assumptions beyond the quality of preprocessing and feature encoding.

Use it as a control model before moving to attention, retrieval, Mamba, or neural tree architectures. A well-tuned MLP is often competitive on medium-size tabular datasets, especially with good numerical scaling and categorical handling.

## Architectural Details

DeepTab's `MLP` implementation follows a simple pipeline:

1. Optionally embed numerical, categorical, and external embedding features with `EmbeddingLayer`.
2. Flatten embedded tokens to a single vector, or concatenate raw/preprocessed input tensors when `use_embeddings=False`.
3. Apply a sequence of linear layers from `layer_sizes`.
4. Optionally apply batch normalization, layer normalization, activation, GLU, dropout, and residual additions when dimensions match.
5. Project the final hidden representation to the task output dimension.

The forward path is:

```text
features -> optional EmbeddingLayer -> flatten/concat -> Linear blocks -> output layer
```

## Main Building Blocks

| Component        | DeepTab implementation                | Role                                              |
| ---------------- | ------------------------------------- | ------------------------------------------------- |
| Feature input    | `torch.cat(...)` or `EmbeddingLayer`  | Builds the vector consumed by the MLP.            |
| Hidden stack     | `nn.Linear` layers from `layer_sizes` | Learns nonlinear feature interactions.            |
| Normalization    | `batch_norm`, `layer_norm`            | Stabilizes training when enabled.                 |
| Activation       | `activation` or `nn.GLU()`            | Controls nonlinear transformation.                |
| Skip connections | `skip_connections`                    | Adds residual connections only when shapes match. |
| Output head      | Final `nn.Linear`                     | Produces logits or regression outputs.            |

## Implementation Notes

The default `MLPConfig` uses `layer_sizes=[256, 128, 32]` and `dropout=0.2`. The model does not require embeddings, so it works well with standard numerical preprocessing and integer/one-hot categorical preprocessing.

`use_glu=True` doubles each hidden projection before `nn.GLU` halves it again, so every `layer_sizes` entry remains the width passed to the next layer. Batch and layer normalization act on the doubled pre-GLU width. Skip connections compare the previous width with that doubled projection, so equal `layer_sizes` receive no residual addition when GLU is enabled.

> **Configuration note:** `MLPConfig(layer_sizes=[8, 4], use_glu=True)` builds linear layers with output widths 16 and 8, and GLU reduces them to 8 and 4. Previously, the projections used the configured widths directly, GLU halved them to 4 and 2, and the next layer failed with a shape mismatch.

## Practical Config

```python
from deeptab.configs import MLPConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MLPClassifier

model = MLPClassifier(
    model_config=MLPConfig(
        layer_sizes=[256, 128, 32],
        dropout=0.2,
        skip_connections=False,
    ),
    preprocessing_config=PreprocessingConfig(numerical_method="standardization"),
    trainer_config=TrainerConfig(lr=1e-3, batch_size=256, max_epochs=100),
    random_state=101,
)
```

Key settings:

| Setting                    | Typical range                    | Effect                                              |
| -------------------------- | -------------------------------- | --------------------------------------------------- |
| `layer_sizes`              | `[128, 64]` to `[512, 256, 128]` | Main capacity control.                              |
| `dropout`                  | `0.0` to `0.5`                   | Regularization; increase on small/noisy data.       |
| `use_embeddings`           | `False` or `True`                | Enables feature token embeddings before flattening. |
| `d_model`                  | `16` to `128`                    | Embedding width when embeddings are used.           |
| `batch_norm`, `layer_norm` | `False` or `True`                | Try when optimization is unstable.                  |

## When To Use

Use MLP when you need a fast sanity check, a strong non-attention baseline, or a low-latency model. It is also a useful ablation target for evaluating whether a more complex architecture is actually adding value.

Avoid treating it as a weak baseline. Many tabular benchmarks show that tuned MLP/ResNet-style models can be difficult to beat without careful preprocessing and hyperparameter search.

## References

- Gorishniy et al., [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959).
- Shwartz-Ziv and Armon, [Tabular Data: Deep Learning is Not All You Need](https://arxiv.org/abs/2106.03253).
