# Tangos

**Tangos** is an MLP-style tabular model with a gradient-attribution regularizer. It encourages hidden units to become specialized and diverse by penalizing latent-unit attributions with respect to input features.

```{warning}
**Experimental model:** Tangos is not covered by stable-model semantic versioning. Pin the exact DeepTab version for reproducible experiments.
```

## Overview

Tangos is not a custom optimizer in the current DeepTab implementation. It is a feedforward network trained with the normal DeepTab optimizer, plus an additional penalty computed from the Jacobian of hidden representations with respect to input features.

The research hypothesis is that tabular MLPs generalize better when hidden units:

- specialize on a sparse subset of input features, and
- avoid learning highly overlapping feature attributions.

| Property                  | DeepTab Tangos                                              |
| ------------------------- | ----------------------------------------------------------- |
| Base architecture         | MLP                                                         |
| Additional mechanism      | Jacobian-based specialization and orthogonalization penalty |
| Training hook             | `penalty_forward`                                           |
| Main cost driver          | `torch.func.jacrev` / Jacobian computation                  |
| Best baseline comparisons | MLP, ResNet, TabM                                           |

## Architectural Details

The forward path is a standard dense network:

```text
raw preprocessed features
    |
Linear -> activation -> dropout
    |
Linear -> activation -> dropout
    |
...
    |
Linear output head
```

During training, Tangos computes a representation Jacobian:

\[
J\_{h,x} = \frac{\partial h(x)}{\partial x}
\]

where \(h(x)\) is the representation before the final output head. The model builds latent-unit attribution vectors from this Jacobian and adds:

- a specialization term, based on the L1 norm of neuron attributions, and
- an orthogonality term, based on cosine similarity between attribution vectors of different hidden units.

The training loss is:

\[
\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda*1 \mathcal{L}*{spec} + \lambda*2 \mathcal{L}*{orth}
\]

## Main Building Blocks

The implementation lives in `deeptab/architectures/experimental/tangos.py`.

| Component                 | Implementation                                                       | Role                                                |
| ------------------------- | -------------------------------------------------------------------- | --------------------------------------------------- |
| Dense body                | `nn.ModuleList` of linear, normalization, activation, dropout layers | Learns tabular representation                       |
| Optional GLU              | `nn.GLU()` when `use_glu=True`                                       | Gated dense transformations                         |
| Optional skip connections | Shape-matched residual additions                                     | Stabilizes deeper MLPs                              |
| Representation function   | `repr_forward`                                                       | Hidden representation used for Jacobian attribution |
| Jacobian computation      | `torch.func.vmap(torch.func.jacrev(...))`                            | Computes per-sample hidden-unit attributions        |
| Specialization loss       | L1 norm of attribution tensor                                        | Encourages sparse feature usage                     |
| Orthogonality loss        | Cosine similarity between neuron attributions                        | Encourages diverse hidden units                     |
| Output head               | `nn.Linear(last_hidden, num_classes)`                                | Task prediction                                     |

## Configuration

| Parameter          | Default                   | Practical Effect                               |
| ------------------ | ------------------------- | ---------------------------------------------- |
| `layer_sizes`      | `[256, 128, 32]`          | Width/depth of the MLP body                    |
| `dropout`          | `0.2`                     | Standard dropout regularization                |
| `activation`       | `nn.ReLU()`               | Hidden activation                              |
| `use_glu`          | `False`                   | Enables gated linear units                     |
| `skip_connections` | `False`                   | Adds residual connections when shapes match    |
| `batch_norm`       | inherited default `False` | Optional batch normalization                   |
| `layer_norm`       | inherited default `False` | Optional layer normalization                   |
| `lamda1`           | `0.5`                     | Weight for specialization penalty              |
| `lamda2`           | `0.1`                     | Weight for orthogonality penalty               |
| `subsample`        | `0.5`                     | Fraction used for regularization pair sampling |

```python
from deeptab.configs import PreprocessingConfig, TangosConfig, TrainerConfig
from deeptab.models.experimental import TangosRegressor

model = TangosRegressor(
    model_config=TangosConfig(
        layer_sizes=[256, 128, 32],
        dropout=0.2,
        lamda1=0.5,
        lamda2=0.1,
        subsample=0.5,
    ),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="standardization"),
    trainer_config=TrainerConfig(lr=1e-3, batch_size=128, max_epochs=100),
    random_state=101,
)
```

## Practical Guide

| Dataset Condition                   | Recommendation                                                    |
| ----------------------------------- | ----------------------------------------------------------------- |
| Small or noisy data                 | Try Tangos against MLP/ResNet; the regularizer may help           |
| Very high feature count             | Watch Jacobian memory and runtime                                 |
| Large batch sizes                   | Reduce batch size if Jacobian computation is slow or memory-heavy |
| Need fast training                  | Prefer MLP, ResNet, or TabM                                       |
| Want attribution diversity analysis | Tangos is a useful research model                                 |

Suggested search space:

```python
param_grid = {
    "preprocessing_config__numerical_preprocessing": ["standardization", "quantile"],
    "model_config__layer_sizes": [[128, 64], [256, 128, 32], [512, 256, 128]],
    "model_config__dropout": [0.0, 0.1, 0.2, 0.3],
    "model_config__lamda1": [0.1, 0.5, 1.0],
    "model_config__lamda2": [0.01, 0.1, 0.5],
    "model_config__subsample": [0.25, 0.5],
    "trainer_config__lr": [3e-4, 1e-3],
    "trainer_config__batch_size": [64, 128, 256],
}
```

## Nuances and Limitations

- The penalty is computed only because `Tangos` implements `penalty_forward`; DeepTab's training module adds the penalty to task loss automatically.
- `lamda1` and `lamda2` are not learning rates. They are regularization weights.
- The Jacobian-based penalty can be substantially more expensive than a plain MLP forward/backward pass.
- The implementation concatenates preprocessed raw feature tensors directly; it does not currently use `EmbeddingLayer` in the active forward path.
- `subsample` controls regularization estimation cost and variance. Report it in experiments.

## When to Use

Use Tangos when the research question is about MLP regularization, feature-attribution structure, or hidden-unit specialization. Prefer MLP/ResNet/TabM when you need a fast production candidate or a strong simple baseline.

## References

- Jeffares, A., Liu, T., Crabbé, J., Imrie, F., & van der Schaar, M. (2023). _TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization_. ICLR 2023. [arXiv:2303.05506](https://arxiv.org/abs/2303.05506)

## See Also

- [MLP](../stable/mlp) - stable dense baseline
- [ResNet](../stable/resnet) - stable residual dense baseline
- [TabM](../stable/tabm) - parameter-efficient ensemble baseline
- [Model Tiers](../../core_concepts/model_tiers) - experimental vs stable models
