# NDTF

**Available as:** `NDTFClassifier`, `NDTFRegressor`, `NDTFLSS` — import from `deeptab.models`.

## Overview

NDTF is DeepTab's neural decision tree forest. It builds an ensemble of differentiable decision trees, applies a convolutional feature interaction layer before the trees, and combines tree predictions with learnable ensemble weights.

Use NDTF when you want a neural forest baseline with explicit ensemble structure and penalty-based regularization.

## Architectural Details

DeepTab's `NDTF` pipeline is:

1. Concatenate all input tensors.
2. Apply a 1D convolution over the feature vector to create transformed feature interactions.
3. Feed feature subsets into an ensemble of `NeuralDecisionTree` modules.
4. Stack tree predictions.
5. Combine predictions with learned `tree_weights`.

```text
features -> Conv1d feature interaction -> NeuralDecisionTree x n_ensembles -> weighted ensemble output
```

## Main Building Blocks

| Component            | DeepTab implementation                         | Role                                           |
| -------------------- | ---------------------------------------------- | ---------------------------------------------- |
| Feature interaction  | `nn.Conv1d`                                    | Produces transformed feature inputs for trees. |
| Tree ensemble        | `nn.ModuleList[NeuralDecisionTree]`            | Differentiable forest members.                 |
| Random tree settings | sampled input dimensions, depths, temperatures | Adds diversity across trees.                   |
| Ensemble weights     | learnable `tree_weights`                       | Combines member predictions.                   |
| Penalty path         | `penalty_forward`                              | Returns prediction and scaled tree penalty.    |

## Implementation Notes

The first tree receives the full input dimension. Remaining trees receive randomly sampled prefix dimensions. Tree depths are sampled with `np.random.randint(min_depth, max_depth)`, so the upper bound is exclusive: with the defaults `min_depth=4` and `max_depth=16`, sampled depths range from 4 to 15 (inclusive). Temperatures are jittered around the configured `temperature`.

`penalty_forward` returns `(prediction, penalty_factor * penalty)`, which can be used by the training module when penalty-aware training is enabled.

## Practical Config

```python
from deeptab.configs import NDTFConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import NDTFClassifier

model = NDTFClassifier(
    model_config=NDTFConfig(
        n_ensembles=12,
        min_depth=4,
        max_depth=12,
        temperature=0.1,
        node_sampling=0.3,
        lamda=0.3,
    ),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="standardization"),
    trainer_config=TrainerConfig(lr=1e-3, batch_size=256, max_epochs=100),
    random_state=101,
)
```

Key settings:

| Setting                  | Typical range     | Effect                              |
| ------------------------ | ----------------- | ----------------------------------- |
| `n_ensembles`            | `4` to `24`       | Number of neural trees.             |
| `min_depth`, `max_depth` | `3` to `16`       | Tree depth distribution.            |
| `temperature`            | `0.05` to `0.5`   | Soft routing sharpness.             |
| `node_sampling`          | `0.1` to `0.8`    | Node-level sampling regularization. |
| `penalty_factor`         | `1e-10` to `1e-6` | Strength of tree penalty term.      |

## When To Use

Use NDTF when you need a neural forest-style model with explicit ensemble aggregation. It can be sensitive to random tree construction, so set `random_state` and evaluate multiple seeds for research reporting.

## References

- Kontschieder et al., [Deep Neural Decision Forests](https://arxiv.org/abs/1505.03424).
- Popov et al., [Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312).
