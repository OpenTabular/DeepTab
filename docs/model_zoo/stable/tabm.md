# TabM

**Available as:** `TabMClassifier`, `TabMRegressor`, `TabMLSS` — import from `deeptab.models`.

## Overview

TabM is a parameter-efficient ensemble model for tabular data. Instead of training many independent networks, it uses BatchEnsemble-style linear layers with shared weights and member-specific scaling factors.

Use TabM when you want strong tabular performance, ensemble-like robustness, and better computational efficiency than training many separate MLPs.

## Architectural Details

DeepTab's `TabM` pipeline is:

1. Use raw concatenated features or `EmbeddingLayer`.
2. If embeddings are used, average feature embeddings or flatten all tokens depending on `average_embeddings`.
3. Apply `LinearBatchEnsembleLayer` blocks over `ensemble_size` members.
4. Apply optional normalization, activation, and dropout.
5. Use an ensemble-aware final layer unless `average_ensembles=True`.

```text
features -> optional embeddings -> BatchEnsemble MLP blocks -> ensemble output/head
```

## Main Building Blocks

| Component | DeepTab implementation | Role |
| --- | --- | --- |
| Feature path | `EmbeddingLayer` or raw concatenation | Builds model input. |
| Ensemble layers | `LinearBatchEnsembleLayer` | Shared weight matrix with member-specific scaling. |
| Final layer | `SNLinear` or `nn.Linear` | Produces per-member or averaged predictions. |
| Ensemble output | `returns_ensemble=True` when not averaged | Lets the training wrapper handle ensemble predictions. |

## Implementation Notes

`model_type="mini"` applies full BatchEnsemble scaling in the input layer and lighter shared transformations in hidden layers. `model_type="full"` uses scaling in hidden layers too.

When `average_ensembles=False`, `TabM` returns one prediction per ensemble member and sets `returns_ensemble=True`. When `average_ensembles=True`, the model averages member states before the final head.

## Practical Config

```python
from deeptab.configs import PreprocessingConfig, TabMConfig, TrainerConfig
from deeptab.models import TabMClassifier

model = TabMClassifier(
    model_config=TabMConfig(
        layer_sizes=[256, 256, 128],
        ensemble_size=32,
        model_type="mini",
        dropout=0.2,
        average_ensembles=False,
    ),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(lr=1e-3, batch_size=256, max_epochs=100),
    random_state=101,
)
```

Key settings:

| Setting | Typical range | Effect |
| --- | --- | --- |
| `ensemble_size` | `8` to `64` | Number of virtual ensemble members. |
| `layer_sizes` | `[128, 128]` to `[512, 256, 128]` | Shared MLP capacity. |
| `model_type` | `"mini"` or `"full"` | Amount of member-specific scaling. |
| `average_ensembles` | `False` or `True` | Return per-member outputs or average internally. |
| `scaling_init` | `"ones"`, `"random-signs"`, `"normal"` | Diversity initialization for scaling factors. |

## When To Use

Use TabM as one of the first strong baselines in a tabular benchmark. It is especially attractive when you want some ensemble benefit but cannot afford many independently trained models.

## References

- Gorishniy et al., [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210).
- Wen et al., [BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning](https://arxiv.org/abs/2002.06715).
