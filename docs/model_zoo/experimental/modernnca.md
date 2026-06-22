# ModernNCA

**ModernNCA** is a differentiable nearest-neighbor model for tabular data. It learns a neural representation of each row, compares query rows to candidate rows in that representation space, and predicts by a softmax-weighted average of candidate labels.

```{warning}
**Experimental model:** ModernNCA is not covered by stable-model semantic versioning. Pin the exact DeepTab version for reproducible experiments.
```

## Overview

ModernNCA revisits Neighborhood Component Analysis (NCA) with modern tabular deep-learning components. In DeepTab, it is implemented as a candidate-based model:

1. Encode each row into a learned representation.
2. Compute Euclidean distances from batch rows to candidate rows.
3. Convert negative distances into weights with a temperature-scaled softmax.
4. Predict by weighting candidate labels.

This makes ModernNCA useful when the target function is locally smooth in a representation space: rows with similar learned embeddings should have similar labels.

| Property                  | DeepTab ModernNCA                                   |
| ------------------------- | --------------------------------------------------- |
| Inductive bias            | Local similarity / soft nearest-neighbor prediction |
| Prediction form           | Weighted candidate labels                           |
| Training mode             | Candidate-aware via `train_with_candidates`         |
| Inference cost            | Pairwise distance to candidate rows                 |
| Best baseline comparisons | TabR, TabM, ResNet, MLP                             |

## Architectural Details

For a query row \(x*i\) and candidate rows \(\{x_j, y_j\}\), ModernNCA learns an encoder \(\phi*\theta\):

```text
raw features
    |
optional DeepTab feature embeddings
    |
linear encoder: input_dim -> dim
    |
residual post-encoder blocks
    |
embedding z = phi(x)
```

Distances are converted to candidate weights:

\[
d*{ij} = \frac{\|\phi*\theta(x*i) - \phi*\theta(x_j)\|\_2}{T}
\]

\[
w*{ij} = \mathrm{softmax}\_j(-d*{ij})
\]

For regression, the output is the weighted average of candidate targets. For classification, candidate labels are one-hot encoded and the weighted class probabilities are log-transformed before loss computation.

During training, DeepTab concatenates the current batch with a sampled subset of training candidates. The diagonal self-match for the current batch is masked to avoid a row predicting from its own label.

## Main Building Blocks

The implementation lives in `deeptab/architectures/experimental/modern_nca.py`.

| Component                  | Implementation                                                   | Role                                                  |
| -------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------- |
| Optional feature embedding | `EmbeddingLayer` when `use_embeddings=True`                      | Converts raw columns into per-feature representations |
| Encoder                    | `nn.Linear(input_dim, config.dim)`                               | Projects the flattened row into metric space          |
| Post-encoder               | Repeated BatchNorm -> Linear -> ReLU -> Dropout -> Linear blocks | Adds nonlinear representation capacity                |
| Candidate weighting        | `torch.cdist` + `softmax(-distance / temperature)`               | Differentiable neighbor weighting                     |
| Candidate prediction       | Matrix multiply between weights and candidate labels             | Produces regression values or class probabilities     |
| Fallback head              | `MLPhead` in `forward`                                           | Allows non-candidate forward compatibility            |

## Configuration

| Parameter                | Default | Practical Effect                                   |
| ------------------------ | ------- | -------------------------------------------------- |
| `dim`                    | `128`   | Metric-space dimension after the encoder           |
| `d_block`                | `512`   | Hidden width inside residual post-encoder blocks   |
| `n_blocks`               | `4`     | Number of post-encoder blocks                      |
| `dropout`                | `0.1`   | Regularization inside post-encoder blocks          |
| `temperature`            | `0.75`  | Softmax sharpness for candidate weighting          |
| `sample_rate`            | `0.5`   | Fraction of candidate rows sampled during training |
| `embedding_type`         | `"plr"` | Default embedding type when embeddings are enabled |
| `n_frequencies`          | `75`    | PLR frequency count                                |
| `frequencies_init_scale` | `0.045` | PLR initialization scale                           |

```python
from deeptab.configs import ModernNCAConfig, PreprocessingConfig, TrainerConfig
from deeptab.models.experimental import ModernNCAClassifier

model = ModernNCAClassifier(
    model_config=ModernNCAConfig(
        dim=128,
        d_block=512,
        n_blocks=4,
        dropout=0.1,
        temperature=0.75,
        sample_rate=0.5,
    ),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(lr=3e-4, batch_size=128, max_epochs=100),
    random_state=101,
)
```

## Practical Guide

| Dataset Condition           | Recommendation                                                                    |
| --------------------------- | --------------------------------------------------------------------------------- |
| Small to medium data        | ModernNCA is worth testing; candidate distance cost is manageable                 |
| Very large candidate pool   | Reduce `sample_rate`, use smaller batches, or prefer TabR/parametric models       |
| Noisy labels                | Increase `temperature` or regularization; very sharp neighbor weights can overfit |
| Strong local clusters       | ModernNCA may be competitive with retrieval models                                |
| Latency-sensitive inference | Prefer MLP/ResNet/TabM unless candidate search is acceptable                      |

Suggested search space:

```python
param_grid = {
    "preprocessing_config__numerical_preprocessing": ["standardization", "quantile", "ple"],
    "model_config__dim": [64, 128, 256],
    "model_config__n_blocks": [2, 4, 6],
    "model_config__d_block": [256, 512],
    "model_config__dropout": [0.0, 0.1, 0.2],
    "model_config__temperature": [0.5, 0.75, 1.0],
    "model_config__sample_rate": [0.25, 0.5, 1.0],
    "trainer_config__lr": [1e-4, 3e-4, 5e-4],
}
```

## Nuances and Limitations

- Candidate construction matters. Validation and test rows should retrieve from training candidates, not from labels that would leak evaluation information.
- `sample_rate` changes the stochastic training objective. Report it in benchmarks.
- `temperature` controls the effective number of neighbors. Lower values make predictions closer to nearest-neighbor behavior.
- Pairwise distance computation is the dominant cost: roughly \(O(B \cdot N_c \cdot dim)\) for batch size \(B\) and candidate count \(N_c\).
- Compared with TabR, ModernNCA uses a simpler soft NCA-style label aggregation rather than TabR's learned context/value transformation.

## When to Use

Use ModernNCA when your hypothesis is that local neighborhoods in a learned representation space carry strong signal. Prefer TabM, ResNet, Mambular, or FTTransformer when you want a purely parametric model with simpler inference.

## References

- Goldberger, J., Roweis, S., Hinton, G., & Salakhutdinov, R. (2004). _Neighbourhood Components Analysis_. NeurIPS 2004.
- Ye, H.-J., Yin, H.-H., Zhan, D.-C., & Chao, W.-L. (2025). _Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later_. ICLR 2025. [OpenReview](https://openreview.net/forum?id=JytL2MrlLT)
- Weinberger, K. Q., & Saul, L. K. (2009). _Distance Metric Learning for Large Margin Nearest Neighbor Classification_. JMLR.

## See Also

- [TabR](../stable/tabr) - stable retrieval-augmented tabular model
- [Recommended Configs](../recommended_configs) - general tuning strategy
- [Model Tiers](../../core_concepts/model_tiers) - experimental vs stable models
