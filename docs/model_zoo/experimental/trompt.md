# Trompt

**Trompt** is a prompt-inspired tabular architecture. It uses learnable prompt/prototype records and feature-importance maps to repeatedly aggregate column representations, producing one prediction per cycle.

```{warning}
**Experimental model:** Trompt is not covered by stable-model semantic versioning. Pin the exact DeepTab version for reproducible experiments.
```

## Overview

Trompt stands for tabular prompt. The original research motivation is to adapt ideas from prompt learning to tabular data by separating table-level feature processing from sample-specific prompt representations.

In DeepTab, Trompt is implemented as a sequence of `TromptCell` modules. Each cell:

1. embeds all input features,
2. expands each feature into `P` prompt slots,
3. computes prompt-to-column importance weights, and
4. aggregates expanded feature representations into updated prompt records.

The model returns predictions from every cycle, so DeepTab treats Trompt as an ensemble-like model (`returns_ensemble=True`).

| Property | DeepTab Trompt |
| -------- | -------------- |
| Inductive bias | Prompt/prototype-mediated feature aggregation |
| Core representation | `P` latent prompt records of width `d_model` |
| Repeated computation | `n_cycles` Trompt cells |
| Output | One decoded prediction per cycle |
| Best baseline comparisons | FTTransformer, Mambular, TabM |

## Architectural Details

The high-level data flow is:

```text
preprocessed row
    |
EmbeddingLayer -> feature embeddings
    |
Expander -> P prompt slots per feature
    |
ImportanceGetter -> prompt-to-feature weights
    |
weighted feature aggregation
    |
updated prompt records O
    |
TromptDecoder -> prediction for this cycle
```

The process is repeated for `n_cycles`. Let \(O^{(c)} \in \mathbb{R}^{P \times d}\) be the prompt records after cycle \(c\), \(C\) the number of columns/tokens, and \(d\) the model width.

The importance module learns prompt and column embeddings and computes a prompt-column attention-like matrix:

\[
M^{(c)} = \mathrm{softmax}(g(O^{(c-1)}, E_p) E_c^\top)
\]

where \(M^{(c)} \in \mathbb{R}^{P \times C}\). The cell uses this matrix to aggregate expanded feature embeddings into the next prompt records.

Unlike FTTransformer, the current DeepTab Trompt implementation does not use a standard multi-head self-attention stack with `n_heads`. Its main controls are `d_model`, `n_cycles`, `n_cells`, and `P`.

## Main Building Blocks

The implementation lives in `deeptab/architectures/experimental/trompt.py` and `deeptab/nn/blocks/trompt.py`.

| Component | Implementation | Role |
| --------- | -------------- | ---- |
| Feature encoder | `EmbeddingLayer` | Produces per-column embeddings |
| Initial prompt records | `init_rec` parameter with shape `(P, d_model)` | Starting latent prompt state |
| Cell stack | `nn.ModuleList(TromptCell(...))` repeated `n_cycles` times | Iterative prompt-feature aggregation |
| Expander | `Expander(P)` | Expands feature embeddings into prompt slots |
| Feature importance | `ImportanceGetter(P, C, d_model)` | Computes prompt-to-column weights |
| Decoder | `TromptDecoder(d_model, num_classes)` | Converts prompt records to predictions |
| Ensemble behavior | `returns_ensemble=True` | Training loss is accumulated across cycle outputs |

```{note}
`n_cells` is present in `TromptConfig`, but the current DeepTab implementation constructs one `TromptCell` per cycle. Treat `n_cycles` and `P` as the primary practical controls.
```

## Configuration

| Parameter | Default | Practical Effect |
| --------- | ------- | ---------------- |
| `d_model` | `128` | Width of feature and prompt representations |
| `n_cycles` | `6` | Number of iterative prompt aggregation cycles |
| `n_cells` | `4` | Config field retained from the Trompt design; limited direct effect in current implementation |
| `P` | `128` | Number of prompt/prototype records |

```python
from deeptab.configs import PreprocessingConfig, TrainerConfig, TromptConfig
from deeptab.models.experimental import TromptClassifier

model = TromptClassifier(
    model_config=TromptConfig(
        d_model=128,
        n_cycles=6,
        n_cells=4,
        P=128,
    ),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(lr=3e-4, batch_size=128, max_epochs=100),
    random_state=101,
)
```

## Practical Guide

| Dataset Condition | Recommendation |
| ----------------- | -------------- |
| Mixed feature types | Trompt can be worth testing because all features pass through `EmbeddingLayer` |
| Need interpretable feature weighting | Inspect prompt-to-column weights conceptually, but internal tooling may require custom hooks |
| Large feature count | Reduce `P` or `d_model`; importance maps scale with prompt slots and columns |
| Need stable transformer baseline | Use FTTransformer |
| Need strong efficient baseline | Use TabM |

Suggested search space:

```python
param_grid = {
    "preprocessing_config__numerical_preprocessing": ["standard", "quantile", "ple"],
    "model_config__d_model": [64, 128, 256],
    "model_config__n_cycles": [2, 4, 6],
    "model_config__P": [32, 64, 128],
    "trainer_config__lr": [1e-4, 3e-4, 5e-4],
    "trainer_config__batch_size": [64, 128, 256],
}
```

## Nuances and Limitations

- Trompt returns a prediction for each cycle. DeepTab's loss handling treats those cycle predictions like an ensemble.
- Increasing `P` increases the number of prompt records and the prompt-column importance map size.
- Increasing `n_cycles` increases iterative refinement cost and adds more cycle predictions to the loss.
- The current implementation is prompt-inspired but not a standard Transformer with attention heads.
- `n_cells` is documented because it exists in `TromptConfig`, but changing it may not have the architectural effect a reader expects from the original paper.

## When to Use

Use Trompt when your research question concerns prompt-style tabular representations or iterative prompt-feature aggregation. Prefer FTTransformer if you want a stable attention baseline, and prefer TabM/ResNet if you need faster practical baselines.

## References

- Chen, K.-Y., Chiang, P.-H., Chou, H.-R., Chen, T.-W., & Chang, T.-H. (2023). _Trompt: Towards a Better Deep Neural Network for Tabular Data_. ICML 2023. [arXiv:2305.18446](https://arxiv.org/abs/2305.18446)
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021. [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)

## See Also

- [FTTransformer](../stable/fttransformer) - stable feature-token Transformer baseline
- [Mambular](../stable/mambular) - stable sequence-style tabular model
- [TabM](../stable/tabm) - strong parameter-efficient baseline
- [Model Tiers](../../core_concepts/model_tiers) - experimental vs stable models
