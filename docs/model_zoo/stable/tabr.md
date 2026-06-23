# TabR

**Available as:** `TabRClassifier`, `TabRRegressor`, `TabRLSS` — import from `deeptab.models`.

## Overview

TabR is a retrieval-augmented tabular model. It encodes the current row and candidate training rows into a latent space, retrieves nearest candidate contexts with FAISS, mixes candidate labels into the representation, and predicts with a neural head.

Use TabR when local neighborhood structure is likely to matter and you can afford train-set candidate retrieval during training, validation, and prediction.

## Architectural Details

DeepTab's `TabR` implementation has three conceptual modules:

1. **Encoder (`E`)**: project input features to `d_main` and optionally apply residual MLP encoder blocks.
2. **Retrieval (`R`)**: compute keys with `K`, search nearest candidate keys using FAISS, encode candidate labels, and compute attention-like weights over contexts.
3. **Predictor (`P`)**: combine retrieved context with the query representation and apply residual predictor blocks plus a normalized output head.

```text
query features -> encoder -> key
candidate features -> encoder -> candidate keys -> FAISS nearest neighbors
candidate labels + key differences -> retrieved context -> predictor -> output
```

## Main Building Blocks

| Component          | DeepTab implementation                  | Role                                                         |
| ------------------ | --------------------------------------- | ------------------------------------------------------------ |
| Optional tokenizer | `EmbeddingLayer`                        | Embeds features before retrieval when `use_embeddings=True`. |
| Encoder            | `linear`, `blocks0`, `K`                | Builds row representation and retrieval key.                 |
| Candidate search   | `faiss.IndexFlatL2` or `GpuIndexFlatL2` | Retrieves nearest candidate keys.                            |
| Label encoder      | `nn.Linear` or `nn.Embedding`           | Converts candidate labels to vectors.                        |
| Context transform  | `T(k - context_k)`                      | Adjusts retrieved values by query-context difference.        |
| Predictor          | `blocks1`, `head`                       | Produces task output.                                        |

## Implementation Notes

TabR sets `uses_candidates=True`, so it has specialized candidate-aware training, validation, and prediction methods. The standard `forward` method exists for baseline compatibility, but proper TabR behavior depends on candidate data.

The implementation lazily imports `delu` and `faiss`. Install the appropriate FAISS package for your hardware before using TabR in experiments.

## Practical Config

```python
from deeptab.configs import PreprocessingConfig, TabRConfig, TrainerConfig
from deeptab.models import TabRRegressor

model = TabRRegressor(
    model_config=TabRConfig(
        d_main=256,
        context_size=96,
        predictor_n_blocks=1,
        encoder_n_blocks=0,
        context_dropout=0.2,
        memory_efficient=False,
    ),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(lr=3e-4, batch_size=128, max_epochs=100),
    random_state=101,
)
```

Key settings:

| Setting                         | Typical range       | Effect                                        |
| ------------------------------- | ------------------- | --------------------------------------------- |
| `d_main`                        | `128` to `512`      | Retrieval and predictor representation width. |
| `context_size`                  | `32` to `256`       | Number of neighbors used per query.           |
| `encoder_n_blocks`              | `0` to `2`          | Query/candidate encoder depth.                |
| `predictor_n_blocks`            | `1` to `3`          | Post-retrieval predictor depth.               |
| `candidate_encoding_batch_size` | `0` or positive int | Chunked candidate encoding.                   |
| `memory_efficient`              | `False` or `True`   | Reduces memory at extra compute cost.         |

## When To Use

Use TabR when nearest-neighbor information is a serious baseline, especially on datasets with local smoothness, repeated profiles, or label neighborhoods. Account for retrieval cost and candidate-set leakage rules in experimental protocols.

## References

- Gorishniy et al., [TabR: Tabular Deep Learning Meets Nearest Neighbors](https://arxiv.org/abs/2307.14338).
- Cover and Hart, [Nearest Neighbor Pattern Classification](https://doi.org/10.1109/TIT.1967.1053964).
