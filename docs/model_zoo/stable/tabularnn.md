# TabulaRNN

**Available as:** `TabulaRNNClassifier`, `TabulaRNNRegressor`, `TabulaRNNLSS` — import from `deeptab.models`.

## Overview

TabulaRNN treats tabular columns as a sequence and processes feature tokens with recurrent layers plus depthwise convolution. It is useful when you want a sequence-model baseline that is simpler than Mamba and different from self-attention.

Use it for experiments on ordered feature sequences, sequentially engineered tabular features, or ablations against Mambular.

## Architectural Details

DeepTab's `TabulaRNN` pipeline is:

1. `EmbeddingLayer` converts features to `(batch, n_features, d_model)` tokens.
2. `ConvRNN` applies depthwise convolution and an RNN-family layer across the sequence.
3. A residual summary `z` is computed by averaging input embeddings and projecting with `linear`.
4. The recurrent output is pooled and added to `z`.
5. Optional normalization and `MLPhead` produce predictions.

```text
feature tokens -> ConvRNN -> pooling
feature tokens -> mean -> Linear
pooled recurrent state + projected mean -> optional norm -> MLPhead
```

## Main Building Blocks

| Component        | DeepTab implementation                    | Role                                        |
| ---------------- | ----------------------------------------- | ------------------------------------------- |
| Tokenizer        | `EmbeddingLayer`                          | Builds sequence tokens.                     |
| Local filter     | depthwise `nn.Conv1d` inside `ConvRNN`    | Adds local token mixing.                    |
| Recurrent block  | `RNN`, `LSTM`, `GRU`, `mLSTM`, or `sLSTM` | Sequential feature processing.              |
| Residual summary | `mean(x)` plus `linear`                   | Preserves direct feature-token information. |
| Head             | `MLPhead`                                 | Final prediction.                           |

## Implementation Notes

The config field `model_type` selects the recurrent cell family. Valid values follow the `ConvRNN` mapping: `"RNN"`, `"LSTM"`, `"GRU"`, `"mLSTM"`, and `"sLSTM"` if the corresponding blocks are available.

The default config uses `d_model=128`, `model_type="RNN"`, `n_layers=4`, `rnn_dropout=0.2`, `dim_feedforward=256`, and `pooling_method="avg"`.

## Practical Config

```python
from deeptab.configs import PreprocessingConfig, TabulaRNNConfig, TrainerConfig
from deeptab.models import TabulaRNNClassifier

model = TabulaRNNClassifier(
    model_config=TabulaRNNConfig(
        d_model=128,
        model_type="GRU",
        n_layers=3,
        rnn_dropout=0.2,
        dim_feedforward=256,
        pooling_method="avg",
    ),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(lr=3e-4, batch_size=128, max_epochs=100),
    random_state=101,
)
```

Key settings:

| Setting           | Typical range              | Effect                            |
| ----------------- | -------------------------- | --------------------------------- |
| `model_type`      | `"RNN"`, `"GRU"`, `"LSTM"` | Recurrent cell family.            |
| `d_model`         | `64` to `192`              | Feature-token width.              |
| `n_layers`        | `1` to `4`                 | Recurrent depth.                  |
| `dim_feedforward` | `128` to `512`             | Hidden size consumed by the head. |
| `d_conv`          | `2` to `8`                 | Depthwise convolution width.      |

## When To Use

Use TabulaRNN when you want a recurrent sequence baseline over feature tokens. Because column order is not always meaningful, compare with shuffled or alternative feature orderings when making architectural claims.

## References

- Hochreiter and Schmidhuber, [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf).
- Cho et al., [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078).
