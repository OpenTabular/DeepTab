# Overview

DeepTab is a Python library that brings modern deep learning architectures to tabular data. It wraps PyTorch and Lightning behind a scikit-learn-compatible interface, so you can use state-of-the-art models without changing how you already work with data.

## Why DeepTab

Tabular data is the most common format in applied machine learning, yet most deep learning tooling is designed for images or text. DeepTab fills that gap by:

- Providing a consistent `fit` / `predict` / `evaluate` API across all models.
- Handling categorical encoding, numerical preprocessing, and batching automatically.
- Supporting regression, classification, and distributional regression from the same model class.
- Integrating with scikit-learn pipelines and hyperparameter search tools.

## Available models

All models support regression, classification, and distributional regression out of the box. Import them as `<ModelName>Regressor`, `<ModelName>Classifier`, or `<ModelName>LSS`.

| Model            | Architecture                                   | Reference                                                                                                                                 |
| ---------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `Mambular`       | Sequential Mamba (SSM) blocks for tabular data | [Thielmann et al. (2024)](https://arxiv.org/abs/2408.06291)                                                                               |
| `MambaTab`       | Mamba block on a joint input representation    | [Ahamed et al. (2024)](https://arxiv.org/abs/2401.08867)                                                                                  |
| `MambAttention`  | Mamba + Transformer hybrid                     | [Thielmann et al. (2025)](https://arxiv.org/pdf/2411.17207)                                                                               |
| `FTTransformer`  | Feature tokeniser + Transformer encoder        | [Gorishniy et al. (2021)](https://arxiv.org/abs/2106.11959)                                                                               |
| `TabTransformer` | Transformer with categorical embeddings        | [Huang et al. (2020)](https://arxiv.org/abs/2012.06678)                                                                                   |
| `SAINT`          | Row attention + contrastive pre-training       | [Somepalli et al. (2021)](https://arxiv.org/pdf/2106.01342)                                                                               |
| `TabM`           | Batch ensembling for MLP                       | [Gorishniy et al. (2024)](https://arxiv.org/abs/2410.24210)                                                                               |
| `TabR`           | Retrieval-augmented tabular model              | —                                                                                                                                         |
| `ResNet`         | ResNet adapted for tabular data                | —                                                                                                                                         |
| `MLP`            | Multi-layer perceptron baseline                | —                                                                                                                                         |
| `NODE`           | Neural oblivious decision ensembles            | [Popov et al. (2019)](https://arxiv.org/abs/1909.06312)                                                                                   |
| `NDTF`           | Neural decision tree forest                    | [Kontschieder et al. (2015)](https://openaccess.thecvf.com/content_iccv_2015/html/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.html) |
| `TabulaRNN`      | Recurrent neural network for tabular data      | [Thielmann et al. (2025)](https://arxiv.org/pdf/2411.17207)                                                                               |
| `ENODE`          | Extended NODE variant                          | —                                                                                                                                         |
| `AutoInt`        | Automatic feature interaction via attention    | —                                                                                                                                         |
| `ModernNCA`      | Modern neural classification architecture      | —                                                                                                                                         |
| `Trompt`         | Tabular-specific prompting model               | —                                                                                                                                         |
| `TANGOS`         | Tabular model with graph-based structure       | —                                                                                                                                         |

## Next steps

- [Installation](installation) — install DeepTab and verify the setup.
- [Key Concepts](key_concepts) — understand the API patterns before writing code.
- [Examples](../examples/classification) — runnable end-to-end workflows.
- [API Reference](../api/models/index) — full parameter documentation.
