# Stable Models

```{important}
Stable model APIs are intended for production use. The pages in this section describe the model idea, the actual DeepTab implementation, and the configuration settings that matter when selecting a model for experiments.
```

DeepTab's stable model zoo contains 15 supervised architectures for classification, regression, and distributional regression. They cover four broad design families:

```{toctree}
:hidden:
:maxdepth: 1

mlp
resnet
tabm
fttransformer
tabtransformer
saint
autoint
mambular
mambatab
mambattention
tabularnn
node
enode
ndtf
tabr
```

| Family                                    | Models                                                                                               | Use when                                                                                              |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| MLP and residual baselines                | [MLP](mlp), [ResNet](resnet), [TabM](tabm)                                                           | You need strong, fast baselines or parameter-efficient ensembles.                                     |
| Transformer and attention models          | [FTTransformer](fttransformer), [TabTransformer](tabtransformer), [SAINT](saint), [AutoInt](autoint) | Feature interactions are important and the dataset is large enough to support attention layers.       |
| State-space and recurrent sequence models | [Mambular](mambular), [MambaTab](mambatab), [MambAttention](mambattention), [TabulaRNN](tabularnn)   | You want to treat columns as a sequence and compare Mamba/RNN-style inductive biases.                 |
| Neural tree and retrieval models          | [NODE](node), [ENODE](enode), [NDTF](ndtf), [TabR](tabr)                                             | You want differentiable tree structure, ensemble behavior, or train-set retrieval at prediction time. |

## Selection Guide

Start with **TabM**, **MLP**, or **ResNet** when building a baseline suite. These models are fast, robust, and usually easier to tune than attention-heavy models.

Use **FTTransformer** when you want a standard feature-token Transformer that embeds both numerical and categorical columns. Use **TabTransformer** when categorical interactions are central; DeepTab's implementation requires categorical features and concatenates normalized numerical features after the categorical Transformer.

Use **Mambular** or **MambAttention** when you want to evaluate state-space sequence modeling over feature tokens. Use **MambaTab** mainly as a lightweight projected-feature baseline in the current implementation; the model object defines a Mamba block, but the current forward path does not apply it.

Use **TabR** when train-set neighbors are expected to carry useful local signal and you can afford candidate retrieval. Use **NODE**, **ENODE**, or **NDTF** when you want differentiable tree/forest inductive bias inside a neural training loop.

## Common Usage Pattern

```python
from deeptab.configs import MLPConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MLPClassifier

model = MLPClassifier(
    model_config=MLPConfig(layer_sizes=[256, 128, 32], dropout=0.2),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="standardization"),
    trainer_config=TrainerConfig(lr=1e-3, batch_size=256, max_epochs=100),
    random_state=101,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Config Layers

DeepTab 2.x separates model, preprocessing, and training settings:

| Config object           | Contains                                                                                                                   |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `*Config` model configs | Architecture fields such as width, depth, dropout, embeddings, heads, pooling, and ensemble size.                          |
| `PreprocessingConfig`   | Numerical/categorical preprocessing choices such as standard scaling, quantile transforms, bins, and categorical encoding. |
| `TrainerConfig`         | Optimizer and training-loop settings such as learning rate, batch size, epochs, patience, and weight decay.                |

## Research Context

The stable zoo intentionally includes simple baselines and specialized models. This is important for tabular research: several broad evaluations show that plain MLP/ResNet-style models, FT-Transformer, retrieval, and tree-based baselines can trade places depending on dataset size, feature types, preprocessing, and tuning budget.

Useful starting references:

- Gorishniy et al., [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959).
- Shwartz-Ziv and Armon, [Tabular Data: Deep Learning is Not All You Need](https://arxiv.org/abs/2106.03253).
- Gorishniy et al., [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210).
