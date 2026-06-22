# Custom Models

DeepTab is not a fixed catalogue of architectures. The same scikit-learn API,
preprocessing pipeline, trainer, and observability stack that power the built-in
models are available to any architecture you write. This page shows how to plug
your own PyTorch module into DeepTab and use it like any other estimator.

## When to write a custom model

Write a custom model when you want a new architecture but still want DeepTab to
handle preprocessing, batching, training loops, checkpointing, metrics, and the
`*Classifier` / `*Regressor` / `*LSS` interface for you. You only implement the
network; DeepTab provides everything around it.

If you only need to change hyperparameters of an existing model, use its config
instead (see [Config System](config_system)). Custom models are for new
_architectures_.

## The three pieces

A DeepTab model is always three small, separate pieces:

| Piece        | Base class                                                          | Responsibility                                          |
| ------------ | ------------------------------------------------------------------- | ------------------------------------------------------- |
| Config       | `BaseModelConfig`                                                   | A dataclass of architecture hyperparameters.            |
| Architecture | `BaseModel`                                                         | The PyTorch module: layers and `forward`.               |
| Estimator    | `SklearnBaseClassifier` / `SklearnBaseRegressor` / `SklearnBaseLSS` | The sklearn-facing wrapper that binds the two together. |

This mirrors exactly how the built-in models are built, so a custom model is a
first-class citizen, not a second-tier extension point.

## 1. The config

Configs are dataclasses that inherit from `BaseModelConfig`. Inheriting matters:
`BaseModelConfig` supplies the shared embedding and architecture fields
(`use_embeddings`, `embedding_type`, `d_model`, `batch_norm`, `layer_norm`,
`activation`, `cat_encoding`, …) that the preprocessing and embedding machinery
rely on. Add only your architecture-specific fields.

```python
from dataclasses import dataclass, field

from deeptab.configs import BaseModelConfig


@dataclass
class MyMLPConfig(BaseModelConfig):
    """Architecture hyperparameters for the custom model."""

    layer_sizes: list = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
```

> **Note:** Use `field(default_factory=...)` for mutable defaults such as lists.
> A plain class (or a non-dataclass) will not integrate with the config system,
> hyperparameter saving, or sklearn introspection.

## 2. The architecture

The architecture subclasses `BaseModel`. Two conventions define the contract:

- The constructor receives a `feature_information` tuple and `num_classes`.
- `forward` receives the three feature groups and returns raw outputs (logits
  for classification, real values for regression). No final activation, because
  DeepTab applies the task-appropriate loss.

### The `feature_information` tuple

Every architecture is built with:

```python
feature_information = (num_feature_info, cat_feature_info, embedding_feature_info)
```

Each element is a dict describing one feature group, where every entry carries a
`"dimension"` key. You rarely inspect these dicts by hand; use the helpers:

- `get_feature_dimensions(*feature_information)` returns the total flattened
  input width when you are **not** using embeddings.
- `EmbeddingLayer(*feature_information, config=config)` builds a learned
  embedding for each feature when you **are** using embeddings.

### The `forward` contract

At training and inference time DeepTab calls `forward` with three positional
tensors: `num_features`, `cat_features`, and `embeddings`. Accepting `*data`
lets you forward the whole group straight into helpers like `EmbeddingLayer`.

```python
import torch
import torch.nn as nn

from deeptab.core import BaseModel, get_feature_dimensions


class MyMLP(BaseModel):
    def __init__(
        self,
        feature_information: tuple,  # (num_info, cat_info, embedding_info)
        num_classes: int = 1,
        config: MyMLPConfig = MyMLPConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        # Persist hyperparameters as self.hparams (skip the runtime-only tuple).
        self.save_hyperparameters(ignore=["feature_information"])

        # Input width is derived from the data, not assumed.
        input_dim = get_feature_dimensions(*feature_information)

        layers: list[nn.Module] = []
        prev = input_dim
        for size in self.hparams.layer_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU(), nn.Dropout(self.hparams.dropout)]
            prev = size
        layers.append(nn.Linear(prev, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, *data) -> torch.Tensor:
        # data == (num_features, cat_features, embeddings); concatenate the
        # non-empty groups into a single dense input.
        x = torch.cat([t for group in data for t in group], dim=1)
        return self.layers(x)
```

> **Why `get_feature_dimensions`?** The number of input columns is only known
> after preprocessing (binning, one-hot encoding, etc.). Hard-coding a width
> such as `config.d_model` is the most common mistake and raises a shape error
> at the first batch. Always derive the input size from `feature_information`.

## 3. The estimator

The estimator binds the architecture and its default config through two class
attributes, `_model_cls` and `_config_cls`. Define one estimator per task you
want to support:

```python
from deeptab.models import (
    SklearnBaseClassifier,
    SklearnBaseRegressor,
    SklearnBaseLSS,
)


class MyMLPClassifier(SklearnBaseClassifier):
    _model_cls = MyMLP
    _config_cls = MyMLPConfig


class MyMLPRegressor(SklearnBaseRegressor):
    _model_cls = MyMLP
    _config_cls = MyMLPConfig


class MyMLPLSS(SklearnBaseLSS):
    _model_cls = MyMLP
    _config_cls = MyMLPConfig
```

That is all the wiring required. The estimators inherit the full DeepTab API:
`fit`, `predict`, `predict_proba`, preprocessing, checkpointing, and
observability.

## Using the custom model

A custom estimator behaves exactly like a built-in one. Pass architecture
hyperparameters through the config and training settings through
`TrainerConfig`:

```python
from deeptab.configs import TrainerConfig

model = MyMLPRegressor(
    model_config=MyMLPConfig(layer_sizes=[256, 128], dropout=0.2),
    trainer_config=TrainerConfig(lr=1e-3),
)
model.fit(X_train, y_train, max_epochs=50)
preds = model.predict(X_test)
```

If you omit `model_config`, DeepTab instantiates `_config_cls()` with its
defaults.

## Optional: use embeddings

To embed categorical and numerical features instead of concatenating raw
columns, set `use_embeddings=True` in the config and build an `EmbeddingLayer`.
This is how the Transformer- and Mamba-family models consume features.

```python
import numpy as np

from deeptab.core import BaseModel
from deeptab.nn.blocks.common import EmbeddingLayer


class MyEmbeddedModel(BaseModel):
    def __init__(self, feature_information, num_classes=1, config=MyMLPConfig(), **kwargs):  # noqa: B008
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["feature_information"])

        self.embedding_layer = EmbeddingLayer(*feature_information, config=config)
        n_features = sum(len(info) for info in feature_information)
        input_dim = n_features * self.hparams.d_model

        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, *data):
        x = self.embedding_layer(*data)   # (batch, n_features, d_model)
        x = x.reshape(x.shape[0], -1)     # flatten to (batch, n_features * d_model)
        return self.head(x)
```

## Checklist

| Piece        | Requirement                                                                                    |
| ------------ | ---------------------------------------------------------------------------------------------- |
| Config       | A `@dataclass` subclassing `BaseModelConfig`.                                                  |
| Config       | Mutable defaults use `field(default_factory=...)`.                                             |
| Architecture | Subclasses `BaseModel` and calls `super().__init__(config=config, **kwargs)`.                  |
| Architecture | Constructor calls `self.save_hyperparameters(ignore=["feature_information"])`.                 |
| Architecture | Input width comes from `get_feature_dimensions(...)` or an `EmbeddingLayer`, never hard-coded. |
| Architecture | `forward` returns raw outputs (no final softmax/sigmoid).                                      |
| Estimator    | Each estimator sets `_model_cls` and `_config_cls`.                                            |

## Next Steps

- [Config System](config_system)
- [scikit-learn API](sklearn_api)
- [Model Tiers](model_tiers)
- [Contributing](../developer_guide/contributing): if you want to upstream a model into DeepTab itself.
