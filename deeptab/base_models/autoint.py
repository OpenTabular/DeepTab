import numpy as np
import torch.nn as nn
import torch.nn.init as nn_init

from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..configs.autoint_config import DefaultAutoIntConfig
from .utils.basemodel import BaseModel


class AutoInt(BaseModel):
    """
    AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks.

    This model uses multi-head self-attention layers to learn feature interactions for tabular data.
    It supports key-value compression for memory efficiency and is compatible with embedding-based
    feature encodings.

    Parameters
    ----------
    feature_information : tuple
        A tuple containing information about numerical features, categorical features,
        and any additional embeddings. Expected format: `(num_feature_info, cat_feature_info, embedding_feature_info)`.
    num_classes : int, default=1
        Number of output classes. For regression, this should be set to `1`.
    config : DefaultAutoIntConfig, optional
        Configuration object containing hyperparameters such as `d_model`, `n_heads`, `n_layers`,
        dropout rates, and compression settings.
    **kwargs : dict
        Additional arguments passed to the `BaseModel`.

    Attributes
    ----------
    embedding_layer : EmbeddingLayer
        Module that processes numerical and categorical features into embeddings.
    kv_compression : float or None
        The proportion of key-value compression. If `None`, no compression is applied.
    kv_compression_sharing : str or None
        Defines how key-value compression is shared across layers. Options:
        - `"layerwise"`: One shared compression layer for all layers.
        - `"headwise"`: Separate key compression per head.
        - `"key-value"`: Separate compression layers for `k` and `v`.
    shared_kv_compression : nn.Linear or None
        Shared key-value compression layer, used when `kv_compression_sharing="layerwise"`.
    layers : nn.ModuleList
        A list of transformer-based attention layers, each consisting of:
        - `attention`: Multi-head self-attention module.
        - `linear`: Fully connected layer for projection.
        - `norm0`: Layer normalization.
    last_norm : nn.LayerNorm or None
        Final normalization layer applied before output if `prenormalization` is enabled.
    head : nn.Linear
        Output layer mapping from the processed feature representation to the final predictions.
    """

    def __init__(
        self,
        feature_information: tuple,  # (num_feature_info, cat_feature_info, embedding_feature_info)
        num_classes=1,
        config: DefaultAutoIntConfig = DefaultAutoIntConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["feature_information"])
        self.returns_ensemble = False

        # Embedding layer
        self.embedding_layer = EmbeddingLayer(*feature_information, config=config)
        n_inputs = np.sum([len(info) for info in feature_information])

        # Key-Value Compression
        self.kv_compression = config.kv_compression
        self.kv_compression_sharing = config.kv_compression_sharing

        def make_kv_compression():
            compression = nn.Linear(
                n_inputs,
                int(n_inputs * config.kv_compression),
                bias=False,
            )
            nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression() if self.kv_compression and self.kv_compression_sharing == "layerwise" else None
        )

        # Transformer-based Interaction Layers
        self.layers = nn.ModuleList()
        for layer_idx in range(config.n_layers):
            layer = nn.ModuleDict(
                {
                    "attention": nn.MultiheadAttention(
                        embed_dim=config.d_model,
                        num_heads=config.n_heads,
                        dropout=config.attn_dropout,
                        batch_first=True,
                    ),
                    "linear": nn.Linear(config.d_model, config.d_model, bias=False),
                    "norm0": nn.LayerNorm(config.d_model),
                }
            )

            if self.kv_compression and self.shared_kv_compression is None:
                layer["key_compression"] = make_kv_compression()
                if self.kv_compression_sharing == "headwise":
                    layer["value_compression"] = make_kv_compression()
                else:
                    assert self.kv_compression_sharing == "key-value"  # noqa: S101

            self.layers.append(layer)

        # Final Normalization & Output Head
        self.last_norm = nn.LayerNorm(config.d_model) if getattr(config, "prenorm", False) else None

        self.head = nn.Linear(config.d_model * n_inputs, num_classes)

    def _get_kv_compressions(self, layer):
        """
        Returns the correct key-value compression layers based on the sharing strategy.

        Parameters
        ----------
        layer : nn.ModuleDict
            The transformer layer containing possible key-value compression modules.

        Returns
        -------
        tuple of (nn.Linear or None, nn.Linear or None)
            The key compression and value compression layers, or `(None, None)` if no compression is applied.
        """
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (
                (layer["key_compression"], layer["value_compression"])
                if "key_compression" in layer and "value_compression" in layer
                else (
                    (layer["key_compression"], layer["key_compression"]) if "key_compression" in layer else (None, None)
                )
            )
        )

    def forward(self, *data):
        """
        Forward pass of the AutoInt model.

        Parameters
        ----------
        *data : tuple
            Input tuple of tensors containing numerical features, categorical features, and embeddings.

        Returns
        -------
        Tensor
            The output predictions of the model.
        """
        x = self.embedding_layer(*data)  # Shape: (N, J, d_model)

        for layer in self.layers:
            x_residual = x  # Store original input for residual connection

            # Apply normalization before attention if prenormalization is enabled
            x_residual = layer["norm0"](x_residual)  # type: ignore[index]

            # Retrieve key-value compression layers
            key_compression, value_compression = self._get_kv_compressions(layer)

            # Multihead Attention
            x_residual, _ = layer["attention"](x_residual, x_residual, x_residual)  # type: ignore[index]

            # Apply residual connection
            x = x + x_residual

            # Apply the linear transformation
            x_residual = layer["linear"](x)  # type: ignore[index]
            x = x + x_residual  # Second residual connection

        if self.last_norm:
            x = self.last_norm(x)  # Final normalization if prenormalization is used

        x = x.flatten(1)  # Flatten from (N, J, d_model) to (N, J * d_model)
        return self.head(x)  # Final prediction
