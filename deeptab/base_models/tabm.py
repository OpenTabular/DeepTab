import numpy as np
import torch
import torch.nn as nn

from ..arch_utils.get_norm_fn import get_normalization_layer
from ..arch_utils.layer_utils.batch_ensemble_layer import LinearBatchEnsembleLayer
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.layer_utils.sn_linear import SNLinear
from ..configs.tabm_config import DefaultTabMConfig
from ..utils.get_feature_dimensions import get_feature_dimensions
from .utils.basemodel import BaseModel


class TabM(BaseModel):
    def __init__(
        self,
        feature_information: tuple,  # Expecting (num_feature_info, cat_feature_info, embedding_feature_info)
        num_classes: int = 1,
        config: DefaultTabMConfig = DefaultTabMConfig(),  # noqa: B008
        **kwargs,
    ):
        # Pass config to BaseModel
        super().__init__(config=config, **kwargs)

        # Save hparams including config attributes
        self.save_hyperparameters(ignore=["feature_information"])
        if not self.hparams.average_ensembles:
            self.returns_ensemble = True  # Directly set ensemble flag
        else:
            self.returns_ensemble = False

        # Initialize layers based on self.hparams
        self.layers = nn.ModuleList()

        # Conditionally initialize EmbeddingLayer based on self.hparams
        if self.hparams.use_embeddings:
            self.embedding_layer = EmbeddingLayer(
                *feature_information,
                config=config,
            )

            if self.hparams.average_embeddings:
                input_dim = self.hparams.d_model
            else:
                input_dim = np.sum([len(info) * self.hparams.d_model for info in feature_information])

        else:
            input_dim = get_feature_dimensions(*feature_information)

        # Input layer with batch ensembling
        self.layers.append(
            LinearBatchEnsembleLayer(
                in_features=input_dim,
                out_features=self.hparams.layer_sizes[0],
                ensemble_size=self.hparams.ensemble_size,
                ensemble_scaling_in=self.hparams.ensemble_scaling_in,
                ensemble_scaling_out=self.hparams.ensemble_scaling_out,
                ensemble_bias=self.hparams.ensemble_bias,
                scaling_init=self.hparams.scaling_init,
            )
        )
        if self.hparams.batch_norm:
            self.layers.append(nn.BatchNorm1d(self.hparams.layer_sizes[0]))

        self.norm_f = get_normalization_layer(config)
        if self.norm_f is not None:
            self.layers.append(self.norm_f(self.hparams.layer_sizes[0]))

        # Optional activation and dropout
        if self.hparams.use_glu:
            self.layers.append(nn.GLU())
        else:
            self.layers.append(self.hparams.activation if hasattr(self.hparams, "activation") else nn.SELU())
        if self.hparams.dropout > 0.0:
            self.layers.append(nn.Dropout(self.hparams.dropout))

        # Hidden layers with batch ensembling
        for i in range(1, len(self.hparams.layer_sizes)):
            if self.hparams.model_type == "mini":
                self.layers.append(
                    LinearBatchEnsembleLayer(
                        in_features=self.hparams.layer_sizes[i - 1],
                        out_features=self.hparams.layer_sizes[i],
                        ensemble_size=self.hparams.ensemble_size,
                        ensemble_scaling_in=False,
                        ensemble_scaling_out=False,
                        ensemble_bias=self.hparams.ensemble_bias,
                        scaling_init="ones",
                    )
                )
            else:
                self.layers.append(
                    LinearBatchEnsembleLayer(
                        in_features=self.hparams.layer_sizes[i - 1],
                        out_features=self.hparams.layer_sizes[i],
                        ensemble_size=self.hparams.ensemble_size,
                        ensemble_scaling_in=self.hparams.ensemble_scaling_in,
                        ensemble_scaling_out=self.hparams.ensemble_scaling_out,
                        ensemble_bias=self.hparams.ensemble_bias,
                        scaling_init="ones",
                    )
                )

            if self.hparams.use_glu:
                self.layers.append(nn.GLU())
            else:
                self.layers.append(self.hparams.activation if hasattr(self.hparams, "activation") else nn.SELU())
            if self.hparams.dropout > 0.0:
                self.layers.append(nn.Dropout(self.hparams.dropout))

        if self.hparams.average_ensembles:
            self.final_layer = nn.Linear(self.hparams.layer_sizes[-1], num_classes)
        else:
            self.final_layer = SNLinear(
                self.hparams.ensemble_size,
                self.hparams.layer_sizes[-1],
                num_classes,
            )

    def forward(self, *data) -> torch.Tensor:
        """Forward pass of the TabM model with batch ensembling.

        Parameters
        ----------
        data : tuple
            Input tuple of tensors of num_features, cat_features, embeddings.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        # Handle embeddings if used
        if self.hparams.use_embeddings:
            x = self.embedding_layer(*data)
            # Option 1: Average over feature dimension (N)
            if self.hparams.average_embeddings:
                x = x.mean(dim=1)  # Shape: (B, D)
            # Option 2: Flatten feature and embedding dimensions
            else:
                B, N, D = x.shape
                x = x.reshape(B, N * D)  # Shape: (B, N * D)

        else:
            x = torch.cat([t for tensors in data for t in tensors], dim=1)

        # Process through layers with optional skip connections
        for i in range(len(self.layers) - 1):
            if isinstance(self.layers[i], LinearBatchEnsembleLayer):
                out = self.layers[i](x)
                # `out` shape is expected to be (batch_size, ensemble_size, out_features)
                if hasattr(self, "skip_connections") and self.skip_connections and x.shape == out.shape:
                    x = x + out
                else:
                    x = out
            else:
                x = self.layers[i](x)

        # Final ensemble output from the last ConfigurableBatchEnsembleLayer
        # Shape (batch_size, ensemble_size, num_classes)
        x = self.layers[-1](x)

        if self.hparams.average_ensembles:
            x = x.mean(axis=1)  # Shape (batch_size, num_classes)
            print(x.shape)
        # Shape (batch_size, (ensemble_size), num_classes) if not averaged
        x = self.final_layer(x)

        if not self.hparams.average_ensembles:
            x = x.squeeze(-1)

        return x
