import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..arch_utils.get_norm_fn import get_normalization_layer
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.mlp_utils import MLPhead
from ..configs.modernnca_config import DefaultModernNCAConfig
from ..utils.get_feature_dimensions import get_feature_dimensions
from .utils.basemodel import BaseModel


class ModernNCA(BaseModel):
    def __init__(
        self,
        feature_information: tuple,
        num_classes=1,
        config: DefaultModernNCAConfig = DefaultModernNCAConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["feature_information"])

        self.returns_ensemble = False
        self.uses_candidates = True

        self.T = config.temperature
        self.sample_rate = config.sample_rate
        if self.hparams.use_embeddings:
            self.embedding_layer = EmbeddingLayer(
                *feature_information,
                config=config,
            )

            input_dim = np.sum([len(info) * self.hparams.d_model for info in feature_information])
        else:
            input_dim = get_feature_dimensions(*feature_information)

        self.encoder = nn.Linear(input_dim, config.dim)

        if config.n_blocks > 0:
            self.post_encoder = nn.Sequential(
                *[self.make_layer(config) for _ in range(config.n_blocks)],
                nn.BatchNorm1d(config.dim),
            )

        self.tabular_head = MLPhead(
            input_dim=config.dim,
            config=config,
            output_dim=num_classes,
        )

        self.hparams.num_classes = num_classes

    def make_layer(self, config):
        return nn.Sequential(
            nn.BatchNorm1d(config.dim),
            nn.Linear(config.dim, config.d_block),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_block, config.dim),
        )

    def forward(self, *data):
        """Standard forward pass without candidate selection (for baseline compatibility)."""
        if self.hparams.use_embeddings:
            x = self.embedding_layer(*data)
            B, S, D = x.shape
            x = x.reshape(B, S * D)
        else:
            x = torch.cat([t for tensors in data for t in tensors], dim=1)
        x = self.encoder(x)
        if hasattr(self, "post_encoder"):
            x = self.post_encoder(x)
        return self.tabular_head(x)

    def train_with_candidates(self, *data, targets, candidate_x, candidate_y):
        """NCA-style training forward pass selecting candidates."""
        if self.hparams.use_embeddings:
            x = self.embedding_layer(*data)
            B, S, D = x.shape
            x = x.reshape(B, S * D)
            candidate_x = self.embedding_layer(*candidate_x)
            B, S, D = candidate_x.shape
            candidate_x = candidate_x.reshape(B, S * D)
        else:
            x = torch.cat([t for tensors in data for t in tensors], dim=1)
            candidate_x = torch.cat([t for tensors in candidate_x for t in tensors], dim=1)

        # Encode input
        x = self.encoder(x)
        candidate_x = self.encoder(candidate_x)

        if hasattr(self, "post_encoder"):
            x = self.post_encoder(x)
            candidate_x = self.post_encoder(candidate_x)

        # Select a subset of candidates
        data_size = candidate_x.shape[0]
        retrieval_size = int(data_size * self.sample_rate)
        sample_idx = torch.randperm(data_size)[:retrieval_size]
        candidate_x = candidate_x[sample_idx]
        candidate_y = candidate_y[sample_idx]

        # Concatenate with training batch
        candidate_x = torch.cat([x, candidate_x], dim=0)
        candidate_y = torch.cat([targets, candidate_y], dim=0)

        # One-hot encode if classification
        if self.hparams.num_classes > 1:
            candidate_y = F.one_hot(candidate_y, num_classes=self.hparams.num_classes).to(x.dtype)
        elif len(candidate_y.shape) == 1:
            candidate_y = candidate_y.unsqueeze(-1)

        # Compute distances
        distances = torch.cdist(x, candidate_x, p=2) / self.T
        # remove the label of training index
        distances = distances.fill_diagonal_(torch.inf)
        distances = F.softmax(-distances, dim=-1)
        logits = torch.mm(distances, candidate_y)
        eps = 1e-7
        if self.hparams.num_classes > 1:
            logits = torch.log(logits + eps)

        return logits

    def validate_with_candidates(self, *data, candidate_x, candidate_y):
        """Validation forward pass with NCA-style candidate selection."""
        if self.hparams.use_embeddings:
            x = self.embedding_layer(*data)
            B, S, D = x.shape
            x = x.reshape(B, S * D)
            candidate_x = self.embedding_layer(*candidate_x)
            B, S, D = candidate_x.shape
            candidate_x = candidate_x.reshape(B, S * D)
        else:
            x = torch.cat([t for tensors in data for t in tensors], dim=1)
            candidate_x = torch.cat([t for tensors in candidate_x for t in tensors], dim=1)

        # Encode input
        x = self.encoder(x)
        candidate_x = self.encoder(candidate_x)

        if hasattr(self, "post_encoder"):
            x = self.post_encoder(x)
            candidate_x = self.post_encoder(candidate_x)

        # One-hot encode if classification
        if self.hparams.num_classes > 1:
            candidate_y = F.one_hot(candidate_y, num_classes=self.hparams.num_classes).to(x.dtype)
        elif len(candidate_y.shape) == 1:
            candidate_y = candidate_y.unsqueeze(-1)

        # Compute distances
        distances = torch.cdist(x, candidate_x, p=2) / self.T
        distances = F.softmax(-distances, dim=-1)

        # Compute logits
        logits = torch.mm(distances, candidate_y)
        eps = 1e-7
        if self.hparams.num_classes > 1:
            logits = torch.log(logits + eps)

        return logits

    def predict_with_candidates(self, *data, candidate_x, candidate_y):
        """Prediction forward pass with candidate selection."""
        if self.hparams.use_embeddings:
            x = self.embedding_layer(*data)
            B, S, D = x.shape
            x = x.reshape(B, S * D)
            candidate_x = self.embedding_layer(*candidate_x)
            B, S, D = candidate_x.shape
            candidate_x = candidate_x.reshape(B, S * D)
        else:
            x = torch.cat([t for tensors in data for t in tensors], dim=1)
            candidate_x = torch.cat([t for tensors in candidate_x for t in tensors], dim=1)

        # Encode input
        x = self.encoder(x)
        candidate_x = self.encoder(candidate_x)

        if hasattr(self, "post_encoder"):
            x = self.post_encoder(x)
            candidate_x = self.post_encoder(candidate_x)

        # One-hot encode if classification
        if self.hparams.num_classes > 1:
            candidate_y = F.one_hot(candidate_y, num_classes=self.hparams.num_classes).to(x.dtype)
        elif len(candidate_y.shape) == 1:
            candidate_y = candidate_y.unsqueeze(-1)

        # Compute distances
        distances = torch.cdist(x, candidate_x, p=2) / self.T
        distances = F.softmax(-distances, dim=-1)

        # Compute logits
        logits = torch.mm(distances, candidate_y)
        eps = 1e-7
        if self.hparams.num_classes > 1:
            logits = torch.log(logits + eps)

        return logits
