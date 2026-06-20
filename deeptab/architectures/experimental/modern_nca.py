import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeptab.configs.experimental.modernnca_config import ModernNCAConfig
from deeptab.core import BaseModel, get_feature_dimensions
from deeptab.nn.blocks.common import EmbeddingLayer
from deeptab.nn.blocks.mlp import MLPhead
from deeptab.nn.normalization import get_normalization_layer


class ModernNCA(BaseModel):
    """Differentiable Neighborhood Component Analysis for tabular data.

    ModernNCA revisits classic Neighborhood Component Analysis with modern
    tabular deep-learning components. Each row is mapped into a learned
    representation by an encoder and optional residual post-encoder blocks.
    Predictions are formed by comparing a query row to a set of candidate
    (training) rows in that representation space and taking a
    temperature-scaled, softmax-weighted aggregate over the neighbors.

    The aggregation target depends on the task:

    - **Classification / regression:** the softmax weights are applied to the
      candidate *labels* (one-hot for classification, raw targets for
      regression), so the prediction is a soft nearest-neighbor vote.
    - **Distributional (LSS):** raw labels cannot describe a distribution, so
      the softmax weights are instead applied to the candidate *representations*
      and the pooled neighbor representation is decoded by ``tabular_head`` into
      the distribution parameters expected by the chosen family.

    Because predictions depend on candidate rows, the model sets
    ``uses_candidates = True`` and exposes candidate-aware
    :meth:`train_with_candidates`, :meth:`validate_with_candidates`, and
    :meth:`predict_with_candidates` methods. The plain :meth:`forward` exists
    only for baseline compatibility.

    Parameters
    ----------
    feature_information : tuple
        A tuple containing feature information for numerical, categorical, and
        embedding features.
    num_classes : int, optional (default=1)
        The output dimension. ``1`` for scalar regression, the number of
        classes for classification, or the distribution parameter count for
        distributional (LSS) models.
    config : ModernNCAConfig, optional (default=ModernNCAConfig())
        Configuration object defining model hyperparameters.
    **kwargs : dict
        Additional arguments for the base model, including the ``lss`` flag.

    Attributes
    ----------
    returns_ensemble : bool
        Whether the model returns an ensemble of predictions. Always ``False``.
    uses_candidates : bool
        Marks the model as candidate-aware so the training loop supplies
        candidate rows. Always ``True``.
    T : float
        Temperature used to scale distances before the softmax.
    sample_rate : float
        Proportion of candidate rows sampled during training.
    embedding_layer : EmbeddingLayer or None
        Optional embedding layer for categorical and embedding features.
    encoder : nn.Linear
        Linear encoder mapping raw feature dimensions to ``config.dim``.
    post_encoder : nn.Sequential or None
        Optional residual blocks applied after the encoder.
    tabular_head : MLPhead
        Output head used for the plain forward pass and for decoding pooled
        neighbor representations in the distributional (LSS) path.
    """

    def __init__(
        self,
        feature_information: tuple,
        num_classes=1,
        config: ModernNCAConfig = ModernNCAConfig(),  # noqa: B008
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
        """Standard forward pass without candidate selection (for baseline compatibility).

        Parameters
        ----------
        data : tuple
            Input tuple of tensors of num_features, cat_features, embeddings.

        Returns
        -------
        Tensor
            The output predictions of the model.
        """
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
        """NCA-style training forward pass selecting candidates.

        Parameters
        ----------
        data : tuple
            Input tuple of tensors of num_features, cat_features, embeddings for
            the query rows.
        targets : Tensor
            Targets for the query rows, concatenated with the candidate pool so
            each query can attend to its own batch.
        candidate_x : tuple
            Input tuple of tensors of num_features, cat_features, embeddings for
            the candidate (training) rows.
        candidate_y : Tensor
            Targets for the candidate rows.

        Returns
        -------
        Tensor
            The output predictions of the model. For classification and
            regression these are softmax-weighted candidate labels; for
            distributional (LSS) models these are the decoded distribution
            parameters.
        """
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

        # Compute distances
        distances = torch.cdist(x, candidate_x, p=2) / self.T
        # remove the label of training index
        distances = distances.fill_diagonal_(torch.inf)
        distances = F.softmax(-distances, dim=-1)

        if self.hparams.lss:
            # Labels cannot describe a distribution, so pool neighbor
            # representations and decode them into distribution parameters.
            context = torch.mm(distances, candidate_x)
            return self.tabular_head(context)

        # One-hot encode if classification
        if self.hparams.num_classes > 1:
            candidate_y = F.one_hot(candidate_y, num_classes=self.hparams.num_classes).to(x.dtype)
        elif len(candidate_y.shape) == 1:
            candidate_y = candidate_y.unsqueeze(-1)

        logits = torch.mm(distances, candidate_y)
        eps = 1e-7
        if self.hparams.num_classes > 1:
            logits = torch.log(logits + eps)

        return logits

    def validate_with_candidates(self, *data, candidate_x, candidate_y):
        """Validation forward pass with NCA-style candidate selection.

        Parameters
        ----------
        data : tuple
            Input tuple of tensors of num_features, cat_features, embeddings for
            the query rows.
        candidate_x : tuple
            Input tuple of tensors of num_features, cat_features, embeddings for
            the candidate (training) rows.
        candidate_y : Tensor
            Targets for the candidate rows.

        Returns
        -------
        Tensor
            The output predictions of the model. For classification and
            regression these are softmax-weighted candidate labels; for
            distributional (LSS) models these are the decoded distribution
            parameters.
        """
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

        # Compute distances
        distances = torch.cdist(x, candidate_x, p=2) / self.T
        distances = F.softmax(-distances, dim=-1)

        if self.hparams.lss:
            # Labels cannot describe a distribution, so pool neighbor
            # representations and decode them into distribution parameters.
            context = torch.mm(distances, candidate_x)
            return self.tabular_head(context)

        # One-hot encode if classification
        if self.hparams.num_classes > 1:
            candidate_y = F.one_hot(candidate_y, num_classes=self.hparams.num_classes).to(x.dtype)
        elif len(candidate_y.shape) == 1:
            candidate_y = candidate_y.unsqueeze(-1)

        # Compute logits
        logits = torch.mm(distances, candidate_y)
        eps = 1e-7
        if self.hparams.num_classes > 1:
            logits = torch.log(logits + eps)

        return logits

    def predict_with_candidates(self, *data, candidate_x, candidate_y):
        """Prediction forward pass with candidate selection.

        Parameters
        ----------
        data : tuple
            Input tuple of tensors of num_features, cat_features, embeddings for
            the query rows.
        candidate_x : tuple
            Input tuple of tensors of num_features, cat_features, embeddings for
            the candidate (training) rows.
        candidate_y : Tensor
            Targets for the candidate rows.

        Returns
        -------
        Tensor
            The output predictions of the model. For classification and
            regression these are softmax-weighted candidate labels; for
            distributional (LSS) models these are the decoded distribution
            parameters.
        """
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

        # Compute distances
        distances = torch.cdist(x, candidate_x, p=2) / self.T
        distances = F.softmax(-distances, dim=-1)

        if self.hparams.lss:
            # Labels cannot describe a distribution, so pool neighbor
            # representations and decode them into distribution parameters.
            context = torch.mm(distances, candidate_x)
            return self.tabular_head(context)

        # One-hot encode if classification
        if self.hparams.num_classes > 1:
            candidate_y = F.one_hot(candidate_y, num_classes=self.hparams.num_classes).to(x.dtype)
        elif len(candidate_y.shape) == 1:
            candidate_y = candidate_y.unsqueeze(-1)

        # Compute logits
        logits = torch.mm(distances, candidate_y)
        eps = 1e-7
        if self.hparams.num_classes > 1:
            logits = torch.log(logits + eps)

        return logits
