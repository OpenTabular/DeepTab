import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..configs.tabr_config import DefaultTabRConfig
from ..utils.get_feature_dimensions import get_feature_dimensions
from .utils.basemodel import BaseModel


class TabR(BaseModel):
    delu = None
    faiss = None
    faiss_torch_utils = None

    def __init__(
        self,
        feature_information: tuple,
        num_classes=1,
        config: DefaultTabRConfig = DefaultTabRConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["feature_information"])

        # lazy import
        if TabR.delu or TabR.faiss or TabR.faiss_torch_utils is None:
            self._lazy_import_dependencies()

        self.returns_ensemble = False
        self.uses_candidates = True

        if self.hparams.use_embeddings:
            self.embedding_layer = EmbeddingLayer(
                *feature_information,
                config=config,
            )
            print(self.embedding_layer)
            input_dim = np.sum([len(info) * self.hparams.d_model for info in feature_information])
        else:
            input_dim = get_feature_dimensions(*feature_information)

        self.hparams.num_classes = num_classes
        memory_efficient = self.hparams.memory_efficient
        mixer_normalization = self.hparams.mixer_normalization
        encoder_n_blocks = self.hparams.encoder_n_blocks
        predictor_n_blocks = self.hparams.predictor_n_blocks
        dropout0 = self.hparams.dropout1
        self.candidate_encoding_batch_size = self.hparams.candidate_encoding_batch_size
        d_main = self.hparams.d_main
        d_multiplier = self.hparams.d_multiplier
        normalization = self.hparams.normalization
        activation = self.hparams.activation
        dropout0 = self.hparams.dropout0
        dropout1 = self.hparams.dropout1
        context_dropout = self.hparams.context_dropout

        if memory_efficient:
            assert self.candidate_encoding_batch_size != 0  # noqa: S101

        if mixer_normalization == "auto":
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization  # noqa: S101

        # Encoder Module: E
        d_in = input_dim
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        self.linear = nn.Linear(d_in, d_main)
        self.context_size = self.hparams.context_size

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                activation,
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        # here in the TabR paper, for first block of Encoder(E),
        # LayerNorm is omitted. In code, we omitted Normalization.
        self.blocks0 = nn.ModuleList([make_block(i > 0) for i in range(encoder_n_blocks)])

        # Retrieval Module: R
        self.normalization = Normalization(d_main) if mixer_normalization else None

        delu = TabR.delu
        self.label_encoder = (
            nn.Linear(1, d_main)
            if num_classes == 1
            else nn.Sequential(
                nn.Embedding(num_classes, d_main),
                # gives depreciation warning
                delu.nn.Lambda(  # type: ignore[union-attr]
                    lambda x: x.squeeze(-2)
                ),  # Removes the unnecessary extra dimension added by the embedding layer
            )
        )
        self.K = nn.Linear(d_main, d_main)  # W_k in paper
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            activation,
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )  # T for T(k-k_i) form the TabR paper.
        self.dropout = nn.Dropout(context_dropout)

        # Predictor Module : P
        self.blocks1 = nn.ModuleList([make_block(True) for _ in range(predictor_n_blocks)])
        self.head = nn.Sequential(
            Normalization(d_main),
            activation,
            nn.Linear(d_main, num_classes),
        )

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):  # if num_classes==1
            bound = 1 / math.sqrt(2.0)  # He initialization (common for layers with ReLU activation)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)  # noqa: S101
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]

    def _lazy_import_dependencies(self):
        """Lazily import external dependencies and store them as class attributes."""
        if TabR.delu is None:
            try:
                import delu  # type: ignore[import-untyped]

                TabR.delu = delu
                print("Successfully lazy imported delu dependency.")

            except ImportError:
                raise ImportError(
                    "Failed to import delu module for TabR. Ensure all dependencies are installed\n"
                    "You can install delu running 'pip install delu'."
                ) from None

        if TabR.faiss is None:
            try:
                import faiss  # type: ignore[import-untyped]
                import faiss.contrib.torch_utils  # type: ignore[import-untyped]

                TabR.faiss = faiss
                TabR.faiss_torch_utils = faiss.contrib.torch_utils
                print("Successfully lazy imported faiss dependency")

            except ImportError as e:
                raise ImportError(
                    "Failed to import faiss module for TabR. Ensure all dependencies are installed\n"
                    "You can install faiss running 'pip install faiss-cpu' for CPU and 'pip install faiss-gpu' for GPU."
                ) from None

    def _encode(self, a):
        # x = x.double() # issue
        x = a.float()
        # x=a.clone().detach().requires_grad_(True)
        x = x.float()
        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))

        return x, k

    def forward(self, *data):
        """
        Standard forward pass without candidate selection (for baseline compatibility).
        """
        if self.hparams.use_embeddings:
            x = self.embedding_layer(*data)
            B, S, D = x.shape
            x = x.reshape(B, S * D)
        else:
            x = torch.cat([t for tensors in data for t in tensors], dim=1)
        x, k = self._encode(x)
        context_k = k.unsqueeze(1).expand(-1, self.context_size, -1)  # using the batch itself as context
        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        context_x = torch.sum(probs.unsqueeze(-1) * context_k, dim=1)
        t = self.T(self.dropout(context_x))
        for block in self.blocks1:
            x = x + block(x + t)
        return self.head(x)

    def train_with_candidates(self, *data, targets, candidate_x, candidate_y):
        """TabR-style training forward pass selecting candidates."""
        assert targets is not None  # noqa: S101

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

        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.memory_efficient):
            candidate_k = (
                self._encode(candidate_x)[1]  # normalized candidate_x
                if self.candidate_encoding_batch_size == 0
                else torch.cat(
                    [
                        self._encode(x)[1]  # normalized x
                        # for x in delu.iter_batches(
                        for x in TabR.delu.iter_batches(candidate_x, self.candidate_encoding_batch_size)  # type: ignore[union-attr]
                    ]
                )
            )

        # Encode input
        x, k = self._encode(x)

        batch_size, d_main = k.shape
        device = k.device
        context_size = self.context_size

        with torch.no_grad():
            # initializing the search index
            if self.search_index is None:
                self.search_index = (
                    TabR.faiss.GpuIndexFlatL2(TabR.faiss.StandardGpuResources(), d_main)  # type: ignore[union-attr]
                    if device.type == "cuda"
                    else TabR.faiss.IndexFlatL2(d_main)  # type: ignore[union-attr]
                )
            # Updating the index is much faster than creating a new one.
            self.search_index.reset()
            self.search_index.add(candidate_k.to(torch.float32))  # type: ignore[code]
            distances: Tensor
            context_idx: Tensor
            distances, context_idx = self.search_index.search(  # type: ignore[code]
                k.to(torch.float32), context_size + 1
            )
            # NOTE: to avoid leakage, the index i must be removed from the i-th row,
            # (because of how candidate_k is constructed).
            distances[context_idx == torch.arange(batch_size, device=device)[:, None]] = torch.inf
            # Not the most elegant solution to remove the argmax, but anyway.
            context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])

        if self.memory_efficient and torch.is_grad_enabled():
            # Repeating the same computation,
            # but now only for the context objects and with autograd on.
            context_k = self._encode(torch.cat([x, candidate_x])[context_idx].flatten(0, 1))[1].reshape(
                batch_size, context_size, -1
            )
        else:
            context_k = candidate_k[context_idx]

        # In theory, when autograd is off, the distances obtained during the search
        # can be reused. However, this is not a bottleneck, so let's keep it simple
        # and use the same code to compute `similarities` during both
        # training and evaluation.
        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        if self.hparams.num_classes > 1:  # for classification
            context_y_emb = self.label_encoder(candidate_y[context_idx][..., None].long())
        else:  # for regression
            context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
            if len(context_y_emb.shape) == 4:
                context_y_emb = context_y_emb[:, :, 0, :]

        # Combine keys and labels with a transformation T.
        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # Predictor has LayerNorm, ReLU and Linear after the N_P number of blocks.
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x

    def validate_with_candidates(self, *data, candidate_x, candidate_y):
        """Validation forward pass with TabR-style candidate selection."""
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

        if not self.memory_efficient:
            candidate_k = (
                self._encode(candidate_x)[1]  # normalized candidate_x
                if self.candidate_encoding_batch_size == 0
                else torch.cat(
                    [
                        self._encode(x)[1]  # normalized x
                        for x in TabR.delu.iter_batches(candidate_x, self.candidate_encoding_batch_size)  # type: ignore[union-attr]
                    ]
                )
            )
        else:
            candidate_x, candidate_k = self._encode(candidate_x)

        x, k = self._encode(x)  # encoded x and k
        batch_size, d_main = k.shape
        device = k.device
        context_size = self.context_size

        if self.search_index is None:
            self.search_index = (
                TabR.faiss.GpuIndexFlatL2(TabR.faiss.StandardGpuResources(), d_main)  # type: ignore[union-attr]
                if device.type == "cuda"
                else TabR.faiss.IndexFlatL2(d_main)  # type: ignore[union-attr]
            )

        # Updating the index is much faster than creating a new one.
        self.search_index.reset()
        self.search_index.add(candidate_k.to(torch.float32))  # type: ignore[code]
        distances: Tensor
        context_idx: Tensor
        distances, context_idx = self.search_index.search(  # type: ignore[code]
            k.to(torch.float32), context_size
        )

        context_k = candidate_k[context_idx]
        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        if self.hparams.num_classes > 1:  # for classification
            context_y_emb = self.label_encoder(candidate_y[context_idx][..., None].long())
        else:  # for regression
            context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
            if len(context_y_emb.shape) == 4:
                context_y_emb = context_y_emb[:, :, 0, :]

        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # Predictor has LayerNorm, ReLU and Linear after the N_P number of blocks.
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x

    def predict_with_candidates(self, *data, candidate_x, candidate_y):
        """Prediction forward pass with TabR-style candidate selection."""
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

        if not self.memory_efficient:
            candidate_k = (
                self._encode(candidate_x)[1]  # normalized candidate_x
                if self.candidate_encoding_batch_size == 0
                else torch.cat(
                    [
                        self._encode(x)[1]  # normalized x
                        for x in TabR.delu.iter_batches(candidate_x, self.candidate_encoding_batch_size)  # type: ignore[union-attr]
                    ]
                )
            )
        else:
            candidate_x, candidate_k = self._encode(candidate_x)

        x, k = self._encode(x)  # encoded x and k
        batch_size, d_main = k.shape
        device = k.device
        context_size = self.context_size

        if self.search_index is None:
            self.search_index = (
                TabR.faiss.GpuIndexFlatL2(TabR.faiss.StandardGpuResources(), d_main)  # type: ignore[union-attr]
                if device.type == "cuda"
                else TabR.faiss.IndexFlatL2(d_main)  # type: ignore[union-attr]
            )

        # Updating the index is much faster than creating a new one.
        self.search_index.reset()
        self.search_index.add(candidate_k.to(torch.float32))  # type: ignore[code]
        distances: Tensor
        context_idx: Tensor
        distances, context_idx = self.search_index.search(  # type: ignore[code]
            k.to(torch.float32), context_size
        )

        context_k = candidate_k[context_idx]
        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        if self.hparams.num_classes > 1:  # for classification
            context_y_emb = self.label_encoder(candidate_y[context_idx][..., None].long())
        else:  # for regression
            context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
            if len(context_y_emb.shape) == 4:
                context_y_emb = context_y_emb[:, :, 0, :]

        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # Predictor has LayerNorm, ReLU and Linear after the N_P number of blocks.
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x
