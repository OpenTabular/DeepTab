from __future__ import annotations

import warnings

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelSummary

from deeptab.core.exceptions import ArchitectureRequirementError


def _validate_pretrainable_model(
    model: object,
    *,
    pool_sequence: bool,
    save_embeddings: bool,
) -> None:
    """Check that *model* has the interface required for contrastive pretraining.

    Parameters
    ----------
    model:
        The architecture instance to validate.
    pool_sequence:
        Whether sequence pooling will be used during pretraining.
    save_embeddings:
        Whether the pretrainer will call ``get_embedding_state_dict()`` at the end.

    Raises
    ------
    ArchitectureRequirementError
        If any required method or attribute is missing.
    """
    missing = []
    if not hasattr(model, "embedding_layer"):
        missing.append("embedding_layer (attribute)")
    if not hasattr(model, "encode"):
        missing.append("encode() method")
    if pool_sequence and not hasattr(model, "pool_sequence"):
        missing.append("pool_sequence() method (required when pool_sequence=True)")
    if save_embeddings and not hasattr(model, "get_embedding_state_dict"):
        missing.append("get_embedding_state_dict() method (required to save embeddings)")

    if missing:
        raise ArchitectureRequirementError(
            "This architecture does not support contrastive pretraining.\n"
            "Missing:\n" + "\n".join(f"  \u2022 {m}" for m in missing) + "\n"
            "Suggestion: use an architecture with embedding layers "
            "(e.g. TabTransformerClassifier, FTTransformerClassifier, MambularClassifier)."
        )


class ContrastivePretrainer(pl.LightningModule):
    def __init__(
        self,
        base_model,
        k_neighbors=5,
        temperature=0.1,
        lr=1e-4,
        regression=True,
        margin=0.5,
        use_positive=True,
        use_negative=True,
        pool_sequence=True,
    ):
        super().__init__()
        self.estimator = base_model
        self.estimator.eval()
        self.k_neighbors = k_neighbors
        self.lr = lr
        self.regression = regression
        self.margin = margin
        self.use_positive = use_positive
        self.use_negative = use_negative
        self.pool_sequence = pool_sequence
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin, reduction="mean")

        if temperature != 0.1:
            warnings.warn(
                "ContrastivePretrainer: temperature is not used with CosineEmbeddingLoss "
                "and has no effect. Set objective='infonce' to use temperature-scaled "
                "contrastive loss (future feature).",
                FutureWarning,
                stacklevel=2,
            )
        self.temperature = temperature

    def _sample_indices(self, indices: torch.Tensor, k: int) -> torch.Tensor:
        """Sample *k* entries from *indices*, with replacement when ``len < k``.

        When *indices* is empty (single-class batch) an empty tensor is returned
        and the caller is responsible for handling that case.

        Parameters
        ----------
        indices:
            1-D tensor of candidate indices.
        k:
            Number of indices to return.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(k,)`` drawn from *indices*, or an empty tensor
            when *indices* is empty.
        """
        n = indices.numel()
        if n == 0:
            return indices  # caller must handle the empty case
        if n >= k:
            perm = torch.randperm(n, device=indices.device)[:k]
            return indices[perm]
        # With replacement to fill the deficit
        extra = torch.randint(n, (k - n,), device=indices.device)
        return torch.cat([indices, indices[extra]])

    def forward(self, x):
        x = self.estimator.encode(x, grad=True)
        if self.pool_sequence:
            return self.estimator.pool_sequence(x)
        return x  # Return unpooled sequence embeddings (N, S, D)

    def get_knn(self, labels):
        batch_size = labels.size(0)
        k_neighbors = min(self.k_neighbors, batch_size - 1)

        if not self.regression:
            knn_indices_list = []
            neg_indices_list = []

            for i in range(batch_size):
                pos = (labels == labels[i]).nonzero(as_tuple=True)[0]
                neg = (labels != labels[i]).nonzero(as_tuple=True)[0]
                pos = pos[pos != i]

                knn_indices_list.append(self._sample_indices(pos, k_neighbors))
                neg_indices_list.append(self._sample_indices(neg, k_neighbors))

            # Filter out samples where either positive or negative set was empty
            valid = [
                i for i in range(batch_size) if knn_indices_list[i].numel() > 0 and neg_indices_list[i].numel() > 0
            ]
            if not valid:
                raise ValueError(
                    "Contrastive pretraining: every sample in this batch has either "
                    "no same-class or no different-class neighbors. "
                    "Use a larger batch size or stratified sampling."
                )
            knn_indices = torch.stack([knn_indices_list[i] for i in valid])
            neg_indices = torch.stack([neg_indices_list[i] for i in valid])
        else:
            with torch.no_grad():
                target_distances = torch.cdist(labels.float(), labels.float(), p=2).squeeze(-1)
            knn_indices = target_distances.topk(k_neighbors + 1, largest=False).indices[:, 1:]
            neg_indices = target_distances.topk(k_neighbors, largest=True).indices

        return knn_indices.to(self.device), neg_indices.to(self.device)

    def contrastive_loss(self, embeddings, knn_indices, neg_indices):
        if not self.pool_sequence:
            N, S, D = embeddings.shape
            loss = 0.0
            for i in range(S):
                embs = embeddings[:, i, :]
                k_neighbors = knn_indices.shape[1]
                embs = F.normalize(embs, p=2, dim=-1)

                positive_pairs = embs[knn_indices] if self.use_positive else None
                negative_pairs = embs[neg_indices] if self.use_negative else None

                pairs = []
                pair_labels = []

                if self.use_positive:
                    pairs.append(positive_pairs.view(-1, D))  # type: ignore[union-attr]
                    pair_labels.append(torch.ones(N * k_neighbors, device=self.device))
                if self.use_negative:
                    pairs.append(negative_pairs.view(-1, D))  # type: ignore[union-attr]
                    pair_labels.append(-torch.ones(N * k_neighbors, device=self.device))

                if not pairs:
                    raise ValueError("At least one of use_positive or use_negative must be True.")

                all_pairs = torch.cat(pairs, dim=0)
                all_pair_labels = torch.cat(pair_labels, dim=0)

                embeddings_s = embs.repeat_interleave(k_neighbors * len(pairs), dim=0)
                _loss = self.loss_fn(embeddings_s, all_pairs, all_pair_labels)
                loss += _loss

            return loss

        else:
            N, D = embeddings.shape
            k_neighbors = knn_indices.shape[1]
            embeddings = F.normalize(embeddings, p=2, dim=-1)

            positive_pairs = embeddings[knn_indices] if self.use_positive else None
            negative_pairs = embeddings[neg_indices] if self.use_negative else None

            pairs = []
            pair_labels = []

            if self.use_positive:
                pairs.append(positive_pairs.view(-1, D))  # type: ignore[union-attr]
                pair_labels.append(torch.ones(N * k_neighbors, device=self.device))
            if self.use_negative:
                pairs.append(negative_pairs.view(-1, D))  # type: ignore[union-attr]
                pair_labels.append(-torch.ones(N * k_neighbors, device=self.device))

            if not pairs:
                raise ValueError("At least one of use_positive or use_negative must be True.")

            all_pairs = torch.cat(pairs, dim=0)
            all_pair_labels = torch.cat(pair_labels, dim=0)

            embeddings_s = embeddings.repeat_interleave(k_neighbors * len(pairs), dim=0)
            loss = self.loss_fn(embeddings_s, all_pairs, all_pair_labels)
            return loss

    def training_step(self, batch, batch_idx):
        self.estimator.embedding_layer.train()

        data, labels = batch
        embeddings = self(data)
        knn_indices, neg_indices = self.get_knn(labels)
        loss = self.contrastive_loss(embeddings, knn_indices, neg_indices)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, labels = batch
        embeddings = self(data)
        knn_indices, neg_indices = self.get_knn(labels)
        loss = self.contrastive_loss(embeddings, knn_indices, neg_indices)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        embeddings = self(data)
        knn_indices, neg_indices = self.get_knn(labels)
        loss = self.contrastive_loss(embeddings, knn_indices, neg_indices)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.estimator.parameters(), lr=self.lr)


def pretrain_embeddings(
    base_model,
    train_dataloader,
    pretrain_epochs=5,
    k_neighbors=5,
    temperature=0.1,
    save_path="pretrained_embeddings.pth",
    regression=True,
    lr=1e-3,
    use_positive=True,
    use_negative=True,
    pool_sequence=True,
):
    _validate_pretrainable_model(
        base_model,
        pool_sequence=pool_sequence,
        save_embeddings=True,
    )

    print("Pretraining embeddings...")
    model = ContrastivePretrainer(
        base_model=base_model,
        k_neighbors=k_neighbors,
        temperature=temperature,
        lr=lr,
        regression=regression,
        use_positive=use_positive,
        use_negative=use_negative,
        pool_sequence=pool_sequence,
    )

    trainer = pl.Trainer(
        max_epochs=pretrain_epochs,
        enable_progress_bar=True,
        callbacks=[
            ModelSummary(max_depth=2),
        ],
    )
    model.train()
    trainer.fit(model, train_dataloader)

    torch.save(base_model.get_embedding_state_dict(), save_path)
    print(f"Embeddings saved to {save_path}")
