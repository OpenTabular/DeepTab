from itertools import chain

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelSummary


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
        self.temperature = temperature
        self.lr = lr
        self.regression = regression
        self.margin = margin
        self.use_positive = use_positive
        self.use_negative = use_negative
        self.pool_sequence = pool_sequence
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin, reduction="mean")

    def forward(self, x):
        x = self.estimator.encode(x, grad=True)
        if self.pool_sequence:
            return self.estimator.pool_sequence(x)
        return x  # Return unpooled sequence embeddings (N, S, D)

    def get_knn(self, labels):
        batch_size = labels.size(0)
        k_neighbors = min(self.k_neighbors, batch_size - 1)

        knn_indices = torch.zeros(batch_size, k_neighbors, dtype=torch.long)
        neg_indices = torch.zeros(batch_size, k_neighbors, dtype=torch.long)

        if not self.regression:
            for i in range(batch_size):
                same_class_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]
                different_class_indices = (labels != labels[i]).nonzero(as_tuple=True)[0]
                same_class_indices = same_class_indices[same_class_indices != i]

                knn_indices[i] = self._sample_indices(same_class_indices, k_neighbors)  # type: ignore[reportCallIssue]
                neg_indices[i] = self._sample_indices(different_class_indices, k_neighbors)  # type: ignore[reportCallIssue]
        else:
            with torch.no_grad():
                target_distances = torch.cdist(labels.float(), labels.float(), p=2).squeeze(-1)

            knn_indices = target_distances.topk(k_neighbors + 1, largest=False).indices[:, 1:]
            neg_indices = target_distances.topk(k_neighbors, largest=True).indices[:, :k_neighbors]

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
                labels = []

                if self.use_positive:
                    pairs.append(positive_pairs.view(-1, D))  # type: ignore[union-attr]
                    labels.append(torch.ones(N * k_neighbors, device=self.device))
                if self.use_negative:
                    pairs.append(negative_pairs.view(-1, D))  # type: ignore[union-attr]
                    labels.append(-torch.ones(N * k_neighbors, device=self.device))

                if not pairs:
                    raise ValueError("At least one of use_positive or use_negative must be True.")

                all_pairs = torch.cat(pairs, dim=0)
                all_labels = torch.cat(labels, dim=0)

                embeddings_s = embs.repeat_interleave(k_neighbors * len(pairs), dim=0)
                _loss = self.loss_fn(embeddings_s, all_pairs, all_labels)
                loss += _loss

            return loss

        else:
            N, D = embeddings.shape
            k_neighbors = knn_indices.shape[1]
            embeddings = F.normalize(embeddings, p=2, dim=-1)

            positive_pairs = embeddings[knn_indices] if self.use_positive else None
            negative_pairs = embeddings[neg_indices] if self.use_negative else None

            pairs = []
            labels = []

            if self.use_positive:
                pairs.append(positive_pairs.view(-1, D))  # type: ignore[union-attr]
                labels.append(torch.ones(N * k_neighbors, device=self.device))
            if self.use_negative:
                pairs.append(negative_pairs.view(-1, D))  # type: ignore[union-attr]
                labels.append(-torch.ones(N * k_neighbors, device=self.device))

            if not pairs:
                raise ValueError("At least one of use_positive or use_negative must be True.")

            all_pairs = torch.cat(pairs, dim=0)
            all_labels = torch.cat(labels, dim=0)

            embeddings_s = embeddings.repeat_interleave(k_neighbors * len(pairs), dim=0)
            loss = self.loss_fn(embeddings_s, all_pairs, all_labels)
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
        params = chain(self.estimator.parameters())
        return torch.optim.Adam(params, lr=self.lr)


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
    print("🚀 Pretraining embeddings...")
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
    print(f"✅ Embeddings saved to {save_path}")
