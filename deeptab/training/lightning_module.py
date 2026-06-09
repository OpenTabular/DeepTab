from collections.abc import Callable

import lightning as pl
import torch
import torch.nn as nn
from tqdm import tqdm

from deeptab.training.optimizers import build_optimizer, normalize_optimizer_kwargs
from deeptab.training.schedulers import build_scheduler


class TaskModel(pl.LightningModule):
    """PyTorch Lightning module that wraps any DeepTab estimator for training.

    ``TaskModel`` is the bridge between a DeepTab architecture (an
    ``nn.Module`` subclass) and PyTorch Lightning's training loop.  It is
    constructed automatically by :meth:`~deeptab.models.base.SklearnBase._build_model`
    and is not normally instantiated directly by end-users.

    Responsibilities
    ----------------
    * Instantiates the model (``self.estimator``) from *model_class* and
      *config*.
    * Selects the default loss function based on *num_classes* / *lss* when
      no *loss_fct* is supplied.
    * Runs training, validation, test, and prediction steps with per-step
      metric logging.
    * Wires the optimizer via :func:`~deeptab.training.optimizers.build_optimizer`
      and the LR scheduler via
      :func:`~deeptab.training.schedulers.build_scheduler`, both of which
      are registry-backed and fully extensible.
    * Supports early-pruning of Optuna trials via *early_pruning_threshold*.

    Parameters
    ----------
    model_class : type[nn.Module]
        Architecture class to instantiate (e.g. ``ResNetModel``).
    config : dataclass
        Architecture configuration dataclass (e.g. ``ResNetConfig``).
    feature_information : tuple
        Three-tuple ``(num_feature_info, cat_feature_info,
        embedding_feature_info)`` produced by
        :class:`~deeptab.data.TabularDataModule`.
    num_classes : int, default=1
        Number of output targets.

        * ``1``  — regression (``MSELoss``).
        * ``2``  — binary classification (``BCEWithLogitsLoss``; model outputs
          a single logit).
        * ``>2`` — multi-class classification (``CrossEntropyLoss``).
    lss : bool, default=False
        When ``True``, the task is distributional (LSS / ``Family``-based)
        and the loss is managed by the *family* object rather than
        ``loss_fct``.
    family : Family or None, default=None
        Distributional family for LSS regression.  Only used when
        *lss* is ``True``.
    loss_fct : callable or None, default=None
        Custom loss function overriding the automatic selection.  Must
        accept ``(predictions, targets)`` and return a scalar tensor.
    early_pruning_threshold : float or None, default=None
        If set, training is stopped once ``val_loss`` exceeds this value
        after *pruning_epoch* epochs (used by Optuna pruners).
    pruning_epoch : int, default=5
        Epoch after which early-pruning logic is applied.
    optimizer_type : str, default="Adam"
        Registered optimizer name.  See
        :func:`~deeptab.training.optimizers.available_optimizers`.
    optimizer_args : dict or None, default=None
        Legacy optimizer kwargs with optional ``"optimizer_"`` prefix
        (e.g. ``{"optimizer_betas": (0.9, 0.95)}``).  Normalised
        automatically via
        :func:`~deeptab.training.optimizers.normalize_optimizer_kwargs`.
    train_metrics : dict[str, Callable] or None, default=None
        Extra metrics to log during training steps.  Keys become the log
        names (prefixed with ``"train_"``).
    val_metrics : dict[str, Callable] or None, default=None
        Extra metrics to log during validation steps (prefixed with
        ``"val_"``).
    lr : float or None, default=None
        Learning rate.  Falls back to ``config.lr`` when ``None``.
    lr_patience : int or None, default=None
        Epochs without improvement before the LR is reduced (used by
        ``ReduceLROnPlateau``).  Falls back to ``config.lr_patience``.
    lr_factor : float or None, default=None
        Multiplicative LR reduction factor.  Falls back to
        ``config.lr_factor``.
    weight_decay : float or None, default=None
        L2 regularisation coefficient.  Falls back to
        ``config.weight_decay``.
    scheduler_type : str or None, default="ReduceLROnPlateau"
        Registered scheduler name or ``None`` to disable.  See
        :func:`~deeptab.training.schedulers.available_schedulers`.
    scheduler_kwargs : dict or None, default=None
        Extra kwargs forwarded to the scheduler constructor.  For
        ``ReduceLROnPlateau``, ``"factor"`` and ``"patience"`` are
        synthesised from *lr_factor* / *lr_patience* when absent.
    monitor : str, default="val_loss"
        Metric monitored by the scheduler (and passed to Lightning so that
        ``ReduceLROnPlateau`` receives the correct value).  Should match
        ``TrainerConfig.monitor``.
    mode : str, default="min"
        ``"min"`` or ``"max"``.  Forwarded to ``ReduceLROnPlateau`` so
        the scheduler and early stopping always track the same direction.
    scheduler_interval : str, default="epoch"
        Lightning scheduling granularity: ``"epoch"`` or ``"step"``.
    scheduler_frequency : int, default=1
        How often to step the scheduler at the given interval.
    no_weight_decay_for_bias_and_norm : bool, default=False
        When ``True``, bias and normalisation-layer parameters receive
        zero weight decay.  Recommended for transformer-style models.
    **kwargs
        Forwarded to *model_class* constructor.

    Attributes
    ----------
    estimator : nn.Module
        The instantiated model architecture.
    val_losses : list of float
        Validation loss recorded at the end of each epoch.

    Examples
    --------
    ``TaskModel`` is normally created via the sklearn-compatible API::

        from deeptab.models import MLP
        from deeptab.configs import TrainerConfig

        model = MLP(trainer_config=TrainerConfig(optimizer_type="AdamW", lr=3e-4))
        model.fit(X_train, y_train)

    For advanced use (e.g. custom Lightning ``Trainer``)::

        from deeptab.training import TaskModel
        from deeptab.architectures import ResNetModel
        from deeptab.configs import ResNetConfig

        task_model = TaskModel(
            model_class=ResNetModel,
            config=ResNetConfig(d_model=64),
            feature_information=(num_info, cat_info, emb_info),
            num_classes=1,
            optimizer_type="AdamW",
            lr=1e-3,
            weight_decay=1e-2,
            no_weight_decay_for_bias_and_norm=True,
            scheduler_type="CosineAnnealingLR",
            scheduler_kwargs={"T_max": 100},
        )

    Notes
    -----
    ``configure_optimizers`` returns either a bare optimizer (when
    *scheduler_type* is ``None``) or the dict
    ``{"optimizer": ..., "lr_scheduler": ...}`` expected by Lightning.

    See Also
    --------
    :class:`~deeptab.configs.TrainerConfig` : All training hyper-parameters
        that feed into ``TaskModel``.
    :func:`~deeptab.training.optimizers.build_optimizer` : Optimizer factory.
    :func:`~deeptab.training.schedulers.build_scheduler` : Scheduler factory.
    :func:`~deeptab.training.losses.build_default_task_loss` : Default loss
        selection logic.
    """

    def __init__(
        self,
        model_class: type[nn.Module],
        config,
        feature_information,
        num_classes=1,
        lss=False,
        family=None,
        loss_fct: Callable | None = None,
        early_pruning_threshold=None,
        pruning_epoch=5,
        optimizer_type: str = "Adam",
        optimizer_args: dict | None = None,
        train_metrics: dict[str, Callable] | None = None,
        val_metrics: dict[str, Callable] | None = None,
        lr: float | None = None,
        lr_patience: int | None = None,
        lr_factor: float | None = None,
        weight_decay: float | None = None,
        scheduler_type: str | None = "ReduceLROnPlateau",
        scheduler_kwargs: dict | None = None,
        monitor: str = "val_loss",
        mode: str = "min",
        scheduler_interval: str = "epoch",
        scheduler_frequency: int = 1,
        no_weight_decay_for_bias_and_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.optimizer_type = optimizer_type
        self.num_classes = num_classes
        self.lss = lss
        self.family = family
        self.loss_fct = loss_fct
        self.early_pruning_threshold = early_pruning_threshold
        self.pruning_epoch = pruning_epoch
        self.val_losses = []

        # Store custom metrics
        self.train_metrics = train_metrics or {}
        self.val_metrics = val_metrics or {}

        # Scheduler / monitoring config
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs
        self.monitor = monitor
        self.mode = mode
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency
        self.no_weight_decay_for_bias_and_norm = no_weight_decay_for_bias_and_norm

        # Normalize legacy optimizer kwargs (strips "optimizer_" prefix; handles None)
        self.optimizer_params = normalize_optimizer_kwargs(optimizer_args)

        if lss:
            pass
        else:
            if num_classes == 2:
                if not self.loss_fct:
                    self.loss_fct = nn.BCEWithLogitsLoss()
                self.num_classes = 1
            elif num_classes > 2:
                if not self.loss_fct:
                    self.loss_fct = nn.CrossEntropyLoss()
            else:
                self.loss_fct = nn.MSELoss()

        self.save_hyperparameters(ignore=["model_class", "loss_fct", "family"])

        self.lr = lr if lr is not None else getattr(config, "lr", 1e-4)
        self.lr_patience = lr_patience if lr_patience is not None else getattr(config, "lr_patience", 10)
        self.weight_decay = weight_decay if weight_decay is not None else getattr(config, "weight_decay", 1e-6)
        self.lr_factor = lr_factor if lr_factor is not None else getattr(config, "lr_factor", 0.1)

        if family is None and num_classes == 2:
            output_dim = 1
        else:
            output_dim = num_classes

        self.estimator = model_class(
            config=config,
            feature_information=feature_information,
            num_classes=output_dim,
            lss=lss,
            **kwargs,
        )

    def setup(self, stage=None):
        if stage == "fit" and hasattr(self.estimator, "uses_candidates"):
            all_train_num = []
            all_train_cat = []
            all_train_embeddings = []
            all_train_targets = []

            device = self.device if hasattr(self, "device") else self.trainer.device  # type: ignore[attr-defined]

            for batch in self.trainer.datamodule.train_dataloader():  # type: ignore[attr-defined]
                (num_features, cat_features, embeddings), labels = batch

                all_train_num.append([f.to(device) for f in num_features])  # Keep lists
                all_train_cat.append([f.to(device) for f in cat_features])  # Keep lists
                if embeddings is not None:
                    all_train_embeddings.append([f.to(device) for f in embeddings])
                all_train_targets.append(labels.to(device))

            # Maintain structure: each feature type remains a list of tensors
            self.train_features = (
                [torch.cat(features, dim=0) for features in zip(*all_train_num, strict=False)],
                [torch.cat(features, dim=0) for features in zip(*all_train_cat, strict=False)],
                (
                    [torch.cat(features, dim=0) for features in zip(*all_train_embeddings, strict=False)]
                    if all_train_embeddings
                    else None
                ),
            )
            self.train_targets = torch.cat(all_train_targets, dim=0)

    def forward(self, num_features, cat_features, embeddings):
        """Forward pass through the model.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the model's forward method.
        **kwargs : dict
            Keyword arguments passed to the model's forward method.

        Returns
        -------
        Tensor
            Model output.
        """

        return self.estimator.forward(num_features, cat_features, embeddings)

    def compute_loss(self, predictions, y_true):
        """Compute the loss for the given predictions and true labels.

        Parameters
        ----------
        predictions : Tensor
            Model predictions. Shape: (batch_size, k, output_dim) for ensembles, or (batch_size, output_dim) otherwise.
        y_true : Tensor
            True labels. Shape: (batch_size, output_dim).

        Returns
        -------
        Tensor
            Computed loss.
        """
        if self.lss:
            if getattr(self.estimator, "returns_ensemble", False):
                loss = 0.0
                for ensemble_member in range(predictions.shape[1]):
                    loss += self.family.compute_loss(  # type: ignore
                        predictions[:, ensemble_member], y_true.squeeze(-1)
                    )
                return loss
            else:
                return self.family.compute_loss(  # type: ignore
                    predictions,
                    y_true.squeeze(-1),
                )

        if getattr(self.estimator, "returns_ensemble", False):  # Ensemble case
            expects_class_indices = getattr(
                self.loss_fct,
                "expects_class_indices",
                self.loss_fct.__class__.__name__ == "CrossEntropyLoss",
            )
            if expects_class_indices and predictions.dim() == 3:
                # Classification case with ensemble: predictions (N, E, k), y_true (N,)
                _, E, _ = predictions.shape
                loss = 0.0
                for ensemble_member in range(E):
                    loss += self.loss_fct(
                        predictions[
                            :,  # type: ignore
                            ensemble_member,
                            :,
                        ],
                        y_true,
                    )
                return loss

            else:
                # Regression case with ensemble (e.g., MSE) or other compatible losses
                y_true_expanded = y_true.expand_as(predictions)
                return self.loss_fct(
                    predictions,  # type: ignore
                    y_true_expanded,
                )
        else:
            # Non-ensemble case
            return self.loss_fct(predictions, y_true)  # type: ignore

    def training_step(self, batch, batch_idx):  # type: ignore
        """Training step for a single batch, incorporating penalty if the model has a penalty_forward method.

        Parameters
        ----------
        batch : tuple
            Batch of data containing numerical features, categorical features, and labels.
        batch_idx : int
            Index of the batch.

        Returns
        ------
        Tensor
            Training loss.
        """
        data, labels = batch

        # Check if the model has a `penalty_forward` method
        if hasattr(self.estimator, "penalty_forward"):
            preds, penalty = self.estimator.penalty_forward(*data)  # type: ignore[reportCallIssue]
            loss = self.compute_loss(preds, labels) + penalty
        elif hasattr(self.estimator, "train_with_candidates"):
            preds = self.estimator.train_with_candidates(  # type: ignore[reportCallIssue]
                *data,
                targets=labels,
                candidate_x=self.train_features,
                candidate_y=self.train_targets,
            )
            loss = self.compute_loss(preds, labels)
        else:
            preds = self(*data)
            loss = self.compute_loss(preds, labels)

        # Log the training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log custom training metrics
        if self.train_metrics:
            # Apply distribution transforms so metrics receive meaningful parameters,
            # not raw logits.  Metrics with needs_raw=True still receive raw preds.
            if self.lss and self.family is not None:
                preds_transformed = self.family(preds)
            else:
                preds_transformed = preds
            for metric_name, metric_fn in self.train_metrics.items():
                needs_raw = getattr(metric_fn, "needs_raw", False)
                metric_value = metric_fn(preds if needs_raw else preds_transformed, labels)
                self.log(
                    f"train_{metric_name}",
                    metric_value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore
        """Validation step for a single batch.

        Parameters
        ----------
        batch : tuple
            Batch of data containing numerical features, categorical features, and labels.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Tensor
            Validation loss.
        """

        data, labels = batch
        if hasattr(self.estimator, "validate_with_candidates") and self.train_features is not None:
            preds = self.estimator.validate_with_candidates(  # type: ignore[reportCallIssue]
                *data, candidate_x=self.train_features, candidate_y=self.train_targets
            )
        else:
            preds = self(*data)
        val_loss = self.compute_loss(preds, labels)

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Log custom validation metrics
        if self.val_metrics:
            # Apply distribution transforms so metrics receive meaningful parameters,
            # not raw logits.  Metrics with needs_raw=True still receive raw preds.
            if self.lss and self.family is not None:
                preds_transformed = self.family(preds)
            else:
                preds_transformed = preds
            for metric_name, metric_fn in self.val_metrics.items():
                needs_raw = getattr(metric_fn, "needs_raw", False)
                metric_value = metric_fn(preds if needs_raw else preds_transformed, labels)
                self.log(
                    f"val_{metric_name}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        return val_loss

    def test_step(self, batch, batch_idx):  # type: ignore
        """Test step for a single batch.

        Parameters
        ----------
        batch : tuple
            Batch of data containing numerical features, categorical features, and labels.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Tensor
            Test loss.
        """
        data, labels = batch
        if hasattr(self.estimator, "predict_with_candidates") and self.train_features is not None:
            preds = self.estimator.predict_with_candidates(  # type: ignore[reportCallIssue]
                *data, candidates_x=self.train_features, candidates_y=self.train_targets
            )
        else:
            preds = self(*data)
        test_loss = self.compute_loss(preds, labels)

        self.log(
            "test_loss",
            test_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return test_loss

    def predict_step(self, batch, batch_idx):
        """Predict step for a single batch.

        Parameters
        ----------
        batch : tuple
            Batch of data containing numerical features, categorical features, and labels.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Tensor
            Predictions.
        """
        if hasattr(self.estimator, "predict_with_candidates") and self.train_features is not None:
            preds = self.estimator.predict_with_candidates(  # type: ignore[reportCallIssue]
                *batch,
                candidate_x=self.train_features,
                candidate_y=self.train_targets,
            )
        else:
            preds = self(*batch)

        return preds

    def on_validation_epoch_end(self):
        """Callback executed at the end of each validation epoch.

        This method retrieves the current validation loss from the trainer's callback metrics
        and stores it in a list for tracking validation losses across epochs. It also applies
        pruning logic to stop training early if the validation loss exceeds a specified threshold.

        Parameters
        ----------
        None

        Attributes
        ----------
        val_loss : torch.Tensor or None
            The validation loss for the current epoch, retrieved from `self.trainer.callback_metrics`.
        val_loss_value : float
            The validation loss for the current epoch, converted to a float.
        val_losses : list of float
            A list storing the validation losses for each epoch.
        pruning_epoch : int
            The epoch after which pruning logic will be applied.
        early_pruning_threshold : float, optional
            The threshold for early pruning based on validation loss. If the current validation
            loss exceeds this value, training will be stopped early.

        Notes
        -----
        If the current epoch is greater than or equal to `pruning_epoch`, and the validation
        loss exceeds the `early_pruning_threshold`, the training is stopped early by setting
        `self.trainer.should_stop` to True.
        """
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            val_loss_value = val_loss.item()
            # Store val_loss for each epoch
            self.val_losses.append(val_loss_value)

            # Apply pruning logic if needed
            if self.current_epoch >= self.pruning_epoch:
                if self.early_pruning_threshold is not None and val_loss_value > self.early_pruning_threshold:
                    print(f"Pruned at epoch {self.current_epoch}, val_loss {val_loss_value}")
                    self.trainer.should_stop = True  # Stop training early

    def epoch_val_loss_at(self, epoch):
        """Retrieve the validation loss at a specific epoch.

        This method allows the user to query the validation loss for any given epoch,
        provided the epoch exists within the range of completed epochs. If the epoch
        exceeds the length of the `val_losses` list, a default value of infinity is returned.

        Parameters
        ----------
        epoch : int
            The epoch number for which the validation loss is requested.

        Returns
        -------
        float
            The validation loss for the requested epoch. If the epoch does not exist,
            the method returns `float("inf")`.

        Notes
        -----
        This method relies on `self.val_losses` which stores the validation loss values
        at the end of each epoch during training.
        """
        if epoch < len(self.val_losses):
            return self.val_losses[epoch]
        else:
            return float("inf")

    def configure_optimizers(self):  # type: ignore
        """Sets up the model's optimizer and learning rate scheduler.

        Uses the :mod:`deeptab.training.optimizers` and
        :mod:`deeptab.training.schedulers` registries so that:

        - Unknown optimizer / scheduler names raise :class:`~deeptab.core.exceptions.InvalidParamError`
          immediately with a helpful list of alternatives.
        - ``monitor`` and ``mode`` are passed through to ``ReduceLROnPlateau``
          so it follows the same metric and direction as early stopping.
        - ``no_weight_decay_for_bias_and_norm`` selectively exempts bias and
          normalisation parameters from weight decay.
        """
        optimizer = build_optimizer(
            self.estimator,
            optimizer_type=self.optimizer_type,
            lr=self.lr,
            weight_decay=self.weight_decay,
            optimizer_kwargs=self.optimizer_params,
            no_weight_decay_for_bias_and_norm=self.no_weight_decay_for_bias_and_norm,
        )

        scheduler_cfg = build_scheduler(
            optimizer,
            scheduler_type=self.scheduler_type,
            scheduler_kwargs=self.scheduler_kwargs,
            lr_factor=self.lr_factor,
            lr_patience=self.lr_patience,
            monitor=self.monitor,
            mode=self.mode,
            interval=self.scheduler_interval,
            frequency=self.scheduler_frequency,
        )

        if scheduler_cfg is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler_cfg}

    def pretrain_embeddings(
        self,
        train_dataloader,
        pretrain_epochs=5,
        k_neighbors=5,
        temperature=0.1,
        save_path="pretrained_embeddings.pth",
        regression=True,
        lr=1e-04,
    ):
        """Pretrain embeddings before full model training.

        Parameters
        ----------
        train_dataloader : DataLoader
            Training dataloader for embedding pretraining.
        pretrain_epochs : int, default=5
            Number of epochs for pretraining the embeddings.
        k_neighbors : int, default=5
            Number of nearest neighbors for positive samples in contrastive learning.
        temperature : float, default=0.1
            Temperature parameter for contrastive loss.
        save_path : str, default="pretrained_embeddings.pth"
            Path to save the pretrained embeddings.
        """
        print("🚀 Pretraining embeddings...")
        self.estimator.train()

        optimizer = torch.optim.Adam(self.estimator.embedding_parameters(), lr=lr)  # type: ignore[reportCallIssue]

        # 🔥 Single tqdm progress bar across all epochs and batches
        total_batches = pretrain_epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_batches, desc="Pretraining", unit="batch")

        for epoch in range(pretrain_epochs):
            total_loss = 0.0

            for batch in train_dataloader:
                data, labels = batch
                optimizer.zero_grad()

                # Forward pass through embeddings only
                embeddings = self.estimator.encode(data, grad=True)  # type: ignore[reportCallIssue]

                # Compute nearest neighbors based on task type
                knn_indices = self.get_knn(labels, k_neighbors, regression)

                # Compute contrastive loss
                loss = self.contrastive_loss(embeddings, knn_indices, temperature)
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                total_loss += batch_loss

                # 🔥 Update tqdm progress bar with loss
                progress_bar.set_postfix(loss=batch_loss)
                progress_bar.update(1)

            avg_loss = total_loss / len(train_dataloader)

        progress_bar.close()

        # Save pretrained embeddings
        torch.save(self.estimator.get_embedding_state_dict(), save_path)  # type: ignore[reportCallIssue]
        print(f"✅ Embeddings saved to {save_path}")

    def get_knn(self, labels, k_neighbors=5, regression=True, device=""):
        """Finds k-nearest neighbors based on class labels (classification) or target distances (regression).

        Parameters
        ----------
        labels : Tensor
            Class labels (classification) or target values (regression) for the batch.
        k_neighbors : int, default=5
            Number of positive pairs to select.
        regression : bool, default=True
            If True, uses target similarity (Euclidean distance). If False, finds neighbors based on class labels.

        Returns
        -------
        Tensor
            Indices of positive samples for each instance.
        """
        batch_size = labels.size(0)

        # Ensure k_neighbors doesn't exceed available samples
        k_neighbors = min(k_neighbors, batch_size - 1)

        knn_indices = torch.zeros(batch_size, k_neighbors, dtype=torch.long, device=labels.device)

        if not regression:
            # Classification: Find samples with the same class label
            for i in range(batch_size):
                same_class_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]
                same_class_indices = same_class_indices[same_class_indices != i]  # Remove self-index

                if len(same_class_indices) >= k_neighbors:
                    knn_indices[i] = same_class_indices[torch.randperm(len(same_class_indices))[:k_neighbors]]
                else:
                    knn_indices[i, : len(same_class_indices)] = same_class_indices
                    knn_indices[i, len(same_class_indices) :] = same_class_indices[
                        torch.randint(
                            len(same_class_indices),
                            (k_neighbors - len(same_class_indices),),
                        )
                    ]

        else:
            # Regression: Find nearest neighbors using Euclidean distance
            with torch.no_grad():
                target_distances = torch.cdist(labels.float(), labels.float(), p=2).squeeze(-1)

            knn_indices = target_distances.topk(k_neighbors + 1, largest=False).indices[:, 1:]  # Exclude self

        return knn_indices

    def contrastive_loss(self, embeddings, knn_indices, temperature=0.1):
        """Computes contrastive loss per token position for embeddings (N, S, D) by looping over sequence axis (S).

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings with shape (N, S, D).
        knn_indices : Tensor
            Indices of k-nearest neighbors for each sample (N, k_neighbors).
        temperature : float, default=0.1
            Temperature parameter for softmax scaling.

        Returns
        -------
        Tensor
            Contrastive loss value.
        """
        _, S, D = embeddings.shape  # Batch size, sequence length, embedding dim
        k_neighbors = knn_indices.shape[1]  # Number of neighbors

        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)  # (N, S, D)

        loss = 0.0  # Accumulate loss across sequence steps
        loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction="mean")

        for s in range(S):  # Loop over sequence length
            embeddings_s = embeddings[:, s, :]  # Shape: (N, D) -> Single token per sample

            # Gather nearest neighbor embeddings for this time step
            positive_pairs = torch.gather(
                embeddings[:, s, :].unsqueeze(1).expand(-1, k_neighbors, -1),
                0,
                knn_indices.unsqueeze(-1).expand(-1, -1, D),
            )  # Shape: (N, k_neighbors, D)

            # Flatten batch and neighbors into a single batch dimension
            embeddings_s = embeddings_s.repeat_interleave(k_neighbors, dim=0)  # (N * k_neighbors, D)
            positive_pairs = positive_pairs.view(-1, D)  # (N * k_neighbors, D)

            # Labels: +1 for positive similarity
            labels = torch.ones(embeddings_s.shape[0], device=embeddings.device)  # Shape: (N * k_neighbors)

            # Compute cosine embedding loss
            loss += -1.0 * loss_fn(embeddings_s, positive_pairs, labels)

        # Average loss across all sequence steps
        loss /= S
        return loss
