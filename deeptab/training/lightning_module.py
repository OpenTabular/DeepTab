from collections.abc import Callable

import lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.trainer.states import TrainerFn
from torchmetrics import Metric as TorchMetric
from torchmetrics.classification import (
    BinaryCalibrationError,
    BinaryConfusionMatrix,
    BinaryPrecisionRecallCurve,
    BinaryStatScores,
    MulticlassCalibrationError,
    MulticlassConfusionMatrix,
    MulticlassPrecisionRecallCurve,
    MulticlassStatScores,
)
from torchmetrics.utilities.distributed import gather_all_tensors
from tqdm import tqdm

from deeptab.core.utils import check_numpy
from deeptab.data.dataset import TabularDataset
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
        Extra optimizer kwargs (e.g. ``{"betas": (0.9, 0.95)}``), forwarded
        from ``TrainerConfig.optimizer_kwargs``. Also accepts keys with the
        legacy ``"optimizer_"`` prefix. Normalised automatically via
        :func:`~deeptab.training.optimizers.normalize_optimizer_kwargs`.
    train_metrics : dict[str, Callable] or None, default=None
        Metrics computed over a complete training epoch. Keys become log
        names prefixed with ``"train_"``; these are not per-batch diagnostics.
        Built-in classification metrics receive probabilities; recognized
        binary TorchMetrics receive positive-class vectors and encoded targets.
        NumPy callables use ``(targets, predictions)``; TorchMetrics use
        ``(predictions, targets)`` on-device. Custom callables retain their
        existing inputs unless they declare ``input_space``. Declared ensembles
        average logits on member axis 1 before classification activation.
    val_metrics : dict[str, Callable] or None, default=None
        Metrics computed over a complete validation pass, prefixed with
        ``"val_"``. Sanity checks and repeated passes have separate state.
    metric_buffer_limit_bytes : int, default=67108864
        Maximum retained callable prediction/target bytes per stage, summed
        across metric names and distributed ranks. Inputs are detached CPU
        copies; concatenation and gathering require additional temporary
        memory. Exceeding the limit raises ``MemoryError`` instead of producing
        an approximate batch average. Native TorchMetrics manage their own
        accumulator memory and synchronization. This is a TaskModel option,
        not a Lightning Trainer or estimator fit argument.
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
        metric_buffer_limit_bytes: int = 64 * 1024 * 1024,
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
        self.train_features = None
        self.train_targets = None

        # Store custom metrics
        self.train_metrics = dict(train_metrics or {})
        self.val_metrics = dict(val_metrics or {})
        seen_metrics = set()
        for metrics in (self.train_metrics, self.val_metrics):
            for name, metric in metrics.items():
                if isinstance(metric, TorchMetric):
                    if id(metric) in seen_metrics:
                        metrics[name] = metric.clone()
                    seen_metrics.add(id(metric))
        if (
            not isinstance(metric_buffer_limit_bytes, int)
            or isinstance(metric_buffer_limit_bytes, bool)
            or metric_buffer_limit_bytes <= 0
        ):
            raise ValueError("metric_buffer_limit_bytes must be positive.")
        self.metric_buffer_limit_bytes = metric_buffer_limit_bytes
        self._metric_buffers: dict[str, dict[str, list[tuple[torch.Tensor, torch.Tensor]]]] = {"train": {}, "val": {}}
        self._metric_buffer_bytes = {"train": 0, "val": 0}
        # torchmetrics ``Metric`` objects hold internal state tensors. Register them
        # as submodules so Lightning moves that state to the training device;
        # otherwise the state stays on CPU and raises a device-mismatch error on
        # GPU/MPS. Plain-callable (DeepTabMetric-style) metrics carry no device
        # state and are left untouched.
        self._train_metric_modules = nn.ModuleDict(
            {name: metric for name, metric in self.train_metrics.items() if isinstance(metric, TorchMetric)}
        )
        self._val_metric_modules = nn.ModuleDict(
            {name: metric for name, metric in self.val_metrics.items() if isinstance(metric, TorchMetric)}
        )

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
                if not self.loss_fct:
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
            dataset = self.trainer.datamodule.train_dataset  # type: ignore[attr-defined]
            if not isinstance(dataset, TabularDataset) or dataset.labels is None:
                raise ValueError("Candidate-aware training requires a labeled TabularDataset.")
            dataset.return_indices = True
            self.train_features = (
                [feature.to(self.device) for feature in dataset.num_features_list],
                [feature.to(self.device) for feature in dataset.cat_features_list],
                [feature.to(self.device) for feature in dataset.embeddings_list]
                if dataset.embeddings_list is not None
                else None,
            )
            self.train_targets = dataset.labels.to(self.device)

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

    def _metric_inputs(self, predictions, labels, metric):
        """Adapt declared classification inputs without changing other task axes."""
        if getattr(metric, "needs_raw", False):
            return predictions, labels
        if self.lss:
            if self.family is None:
                return predictions, labels
            parameters = predictions.mean(dim=1) if getattr(self.estimator, "returns_ensemble", False) else predictions
            return self.family(parameters), labels
        binary_metric = isinstance(
            metric, (BinaryStatScores, BinaryPrecisionRecallCurve, BinaryCalibrationError, BinaryConfusionMatrix)
        )
        multiclass_metric = isinstance(
            metric,
            (
                MulticlassStatScores,
                MulticlassPrecisionRecallCurve,
                MulticlassCalibrationError,
                MulticlassConfusionMatrix,
            ),
        )
        input_space = getattr(metric, "input_space", None)
        if input_space is None and (binary_metric or multiclass_metric):
            input_space = "probabilities"
        if input_space is None or input_space == "logits":
            return predictions, labels
        if input_space not in {"probabilities", "labels"}:
            raise ValueError("Metric input_space must be 'logits', 'probabilities', 'labels', or None.")
        logits = predictions.mean(dim=1) if getattr(self.estimator, "returns_ensemble", False) else predictions
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        if logits.ndim != 2:
            raise ValueError("Classification metrics require per-row logits after ensemble reduction.")
        if logits.shape[1] == 1:
            positive = torch.sigmoid(logits)
            probabilities = torch.cat([1 - positive, positive], dim=1)
        else:
            probabilities = torch.softmax(logits, dim=1)
        metric_predictions = probabilities.argmax(dim=1) if input_space == "labels" else probabilities
        if isinstance(metric, TorchMetric):
            if binary_metric and input_space == "probabilities":
                if probabilities.shape[1] != 2:
                    raise ValueError("Binary TorchMetrics require exactly two probability columns.")
                metric_predictions = probabilities[:, 1]
            return metric_predictions, labels.reshape(-1).long()
        return metric_predictions, labels

    def _reset_epoch_metrics(self, stage):
        self._metric_buffers[stage].clear()
        self._metric_buffer_bytes[stage] = 0
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        for metric in metrics.values():
            if isinstance(metric, TorchMetric):
                metric.reset()

    def _update_epoch_metrics(self, stage, predictions, labels):
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        for name, metric in metrics.items():
            metric_predictions, metric_labels = self._metric_inputs(predictions, labels, metric)
            if isinstance(metric, TorchMetric):
                metric.update(metric_predictions.detach(), metric_labels.detach())
            else:
                size = metric_predictions.numel() * metric_predictions.element_size()
                size += metric_labels.numel() * metric_labels.element_size()
                if self._metric_buffer_bytes[stage] + size > self.metric_buffer_limit_bytes:
                    self._reset_epoch_metrics(stage)
                    raise MemoryError(
                        f"{stage} callable metric inputs exceed metric_buffer_limit_bytes "
                        f"({self.metric_buffer_limit_bytes}). Use an incremental TorchMetric or a larger TaskModel limit."
                    )
                self._metric_buffers[stage].setdefault(name, []).append(
                    (metric_predictions.detach().cpu().clone(), metric_labels.detach().cpu().clone())
                )
                self._metric_buffer_bytes[stage] += size

    def _finish_epoch_metrics(self, stage):
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed and any(not isinstance(metric, TorchMetric) for metric in metrics.values()):
            total_bytes = torch.tensor(self._metric_buffer_bytes[stage], device=self.device, dtype=torch.long)
            torch.distributed.all_reduce(total_bytes)
            if total_bytes.item() > self.metric_buffer_limit_bytes:
                self._reset_epoch_metrics(stage)
                raise MemoryError("Global callable metric inputs exceed metric_buffer_limit_bytes across ranks.")
        try:
            for name, metric in metrics.items():
                if isinstance(metric, TorchMetric):
                    if metric.update_called:
                        value = metric.compute()
                    else:
                        continue
                else:
                    chunks = self._metric_buffers[stage].get(name)
                    if not chunks:
                        continue
                    predictions = torch.cat([chunk[0] for chunk in chunks], dim=0)
                    targets = torch.cat([chunk[1] for chunk in chunks], dim=0)
                    if distributed:
                        predictions = torch.cat(gather_all_tensors(predictions.to(self.device)), dim=0).cpu()
                        targets = torch.cat(gather_all_tensors(targets.to(self.device)), dim=0).cpu()
                    value = metric(check_numpy(targets), check_numpy(predictions))
                self.log(
                    f"{stage}_{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=False,
                )
        finally:
            self._reset_epoch_metrics(stage)

    def on_train_epoch_start(self):
        self._reset_epoch_metrics("train")

    def on_train_epoch_end(self):
        self._finish_epoch_metrics("train")

    def on_validation_epoch_start(self):
        self._reset_epoch_metrics("val")

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
        if hasattr(self.estimator, "train_with_candidates"):
            if len(batch) != 3:
                raise ValueError("Candidate-aware training requires dataset row indices in each batch.")
            data, labels, query_indices = batch
        else:
            data, labels = batch

        # Check if the model has a `penalty_forward` method
        if hasattr(self.estimator, "penalty_forward"):
            preds, penalty = self.estimator.penalty_forward(*data)  # type: ignore[reportCallIssue]
            loss = self.compute_loss(preds, labels) + penalty
        elif hasattr(self.estimator, "train_with_candidates"):
            if self.train_features is None or self.train_targets is None:
                raise RuntimeError("Candidate-aware training requires a training bank initialized by setup('fit').")
            keep = torch.ones(len(self.train_targets), dtype=torch.bool, device=self.train_targets.device)
            keep[query_indices.to(keep.device)] = False
            candidate_features = tuple(
                [feature[keep] for feature in group] if group is not None else None for group in self.train_features
            )
            preds = self.estimator.train_with_candidates(  # type: ignore[reportCallIssue]
                *data,
                targets=labels,
                candidate_x=candidate_features,
                candidate_y=self.train_targets[keep],
                query_indices=query_indices,
            )
            loss = self.compute_loss(preds, labels)
        else:
            preds = self(*data)
            loss = self.compute_loss(preds, labels)

        # Log the training loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(labels),
            sync_dist=True,
        )

        # Log custom training metrics
        if self.train_metrics:
            self._update_epoch_metrics("train", preds, labels)

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
            batch_size=len(labels),
            sync_dist=True,
        )

        # Log custom validation metrics
        if self.val_metrics:
            self._update_epoch_metrics("val", preds, labels)

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
                *data, candidate_x=self.train_features, candidate_y=self.train_targets
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

        Computes completed-pass metrics and releases stage state. During
        fitting, stores the current validation loss at its zero-based
        training epoch index and applies early pruning. Sanity checks and
        standalone validation leave the history and pruning state unchanged.

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
            Validation losses indexed by training epoch. Epochs without a
            validation measurement contain infinity.
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
        self._finish_epoch_metrics("val")
        if self.trainer.sanity_checking or self.trainer.state.fn != TrainerFn.FITTING:
            return

        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            val_loss_value = val_loss.item()
            if self.current_epoch >= len(self.val_losses):
                self.val_losses.extend([float("inf")] * (self.current_epoch + 1 - len(self.val_losses)))
            self.val_losses[self.current_epoch] = val_loss_value

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
            Zero-based training epoch number. Sanity checks and standalone
            validation are not included.

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
        if 0 <= epoch < len(self.val_losses):
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

        .. deprecated::
            Use :func:`deeptab.training.pretrain_embeddings` instead::

                from deeptab.training import pretrain_embeddings
                pretrain_embeddings(model.estimator, train_dataloader, ...)
        """
        import warnings

        from deeptab.training.pretraining import pretrain_embeddings

        warnings.warn(
            "TaskModel.pretrain_embeddings is deprecated. "
            "Call deeptab.training.pretrain_embeddings(model.estimator, ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return pretrain_embeddings(
            base_model=self.estimator,
            train_dataloader=train_dataloader,
            pretrain_epochs=pretrain_epochs,
            k_neighbors=k_neighbors,
            temperature=temperature,
            save_path=save_path,
            regression=regression,
            lr=lr,
        )

    def get_knn(self, labels, k_neighbors=5, regression=True, device=""):
        """Find k-nearest neighbours.

        .. deprecated::
            Use :class:`deeptab.training.ContrastivePretrainer` directly.
        """
        import warnings

        warnings.warn(
            "TaskModel.get_knn is deprecated. Use deeptab.training.ContrastivePretrainer directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        from deeptab.training.pretraining import ContrastivePretrainer

        pt = ContrastivePretrainer(self.estimator, k_neighbors=k_neighbors, regression=regression)
        knn_indices, _ = pt.get_knn(labels)
        return knn_indices

    def contrastive_loss(self, embeddings, knn_indices, temperature=0.1):
        """Compute contrastive loss.

        .. deprecated::
            Use :class:`deeptab.training.ContrastivePretrainer` directly.
        """
        import warnings

        warnings.warn(
            "TaskModel.contrastive_loss is deprecated. Use deeptab.training.ContrastivePretrainer directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Provide a minimal neg_indices (same as knn_indices, fallback)
        neg_indices = knn_indices
        from deeptab.training.pretraining import ContrastivePretrainer

        pt = ContrastivePretrainer(self.estimator, pool_sequence=embeddings.dim() == 2)
        return pt.contrastive_loss(embeddings, knn_indices, neg_indices)
