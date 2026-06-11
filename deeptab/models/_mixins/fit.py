"""Model construction and training-loop logic for all DeepTab estimators."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from pretab.preprocessor import Preprocessor

from deeptab.core.sklearn_compat import ensure_dataframe, set_input_feature_attributes
from deeptab.training import pretrain_embeddings

if TYPE_CHECKING:
    from deeptab.configs import PreprocessingConfig, TrainerConfig
    from deeptab.core.default_factories import DefaultDataModuleFactory, DefaultTaskModelFactory
    from deeptab.models._mixins.observability import _SupportsInfo


class _FitMixin:
    # ---------------------------------------------------------------------------
    # Attributes provided by SklearnBase when this mixin is composed.
    # Declared here for static type-checkers only; never initialised in this class.
    # ---------------------------------------------------------------------------
    if TYPE_CHECKING:
        random_state: int | None
        trainer_config: TrainerConfig | None
        preprocessing_config: PreprocessingConfig | None
        config: Any
        input_columns_: list[str] | None
        _data_module_factory: DefaultDataModuleFactory
        _task_model_factory: DefaultTaskModelFactory
        _optimizer_type: str | None
        _optimizer_kwargs: dict | None
        _event_logger: _SupportsInfo | None

        def _emit_event(self, event: str, **kwargs: Any) -> None: ...

    """Model construction and training loop.

    Responsibilities
    ----------------
    * ``_build_model`` — creates and configures the ``IDataModule`` and
      ``ITaskModel`` collaborators using the injected factories.
    * ``fit`` — orchestrates data validation, model construction, Lightning
      Trainer setup, weight checkpointing, and best-weight restoration.
    * ``get_number_of_params`` — counts trainable / total parameters after a
      model has been built.
    * ``_pretrain`` — contrastive pre-training pass (optional, used for
      embedding warm-start).
    """

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(
        self,
        X,
        y,
        regression: bool,
        val_size: float = 0.2,
        X_val=None,
        y_val=None,
        embeddings=None,
        embeddings_val=None,
        num_classes: int | None = None,
        random_state: int = 101,
        batch_size: int = 128,
        shuffle: bool = True,
        lr: float | None = None,
        lr_patience: int | None = None,
        lr_factor: float | None = None,
        weight_decay: float | None = None,
        train_metrics: dict[str, Callable] | None = None,
        val_metrics: dict[str, Callable] | None = None,
        dataloader_kwargs={},
        loss_fct: Callable | None = None,
        sampler=None,
    ):
        """Builds the model using the provided training data."""
        # When trainer_config is active, use its values for lr / weight_decay / scheduler
        if self.trainer_config is not None:
            tc = self.trainer_config
            if lr is None:
                lr = tc.lr
            if lr_patience is None:
                lr_patience = tc.lr_patience
            if lr_factor is None:
                lr_factor = tc.lr_factor
            if weight_decay is None:
                weight_decay = tc.weight_decay

        # Collect new scheduler/optimizer fields from TrainerConfig
        _tc = self.trainer_config
        _scheduler_type = (
            getattr(_tc, "scheduler_type", "ReduceLROnPlateau") if _tc is not None else "ReduceLROnPlateau"
        )
        _scheduler_kwargs = getattr(_tc, "scheduler_kwargs", None) if _tc is not None else None
        _scheduler_monitor = getattr(_tc, "scheduler_monitor", None) if _tc is not None else None
        _scheduler_interval = getattr(_tc, "scheduler_interval", "epoch") if _tc is not None else "epoch"
        _scheduler_frequency = getattr(_tc, "scheduler_frequency", 1) if _tc is not None else 1
        _no_wd_bn = getattr(_tc, "no_weight_decay_for_bias_and_norm", False) if _tc is not None else False
        _optimizer_kwargs = getattr(_tc, "optimizer_kwargs", None) if _tc is not None else None

        # Re-sync preprocessor from current preprocessing_config state so that
        # direct mutations (e.g. clf.preprocessing_config.n_bins = 8) are
        # honoured on the next fit(), consistent with set_params() behaviour.
        if self.preprocessing_config is not None:
            self._preprocessor_kwargs = self.preprocessing_config.to_preprocessor_kwargs()
            self._preprocessor = Preprocessor(**self._preprocessor_kwargs)

        X = ensure_dataframe(X)
        set_input_feature_attributes(self, X)
        if hasattr(y, "values"):
            y = y.values
        if X_val is not None:
            X_val = ensure_dataframe(X_val)
            if y_val is not None and hasattr(y_val, "values"):
                y_val = y_val.values

        self._data_module = self._data_module_factory.create(
            preprocessor=self._preprocessor,
            batch_size=batch_size,
            shuffle=shuffle,
            X_val=X_val,
            y_val=y_val,
            val_size=val_size,
            random_state=random_state,
            regression=regression,
            sampler=sampler,
            **dataloader_kwargs,
        )
        self._data_module.input_columns_ = self.input_columns_

        self._data_module.preprocess_data(
            X,
            y,
            X_val=X_val,
            y_val=y_val,
            embeddings_train=embeddings,
            embeddings_val=embeddings_val,
            val_size=val_size,
            random_state=random_state,
        )
        self._emit_event("data_module_created")

        # Derive split sizes for the data_prepared event; fall back gracefully
        # when the data module doesn't expose dataset sizes yet.
        _n_train = getattr(getattr(self._data_module, "train_dataset", None), "__len__", lambda: None)()
        _n_val = getattr(getattr(self._data_module, "val_dataset", None), "__len__", lambda: None)()
        self._emit_event("data_prepared", n_train=_n_train, n_val=_n_val)

        self._task_model = self._task_model_factory.create(
            model_class=self._estimator,  # type: ignore
            config=self.config,
            feature_information=(
                self._data_module.num_feature_info,  # type: ignore[arg-type]
                self._data_module.cat_feature_info,  # type: ignore[arg-type]
                self._data_module.embedding_feature_info,  # type: ignore[arg-type]
            ),
            lr=lr if lr is not None else getattr(self.config, "lr", None),
            lr_patience=(lr_patience if lr_patience is not None else getattr(self.config, "lr_patience", None)),
            lr_factor=lr_factor if lr_factor is not None else getattr(self.config, "lr_factor", None),
            weight_decay=(weight_decay if weight_decay is not None else getattr(self.config, "weight_decay", None)),
            num_classes=num_classes,  # type: ignore[arg-type]
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            optimizer_type=(
                self.trainer_config.optimizer_type if self.trainer_config is not None else self._optimizer_type
            ),
            optimizer_args=_optimizer_kwargs if _optimizer_kwargs is not None else self._optimizer_kwargs,
            scheduler_type=_scheduler_type,
            scheduler_kwargs=_scheduler_kwargs,
            monitor=_scheduler_monitor
            if _scheduler_monitor is not None
            else (
                getattr(self.trainer_config, "monitor", "val_loss") if self.trainer_config is not None else "val_loss"
            ),
            mode=getattr(self.trainer_config, "mode", "min") if self.trainer_config is not None else "min",
            scheduler_interval=_scheduler_interval,
            scheduler_frequency=_scheduler_frequency,
            no_weight_decay_for_bias_and_norm=_no_wd_bn,
            loss_fct=loss_fct,
        )

        self._built = True
        self._estimator = self._task_model.estimator
        self._emit_event("task_model_created")

        return self

    def get_number_of_params(self, requires_grad=True):
        """Calculate the number of parameters in the model.

        Parameters
        ----------
        requires_grad : bool, optional
            If True, only count the parameters that require gradients (trainable parameters).
            If False, count all parameters. Default is True.

        Returns
        -------
        int
            The total number of parameters in the model.

        Raises
        ------
        ValueError
            If the model has not been built prior to calling this method.
        """
        if not self._built:
            raise ValueError("The model must be built before the number of parameters can be estimated")
        if requires_grad:
            return sum(p.numel() for p in self._task_model.parameters() if p.requires_grad)  # type: ignore
        return sum(p.numel() for p in self._task_model.parameters())  # type: ignore

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        regression: bool,
        val_size: float = 0.2,
        X_val=None,
        y_val=None,
        embeddings=None,
        embeddings_val=None,
        num_classes: int | None = None,
        max_epochs: int = 100,
        random_state: int = 101,
        batch_size: int = 128,
        shuffle: bool = True,
        patience: int = 15,
        monitor: str = "val_loss",
        mode: str = "min",
        lr: float | None = None,
        lr_patience: int | None = None,
        lr_factor: float | None = None,
        weight_decay: float | None = None,
        checkpoint_path="model_checkpoints",
        dataloader_kwargs={},
        train_metrics: dict[str, Callable] | None = None,
        val_metrics: dict[str, Callable] | None = None,
        rebuild=True,
        loss_fct: Callable | None = None,
        sampler=None,
        **trainer_kwargs,
    ):
        """Trains the model using the provided training data.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values.
        regression : bool
            Whether this is a regression task.
        val_size : float, default=0.2
            Proportion of the dataset for the validation split when ``X_val``
            is ``None``.
        X_val : DataFrame or array-like, optional
            Explicit validation features.
        y_val : array-like, optional
            Explicit validation targets.
        embeddings : array-like, optional
            Pre-computed embeddings for training samples.
        embeddings_val : array-like, optional
            Pre-computed embeddings for validation samples.
        num_classes : int or None, optional
            Number of target classes (classification only).
        max_epochs : int, default=100
            Maximum number of training epochs.
        random_state : int, default=101
            RNG seed for reproducibility.
        batch_size : int, default=128
            Mini-batch size.
        shuffle : bool, default=True
            Whether to shuffle training data each epoch.
        patience : int, default=15
            Early-stopping patience (epochs without validation improvement).
        monitor : str, default="val_loss"
            Metric to monitor for early stopping.
        mode : str, default="min"
            Whether the monitored metric should be minimised (``"min"``) or
            maximised (``"max"``).
        lr : float or None, optional
            Learning rate override.
        lr_patience : int or None, optional
            LR scheduler patience override.
        lr_factor : float or None, optional
            LR scheduler reduction factor override.
        weight_decay : float or None, optional
            Weight-decay (L2 penalty) override.
        checkpoint_path : str, default="model_checkpoints"
            Directory for Lightning checkpoints.
        dataloader_kwargs : dict, default={}
            Extra kwargs forwarded to the PyTorch DataLoader.
        train_metrics : dict or None, optional
            TorchMetrics to log during training.
        val_metrics : dict or None, optional
            TorchMetrics to log during validation.
        rebuild : bool, default=True
            Whether to rebuild the model when already built.
        loss_fct : Callable or None, optional
            Custom loss function override.
        sampler : {"balanced", True}, array-like, or None, optional
            Weighted-sampling specification.
        **trainer_kwargs
            Additional keyword arguments forwarded to ``pl.Trainer``.

        Returns
        -------
        self
        """
        # When trainer_config is active, override all training-loop params from it
        if self.trainer_config is not None:
            tc = self.trainer_config
            max_epochs = tc.max_epochs
            batch_size = tc.batch_size
            val_size = tc.val_size
            shuffle = tc.shuffle
            patience = tc.patience
            monitor = tc.monitor
            mode = tc.mode
            checkpoint_path = tc.checkpoint_path

        # Validate inputs before any preprocessing or model construction
        from deeptab.models.base import _validate_fit_inputs

        _validate_fit_inputs(X, y, regression=regression)

        # When random_state was fixed at construction time, honour it
        if self.random_state is not None:
            random_state = self.random_state

        # Seed all RNGs so that weight init, dropout masks, and DataLoader
        # shuffling are all deterministic when a random_state is provided.
        if random_state is not None:
            from deeptab.core.reproducibility import set_seed

            set_seed(random_state)

        self._emit_event("fit_started", n_samples=len(X), n_features=getattr(self, "n_features_in_", None))

        if rebuild:
            self._build_model(
                X=X,
                y=y,
                regression=regression,
                val_size=val_size,
                X_val=X_val,
                y_val=y_val,
                embeddings=embeddings,
                embeddings_val=embeddings_val,
                num_classes=num_classes,
                random_state=random_state,  # type: ignore[arg-type]
                batch_size=batch_size,
                shuffle=shuffle,
                lr=lr,
                lr_patience=lr_patience,
                lr_factor=lr_factor,
                weight_decay=weight_decay,
                dataloader_kwargs=dataloader_kwargs,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                loss_fct=loss_fct,
                sampler=sampler,
            )
        else:
            if not self._built:
                raise ValueError(
                    "The model must be built before calling the fit method. "
                    "Either call .build_model() or set rebuild=True"
                )

        self._emit_event(
            "model_built",
            n_params=sum(p.numel() for p in self._task_model.parameters() if p.requires_grad),  # type: ignore
        )

        early_stop_callback = EarlyStopping(
            monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=checkpoint_path,
            filename="best_model",
        )

        self._trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                early_stop_callback,
                checkpoint_callback,
                ModelSummary(max_depth=2),
            ],
            **trainer_kwargs,
        )
        self._task_model.train()  # type: ignore[union-attr]
        self._task_model.estimator.train()  # type: ignore[union-attr]

        self._emit_event("training_started", max_epochs=max_epochs, batch_size=batch_size)
        self._trainer.fit(self._task_model, self._data_module)  # type: ignore

        self._best_model_path = checkpoint_callback.best_model_path
        if self._best_model_path:
            torch.serialization.add_safe_globals([type(self.config)])
            checkpoint = torch.load(self._best_model_path, weights_only=False)
            self._task_model.load_state_dict(checkpoint["state_dict"])  # type: ignore

        # Retrieve best epoch and best val_loss from the checkpoint callback
        # (both are None before training and when no checkpoint was saved).
        self._emit_event(
            "training_completed",
            best_epoch=getattr(checkpoint_callback, "best_k_models", {})
            and getattr(self._trainer, "current_epoch", None),
            best_val_loss=checkpoint_callback.best_model_score.item()
            if checkpoint_callback.best_model_score is not None
            else None,
        )

        self.is_fitted_ = True
        self._emit_event("fit_completed")
        return self

    # ------------------------------------------------------------------
    # Pre-training
    # ------------------------------------------------------------------

    def _pretrain(
        self,
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
        """Run a contrastive pre-training pass to warm-start embeddings.

        Delegates to :func:`~deeptab.training.pretrain_embeddings`. Call
        this before :meth:`fit` when you want to initialise the backbone
        with representation learning before fine-tuning on the target task.

        Parameters
        ----------
        base_model :
            The backbone model to pre-train.
        train_dataloader : DataLoader
            DataLoader that yields batches of tabular features.
        pretrain_epochs : int, default=5
            Number of contrastive pre-training epochs.
        k_neighbors : int, default=5
            Number of nearest neighbours used to construct positive pairs.
        temperature : float, default=0.1
            Softmax temperature for the contrastive loss.
        save_path : str, default="pretrained_embeddings.pth"
            Path to save the pre-trained weights.
        regression : bool, default=True
            Whether the downstream task is regression.
        lr : float, default=1e-3
            Learning rate for the pre-training optimiser.
        use_positive : bool, default=True
            Whether to include positive-pair terms in the loss.
        use_negative : bool, default=True
            Whether to include negative-pair terms in the loss.
        pool_sequence : bool, default=True
            Whether to pool sequence-dimension embeddings before computing
            the contrastive loss.
        """
        pretrain_embeddings(
            base_model=base_model,
            train_dataloader=train_dataloader,
            pretrain_epochs=pretrain_epochs,
            k_neighbors=k_neighbors,
            temperature=temperature,
            save_path=save_path,
            regression=regression,
            lr=lr,
            use_positive=use_positive,
            use_negative=use_negative,
            pool_sequence=pool_sequence,
        )
