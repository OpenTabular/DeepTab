"""Public methods shared by the point-prediction estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from deeptab.core.exceptions import not_fitted_error

if TYPE_CHECKING:
    from deeptab.core.interfaces import IDataModule, ITaskModel


class _PointEstimatorMixin:
    """Behaviour the classifier and regressor bases share but the LSS base does not.

    ``SklearnBaseClassifier`` and ``SklearnBaseRegressor`` list this mixin in
    front of ``SklearnBase``. The task is read from the ``_task`` class
    attribute, so one implementation serves both.

    Responsibilities
    ----------------
    * ``pretrain`` — public contrastive pre-training of the embedding layer.
    * ``optimize_hparams`` — hyperparameter search that also takes external
      embeddings.
    """

    if TYPE_CHECKING:
        # Attributes provided by SklearnBase when this mixin is composed.
        # Declared here for static type-checkers only; never initialised in this class.
        _task: ClassVar[str]
        _built: bool
        _is_pretrained: bool
        _task_model: ITaskModel | None
        _data_module: IDataModule | None

        def _pretrain(self, base_model: Any, train_dataloader: Any, **kwargs: Any) -> None: ...

    def pretrain(
        self,
        pretrain_epochs=15,
        k_neighbors=10,
        temperature=0.1,
        save_path="pretrained_embeddings.pth",
        lr=1e-3,
        use_positive=True,
        use_negative=False,
        pool_sequence=True,
    ):
        """
        Pretrains the embedding layer of the model using a contrastive learning approach.

        This method performs pretraining by optimizing the embeddings with respect to
        neighborhood structure in the feature space. The embeddings are saved after training.

        Parameters
        ----------
        pretrain_epochs : int, default=15
            Number of epochs to run pretraining.
        k_neighbors : int, default=10
            Number of neighbors used in the contrastive loss computation.
        temperature : float, default=0.1
            Temperature parameter for contrastive loss scaling.
        save_path : str, default="pretrained_embeddings.pth"
            Path to save the pretrained embeddings.
        lr : float, default=1e-3
            Learning rate for the pretraining optimizer.
        use_positive : bool, default=True
            Whether to include positive pairs in contrastive learning.
        use_negative : bool, default=False
            Whether to include negative pairs in contrastive learning.
        pool_sequence : bool, default=True
            Whether to apply sequence pooling before computing contrastive loss.

        Raises
        ------
        ValueError
            If the model has not been built before calling this method.
        ValueError
            If the model does not contain an embedding layer.

        Notes
        -----
        - This function requires that `self.build_model()` has been called beforehand.
        - The pretraining method uses `self.task_model.estimator.embedding_layer`.
        - Classifiers pretrain with regression mode disabled, so class labels define the
          positive and negative pairs. Regressors pretrain with regression mode enabled.

        """
        if not self._built:
            raise ValueError("The model has not been built yet. Call model.build_model(**args) first.")

        if not hasattr(self._task_model.estimator, "embedding_layer"):  # type: ignore[union-attr]
            raise ValueError("The model does not have an embedding layer")

        if self._data_module is None:
            raise not_fitted_error(type(self).__name__, "_pretrain")
        self._data_module.setup("fit")

        self._pretrain(
            self._task_model.estimator,  # type: ignore[union-attr]
            self._data_module,
            pretrain_epochs=pretrain_epochs,
            k_neighbors=k_neighbors,
            temperature=temperature,
            save_path=save_path,
            regression=type(self)._task == "regression",
            lr=lr,
            use_positive=use_positive,
            use_negative=use_negative,
            pool_sequence=pool_sequence,
        )
        self._is_pretrained = True

    def optimize_hparams(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        embeddings=None,
        embeddings_val=None,
        time=100,
        max_epochs=200,
        prune_by_epoch=True,
        prune_epoch=5,
        fixed_params=None,
        custom_search_space=None,
        **optimize_kwargs,
    ):
        """Optimizes hyperparameters using Bayesian optimization with optional pruning.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Training labels.
        X_val, y_val : array-like, optional
            Validation data and labels.
        time : int
            The number of optimization trials to run.
        max_epochs : int
            Maximum number of epochs for training.
        prune_by_epoch : bool
            Whether to prune based on a specific epoch (True) or the best validation loss (False).
        prune_epoch : int
            The specific epoch to prune by when prune_by_epoch is True.
        **optimize_kwargs : dict
            Additional keyword arguments passed to the fit method.

        Returns
        -------
        best_hparams : list
            Best hyperparameters found during optimization.
        """

        return super().optimize_hparams(  # type: ignore[misc]
            X,
            y,
            regression=type(self)._task == "regression",
            X_val=X_val,
            y_val=y_val,
            embeddings=embeddings,
            embeddings_val=embeddings_val,
            time=time,
            max_epochs=max_epochs,
            prune_by_epoch=prune_by_epoch,
            prune_epoch=prune_epoch,
            fixed_params=fixed_params,
            custom_search_space=custom_search_space,
            **optimize_kwargs,
        )
