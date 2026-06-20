"""Abstract interface Protocols for DeepTab's two core collaborators.

``SklearnBase`` depends on these abstractions rather than on the concrete
``TabularDataModule`` and ``TaskModel`` classes.  Because the Protocols use
structural sub-typing (``typing.Protocol``), the concrete classes satisfy
them implicitly — no inheritance required.

Replace either collaborator by assigning a compatible factory::

    from deeptab.core.interfaces import IDataModuleFactory

    class MyDataModuleFactory:
        def create(self, preprocessor, batch_size, shuffle, regression, **kw):
            return MyDataModule(preprocessor, batch_size, shuffle, regression)

    clf._data_module_factory = MyDataModuleFactory()
    clf.fit(X, y)   # uses MyDataModule internally
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Data-module interface
# ---------------------------------------------------------------------------


@runtime_checkable
class IDataModule(Protocol):
    """Minimal data-handling interface required by ``SklearnBase``.

    Any object that exposes these attributes and methods can be used as
    the data module, including test doubles and custom implementations.
    """

    num_feature_info: dict | None
    """Per-feature metadata dict for numerical features."""
    cat_feature_info: dict | None
    """Per-feature metadata dict for categorical features."""
    embedding_feature_info: dict | None
    """Per-feature metadata dict for pre-computed embedding features."""
    input_columns_: list[str] | None
    """Ordered column names seen during ``fit``; ``None`` before fitting."""

    def preprocess_data(self, *args: Any, **kwargs: Any) -> None:
        """Fit the preprocessor on training data and store the result."""
        ...

    def preprocess_new_data(self, *args: Any, **kwargs: Any) -> Any:
        """Transform new data using the already-fitted preprocessor."""
        ...

    def assign_predict_dataset(self, *args: Any, **kwargs: Any) -> None:
        """Prepare the dataset used during predict / inference."""
        ...

    def setup(self, *args: Any, **kwargs: Any) -> None:
        """Lightning ``DataModule.setup`` — called before dataloaders are created."""
        ...

    def train_dataloader(self) -> Any:
        """Return the training ``DataLoader``."""
        ...

    def val_dataloader(self) -> Any:
        """Return the validation ``DataLoader``."""
        ...


# ---------------------------------------------------------------------------
# Task-model interface
# ---------------------------------------------------------------------------


@runtime_checkable
class ITaskModel(Protocol):
    """Minimal neural-network interface required by ``SklearnBase``.

    Any object that exposes these attributes and methods can be used as
    the task model, including Lightning modules and test doubles.
    """

    estimator: Any
    """The underlying architecture module (e.g. an ``nn.Module``)."""

    def train(self, mode: bool = True) -> Any:
        """Switch the model to training mode."""
        ...

    def eval(self) -> Any:
        """Switch the model to evaluation mode."""
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        """Load weights from a state dict (e.g. from a checkpoint)."""
        ...

    def parameters(self) -> Any:
        """Return an iterator over model parameters."""
        ...


# ---------------------------------------------------------------------------
# Factory interfaces
# ---------------------------------------------------------------------------


@runtime_checkable
class IDataModuleFactory(Protocol):
    """Creates ``IDataModule``-compatible objects on demand.

    Implement this Protocol to supply a custom data-module implementation
    without subclassing ``SklearnBase``.
    """

    def create(
        self,
        preprocessor: Any,
        batch_size: int,
        shuffle: bool,
        regression: bool,
        **kwargs: Any,
    ) -> IDataModule:
        """Construct and return a data module.

        Parameters
        ----------
        preprocessor :
            Fitted or unfitted ``Preprocessor`` instance.
        batch_size : int
            Mini-batch size for the DataLoader.
        shuffle : bool
            Whether to shuffle training samples each epoch.
        regression : bool
            ``True`` for regression tasks, ``False`` for classification.
        **kwargs
            Additional arguments forwarded to the concrete constructor
            (e.g. ``val_size``, ``sampler``, ``random_state``).

        Returns
        -------
        IDataModule
            A configured data module ready for ``preprocess_data``.
        """
        ...


@runtime_checkable
class ITaskModelFactory(Protocol):
    """Creates ``ITaskModel``-compatible objects on demand.

    Implement this Protocol to supply a custom Lightning module without
    subclassing ``SklearnBase``.
    """

    def create(
        self,
        model_class: Any,
        config: Any,
        feature_information: tuple[dict, dict, dict],
        **kwargs: Any,
    ) -> ITaskModel:
        """Construct and return a task model.

        Parameters
        ----------
        model_class :
            The backbone ``nn.Module`` class (not an instance).
        config :
            Config dataclass instance for the backbone.
        feature_information : (num_info, cat_info, emb_info)
            Tuple of three dicts describing the feature schema, as produced
            by ``TabularDataModule`` after ``preprocess_data``.
        **kwargs
            Additional arguments forwarded to the concrete constructor
            (e.g. ``lr``, ``optimizer_type``, ``loss_fct``).

        Returns
        -------
        ITaskModel
            A configured task model ready to be passed to ``pl.Trainer.fit``.
        """
        ...
