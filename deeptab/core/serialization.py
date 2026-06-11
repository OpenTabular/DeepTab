"""Serialization helpers for model weights and fitted estimator artifacts."""

from __future__ import annotations

import platform
import warnings
from dataclasses import fields, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import torch

RECOMMENDED_EXTENSION = ".deeptab"
ARTIFACT_FORMAT_VERSION = 2


def _warn_extension(path: str) -> None:
    """Emit a warning when *path* does not use the recommended ``.deeptab`` extension.

    This is a soft advisory only — any path is still accepted.

    Parameters
    ----------
    path : str
        The file path passed to :meth:`save` or :meth:`load`.
    """
    if not str(path).endswith(RECOMMENDED_EXTENSION):
        warnings.warn(
            f"DeepTab artifacts should use the '{RECOMMENDED_EXTENSION}' extension "
            f"(e.g. 'model.deeptab'). "
            f"Got: '{path}'. "
            f"The file will still be saved/loaded correctly, but using '{RECOMMENDED_EXTENSION}' "
            "makes the artifact type unambiguous and future-proof.",
            UserWarning,
            stacklevel=3,
        )


def save_state_dict(model: torch.nn.Module, path: str) -> None:
    """Save a module state dict to disk."""
    torch.save(model.state_dict(), path)


def load_state_dict(model: torch.nn.Module, path: str, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Load a module state dict and move the module to ``device``."""
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def collect_version_metadata() -> dict[str, Any]:
    """Collect package versions that are useful when debugging saved artifacts."""
    packages = {
        "deeptab": "deeptab",
        "torch": "torch",
        "lightning": "lightning",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "pretab": "pretab",
        "torchmetrics": "torchmetrics",
        "scipy": "scipy",
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _package_version(distribution) for name, distribution in packages.items()},
    }


def build_artifact_metadata(
    *,
    estimator: Any,
    model_class: type,
    config: Any,
    data_module: Any,
    preprocessor: Any,
    preprocessor_kwargs: dict[str, Any] | None,
    task: str,
    regression: bool,
    lss: bool,
    family: str | None,
    num_classes: int | None,
    classes_: Any = None,
) -> dict[str, Any]:
    """Build the standard metadata block stored with fitted estimators."""
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "architecture": build_architecture_metadata(model_class=model_class, config=config, estimator=estimator),
        "feature_schema": build_feature_schema_metadata(data_module),
        "preprocessing": build_preprocessing_metadata(preprocessor, preprocessor_kwargs),
        "task": build_task_metadata(
            task=task,
            regression=regression,
            lss=lss,
            family=family,
            num_classes=num_classes,
            classes_=classes_,
        ),
        "versions": collect_version_metadata(),
    }


def build_architecture_metadata(*, model_class: type, config: Any, estimator: Any = None) -> dict[str, Any]:
    """Describe the architecture from central registry/config state."""
    architecture_name = model_class.__name__
    metadata = {
        "name": architecture_name,
        "class_name": architecture_name,
        "module": model_class.__module__,
        "registry": None,
        "config_class": type(config).__name__ if config is not None else None,
        "config_module": type(config).__module__ if config is not None else None,
        "config": _simplify(config),
    }

    try:
        from deeptab.core.registry import MODEL_REGISTRY

        registry_info = MODEL_REGISTRY.get(architecture_name)
        if registry_info is not None:
            metadata["registry"] = {
                "name": registry_info.name,
                "status": registry_info.status,
                "import_path": registry_info.import_path,
            }
    except Exception:
        metadata["registry"] = None

    if estimator is not None:
        metadata["estimator_class"] = type(estimator).__name__
        metadata["estimator_module"] = type(estimator).__module__
    return metadata


def build_feature_schema_metadata(data_module: Any) -> dict[str, Any]:
    """Serialize feature order, groups, and preprocessing-derived schema."""
    num_info = getattr(data_module, "num_feature_info", None) or {}
    cat_info = getattr(data_module, "cat_feature_info", None) or {}
    emb_info = getattr(data_module, "embedding_feature_info", None) or {}
    input_columns = getattr(data_module, "input_columns_", None)

    schema = getattr(data_module, "schema", None)
    schema_dict = schema.to_dict() if schema is not None and hasattr(schema, "to_dict") else None

    return {
        "column_order": _simplify(input_columns),
        "feature_groups": {
            "numerical": _simplify(list(num_info.keys())),
            "categorical": _simplify(list(cat_info.keys())),
            "embedding": _simplify(list(emb_info.keys())),
        },
        "feature_info": {
            "num": _simplify(num_info),
            "cat": _simplify(cat_info),
            "emb": _simplify(emb_info),
        },
        "schema": schema_dict,
    }


def build_preprocessing_metadata(
    preprocessor: Any, preprocessor_kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Describe the fitted preprocessing object stored in the artifact."""
    return {
        "class_name": type(preprocessor).__name__ if preprocessor is not None else None,
        "module": type(preprocessor).__module__ if preprocessor is not None else None,
        "kwargs": _simplify(preprocessor_kwargs or {}),
        "fitted_state_persisted": preprocessor is not None,
    }


def build_task_metadata(
    *,
    task: str,
    regression: bool,
    lss: bool,
    family: str | None,
    num_classes: int | None,
    classes_: Any = None,
) -> dict[str, Any]:
    """Describe target/task semantics persisted with an estimator."""
    return {
        "task": task,
        "regression": regression,
        "lss": lss,
        "family": family,
        "num_classes": num_classes,
        "classes_": _simplify(classes_),
    }


_PREPROCESSOR_ARG_NAMES: list[str] = [
    "n_bins",
    "feature_preprocessing",
    "numerical_preprocessing",
    "categorical_preprocessing",
    "use_decision_tree_bins",
    "binning_strategy",
    "task",
    "cat_cutoff",
    "treat_all_integers_as_numerical",
    "degree",
    "scaling_strategy",
    "n_knots",
    "use_decision_tree_knots",
    "knots_strategy",
    "spline_implementation",
]


def build_save_bundle(
    estimator: Any,
    *,
    lss: bool,
    family: str | None,
) -> dict[str, Any]:
    """Build the complete save bundle for a fitted estimator.

    This is the single source of truth for what gets written to disk by
    :meth:`~deeptab.models.base.SklearnBase.save` and
    :meth:`~deeptab.models.lss_base.SklearnBaseLSS.save`.  Both the standard
    estimator base and the LSS base delegate to this function, ensuring a
    consistent artifact structure across all model variants.

    Parameters
    ----------
    estimator : fitted estimator
        The estimator whose state should be serialized.  Must have
        ``is_fitted_`` set to ``True`` and a non-``None`` ``task_model``.
    lss : bool
        Whether the estimator is a distributional (LSS) model.
    family : str or None
        Distribution family name for LSS models; ``None`` otherwise.

    Returns
    -------
    bundle : dict
        A plain dictionary ready to be passed to ``torch.save(bundle, path)``.

    Raises
    ------
    ValueError
        If the estimator has not been fitted.
    RuntimeError
        If ``task_model`` is unexpectedly ``None`` after fitting.

    Notes
    -----
    The bundle always contains the following top-level keys:

    * ``_class`` — the Python class of the estimator (used to reconstruct
      the object on load).
    * ``artifact_metadata`` — the full structured metadata block produced by
      :func:`build_artifact_metadata`, including architecture, feature schema,
      preprocessing, task, and version information.
    * ``task_model_state_dict`` — the Lightning module weights.
    * ``preprocessor`` — the fitted preprocessing object.
    * ``feature_info`` — numerical, categorical, and embedding feature dicts.
    * ``classes_``, ``n_features_in_``, ``feature_names_in_`` — sklearn-style
      fitted attributes.

    Examples
    --------
    This function is used internally by ``save()``; typical users should call
    ``model.save(path)`` directly rather than using this helper:

    >>> model = MLPClassifier()
    >>> model.fit(X_train, y_train)
    >>> model.save("my_model.pt")           # internally calls build_save_bundle
    >>> loaded = MLPClassifier.load("my_model.pt")
    """
    if not getattr(estimator, "is_fitted_", False):
        raise ValueError("Model must be fitted before saving.")
    if estimator._task_model is None:
        raise RuntimeError("_task_model is unexpectedly None after fitting.")

    if lss:
        task = (
            "classification"
            if getattr(estimator, "family_name", None) == "categorical"
            else "distributional_regression"
        )
    else:
        task = "regression" if estimator._data_module.regression else "classification"

    artifact_metadata = build_artifact_metadata(
        estimator=estimator,
        model_class=type(estimator._estimator),
        config=estimator.config,
        data_module=estimator._data_module,
        preprocessor=estimator._preprocessor,
        preprocessor_kwargs=getattr(estimator, "_preprocessor_kwargs", {}),
        task=task,
        regression=estimator._data_module.regression,
        lss=lss,
        family=family,
        num_classes=estimator._task_model.num_classes,
        classes_=getattr(estimator, "classes_", None),
    )
    feature_schema = artifact_metadata["feature_schema"]

    return {
        "_class": type(estimator),
        "config": estimator.config,
        "config_kwargs": estimator._config_kwargs,
        "preprocessor_kwargs": getattr(estimator, "_preprocessor_kwargs", {}),
        "preprocessor": estimator._preprocessor,
        "feature_info": {
            "num": estimator._data_module.num_feature_info,
            "cat": estimator._data_module.cat_feature_info,
            "emb": estimator._data_module.embedding_feature_info,
        },
        "batch_size": estimator._data_module.batch_size,
        "regression": estimator._data_module.regression,
        "model_class": type(estimator._estimator),
        "num_classes": estimator._task_model.num_classes,
        "lss": lss,
        "family": family,
        "optimizer_type": estimator._optimizer_type,
        "optimizer_kwargs": estimator._optimizer_kwargs,
        "lr": estimator._task_model.lr,
        "lr_patience": estimator._task_model.lr_patience,
        "lr_factor": estimator._task_model.lr_factor,
        "weight_decay": estimator._task_model.weight_decay,
        "task_model_state_dict": estimator._task_model.state_dict(),
        "artifact_metadata": artifact_metadata,
        "architecture_metadata": artifact_metadata["architecture"],
        "feature_schema": feature_schema,
        "input_columns": feature_schema["column_order"],
        "preprocessing_metadata": artifact_metadata["preprocessing"],
        "task_info": artifact_metadata["task"],
        "classes_": getattr(estimator, "classes_", None),
        "n_features_in_": getattr(estimator, "n_features_in_", None),
        "feature_names_in_": getattr(estimator, "feature_names_in_", None),
        "versions": artifact_metadata["versions"],
    }


def restore_base_state(obj: Any, bundle: dict[str, Any]) -> None:
    """Restore the common estimator state from a loaded bundle.

    Called by both :meth:`~deeptab.models.base.SklearnBase.load` and
    :meth:`~deeptab.models.lss_base.SklearnBaseLSS.load` to set all fields
    that are identical between the two base classes, keeping load logic
    in one place.

    Parameters
    ----------
    obj : estimator instance
        A freshly allocated (``__new__``) estimator object to populate.
    bundle : dict
        The bundle dictionary loaded from disk via ``torch.load``.

    Notes
    -----
    This function sets:

    * Core config and preprocessor state (``config``, ``preprocessor``,
      ``preprocessor_kwargs``, ``optimizer_type``, ``optimizer_kwargs``).
    * Fitted-state flags (``built``, ``is_fitted_``).
    * Config API attributes (``model_config``, ``preprocessing_config``,
      ``trainer_config``, ``random_state``).
    * The canonical ``preprocessor_arg_names`` list.

    It does **not** reconstruct the ``data_module``, ``task_model``, or
    ``trainer`` — those require task-specific wiring handled by each
    ``load()`` classmethod.
    """
    obj.config = bundle["config"]
    obj._config_kwargs = bundle["config_kwargs"]
    obj._preprocessor_kwargs = bundle.get("preprocessor_kwargs", {})
    obj._preprocessor = bundle["preprocessor"]
    obj._optimizer_type = bundle["optimizer_type"]
    obj._optimizer_kwargs = bundle["optimizer_kwargs"]
    obj._built = True
    obj.is_fitted_ = True
    obj.model_config = None
    obj.preprocessing_config = None
    obj.trainer_config = None
    obj.random_state = None
    obj._preprocessor_arg_names = list(_PREPROCESSOR_ARG_NAMES)


def restore_loaded_metadata(obj: Any, bundle: dict[str, Any]) -> None:
    """Attach metadata fields to an estimator restored from a saved artifact.

    Called as the final step of every ``load()`` classmethod. Populates all
    sklearn-style fitted attributes and the richer metadata fields that make
    loaded models introspectable without needing to re-fit.

    Parameters
    ----------
    obj : estimator instance
        The partially reconstructed estimator (weights and data module already
        set) to attach metadata to.
    bundle : dict
        The bundle dictionary loaded from disk via ``torch.load``.

    Notes
    -----
    After this function runs, the following attributes are available on *obj*:

    * ``artifact_metadata_`` — the full structured metadata block (architecture,
      feature schema, preprocessing, task, versions).
    * ``architecture_metadata_`` — architecture name, config class, registry info.
    * ``feature_schema_`` — column order, feature groups, and per-feature info.
    * ``preprocessing_metadata_`` — preprocessor class, kwargs, and fitted state flag.
    * ``task_info_`` — task type, regression flag, LSS flag, family, num_classes,
      and ``classes_`` for classification tasks.
    * ``versions_`` — Python, platform, and package version snapshot at save time.
    * ``classes_`` — numpy array of class labels (classification only; ``None`` otherwise).
    * ``input_columns_`` — ordered list of feature column names seen during fit.
    * ``n_features_in_`` — number of features the model was trained on.
    * ``feature_names_in_`` — numpy array of feature names (when all columns are strings).

    Examples
    --------
    Inspect a loaded model's metadata without re-fitting:

    >>> loaded = MLPClassifier.load("my_model.pt")

    Check task and class information:

    >>> loaded.task_info_["task"]
    'classification'
    >>> loaded.classes_
    array([0, 1, 2])

    Verify the feature schema matches your inference data:

    >>> loaded.input_columns_
    ['age', 'income', 'score']
    >>> loaded.n_features_in_
    3

    Inspect the version snapshot from when the model was saved:

    >>> loaded.versions_["packages"]["torch"]
    '2.7.0'
    >>> loaded.versions_["python"]
    '3.11.9'

    Check the architecture that was saved:

    >>> loaded.architecture_metadata_["name"]
    'MLP'
    >>> loaded.architecture_metadata_["config_class"]
    'MLPConfig'
    """
    artifact_metadata = bundle.get("artifact_metadata", {})
    task_info = bundle.get("task_info") or artifact_metadata.get("task", {})
    feature_schema = bundle.get("feature_schema") or artifact_metadata.get("feature_schema")

    obj.artifact_metadata_ = artifact_metadata
    obj.architecture_metadata_ = bundle.get("architecture_metadata") or artifact_metadata.get("architecture")
    obj.feature_schema_ = feature_schema
    obj.preprocessing_metadata_ = bundle.get("preprocessing_metadata") or artifact_metadata.get("preprocessing")
    obj.task_info_ = task_info
    obj.versions_ = bundle.get("versions") or artifact_metadata.get("versions")
    classes = bundle.get("classes_", task_info.get("classes_") if isinstance(task_info, dict) else None)
    obj.classes_ = np.asarray(classes) if classes is not None else None
    obj.input_columns_ = bundle.get("input_columns")
    if obj.input_columns_ is None and isinstance(feature_schema, dict):
        obj.input_columns_ = feature_schema.get("column_order")
    obj.n_features_in_ = bundle.get("n_features_in_")
    if obj.n_features_in_ is None and obj.input_columns_ is not None:
        obj.n_features_in_ = len(obj.input_columns_)
    feature_names = bundle.get("feature_names_in_")
    if (
        feature_names is None
        and obj.input_columns_ is not None
        and all(isinstance(column, str) for column in obj.input_columns_)
    ):
        feature_names = obj.input_columns_
    if feature_names is not None:
        obj.feature_names_in_ = np.asarray(feature_names, dtype=object)


def _package_version(distribution_name: str) -> str | None:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return None


def _simplify(value: Any) -> Any:
    """Convert common Python/scientific objects into metadata-friendly values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {_simplify_dict_key(key): _simplify(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set):
        return [_simplify(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _simplify(value.tolist())
        except Exception:
            return repr(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _simplify(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, type):
        return {"class_name": value.__name__, "module": value.__module__}
    return repr(value)


def _simplify_dict_key(value: Any) -> Any:
    simplified = _simplify(value)
    if isinstance(simplified, dict | list):
        return repr(simplified)
    return simplified
