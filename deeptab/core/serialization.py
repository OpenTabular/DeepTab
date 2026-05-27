"""Serialization helpers for model weights and fitted estimator artifacts."""

from __future__ import annotations

import platform
from dataclasses import fields, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

ARTIFACT_FORMAT_VERSION = 2


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


def restore_loaded_metadata(obj: Any, bundle: dict[str, Any]) -> None:
    """Attach metadata fields to an estimator restored from a saved artifact."""
    artifact_metadata = bundle.get("artifact_metadata", {})
    task_info = bundle.get("task_info") or artifact_metadata.get("task", {})
    feature_schema = bundle.get("feature_schema") or artifact_metadata.get("feature_schema")

    obj.artifact_metadata_ = artifact_metadata
    obj.architecture_metadata_ = bundle.get("architecture_metadata") or artifact_metadata.get("architecture")
    obj.feature_schema_ = feature_schema
    obj.preprocessing_metadata_ = bundle.get("preprocessing_metadata") or artifact_metadata.get("preprocessing")
    obj.task_info_ = task_info
    obj.versions_ = bundle.get("versions") or artifact_metadata.get("versions")
    obj.classes_ = bundle.get("classes_", task_info.get("classes_") if isinstance(task_info, dict) else None)
    obj.input_columns_ = bundle.get("input_columns")
    if obj.input_columns_ is None and isinstance(feature_schema, dict):
        obj.input_columns_ = feature_schema.get("column_order")


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
