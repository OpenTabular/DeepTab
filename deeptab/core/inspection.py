from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd
import torch
import torch.nn as nn


class ImportanceGetter(nn.Module):  # Figure 3 part 1
    def __init__(self, P, C, d):
        super().__init__()
        self.colemb = nn.Parameter(torch.empty(C, d))
        self.pemb = nn.Parameter(torch.empty(P, d))
        torch.nn.init.normal_(self.colemb, std=0.01)
        torch.nn.init.normal_(self.pemb, std=0.01)
        self.C = C
        self.P = P
        self.d = d
        self.dense = nn.Linear(2 * self.d, self.d)
        self.laynorm1 = nn.LayerNorm(self.d)
        self.laynorm2 = nn.LayerNorm(self.d)

    def forward(self, O):  # noqa: E741
        eprompt = self.pemb.unsqueeze(0).repeat(O.shape[0], 1, 1)

        dense_out = self.dense(torch.cat((self.laynorm1(eprompt), O), dim=-1))

        dense_out = dense_out + eprompt + O

        ecolumn = self.laynorm2(self.colemb.unsqueeze(0).repeat(O.shape[0], 1, 1))

        return torch.softmax(dense_out @ ecolumn.transpose(1, 2), dim=-1)


def get_feature_dimensions(num_feature_info, cat_feature_info, embedding_info):
    input_dim = 0
    for _, feature_info in num_feature_info.items():
        input_dim += feature_info["dimension"]
    for _, feature_info in cat_feature_info.items():
        input_dim += feature_info["dimension"]
    for _, feature_info in embedding_info.items():
        input_dim += feature_info["dimension"]

    return input_dim


def _safe_class_name(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, type):
        return obj.__name__
    return type(obj).__name__


def _first_parameter(module: nn.Module | None):
    if module is None:
        return None
    return next(module.parameters(), None)


def _config_to_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    get_params = getattr(config, "get_params", None)
    if callable(get_params):
        return get_params(deep=False)  # type: ignore[return-value]
    config_vars: dict[str, Any] = getattr(config, "__dict__", {})
    return {key: value for key, value in config_vars.items() if not key.startswith("_") and not callable(value)}


class InspectionMixin:
    """Shared model-inspection interface for sklearn-style DeepTab estimators."""

    def _require_built_for_inspection(self) -> None:
        if not getattr(self, "built", False) or getattr(self, "task_model", None) is None:
            raise ValueError("The model must be built or fitted before this inspection method can be used.")

    def _architecture(self) -> nn.Module | None:
        task_model = getattr(self, "task_model", None)
        if task_model is not None:
            return getattr(task_model, "estimator", None)
        estimator = getattr(self, "estimator", None)
        return estimator if isinstance(estimator, nn.Module) else None

    def _parameter_counts(self) -> dict[str, int]:
        task_model = getattr(self, "task_model", None)
        if task_model is None:
            return {"total": 0, "trainable": 0, "non_trainable": 0}

        total = sum(p.numel() for p in task_model.parameters())
        trainable = sum(p.numel() for p in task_model.parameters() if p.requires_grad)
        return {
            "total": int(total),
            "trainable": int(trainable),
            "non_trainable": int(total - trainable),
        }

    def describe(self) -> dict[str, Any]:
        """Return a structured description of the estimator and fitted model.

        The method is safe to call before fitting. Parameter counts and feature
        metadata are included only after the model has been built.
        """
        data_module = getattr(self, "data_module", None)
        task_model = getattr(self, "task_model", None)
        architecture = self._architecture()
        config = getattr(self, "config", None)

        feature_counts = None
        if data_module is not None:
            feature_counts = {
                "numerical": len(getattr(data_module, "num_feature_info", {}) or {}),
                "categorical": len(getattr(data_module, "cat_feature_info", {}) or {}),
                "embedding": len(getattr(data_module, "embedding_feature_info", {}) or {}),
            }
            feature_counts["total"] = sum(feature_counts.values())

        task = "unknown"
        if task_model is not None and getattr(task_model, "lss", False):
            task = "distributional_regression"
        elif data_module is not None:
            task = "regression" if getattr(data_module, "regression", False) else "classification"
        elif type(self).__name__.endswith("Regressor"):
            task = "regression"
        elif type(self).__name__.endswith("Classifier"):
            task = "classification"
        elif type(self).__name__.endswith("LSS"):
            task = "distributional_regression"

        return {
            "estimator": type(self).__name__,
            "architecture": _safe_class_name(architecture) or _safe_class_name(getattr(self, "estimator", None)),
            "task": task,
            "built": bool(getattr(self, "built", False)),
            "fitted": bool(getattr(self, "is_fitted_", False)),
            "model_config": _safe_class_name(config),
            "preprocessing_config": _safe_class_name(getattr(self, "preprocessing_config", None)),
            "trainer_config": _safe_class_name(getattr(self, "trainer_config", None)),
            "feature_counts": feature_counts,
            "num_classes": getattr(task_model, "num_classes", None),
            "family": getattr(self, "family_name", None) or _safe_class_name(getattr(task_model, "family", None)),
            "returns_ensemble": getattr(architecture, "returns_ensemble", None),
            "parameters": self._parameter_counts() if task_model is not None else None,
        }

    def summary(self) -> str:
        """Return a compact human-readable model summary."""
        info = self.describe()
        lines = [
            f"{info['estimator']} summary",
            f"  Architecture: {info['architecture']}",
            f"  Task: {info['task']}",
            f"  Built: {info['built']}",
            f"  Fitted: {info['fitted']}",
            f"  Model config: {info['model_config']}",
        ]

        if info["feature_counts"] is not None:
            counts = info["feature_counts"]
            lines.append(
                "  Features: "
                f"{counts['total']} total "
                f"({counts['numerical']} numerical, "
                f"{counts['categorical']} categorical, "
                f"{counts['embedding']} embedding)"
            )

        if info["parameters"] is not None:
            params = info["parameters"]
            lines.append(
                "  Parameters: "
                f"{params['total']:,} total, "
                f"{params['trainable']:,} trainable, "
                f"{params['non_trainable']:,} non-trainable"
            )

        runtime = self.runtime_info()
        if runtime["device"] is not None:
            lines.append(f"  Device: {runtime['device']}")
        if runtime["precision"] is not None:
            lines.append(f"  Precision: {runtime['precision']}")
        if runtime["accelerator"] is not None:
            lines.append(f"  Accelerator: {runtime['accelerator']}")

        return "\n".join(lines)

    def parameter_table(self, trainable_only: bool = False) -> pd.DataFrame:
        """Return one row per model parameter as a pandas DataFrame.

        Parameters
        ----------
        trainable_only : bool, default=False
            If True, include only parameters with ``requires_grad=True``.
        """
        self._require_built_for_inspection()
        task_model = self.task_model
        if task_model is None:
            raise RuntimeError("The model must be built before calling parameter_table.")

        rows = []
        for name, param in task_model.named_parameters():
            if trainable_only and not param.requires_grad:
                continue
            module = name.rsplit(".", 1)[0] if "." in name else ""
            rows.append(
                {
                    "name": name,
                    "module": module,
                    "shape": tuple(param.shape),
                    "num_params": int(param.numel()),
                    "trainable": bool(param.requires_grad),
                    "dtype": str(param.dtype).replace("torch.", ""),
                    "device": str(param.device),
                }
            )

        return pd.DataFrame(
            rows,
            columns=["name", "module", "shape", "num_params", "trainable", "dtype", "device"],  # type: ignore[call-overload]
        )

    def runtime_info(self) -> dict[str, Any]:
        """Return runtime setup information for the estimator.

        The method is safe to call before fitting. Device and dtype are inferred
        from model parameters when a model has been built.
        """
        task_model = getattr(self, "task_model", None)
        trainer = getattr(self, "trainer", None)
        data_module = getattr(self, "data_module", None)
        first_param = _first_parameter(task_model)

        accelerator = getattr(trainer, "accelerator", None)
        strategy = getattr(trainer, "strategy", None)
        precision_plugin = getattr(trainer, "precision_plugin", None)
        logger = getattr(trainer, "logger", None)

        trainer_config = getattr(self, "trainer_config", None)
        trainer_config_values = _config_to_dict(trainer_config)

        return {
            "built": bool(getattr(self, "built", False)),
            "fitted": bool(getattr(self, "is_fitted_", False)),
            "device": str(first_param.device) if first_param is not None else None,
            "dtype": str(first_param.dtype).replace("torch.", "") if first_param is not None else None,
            "precision": getattr(trainer, "precision", None) or getattr(precision_plugin, "precision", None),
            "accelerator": _safe_class_name(accelerator),
            "strategy": _safe_class_name(strategy),
            "num_devices": getattr(trainer, "num_devices", None),
            "root_device": str(getattr(strategy, "root_device", "")) if strategy is not None else None,
            "max_epochs": getattr(trainer, "max_epochs", None)
            if trainer is not None
            else trainer_config_values.get("max_epochs"),
            "current_epoch": getattr(trainer, "current_epoch", None),
            "global_step": getattr(trainer, "global_step", None),
            "batch_size": getattr(data_module, "batch_size", None) or trainer_config_values.get("batch_size"),
            "optimizer_type": getattr(self, "optimizer_type", None),
            "lr": getattr(task_model, "lr", None) if task_model is not None else trainer_config_values.get("lr"),
            "weight_decay": getattr(task_model, "weight_decay", None)
            if task_model is not None
            else trainer_config_values.get("weight_decay"),
            "logger": _safe_class_name(logger),
            "deterministic": getattr(trainer, "deterministic", None),
        }
