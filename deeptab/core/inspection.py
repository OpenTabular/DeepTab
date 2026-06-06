from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
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
        task_model = self.task_model  # pyright: ignore[reportAttributeAccessIssue]
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

    def profile(
        self,
        X,
        y,
        dry_run: bool = True,
        n_forward_passes: int = 3,
        batch_size: int | None = None,
        random_state: int = 0,
    ) -> dict[str, Any]:
        """Build the model on a small data sample and run a dry forward pass.

        Combines :meth:`describe`, :meth:`runtime_info`, and a timed forward
        pass to give a complete pre-training picture without running any
        gradient updates.

        Parameters
        ----------
        X : DataFrame or array-like
            Feature matrix. The first ``min(256, len(X))`` rows are used for
            the dry-run build.
        y : array-like
            Target vector aligned with *X*.
        dry_run : bool, default=True
            When ``True`` the temporary model is discarded after profiling so
            the estimator's state is left unchanged (unless the model was
            already built, in which case the existing model is used directly).
        n_forward_passes : int, default=3
            Number of forward passes used to estimate per-batch runtime. The
            median is reported to reduce noise.
        batch_size : int or None, default=None
            Override the batch size used for timing. Defaults to the value in
            ``trainer_config`` or 64.
        random_state : int, default=0
            Seed passed to the dry-run build for reproducibility.

        Returns
        -------
        dict
            Keys:

            ``builds``
                ``True`` if the model constructed without error.
            ``error``
                Exception message when ``builds`` is ``False``, else ``None``.
            ``device``
                Device string (e.g. ``"cpu"``, ``"mps:0"``, ``"cuda:0"``).
            ``dtype``
                Parameter dtype string (e.g. ``"float32"``).
            ``total_params``
                Total number of model parameters.
            ``trainable_params``
                Number of trainable parameters.
            ``memory_mb``
                Estimated parameter memory in megabytes.
            ``batch_shape``
                Shape of the first dummy batch drawn from the data module.
            ``output_shape``
                Shape of the model output for that dummy batch (``None`` on error).
            ``loss_fct``
                Class name of the loss function.
            ``forward_ms_median``
                Median forward-pass wall time in milliseconds (``None`` on error).
            ``forward_ms_min``
                Minimum forward-pass wall time in milliseconds (``None`` on error).
            ``describe``
                Full :meth:`describe` dict (populated after build).
            ``runtime``
                Full :meth:`runtime_info` dict (populated after build).
        """
        was_already_built = bool(getattr(self, "built", False))

        result: dict[str, Any] = {
            "builds": False,
            "error": None,
            "device": None,
            "dtype": None,
            "total_params": None,
            "trainable_params": None,
            "memory_mb": None,
            "batch_shape": None,
            "output_shape": None,
            "loss_fct": None,
            "forward_ms_median": None,
            "forward_ms_min": None,
            "describe": None,
            "runtime": None,
        }

        try:
            # ── 1. Build on a small sample if not already built ──────────────
            if not was_already_built:
                n_sample = min(256, len(y))
                idx = np.random.default_rng(random_state).choice(len(y), size=n_sample, replace=False)
                X_sample = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
                y_sample = y[idx] if isinstance(y, np.ndarray) else np.asarray(y)[idx]

                # Determine task type from class hierarchy — used by build_fn
                # internally; we only need to detect it for build dispatch.
                build_fn = getattr(self, "build_model", None)
                if build_fn is None:
                    raise RuntimeError("Estimator does not expose a build_model() method.")

                tc = getattr(self, "trainer_config", None)
                _bs = batch_size or (tc.batch_size if tc is not None else 64)

                build_fn(
                    X_sample,
                    y_sample,
                    val_size=0.2,
                    batch_size=_bs,
                    random_state=random_state,
                )
            else:
                tc = getattr(self, "trainer_config", None)
                _bs = batch_size or (tc.batch_size if tc is not None else 64)

            result["builds"] = True

            # ── 2. Parameter counts & memory ─────────────────────────────────
            task_model = getattr(self, "task_model", None)
            counts = self._parameter_counts()
            result["total_params"] = counts["total"]
            result["trainable_params"] = counts["trainable"]

            first_param = _first_parameter(task_model)
            if first_param is not None:
                result["device"] = str(first_param.device)
                dtype_str = str(first_param.dtype).replace("torch.", "")
                result["dtype"] = dtype_str
                _bytes_per_elem = {"float32": 4, "float16": 2, "bfloat16": 2, "float64": 8}.get(dtype_str, 4)
                result["memory_mb"] = round(counts["total"] * _bytes_per_elem / (1024**2), 3)

            # ── 3. Loss function info ─────────────────────────────────────────
            if task_model is not None:
                result["loss_fct"] = _safe_class_name(getattr(task_model, "loss_fct", None))

            # ── 4. Dummy forward pass — shape + timing ────────────────────────
            data_module = getattr(self, "data_module", None)
            if task_model is not None and data_module is not None:
                try:
                    data_module.setup("fit")
                    train_loader = data_module.train_dataloader()
                    raw_batch = next(iter(train_loader))

                    # Batch format: ((num_feats, cat_feats, embeddings), labels)
                    feat_tuple, _labels = raw_batch
                    num_feats, cat_feats, embeddings = feat_tuple

                    result["batch_shape"] = {
                        "num_features": [list(t.shape) for t in num_feats] if num_feats else [],
                        "cat_features": [list(t.shape) for t in cat_feats] if cat_feats else [],
                        "labels": list(_labels.shape),
                    }

                    task_model.eval()
                    device = first_param.device if first_param is not None else torch.device("cpu")

                    num_feats_dev = [t.to(device) for t in num_feats] if num_feats else []
                    cat_feats_dev = [t.to(device) for t in cat_feats] if cat_feats else []
                    # Embeddings: pass through as-is (may be None or [None, ...]);
                    # the estimator handles both just as training_step does.
                    emb_dev = (
                        [t.to(device) for t in embeddings]
                        if embeddings and all(t is not None for t in embeddings)
                        else embeddings
                    )

                    timings: list[float] = []
                    with torch.no_grad():
                        for _ in range(n_forward_passes):
                            t0 = time.perf_counter()
                            task_model.estimator(num_feats_dev, cat_feats_dev, emb_dev)
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            timings.append((time.perf_counter() - t0) * 1000)

                    # Capture output shape from a final pass
                    with torch.no_grad():
                        out = task_model.estimator(num_feats_dev, cat_feats_dev, emb_dev)
                    result["output_shape"] = list(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__
                    result["forward_ms_median"] = round(float(np.median(timings)), 3)
                    result["forward_ms_min"] = round(float(np.min(timings)), 3)
                except Exception as fwd_err:
                    result["output_shape"] = None
                    result["error"] = f"forward pass failed: {fwd_err}"

            # ── 5. Attach describe / runtime snapshots ────────────────────────
            result["describe"] = self.describe()
            result["runtime"] = self.runtime_info()

        except Exception as build_err:
            result["builds"] = False
            result["error"] = str(build_err)

        finally:
            # Tear down the temporary build so the estimator is left unfitted
            if dry_run and not was_already_built:
                self.task_model = None
                self.built = False
                if hasattr(self, "data_module"):
                    self.data_module = None  # type: ignore[assignment]
                if hasattr(self, "is_fitted_"):
                    self.is_fitted_ = False

        return result
