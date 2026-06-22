"""Observability configuration and backend construction for DeepTab.

Provides:
- ``ObservabilityConfig`` — dataclass that controls all logging and
  experiment-tracking behaviour.
- ``build_structlog_logger`` — configures and returns a structlog-backed
  logger when ``structured_logging=True``.
- ``build_lightning_loggers`` — constructs the list of Lightning loggers
  from an ``ObservabilityConfig``.
- ``create_run_dir`` — creates the per-run output directory tree.
- ``write_run_config`` — serialises estimator params to ``config.yaml``.
- ``write_run_summary`` — writes final metrics to ``summary.json``.

All optional dependencies (structlog, mlflow, tensorboard) are imported
lazily inside their respective factory functions, never at module level.
The core library therefore remains zero-dependency by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Verbosity level constants
# ---------------------------------------------------------------------------

#: Events emitted at verbosity=1 (important milestones only).
_VERBOSITY_1: frozenset[str] = frozenset(
    {
        "fit.started",
        "model.created",
        "train.completed",
        "fit.completed",
    }
)

#: Events emitted at verbosity=2 (adds data/training setup details).
_VERBOSITY_2: frozenset[str] = _VERBOSITY_1 | frozenset(
    {
        "data.created",
        "train.started",
    }
)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class ObservabilityConfig:
    """Controls all logging and experiment-tracking behaviour.

    All output paths are derived from ``root_dir`` by default, producing
    a single organised directory tree::

        <root_dir>/
        ├── runs/
        │   └── <experiment_name>/
        │       └── <YYYYMMDD_HHMMSS>_<run_id>/
        │           ├── config.yaml     ← estimator hyperparams
        │           ├── lifecycle.jsonl ← structured event log
        │           ├── summary.json    ← final metrics
        │           └── checkpoints/
        │               └── best.ckpt
        ├── tensorboard/
        │   └── <experiment_name>/
        │       └── <YYYYMMDD_HHMMSS>_<run_id>/
        │           └── events.out.tfevents…
        └── mlflow/
            ├── backend/
            │   └── mlflow.db
            └── artifacts/

    Parameters
    ----------
    root_dir : str, default="deeptab_runs"
        Base directory for all observability outputs.
    experiment_name : str, default="default"
        Logical experiment label used to group related runs.
    structured_logging : bool, default=False
        Enable structured runtime logging via ``structlog``.
        Lifecycle events are emitted as structured log records.
        Requires ``structlog``: ``pip install 'deeptab[logs]'``.
    log_to_console : bool, default=True
        Stream compact human-readable output to stdout.
    log_to_file : bool, default=False
        Write a per-run ``lifecycle.jsonl`` inside the run directory.
    verbosity : int, default=1
        Controls which lifecycle events are emitted when
        ``structured_logging=True``.  Levels:

        * ``0`` — silent.
        * ``1`` — milestones: ``fit.started``, ``model.created``,
          ``train.completed``, ``fit.completed``.
        * ``2`` — detailed: level-1 plus ``data.created``,
          ``train.started``.
        * ``3`` — debug: all events.
    experiment_trackers : list of str, default=[]
        Lightning loggers to activate.  Supported values:
        ``"mlflow"``, ``"tensorboard"``.
    tensorboard_save_dir : str, default=""
        Root directory for TensorBoard event files.  Resolved to
        ``<root_dir>/tensorboard`` when empty.
    tensorboard_name : str, default="deeptab"
        Sub-directory / experiment label inside ``tensorboard_save_dir``.
    mlflow_experiment_name : str, default="deeptab"
        Name of the MLflow experiment.
    mlflow_tracking_uri : str, default=""
        MLflow tracking-server URI.  Resolved to
        ``sqlite:///<root_dir>/mlflow/backend/mlflow.db`` when empty.
    mlflow_artifact_location : str, default=""
        Root artifact store path.  Resolved to
        ``<root_dir>/mlflow/artifacts`` when empty.
    mlflow_run_name : str or None, default=None
        Human-readable label for the run.
    mlflow_log_model : bool, default=True
        Upload model checkpoints as MLflow artifacts.
    logger : Any, default=None
        A user-provided Lightning logger appended alongside any
        built-in trackers.

    Examples
    --------
    >>> obs = ObservabilityConfig(
    ...     root_dir="deeptab_runs",
    ...     experiment_name="iris_debug",
    ...     structured_logging=True,
    ...     log_to_file=True,
    ...     verbosity=2,
    ...     experiment_trackers=["tensorboard", "mlflow"],
    ... )

    Passing *obs* to an estimator and calling ``clf.fit(X, y)`` creates::

        deeptab_runs/runs/iris_debug/20260611_174830_8f3a2c/
        deeptab_runs/tensorboard/iris_debug/20260611_174830_8f3a2c/
        deeptab_runs/mlflow/backend/mlflow.db
    """

    # --- Root ---
    root_dir: str = "deeptab_runs"
    experiment_name: str = "default"

    # --- Structured runtime logging ---
    structured_logging: bool = False
    log_to_console: bool = True
    log_to_file: bool = False
    verbosity: int = 1

    # --- Experiment tracking ---
    experiment_trackers: list[str] = field(default_factory=list)

    # --- TensorBoard ---
    tensorboard_save_dir: str = ""  # resolved to {root_dir}/tensorboard
    tensorboard_name: str = "deeptab"

    # --- MLflow ---
    mlflow_experiment_name: str = "deeptab"
    mlflow_tracking_uri: str = ""  # resolved to sqlite:///{root_dir}/mlflow/backend/mlflow.db
    mlflow_artifact_location: str = ""  # resolved to {root_dir}/mlflow/artifacts
    mlflow_run_name: str | None = None
    mlflow_log_model: bool = True

    # --- Custom logger ---
    logger: Any = None

    def __post_init__(self) -> None:
        """Resolve empty sub-paths relative to ``root_dir``."""
        if not self.tensorboard_save_dir:
            self.tensorboard_save_dir = f"{self.root_dir}/tensorboard"
        if not self.mlflow_tracking_uri:
            self.mlflow_tracking_uri = f"sqlite:///{self.root_dir}/mlflow/backend/mlflow.db"
        if not self.mlflow_artifact_location:
            self.mlflow_artifact_location = f"{self.root_dir}/mlflow/artifacts"


# ---------------------------------------------------------------------------
# Per-run directory helpers
# ---------------------------------------------------------------------------


def create_run_dir(config: ObservabilityConfig, run_id: str) -> tuple[str, str]:
    """Create the per-run output directory tree and return ``(run_dir, run_dir_name)``.

    The directory is created at::

        <root_dir>/runs/<experiment_name>/<YYYYMMDD_HHMMSS>_<run_id>/

    Sub-directories ``checkpoints/`` and ``artifacts/`` are created inside.

    Parameters
    ----------
    config : ObservabilityConfig
        Provides ``root_dir`` and ``experiment_name``.
    run_id : str
        Short hex string identifying this fit call (e.g. ``"8f3a2c"``),
        typically ``uuid.uuid4().hex[:8]``.

    Returns
    -------
    tuple[str, str]
        ``(run_dir, run_dir_name)`` where *run_dir* is the absolute-or-relative
        path and *run_dir_name* is just the leaf component
        (``"<timestamp>_<run_id>"``).
    """
    import os
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{ts}_{run_id}"
    run_dir = os.path.join(config.root_dir, "runs", config.experiment_name, run_dir_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=True)
    return run_dir, run_dir_name


def write_run_config(run_dir: str, params: dict[str, Any]) -> None:
    """Serialise estimator *params* to ``config.yaml`` in *run_dir*.

    Non-serialisable values (activation functions, custom objects) are
    converted to their string representation.  Only top-level params
    (those without ``__`` in the key) are written to avoid redundancy with
    the flattened sklearn sub-params.

    Falls back to ``config.json`` when PyYAML is not available.
    """
    import json
    import os

    def _to_primitive(v: Any) -> Any:
        """Recursively convert *v* to a YAML/JSON-safe primitive."""
        if v is None or isinstance(v, bool | int | float | str):
            return v
        if isinstance(v, list | tuple):
            return [_to_primitive(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _to_primitive(vv) for k, vv in v.items()}
        # Dataclass → flatten one level
        try:
            from dataclasses import asdict, is_dataclass

            if is_dataclass(v) and not isinstance(v, type):
                return {f: _to_primitive(fv) for f, fv in asdict(v).items()}
        except Exception:  # noqa: S110
            pass
        # nn.Module (e.g. ReLU(), Identity()) → emit class name only
        try:
            import torch.nn as _nn

            if isinstance(v, _nn.Module):
                return type(v).__name__
        except ImportError:
            pass
        return str(v)

    # Keep only top-level keys (skip flattened sub-params like model_config__lr)
    top_level = {k: _to_primitive(v) for k, v in params.items() if "__" not in k}

    try:
        import yaml  # type: ignore[import-untyped]

        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as fh:
            yaml.safe_dump(top_level, fh, default_flow_style=False, sort_keys=True)
    except ImportError:
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(top_level, fh, indent=2, default=str)


def write_run_summary(run_dir: str, summary: dict[str, Any]) -> None:
    """Write final training metrics to ``summary.json`` in *run_dir*."""
    import json
    import os

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)


def build_structlog_logger(config: ObservabilityConfig, run_dir: str | None = None) -> Any:
    """Configure and return a dual-output event logger for *config*.

    Verbosity controls which events are emitted (see ``ObservabilityConfig.verbosity``).

    * **Console** (``log_to_console=True``) — compact human-readable lines
      with a short ``run=XXXXXXXX`` prefix and dot-namespaced event names.
    * **Per-run JSONL** (``log_to_file=True``, *run_dir* provided) — one
      JSON object per line written to ``<run_dir>/lifecycle.jsonl``.

    Parameters
    ----------
    config : ObservabilityConfig
        Observability settings.
    run_dir : str or None, default=None
        Path to the per-run output directory.  When ``None``, file output
        is silently skipped even if ``log_to_file=True``.

    Raises
    ------
    ImportError
        If ``structlog`` is not installed, with an actionable install hint.
    """
    from deeptab.core.optional_deps import require_structlog

    structlog = require_structlog()

    import json
    import os
    from datetime import datetime

    # -----------------------------------------------------------------------
    # Console rendering — short field aliases and value formatting
    # -----------------------------------------------------------------------
    _ALIASES: dict[str, str] = {
        "model_class": "model",
        "n_samples": "samples",
        "n_features": "features",
        "random_state": "seed",
        "n_train": "train",
        "n_val": "val",
        "n_num_features": "num",
        "n_cat_features": "cat",
        "n_params": "params",
        "max_epochs": "epochs",
        "batch_size": "batch",
        "n_epochs_run": "epochs_run",
    }

    def _fmt_console(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        if isinstance(v, int) and v >= 1_000:
            return f"{v:_}"
        if v is None:
            return "null"
        return str(v)

    def _render_console(event: str, kwargs: dict[str, Any]) -> str:
        run_id = kwargs.get("run_id", "")
        prefix = f"run={run_id}  " if run_id else ""
        kv = "  ".join(f"{_ALIASES.get(k, k)}={_fmt_console(v)}" for k, v in kwargs.items() if k != "run_id")
        # Pad event name to 16 chars so columns align across events
        return f"{prefix}{event:<16}  {kv}" if kv else f"{prefix}{event}"

    # -----------------------------------------------------------------------
    # JSONL rendering — full precision, numpy-safe
    # -----------------------------------------------------------------------
    class _JsonEncoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            try:
                import numpy as _np

                if isinstance(o, _np.integer):
                    return int(o)
                if isinstance(o, _np.floating):
                    return float(o)
            except ImportError:
                pass
            return super().default(o)

    # -----------------------------------------------------------------------
    # File handle — opened once per run, line-buffered
    # -----------------------------------------------------------------------
    _fh = None
    if config.log_to_file and run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        _fh = open(os.path.join(run_dir, "lifecycle.jsonl"), "a", encoding="utf-8", buffering=1)

    # -----------------------------------------------------------------------
    # Verbosity event filter
    # -----------------------------------------------------------------------
    _verbosity = config.verbosity

    def _is_allowed(event: str) -> bool:
        if _verbosity <= 0:
            return False
        if _verbosity == 1:
            return event in _VERBOSITY_1
        if _verbosity == 2:
            return event in _VERBOSITY_2
        return True  # verbosity >= 3: all events

    # -----------------------------------------------------------------------
    # Logger class
    # -----------------------------------------------------------------------
    class _StructlogEventLogger:
        def __del__(self) -> None:
            if _fh is not None and not _fh.closed:
                _fh.close()

        def info(self, event: str, **kwargs: Any) -> None:
            if not _is_allowed(event):
                return

            now = datetime.now()

            if config.log_to_console:
                ts = now.strftime("%Y-%m-%d %H:%M:%S")
                print(f"{ts} [info] {_render_console(event, kwargs)}")

            if config.log_to_file and _fh is not None:
                # Canonical order: timestamp, level, run_id (if present), event, then payload
                record: dict[str, Any] = {
                    "timestamp": now.isoformat(timespec="seconds"),
                    "level": "info",
                }
                if "run_id" in kwargs:
                    record["run_id"] = kwargs["run_id"]
                record["event"] = event
                for k, v in kwargs.items():
                    if k != "run_id":
                        record[k] = v
                _fh.write(json.dumps(record, cls=_JsonEncoder) + "\n")

    return _StructlogEventLogger()


# ---------------------------------------------------------------------------
# Lightning logger construction
# ---------------------------------------------------------------------------


def build_lightning_loggers(
    config: ObservabilityConfig,
    run_dir_name: str | None = None,
) -> list[Any]:
    """Construct the list of Lightning loggers described by *config*.

    Returns an empty list when no trackers are configured, which causes
    ``pl.Trainer`` to fall back to its default CSV logger.

    Parameters
    ----------
    config : ObservabilityConfig
        Observability configuration from the estimator.
    run_dir_name : str or None, default=None
        Leaf directory name for the current run
        (e.g. ``"20260611_174830_8f3a2c"``).  When provided, TensorBoard
        event files are written under
        ``<tensorboard_save_dir>/<experiment_name>/<run_dir_name>/``.

    Returns
    -------
    list
        Zero or more Lightning logger instances ready to be passed to
        ``pl.Trainer(logger=...)``.

    Raises
    ------
    ImportError
        If a requested tracker's package is not installed, with an
        actionable install hint.
    ValueError
        If ``experiment_trackers`` contains an unrecognised tracker name.
    """
    import os

    from deeptab.core.optional_deps import require_mlflow, require_tensorboard

    loggers: list[Any] = []

    for tracker in config.experiment_trackers:
        if tracker == "mlflow":
            require_mlflow()
            from lightning.pytorch.loggers import MLFlowLogger

            # Ensure the artifact location directory exists
            if config.mlflow_artifact_location:
                os.makedirs(config.mlflow_artifact_location, exist_ok=True)
            loggers.append(
                MLFlowLogger(
                    experiment_name=config.mlflow_experiment_name,
                    tracking_uri=config.mlflow_tracking_uri,
                    run_name=config.mlflow_run_name,
                    artifact_location=config.mlflow_artifact_location or None,
                    log_model=config.mlflow_log_model,
                )
            )

        elif tracker == "tensorboard":
            require_tensorboard()
            from lightning.pytorch.loggers import TensorBoardLogger

            loggers.append(
                TensorBoardLogger(
                    save_dir=config.tensorboard_save_dir,
                    name=config.experiment_name,
                    version=run_dir_name,
                )
            )

        else:
            raise ValueError(f"Unknown experiment tracker: {tracker!r}. Supported values are: 'mlflow', 'tensorboard'.")

    if config.logger is not None:
        loggers.append(config.logger)

    return loggers
