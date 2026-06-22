"""Guards for optional third-party dependencies.

DeepTab keeps its default install lightweight: structured logging and the
experiment-tracking backends are shipped as optional extras. Each helper here
imports one optional dependency on demand and raises a clear, actionable
:class:`ImportError` — pointing at the matching ``pip install 'deeptab[...]'``
command — when the package is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType


def require_structlog() -> ModuleType:
    """Return the :mod:`structlog` module.

    Returns
    -------
    module
        The imported ``structlog`` module.

    Raises
    ------
    ImportError
        If ``structlog`` is not installed, with an actionable install hint.
    """
    try:
        import structlog  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Structured logging requires the optional 'structlog' dependency. "
            "Install it with: pip install 'deeptab[logs]'"
        ) from exc

    return structlog


def require_mlflow() -> ModuleType:
    """Return the :mod:`mlflow` module.

    Returns
    -------
    module
        The imported ``mlflow`` module.

    Raises
    ------
    ImportError
        If ``mlflow`` is not installed, with an actionable install hint.
    """
    try:
        import mlflow  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "MLflow tracking requires the optional 'mlflow' dependency. Install it with: pip install 'deeptab[mlflow]'"
        ) from exc

    return mlflow


def require_tensorboard() -> Any:
    """Return ``torch.utils.tensorboard.SummaryWriter``.

    Returns
    -------
    type
        The ``SummaryWriter`` class from ``torch.utils.tensorboard``.

    Raises
    ------
    ImportError
        If ``tensorboard`` is not installed, with an actionable install hint.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard logging requires the optional 'tensorboard' dependency. "
            "Install it with: pip install 'deeptab[tensorboard]'"
        ) from exc

    return SummaryWriter
