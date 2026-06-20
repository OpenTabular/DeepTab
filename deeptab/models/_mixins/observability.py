"""Lifecycle event dispatch for all DeepTab estimators.

All estimators emit named events at key points in the fit / predict /
serialise lifecycle via ``_emit_event``.  This module provides the default
no-op implementation so the call sites work without any configuration.

To receive events, pass an ``ObservabilityConfig`` at construction time::

    from deeptab.core.observability import ObservabilityConfig

    obs = ObservabilityConfig(structured_logging=True)
    clf = MLPClassifier(observability_config=obs)
    clf.fit(X, y)   # fit_started, model_built, … are now logged

Or configure after construction::

    clf.configure_observability(obs)

The full event inventory is documented in the architecture plan:
``dev/documentation/deeptab-modules/architecture_improvement_v0.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from deeptab.core.observability import ObservabilityConfig


class _SupportsInfo(Protocol):
    """Structural type for any logger that accepts named lifecycle events.

    Any object with an ``info(event: str, **kwargs) -> None`` method satisfies
    this Protocol — ``structlog`` bound-loggers, ``logging.Logger`` adapters,
    or simple test doubles all qualify.
    """

    def info(self, event: str, **kwargs: Any) -> None: ...


class _NoOpEventLogger:
    """Logger that silently discards every event.

    Used as the default when no real logger has been attached to an
    estimator.  Its interface mirrors the ``structlog`` bound-logger API
    so that swapping in a real backend requires no changes at the call
    site.
    """

    def info(self, event: str, **kwargs: Any) -> None:
        pass


class _ObservabilityMixin:
    """Provide lifecycle event dispatch to all DeepTab estimators.

    Use ``configure_observability`` to attach a backend::

        from deeptab.core.observability import ObservabilityConfig
        clf.configure_observability(ObservabilityConfig(structured_logging=True))

    When ``_event_logger`` is ``None`` (the default) all events are
    silently discarded via ``_NoOpEventLogger`` semantics.
    """

    _event_logger: _SupportsInfo | None = None
    _run_id: str | None = None  # set per fit() call; auto-injected into every event
    _run_dir: str | None = None  # per-run output directory (set at fit start)
    _fit_start_ms: float = 0.0  # monotonic timestamp at fit() start

    def configure_observability(self, config: ObservabilityConfig) -> None:
        """Wire up logging backends described by *config*.

        Can be called at any point — before or after ``fit()``.  Changes take
        effect on the next lifecycle event emitted (i.e. the next ``fit()``
        or ``predict()`` call).

        Parameters
        ----------
        config : ObservabilityConfig
            Observability settings.  Imports optional dependencies lazily;
            raises ``ImportError`` with install hints if they are absent.
        """
        from deeptab.core.observability import build_structlog_logger

        # Always store the config so fit() can access it for run-dir creation,
        # Lightning loggers, and MLflow metadata logging.
        self._observability_config = config  # type: ignore[attr-defined]

        if config.structured_logging:
            self._event_logger = build_structlog_logger(config)

    def _emit_event(self, event: str, **kwargs: Any) -> None:
        """Dispatch a named lifecycle event to the attached logger.

        Automatically prepends ``run_id`` from the current fit run when
        one is active, so call sites never need to pass it explicitly.

        Parameters
        ----------
        event : str
            Dot-namespaced event name, e.g. ``"fit.started"``, ``"train.completed"``.
        **kwargs
            Arbitrary key-value context attached to the event.
        """
        if self._event_logger is not None:
            run_id = getattr(self, "_run_id", None)
            if run_id is not None and "run_id" not in kwargs:
                self._event_logger.info(event, run_id=run_id, **kwargs)
            else:
                self._event_logger.info(event, **kwargs)
