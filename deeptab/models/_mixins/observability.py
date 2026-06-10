"""Lifecycle event dispatch for all DeepTab estimators.

All estimators emit named events at key points in the fit / predict /
serialise lifecycle via ``_emit_event``.  This module provides the default
no-op implementation so the call sites work without any configuration.

To receive events, replace ``_event_logger`` on an estimator instance with
any object that exposes ``info(event: str, **kwargs) -> None``::

    import structlog
    clf._event_logger = structlog.get_logger()
    clf.fit(X, y)   # fit_started, model_built, … are now logged

The full event inventory is documented in the architecture plan:
``dev/documentation/deeptab-modules/architecture_improvement_v0.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


class _SupportsInfo(Protocol):
    """Structural type for any logger that accepts named lifecycle events.

    Any object with an ``info(event: str, **kwargs) -> None`` method satisfies
    this Protocol — ``structlog`` bound-loggers, ``logging.Logger`` adapters,
    or simple test doubles all qualify.
    """

    def info(self, event: str, **kwargs) -> None: ...


class _NoOpEventLogger:
    """Logger that silently discards every event.

    Used as the default when no real logger has been attached to an
    estimator.  Its interface mirrors the ``structlog`` bound-logger API
    so that swapping in a real backend requires no changes at the call
    site.
    """

    def info(self, event: str, **kwargs) -> None:
        pass


class _ObservabilityMixin:
    """Provide lifecycle event dispatch to all DeepTab estimators.

    Attach a logger to start receiving events::

        clf._event_logger = structlog.get_logger()

    Any object with an ``info(event: str, **kwargs) -> None`` method is
    accepted — standard ``logging.Logger``, ``structlog`` loggers, and
    simple callables all work.

    When ``_event_logger`` is ``None`` (the default) all events are
    silently discarded via ``_NoOpEventLogger`` semantics.
    """

    _event_logger: _SupportsInfo | None = None

    def _emit_event(self, event: str, **kwargs) -> None:
        """Dispatch a named lifecycle event to the attached logger.

        Parameters
        ----------
        event : str
            Event name, e.g. ``"fit_started"``, ``"model_built"``.
        **kwargs
            Arbitrary key-value context attached to the event
            (e.g. ``n_samples=1000``, ``path="model.pt"``).
        """
        if self._event_logger is not None:
            self._event_logger.info(event, **kwargs)
