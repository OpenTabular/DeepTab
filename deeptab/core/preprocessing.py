"""Central boundary for constructing PreTab's ``Preprocessor``.

Every DeepTab estimator builds its preprocessor through :func:`build_preprocessor`
instead of importing and instantiating ``pretab.Preprocessor`` directly, so PreTab
stays isolated behind one internal contract.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pretab import Preprocessor
from pretab import list_representations as _list_representations

from deeptab.core.exceptions import warn_config

if TYPE_CHECKING:
    from deeptab.configs import PreprocessingConfig
    from deeptab.core.observability import ObservabilityConfig

__all__ = ["build_preprocessor", "list_available_representations"]

# DeepTab's tensor pipeline (deeptab/data/datamodule.py) consumes the preprocessor's
# transformed output as a dict of per-feature blocks (`num_<col>`, `cat_<col>`).
# PreTab defaults to returning a single stacked matrix instead, so these are forced
# explicitly rather than left to PreTab's own default.
_FORCED_PREPROCESSOR_KWARGS = {"output_structure": "blocks", "output_format": "dense"}

_PRETAB_LOGGER_NAME = "pretab"


class _PretabConsoleHandler(logging.Handler):
    """Routes PreTab's shared ``"pretab"`` logger through DeepTab's own console output.

    PreTab only attaches its own default handler (and sets the logger's level)
    when the ``"pretab"`` logger has none yet; attaching this one first keeps a
    single ``ObservabilityConfig`` knob authoritative instead of exposing a
    separate PreTab verbose flag.
    """

    def emit(self, record: logging.LogRecord) -> None:
        print(self.format(record))


def _attach_pretab_console_logging(observability_config: ObservabilityConfig | None) -> None:
    """Attach :class:`_PretabConsoleHandler` once, when structured console logging is on."""
    if observability_config is None or not getattr(observability_config, "structured_logging", False):
        return
    if not getattr(observability_config, "log_to_console", True):
        return
    pretab_logger = logging.getLogger(_PRETAB_LOGGER_NAME)
    if any(isinstance(handler, _PretabConsoleHandler) for handler in pretab_logger.handlers):
        return
    handler = _PretabConsoleHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    pretab_logger.addHandler(handler)
    pretab_logger.propagate = False
    # PreTab's own `verbose` int already gates which messages it emits; without
    # this, Python's logging level (WARNING by default) would filter them out
    # before they ever reach the handler above.
    pretab_logger.setLevel(logging.DEBUG)


def build_preprocessor(
    preprocessing_config: PreprocessingConfig | None = None,
    *,
    task: str | None = None,
    random_state: int | None = None,
    observability_config: ObservabilityConfig | None = None,
    for_external_embeddings: bool = False,
) -> Preprocessor:
    """Construct a PreTab ``Preprocessor`` from a resolved ``PreprocessingConfig``.

    Parameters
    ----------
    preprocessing_config : PreprocessingConfig or None, default=None
        Resolved preprocessing options. ``None`` falls back to PreTab's own
        defaults.
    task : str or None, default=None
        Task resolved from the calling estimator's type (e.g. ``"classification"``
        for a classifier, ``"regression"`` for a regressor or LSS estimator). This
        always takes precedence over a task set directly on *preprocessing_config*;
        a conflicting value there emits a ``ConfigWarning`` rather than being used.
    random_state : int or None, default=None
        Seed resolved by the calling estimator, forwarded to PreTab's
        ``random_state`` argument.
    observability_config : ObservabilityConfig or None, default=None
        When given, ``observability_config.verbosity`` is forwarded as PreTab's
        ``verbose`` level (both use the same 0-3 scale), and, when
        ``structured_logging=True``, DeepTab's own console handler is attached
        to PreTab's shared ``"pretab"`` logger. This is the only knob that
        controls PreTab's fit-time logging; no separate PreTab verbose flag is
        exposed.
    for_external_embeddings : bool, default=False
        Reserved for the external-embedding-group construction path.

    Returns
    -------
    Preprocessor
        A configured, unfitted PreTab preprocessor. Always uses block-structured,
        dense output (``output_structure="blocks"``, ``output_format="dense"``);
        these are not currently user-configurable.
    """
    kwargs = (
        preprocessing_config.to_preprocessor_kwargs()
        if preprocessing_config is not None and hasattr(preprocessing_config, "to_preprocessor_kwargs")
        else {}
    )
    if task is not None:
        user_task = kwargs.get("task")
        if user_task is not None and user_task != task:
            warn_config(
                f"PreprocessingConfig.task={user_task!r} conflicts with the task resolved from the "
                f"estimator ({task!r}). The estimator's task always takes precedence; remove "
                "PreprocessingConfig.task to silence this warning.",
                stacklevel=4,
            )
        kwargs["task"] = task
    if random_state is not None:
        kwargs.setdefault("random_state", random_state)
    if observability_config is not None and hasattr(observability_config, "verbosity"):
        kwargs["verbose"] = max(0, min(observability_config.verbosity, 3))
        _attach_pretab_console_logging(observability_config)
    kwargs.update(_FORCED_PREPROCESSOR_KWARGS)
    return Preprocessor(**kwargs)


def list_available_representations(
    *,
    feature_kind: str | None = None,
    scope: str | None = None,
    supervised: bool | None = None,
    adaptive: bool | None = None,
) -> list[str]:
    """List PreTab representation names usable as `numerical_method`/`categorical_method`.

    A thin, read-only pass-through to PreTab's own representation registry, kept
    here so discovering available methods does not require importing ``pretab``
    directly. Every name returned is queried live from PreTab's registry, so it
    stays current as PreTab adds representations; it is not cross-checked against
    :class:`~deeptab.configs.PreprocessingConfig`'s own accepted values here, since
    not every registered representation is valid as a `Preprocessor` constructor
    argument (some are standalone-only transformers).

    Parameters
    ----------
    feature_kind : {"numerical", "categorical"} or None, default=None
        Keep only representations that apply to this column kind.
    scope : {"univariate", "multivariate"} or None, default=None
        Keep only representations with this arity.
    supervised : bool or None, default=None
        Keep only representations that can (``True``) or cannot (``False``)
        consume the target.
    adaptive : bool or None, default=None
        Keep only representations whose adaptive-resolution support matches.

    Returns
    -------
    list of str
        Matching canonical representation names, sorted.
    """
    return _list_representations(
        feature_kind=feature_kind,
        scope=scope,
        supervised=supervised,
        adaptive=adaptive,
    )
