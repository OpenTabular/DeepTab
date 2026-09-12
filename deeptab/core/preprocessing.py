"""Central boundary for constructing PreTab's ``Preprocessor``.

Every DeepTab estimator builds its preprocessor through :func:`build_preprocessor`
instead of importing and instantiating ``pretab.Preprocessor`` directly, so PreTab
stays isolated behind one internal contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pretab import Preprocessor

from deeptab.core.exceptions import warn_config

if TYPE_CHECKING:
    from deeptab.configs import PreprocessingConfig

__all__ = ["build_preprocessor"]

# DeepTab's tensor pipeline (deeptab/data/datamodule.py) consumes the preprocessor's
# transformed output as a dict of per-feature blocks (`num_<col>`, `cat_<col>`).
# PreTab defaults to returning a single stacked matrix instead, so these are forced
# explicitly rather than left to PreTab's own default.
_FORCED_PREPROCESSOR_KWARGS = {"output_structure": "blocks", "output_format": "dense"}


def build_preprocessor(
    preprocessing_config: PreprocessingConfig | None = None,
    *,
    task: str | None = None,
    random_state: int | None = None,
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
    kwargs.update(_FORCED_PREPROCESSOR_KWARGS)
    return Preprocessor(**kwargs)
