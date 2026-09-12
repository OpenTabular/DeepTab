"""Central boundary for constructing PreTab's ``Preprocessor``.

Every DeepTab estimator builds its preprocessor through :func:`build_preprocessor`
instead of importing and instantiating ``pretab.Preprocessor`` directly, so PreTab
stays isolated behind one internal contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pretab import Preprocessor

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
        Task hint forwarded to PreTab's ``task`` argument when
        *preprocessing_config* does not already resolve one.
    random_state : int or None, default=None
        Seed forwarded to PreTab's ``random_state`` argument when
        *preprocessing_config* does not already resolve one.
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
        kwargs.setdefault("task", task)
    if random_state is not None:
        kwargs.setdefault("random_state", random_state)
    kwargs.update(_FORCED_PREPROCESSOR_KWARGS)
    return Preprocessor(**kwargs)
