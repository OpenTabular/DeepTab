from dataclasses import dataclass

from sklearn.base import BaseEstimator


@dataclass
class PreprocessingConfig(BaseEstimator):
    """Configuration for input feature preprocessing.

    All fields map directly to arguments accepted by ``pretab.preprocessor.Preprocessor``.
    Using ``None`` for any field leaves the preprocessor default in effect.

    Parameters
    ----------
    numerical_preprocessing : str or None, default=None
        Strategy for transforming numerical features (e.g. ``"ple"``, ``"quantile"``,
        ``"standard"``).  ``None`` uses the preprocessor's built-in default.
    categorical_preprocessing : str or None, default=None
        Strategy for transforming categorical features (e.g. ``"int"``, ``"one-hot"``).
        ``None`` uses the preprocessor's built-in default.
    n_bins : int or None, default=None
        Number of bins for numerical binning.  ``None`` uses the preprocessor default.
    feature_preprocessing : str or None, default=None
        General feature-level preprocessing override.
    use_decision_tree_bins : bool or None, default=None
        Whether to use decision-tree-derived bin edges.
    binning_strategy : str or None, default=None
        Strategy for choosing bin edges (e.g. ``"uniform"``, ``"quantile"``).
    task : str or None, default=None
        Task type passed to the preprocessor for task-aware transformations
        (e.g. ``"regression"``, ``"classification"``).
    cat_cutoff : float or None, default=None
        Threshold for treating integer columns as categorical.
    treat_all_integers_as_numerical : bool or None, default=None
        When ``True``, integer columns are never converted to categorical.
    degree : int or None, default=None
        Polynomial / spline degree for numerical feature expansion.
    scaling_strategy : str or None, default=None
        Scaling method applied to numerical features (e.g. ``"standard"``,
        ``"minmax"``, ``"robust"``).
    n_knots : int or None, default=None
        Number of knots for spline preprocessing.
    use_decision_tree_knots : bool or None, default=None
        Whether to use decision-tree-derived knot positions.
    knots_strategy : str or None, default=None
        Strategy for knot placement.
    spline_implementation : str or None, default=None
        Backend used for spline transformations.
    """

    numerical_preprocessing: str | None = None
    categorical_preprocessing: str | None = None
    n_bins: int | None = None
    feature_preprocessing: str | None = None
    use_decision_tree_bins: bool | None = None
    binning_strategy: str | None = None
    task: str | None = None
    cat_cutoff: float | None = None
    treat_all_integers_as_numerical: bool | None = None
    degree: int | None = None
    scaling_strategy: str | None = None
    n_knots: int | None = None
    use_decision_tree_knots: bool | None = None
    knots_strategy: str | None = None
    spline_implementation: str | None = None

    def to_preprocessor_kwargs(self) -> dict:
        """Return a dict of non-None fields suitable for passing to ``Preprocessor(**...)``.

        Returns
        -------
        dict
            Mapping of field name → value for every field that is not ``None``.
        """
        return {k: v for k, v in self.get_params(deep=False).items() if v is not None}
