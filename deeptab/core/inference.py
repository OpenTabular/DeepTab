"""Deployment-only inference interface for fitted DeepTab artifacts."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from deeptab.core.sklearn_compat import ensure_dataframe

if TYPE_CHECKING:
    pass

__all__ = ["InferenceModel"]


class InferenceModel:
    """Deployment-only inference wrapper for a fitted DeepTab estimator.

    :class:`InferenceModel` is a thin, immutable wrapper around a loaded
    estimator.  It exposes exactly the surface needed in production —
    schema validation, inference, and introspection — while intentionally
    omitting ``fit()``, ``optimize_hparams()``, and other training methods
    so that deployment code cannot accidentally retrain a model.

    Do not instantiate directly.  Use :meth:`from_path` to load an artifact
    from disk or :meth:`from_estimator` to wrap an already-fitted estimator.

    Parameters
    ----------
    estimator : fitted DeepTab estimator
        Must have ``is_fitted_`` set to ``True``.  Prefer :meth:`from_path`
        or :meth:`from_estimator` over calling this constructor directly.

    Attributes
    ----------
    task : str
        ``"classification"``, ``"regression"``, or
        ``"distributional_regression"``.
    feature_names : list[str] or None
        Ordered feature names seen during training, or *None* when the
        artifact was saved without string column names.
    n_features : int or None
        Number of features the model was trained on.
    classes_ : ndarray or None
        Class labels (classification only).
    task_info : dict
        Task metadata dict (``task``, ``regression``, ``lss``, ``family``,
        ``num_classes``, ``classes_``).
    feature_schema : dict
        Full feature-schema metadata block from the artifact.

    Notes
    -----
    The following methods are available on every :class:`InferenceModel`:

    * :meth:`from_path` / :meth:`from_estimator` — construction
    * :meth:`validate_input` — column-level schema enforcement
    * :meth:`predict` / :meth:`predict_proba` / :meth:`predict_params` — inference
    * :meth:`describe` / :meth:`runtime_info` / :meth:`parameter_table` — introspection

    :meth:`predict_proba` is only available when ``task == "classification"``.
    :meth:`predict_params` is only available when
    ``task == "distributional_regression"``.

    Examples
    --------
    Load a saved artifact and run predictions:

    >>> from deeptab import InferenceModel
    >>> model = InferenceModel.from_path("my_model.deeptab")
    >>> model.validate_input(X_new)           # raises on schema mismatch
    >>> predictions = model.predict(X_new)
    >>> probabilities = model.predict_proba(X_new)  # classifiers only

    Wrap an already-fitted estimator without saving to disk:

    >>> clf = MLPClassifier()
    >>> clf.fit(X_train, y_train)
    >>> model = InferenceModel.from_estimator(clf)
    >>> proba = model.predict_proba(X_test)

    Inspect a loaded model before predicting:

    >>> model = InferenceModel.from_path("my_model.deeptab")
    >>> print(model)
    InferenceModel(task='classification', estimator='MLPClassifier', ...)
    >>> info = model.describe()
    >>> rt = model.runtime_info()
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, estimator: Any) -> None:
        """Wrap a fitted estimator.

        Parameters
        ----------
        estimator : fitted DeepTab estimator
            Must have ``is_fitted_`` set to ``True``.

        Raises
        ------
        ValueError
            If the estimator has not been fitted.
        """
        if not getattr(estimator, "is_fitted_", False):
            raise ValueError(
                "Cannot wrap an unfitted estimator in InferenceModel. "
                "Call estimator.fit() first, or load from a saved artifact "
                "with InferenceModel.from_path()."
            )
        self._estimator = estimator
        self._task = self._detect_task()

    @classmethod
    def from_path(cls, path: str | os.PathLike) -> InferenceModel:
        """Load a DeepTab artifact and return an :class:`InferenceModel`.

        Parameters
        ----------
        path : str or path-like
            Path to a ``.deeptab`` file written by
            :meth:`~deeptab.models.base.SklearnBase.save`.

        Returns
        -------
        InferenceModel

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the loaded artifact was not fitted.

        Examples
        --------
        >>> model = InferenceModel.from_path("my_model.deeptab")
        >>> predictions = model.predict(X_new)
        """
        path = os.fspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact not found: {path!r}")

        import torch

        from deeptab.core.serialization import _warn_extension

        _warn_extension(path)
        bundle = torch.load(path, weights_only=False)

        estimator_class = bundle.get("_class")
        if estimator_class is None:
            raise ValueError(
                f"The artifact at {path!r} does not contain a '_class' key. "
                "It may have been saved by an older version of DeepTab."
            )

        estimator = estimator_class.load(path)
        return cls(estimator)

    @classmethod
    def from_estimator(cls, estimator: Any) -> InferenceModel:
        """Wrap an already-fitted estimator in an :class:`InferenceModel`.

        Parameters
        ----------
        estimator : fitted DeepTab estimator

        Returns
        -------
        InferenceModel

        Examples
        --------
        >>> clf = MLPClassifier()
        >>> clf.fit(X_train, y_train)
        >>> model = InferenceModel.from_estimator(clf)
        >>> predictions = model.predict(X_test)
        """
        return cls(estimator)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_task(self) -> str:
        """Infer the task type from the wrapped estimator."""
        task_info = getattr(self._estimator, "task_info_", None)
        if task_info is not None:
            if task_info.get("lss"):
                return "distributional_regression"
            if task_info.get("regression"):
                return "regression"
            return "classification"

        # Fall back to class name heuristic
        name = type(self._estimator).__name__
        if name.endswith("LSS"):
            return "distributional_regression"
        if name.endswith("Regressor"):
            return "regression"
        return "classification"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def task(self) -> str:
        """Task type: ``"classification"``, ``"regression"``, or
        ``"distributional_regression"``."""
        return self._task

    @property
    def feature_names(self) -> list[str] | None:
        """Ordered list of feature names from the training run, or *None*."""
        names = getattr(self._estimator, "input_columns_", None)
        if names is None:
            fn = getattr(self._estimator, "feature_names_in_", None)
            if fn is not None:
                names = list(fn)
        return list(names) if names is not None else None

    @property
    def n_features(self) -> int | None:
        """Number of features the model was trained on."""
        return getattr(self._estimator, "n_features_in_", None)

    @property
    def classes_(self) -> np.ndarray | None:
        """Class labels for classification models, *None* otherwise."""
        return getattr(self._estimator, "classes_", None)

    @property
    def task_info(self) -> dict[str, Any]:
        """Task metadata dict from the artifact."""
        return dict(getattr(self._estimator, "task_info_", {}))

    @property
    def feature_schema(self) -> dict[str, Any]:
        """Full feature-schema metadata block from the artifact."""
        return dict(getattr(self._estimator, "feature_schema_", {}))

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def validate_input(
        self,
        X: Any,
        *,
        allow_extra_columns: bool = False,
    ) -> pd.DataFrame:
        """Validate *X* against the training schema and return a ready DataFrame.

        Performs the following checks in order:

        1. **Feature names** — if the artifact stores named columns, every
           expected column must be present in *X*.
        2. **Missing columns** — any column seen during training but absent
           from *X* raises :exc:`ValueError`.
        3. **Extra columns** — columns in *X* that were not seen during
           training raise :exc:`ValueError` by default.  Pass
           ``allow_extra_columns=True`` to drop them with a warning instead.
        4. **Column order** — when feature names are available the returned
           DataFrame always uses the training column order.
        5. **Feature count** — when only the column count is known (no names),
           a mismatch raises :exc:`ValueError`.

        Parameters
        ----------
        X : DataFrame or array-like
            Input to validate.
        allow_extra_columns : bool, default=False
            When *True*, columns not seen during training are silently dropped
            with a :exc:`UserWarning`.  When *False* (default) their presence
            raises :exc:`ValueError`.

        Returns
        -------
        pd.DataFrame
            Validated DataFrame with columns reordered to the training order.

        Raises
        ------
        ValueError
            On any schema violation that cannot be auto-corrected.

        Examples
        --------
        >>> model = InferenceModel.from_path("my_model.deeptab")
        >>> X_valid = model.validate_input(X_new)
        >>> predictions = model.predict(X_valid)
        """
        X_df = ensure_dataframe(X)

        expected_names = self.feature_names

        if expected_names is None:
            # Only a count check is possible
            n = self.n_features
            if n is not None and X_df.shape[1] != n:
                raise ValueError(
                    f"Expected {n} feature(s) (no column names available for detailed validation), got {X_df.shape[1]}."
                )
            return X_df

        actual_cols: set[Any] = set(X_df.columns)
        expected_set: set[str] = set(expected_names)

        missing = sorted(expected_set - actual_cols)
        extra = sorted(actual_cols - expected_set)

        if missing:
            raise ValueError(f"Input is missing {len(missing)} column(s) that were present during training: {missing}.")

        if extra:
            if not allow_extra_columns:
                raise ValueError(
                    f"Input has {len(extra)} unexpected column(s) not seen during "
                    f"training: {extra}. "
                    f"To drop them automatically, pass allow_extra_columns=True."
                )
            warnings.warn(
                f"Input has {len(extra)} column(s) not seen during training ({extra}); they will be dropped.",
                UserWarning,
                stacklevel=2,
            )

        # Always return in training column order
        return X_df[expected_names]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        """Run inference and return the primary predictions.

        For **classification** returns integer class labels (same dtype as
        ``classes_``).  For **regression** returns a float array of target
        values.  For **distributional regression** (LSS) returns the
        distribution mean / mode as a float array.

        *X* is passed through :meth:`validate_input` before prediction.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, n_outputs)

        Raises
        ------
        ValueError
            If *X* does not match the training schema.

        Examples
        --------
        >>> model = InferenceModel.from_path("my_model.deeptab")
        >>> predictions = model.predict(X_new)
        """
        X_validated = self.validate_input(X)
        return self._estimator.predict(X_validated)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return predicted class probabilities (classification only).

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)

        Raises
        ------
        TypeError
            If the wrapped model is not a classifier.
        ValueError
            If *X* does not match the training schema.

        Examples
        --------
        >>> model = InferenceModel.from_path("my_model.deeptab")
        >>> proba = model.predict_proba(X_new)
        """
        if self._task != "classification":
            raise TypeError(
                f"predict_proba() is only available for classification models, but this model's task is '{self._task}'."
            )
        if not callable(getattr(self._estimator, "predict_proba", None)):
            raise TypeError(f"{type(self._estimator).__name__} does not expose predict_proba().")
        X_validated = self.validate_input(X)
        return self._estimator.predict_proba(X_validated)

    def predict_params(self, X: Any, *, raw: bool = False) -> np.ndarray:
        """Return distribution parameters (distributional regression only).

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        raw : bool, default=False
            When *True*, return raw network outputs before the inverse-link
            transform.

        Returns
        -------
        ndarray of shape (n_samples, n_params)

        Raises
        ------
        TypeError
            If the wrapped model is not a distributional regression (LSS) model.
        ValueError
            If *X* does not match the training schema.

        Examples
        --------
        >>> model = InferenceModel.from_path("lss_model.deeptab")
        >>> params = model.predict_params(X_new)
        """
        if self._task != "distributional_regression":
            raise TypeError(
                f"predict_params() is only available for distributional regression "
                f"(LSS) models, but this model's task is '{self._task}'."
            )
        X_validated = self.validate_input(X)
        return self._estimator.predict(X_validated, raw=raw)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Return a structured metadata summary.

        Delegates to the wrapped estimator's
        :meth:`~deeptab.core.inspection.InspectionMixin.describe` when
        available, then augments with an ``inference_task`` key.

        Returns
        -------
        dict
        """
        info: dict[str, Any]
        describe_fn = getattr(self._estimator, "describe", None)
        if callable(describe_fn):
            info = describe_fn()  # type: ignore[assignment]
        else:
            info = {
                "estimator": type(self._estimator).__name__,
                "fitted": True,
            }
        info["inference_task"] = self._task
        return info

    def runtime_info(self) -> dict[str, Any]:
        """Return device / precision / training-loop runtime information.

        Delegates to the wrapped estimator's
        :meth:`~deeptab.core.inspection.InspectionMixin.runtime_info`.

        Returns
        -------
        dict
        """
        runtime_fn = getattr(self._estimator, "runtime_info", None)
        if callable(runtime_fn):
            return runtime_fn()  # type: ignore[return-value]
        return {}

    def parameter_table(self, trainable_only: bool = False) -> pd.DataFrame:
        """Return one row per model parameter as a DataFrame.

        Delegates to the wrapped estimator's
        :meth:`~deeptab.core.inspection.InspectionMixin.parameter_table`.

        Parameters
        ----------
        trainable_only : bool, default=False
            When *True*, include only parameters with ``requires_grad=True``.

        Returns
        -------
        pd.DataFrame
        """
        pt_fn = getattr(self._estimator, "parameter_table", None)
        if callable(pt_fn):
            return pt_fn(trainable_only=trainable_only)  # type: ignore[return-value]
        raise AttributeError(f"{type(self._estimator).__name__} does not expose parameter_table().")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = self.n_features
        names_preview = ""
        names = self.feature_names
        if names is not None:
            preview = names[:3]
            suffix = ", ..." if len(names) > 3 else ""
            names_preview = f", features=[{', '.join(repr(c) for c in preview)}{suffix}]"
        classes_info = ""
        if self._task == "classification" and self.classes_ is not None:
            classes_info = f", n_classes={len(self.classes_)}"
        return (
            f"InferenceModel("
            f"task={self._task!r}"
            f", estimator={type(self._estimator).__name__!r}"
            f", n_features={n}"
            f"{names_preview}"
            f"{classes_info}"
            f")"
        )
