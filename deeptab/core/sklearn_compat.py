"""Small sklearn-compatibility helpers shared by estimator bases."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from deeptab.core.exceptions import (
    ColumnDtypeError,
    column_count_error,
    column_dtype_error,
    column_name_error,
    empty_data_error,
    warn_data,
)


def ensure_dataframe(X: Any, context: str = "fit") -> pd.DataFrame:
    """Return ``X`` as a DataFrame, casting dtypes that sklearn preprocessing cannot handle.

    - 1-D arrays raise :exc:`ValueError` following sklearn convention.
    - Empty DataFrames raise :exc:`~deeptab.core.exceptions.EmptyDataError`.
    - ``bool`` columns are silently cast to ``int8``; they represent valid binary
      features but sklearn's ``SimpleImputer`` rejects the ``bool`` dtype.
    - ``category`` columns are silently cast to ``object`` so they are detected and
      preprocessed as categorical features (the underlying categories are kept).
    - Any remaining non-numeric, non-object column dtype raises
      :exc:`~deeptab.core.exceptions.ColumnDtypeError` naming each offending column.
    - Columns where every value is NaN issue a
      :class:`~deeptab.core.exceptions.DataWarning`.

    Parameters
    ----------
    X:
        Input data.  Converted to :class:`pandas.DataFrame` if necessary.
    context:
        Name of the calling method (used in error messages).
    """
    # Reject 1-D input early: sklearn convention requires 2-D feature arrays.
    _arr = np.asarray(X) if not isinstance(X, pd.DataFrame | pd.Series) else X
    if getattr(_arr, "ndim", 2) == 1:
        raise ValueError(
            "Expected 2D array, got 1D array instead.\n"
            "Reshape your data either using array.reshape(-1, 1) if your data has "
            "a single feature or array.reshape(1, -1) if it contains a single sample."
        )

    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise empty_data_error(context)

    # bool → int8: valid binary feature, but SimpleImputer rejects bool dtype
    bool_cols = [c for c, dt in df.dtypes.items() if dt is np.dtype(bool)]
    if bool_cols:
        df = df.copy()
        df[bool_cols] = df[bool_cols].astype("int8")

    # category → object: treat pandas categoricals as categorical features.
    # The dtype detector downstream keys off object/string dtype, so cast here
    # while preserving the underlying category values.
    cat_cols = [c for c, dt in df.dtypes.items() if isinstance(dt, pd.CategoricalDtype)]
    if cat_cols:
        if not bool_cols:
            df = df.copy()
        df = df.astype(dict.fromkeys(cat_cols, "object"))

    # Catch any other dtype that is neither numeric nor object/string
    bad_cols = [
        (str(c), dt)
        for c, dt in df.dtypes.items()
        if not (
            pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_object_dtype(dt) or pd.api.types.is_string_dtype(dt)
        )
    ]
    if bad_cols:
        raise column_dtype_error(bad_cols)

    # Warn about all-NaN columns — imputation will produce a column of constants
    all_nan_cols = [str(c) for c in df.columns if bool(df[c].isna().all())]
    if all_nan_cols:
        warn_data(
            f"The following column(s) are entirely NaN and will be imputed with a "
            f"constant: {all_nan_cols}. Consider dropping them before calling fit().",
            stacklevel=4,
        )

    return df


def set_input_feature_attributes(estimator: Any, X: pd.DataFrame) -> None:
    """Set fitted-input attributes following sklearn conventions."""
    estimator.n_features_in_ = X.shape[1]
    estimator.input_columns_ = list(X.columns)

    if all(isinstance(column, str) for column in X.columns):
        estimator.feature_names_in_ = np.asarray(X.columns, dtype=object)
    elif hasattr(estimator, "feature_names_in_"):
        delattr(estimator, "feature_names_in_")


def validate_input_features(estimator: Any, X: Any) -> pd.DataFrame:
    """Validate prediction input against fitted feature count and names.

    Raises
    ------
    ColumnCountError
        If the number of columns differs from what was seen during fit.
    ColumnNameError
        If column names differ from what was seen during fit.
    """
    X_df = ensure_dataframe(X, context="predict")

    expected_n_features = getattr(estimator, "n_features_in_", None)
    if expected_n_features is not None and X_df.shape[1] != expected_n_features:
        raise column_count_error(expected_n_features, X_df.shape[1])

    expected_names = getattr(estimator, "feature_names_in_", None)
    if expected_names is not None:
        if not all(isinstance(column, str) for column in X_df.columns):
            raise column_name_error(
                missing=list(expected_names),
                extra=[],
            )
        expected = list(expected_names)
        actual = list(X_df.columns)
        if actual != expected:
            expected_set = set(expected)
            actual_set = set(actual)
            raise column_name_error(
                missing=sorted(expected_set - actual_set),
                extra=sorted(actual_set - expected_set),
            )

    return X_df
