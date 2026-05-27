"""Small sklearn-compatibility helpers shared by estimator bases."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def ensure_dataframe(X: Any) -> pd.DataFrame:
    """Return ``X`` as a DataFrame while preserving existing DataFrames."""
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def set_input_feature_attributes(estimator: Any, X: pd.DataFrame) -> None:
    """Set fitted-input attributes following sklearn conventions."""
    estimator.n_features_in_ = X.shape[1]
    estimator.input_columns_ = list(X.columns)

    if all(isinstance(column, str) for column in X.columns):
        estimator.feature_names_in_ = np.asarray(X.columns, dtype=object)
    elif hasattr(estimator, "feature_names_in_"):
        delattr(estimator, "feature_names_in_")


def validate_input_features(estimator: Any, X: Any) -> pd.DataFrame:
    """Validate prediction input against fitted feature count and names."""
    X_df = ensure_dataframe(X)

    expected_n_features = getattr(estimator, "n_features_in_", None)
    if expected_n_features is not None and X_df.shape[1] != expected_n_features:
        raise ValueError(
            f"X has {X_df.shape[1]} features, but this estimator was fitted with {expected_n_features} features."
        )

    expected_names = getattr(estimator, "feature_names_in_", None)
    if expected_names is not None:
        if not all(isinstance(column, str) for column in X_df.columns):
            raise ValueError(
                "X does not contain valid feature names, but this estimator was fitted with feature names."
            )
        expected = list(expected_names)
        actual = list(X_df.columns)
        if actual != expected:
            raise ValueError("X feature names must match the names and order seen during fit.")

    return X_df
