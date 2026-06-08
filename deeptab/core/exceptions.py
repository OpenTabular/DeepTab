"""User-facing exception types and message factories for DeepTab.

All user-facing errors and warnings are defined here.  Internal modules should
import from this module rather than raising bare ``ValueError`` / ``TypeError``
with ad-hoc strings.

Exception hierarchy
-------------------
DeepTabError
├── DataError
│   ├── ColumnDtypeError
│   ├── ColumnCountError
│   ├── ColumnNameError
│   ├── EmptyDataError
│   └── InsufficientSamplesError
├── ModelError
│   ├── NotFittedError
│   └── ArchitectureRequirementError
└── ConfigError
    ├── InvalidParamError
    └── IncompatibleParamsError

Warning hierarchy
-----------------
DeepTabWarning (UserWarning)
├── DataWarning
├── ConfigWarning
└── PerformanceWarning
"""

from __future__ import annotations

import warnings
from typing import Any

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class DeepTabError(Exception):
    """Base class for all DeepTab user-facing errors."""


# -- Data errors -------------------------------------------------------------


class DataError(DeepTabError):
    """Problem with the input DataFrame (shape, dtypes, missing columns, or values)."""


class ColumnDtypeError(DataError):
    """One or more columns have an unsupported dtype."""


class ColumnCountError(DataError):
    """Wrong number of feature columns at predict time vs. fit time."""


class ColumnNameError(DataError):
    """Feature column names don't match what was seen at fit time."""


class EmptyDataError(DataError):
    """The input DataFrame is empty (0 rows or 0 columns)."""


class InsufficientSamplesError(DataError):
    """Not enough rows for the requested operation (e.g. PLE decision-tree binning)."""


# -- Model errors ------------------------------------------------------------


class ModelError(DeepTabError):
    """Problem with model construction or state."""


class NotFittedError(ModelError):
    """A method was called before fit() completed."""


class ArchitectureRequirementError(ModelError):
    """The chosen architecture cannot operate on the provided data."""


# -- Config errors -----------------------------------------------------------


class ConfigError(DeepTabError):
    """Invalid configuration value or combination."""


class InvalidParamError(ConfigError):
    """A single config field is out of range or not a valid choice."""


class IncompatibleParamsError(ConfigError):
    """Two or more config fields conflict with each other."""


# ---------------------------------------------------------------------------
# Warning hierarchy
# ---------------------------------------------------------------------------


class DeepTabWarning(UserWarning):
    """Base class for all DeepTab warnings."""


class DataWarning(DeepTabWarning):
    """Non-fatal data issue (e.g. constant column, high NaN rate)."""


class ConfigWarning(DeepTabWarning):
    """Potentially suboptimal or surprising configuration."""


class PerformanceWarning(DeepTabWarning):
    """Expected slow execution (e.g. no GPU, very large dataset)."""


# ---------------------------------------------------------------------------
# Message factories — Data
# ---------------------------------------------------------------------------


def column_dtype_error(bad_cols: list[tuple[str, Any]]) -> ColumnDtypeError:
    """Return a :class:`ColumnDtypeError` for columns with unsupported dtypes.

    Parameters
    ----------
    bad_cols:
        List of ``(column_name, dtype)`` pairs that are unsupported.
    """
    lines = [f"  • {col!r}: {dt}" for col, dt in bad_cols]
    return ColumnDtypeError(
        "Input contains columns with unsupported dtypes:\n"
        + "\n".join(lines)
        + "\n\nDeepTab preprocessing accepts: numeric (int / float), object, "
        "string, or bool.\n"
        "Fix: cast the column before calling fit(), e.g.\n"
        "  df['col'] = df['col'].astype('float32')"
    )


def column_count_error(expected: int, got: int) -> ColumnCountError:
    """Return a :class:`ColumnCountError` for a feature-count mismatch."""
    return ColumnCountError(
        f"Expected {expected} feature column(s) (as seen during fit), "
        f"but got {got}.\n"
        "Fix: pass the same columns in the same order as during fit()."
    )


def column_name_error(missing: list[str], extra: list[str]) -> ColumnNameError:
    """Return a :class:`ColumnNameError` listing missing and extra columns."""
    parts: list[str] = []
    if missing:
        parts.append(f"  Missing : {missing}")
    if extra:
        parts.append(f"  Extra   : {extra}")
    return ColumnNameError(
        "Feature column names do not match what was seen during fit.\n"
        + "\n".join(parts)
        + "\nFix: align column names with the training DataFrame."
    )


def empty_data_error(context: str = "fit") -> EmptyDataError:
    """Return an :class:`EmptyDataError` for a zero-row or zero-column DataFrame."""
    return EmptyDataError(
        f"Input DataFrame passed to {context}() is empty (0 rows or 0 columns).\nFix: pass a non-empty DataFrame."
    )


def insufficient_samples_error(
    n_rows: int,
    min_required: int,
    reason: str,
) -> InsufficientSamplesError:
    """Return an :class:`InsufficientSamplesError` with context about the requirement."""
    return InsufficientSamplesError(
        f"Dataset has {n_rows} row(s) but at least {min_required} are needed "
        f"for {reason}.\n"
        "Fix: use a larger dataset, or switch to a simpler preprocessing method "
        "(e.g. PreprocessingConfig(numerical_preprocessing='quantile'))."
    )


def target_nan_error() -> DataError:
    """Return a :class:`DataError` when ``y`` contains NaN values."""
    return DataError("y contains NaN values.\nFix: remove or impute missing target values before calling fit().")


def target_range_error(family: str, constraint: str) -> DataError:
    """Return a :class:`DataError` when ``y`` violates a distribution family's range."""
    return DataError(
        f"family='{family}' requires {constraint} target values, "
        "but y does not satisfy this constraint.\n"
        "Fix: filter or transform y before calling fit()."
    )


def xy_length_mismatch_error(n_X: int, n_y: int) -> DataError:
    """Return a :class:`DataError` when X and y have different row counts."""
    return DataError(
        f"X has {n_X} row(s) but y has {n_y} element(s). They must match.\n"
        "Fix: ensure X and y are derived from the same dataset without dropping rows."
    )


# ---------------------------------------------------------------------------
# Message factories — Model
# ---------------------------------------------------------------------------


def not_fitted_error(estimator_name: str, method: str) -> NotFittedError:
    """Return a :class:`NotFittedError` for a method called before fit()."""
    return NotFittedError(
        f"{estimator_name}.{method}() was called before fit().\nFix: call {estimator_name}.fit(X_train, y_train) first."
    )


def architecture_requirement_error(
    arch: str,
    requirement: str,
    suggestion: str,
) -> ArchitectureRequirementError:
    """Return an :class:`ArchitectureRequirementError` with a concrete suggestion."""
    return ArchitectureRequirementError(
        f"{arch} cannot be used with this data: {requirement}\nSuggestion: {suggestion}"
    )


# ---------------------------------------------------------------------------
# Message factories — Config
# ---------------------------------------------------------------------------


def invalid_param_error(
    config_cls: str,
    param: str,
    value: Any,
    constraint: str,
    valid_values: list[Any] | None = None,
) -> InvalidParamError:
    """Return an :class:`InvalidParamError` for a single out-of-range or bad-choice field."""
    msg = f"{config_cls}.{param} = {value!r} is invalid.\nConstraint: {constraint}"
    if valid_values is not None:
        msg += f"\nValid values: {valid_values}"
    return InvalidParamError(msg)


def incompatible_params_error(
    config_cls: str,
    details: str,
) -> IncompatibleParamsError:
    """Return an :class:`IncompatibleParamsError` describing conflicting fields."""
    return IncompatibleParamsError(f"Incompatible parameters in {config_cls}:\n{details}")


# ---------------------------------------------------------------------------
# Warning helpers
# ---------------------------------------------------------------------------


def warn_data(msg: str, stacklevel: int = 3) -> None:
    """Issue a :class:`DataWarning`."""
    warnings.warn(msg, DataWarning, stacklevel=stacklevel)


def warn_config(msg: str, stacklevel: int = 3) -> None:
    """Issue a :class:`ConfigWarning`."""
    warnings.warn(msg, ConfigWarning, stacklevel=stacklevel)


def warn_performance(msg: str, stacklevel: int = 3) -> None:
    """Issue a :class:`PerformanceWarning`."""
    warnings.warn(msg, PerformanceWarning, stacklevel=stacklevel)
