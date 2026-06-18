"""Tests for deeptab.core.exceptions — exception hierarchy, factories, and integration.

Covers:
- Exception class hierarchy (is-a relationships)
- Warning class hierarchy
- Every factory function produces the right type with the right message fragment
- PreprocessingConfig validation (__post_init__)
- TrainerConfig validation (__post_init__)
- Misplaced-config warning from the estimator constructor
- BaseModelConfig / per-model config validation (__post_init__)
- sklearn_compat.ensure_dataframe() guards (empty, bad dtype, all-NaN warning)
- sklearn_compat.validate_input_features() guards (column count, column names)
- _validate_fit_inputs() guards (length mismatch, NaN y, family range)
- Distribution registry: unknown family raises InvalidParamError
- TabTransformer architecture requirement
- Public API exports from deeptab and deeptab.core
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import deeptab
from deeptab.core.exceptions import (
    ArchitectureRequirementError,
    ColumnCountError,
    ColumnDtypeError,
    ColumnNameError,
    ConfigError,
    ConfigWarning,
    DataError,
    DataWarning,
    DeepTabError,
    DeepTabWarning,
    EmptyDataError,
    IncompatibleParamsError,
    InsufficientSamplesError,
    InvalidParamError,
    ModelError,
    NotFittedError,
    PerformanceWarning,
    architecture_requirement_error,
    column_count_error,
    column_dtype_error,
    column_name_error,
    empty_data_error,
    incompatible_params_error,
    insufficient_samples_error,
    invalid_param_error,
    not_fitted_error,
    target_nan_error,
    target_range_error,
    warn_config,
    warn_data,
    warn_performance,
    xy_length_mismatch_error,
)

# ===========================================================================
# 1 — Exception hierarchy
# ===========================================================================


class TestExceptionHierarchy:
    def test_data_error_is_deeptab_error(self):
        assert issubclass(DataError, DeepTabError)

    def test_column_dtype_error_is_data_error(self):
        assert issubclass(ColumnDtypeError, DataError)

    def test_column_count_error_is_data_error(self):
        assert issubclass(ColumnCountError, DataError)

    def test_column_name_error_is_data_error(self):
        assert issubclass(ColumnNameError, DataError)

    def test_empty_data_error_is_data_error(self):
        assert issubclass(EmptyDataError, DataError)

    def test_insufficient_samples_error_is_data_error(self):
        assert issubclass(InsufficientSamplesError, DataError)

    def test_model_error_is_deeptab_error(self):
        assert issubclass(ModelError, DeepTabError)

    def test_not_fitted_error_is_model_error(self):
        assert issubclass(NotFittedError, ModelError)

    def test_architecture_requirement_error_is_model_error(self):
        assert issubclass(ArchitectureRequirementError, ModelError)

    def test_config_error_is_deeptab_error(self):
        assert issubclass(ConfigError, DeepTabError)

    def test_invalid_param_error_is_config_error(self):
        assert issubclass(InvalidParamError, ConfigError)

    def test_incompatible_params_error_is_config_error(self):
        assert issubclass(IncompatibleParamsError, ConfigError)

    def test_all_errors_are_exceptions(self):
        for cls in (
            DeepTabError,
            DataError,
            ColumnDtypeError,
            ColumnCountError,
            ColumnNameError,
            EmptyDataError,
            InsufficientSamplesError,
            ModelError,
            NotFittedError,
            ArchitectureRequirementError,
            ConfigError,
            InvalidParamError,
            IncompatibleParamsError,
        ):
            assert issubclass(cls, Exception)


class TestWarningHierarchy:
    def test_deeptab_warning_is_user_warning(self):
        assert issubclass(DeepTabWarning, UserWarning)

    def test_data_warning_is_deeptab_warning(self):
        assert issubclass(DataWarning, DeepTabWarning)

    def test_config_warning_is_deeptab_warning(self):
        assert issubclass(ConfigWarning, DeepTabWarning)

    def test_performance_warning_is_deeptab_warning(self):
        assert issubclass(PerformanceWarning, DeepTabWarning)


# ===========================================================================
# 2 — Factory functions: return type and message content
# ===========================================================================


class TestDataFactories:
    def test_column_dtype_error_type_and_message(self):
        exc = column_dtype_error([("col_a", "datetime64[ns]"), ("col_b", "timedelta64")])
        assert isinstance(exc, ColumnDtypeError)
        assert "col_a" in str(exc)
        assert "col_b" in str(exc)
        assert "Fix:" in str(exc)

    def test_column_count_error_type_and_message(self):
        exc = column_count_error(expected=10, got=8)
        assert isinstance(exc, ColumnCountError)
        assert "10" in str(exc)
        assert "8" in str(exc)
        assert "Fix:" in str(exc)

    def test_column_name_error_missing_and_extra(self):
        exc = column_name_error(missing=["age", "income"], extra=["AGE"])
        assert isinstance(exc, ColumnNameError)
        assert "age" in str(exc)
        assert "income" in str(exc)
        assert "AGE" in str(exc)
        assert "Fix:" in str(exc)

    def test_column_name_error_missing_only(self):
        exc = column_name_error(missing=["x"], extra=[])
        assert "x" in str(exc)
        assert "Extra" not in str(exc)

    def test_empty_data_error_default_context(self):
        exc = empty_data_error()
        assert isinstance(exc, EmptyDataError)
        assert "fit" in str(exc)

    def test_empty_data_error_custom_context(self):
        exc = empty_data_error("predict")
        assert "predict" in str(exc)

    def test_insufficient_samples_error(self):
        exc = insufficient_samples_error(n_rows=5, min_required=50, reason="PLE binning")
        assert isinstance(exc, InsufficientSamplesError)
        assert "5" in str(exc)
        assert "50" in str(exc)
        assert "PLE binning" in str(exc)
        assert "Fix:" in str(exc)

    def test_target_nan_error(self):
        exc = target_nan_error()
        assert isinstance(exc, DataError)
        assert "NaN" in str(exc)
        assert "Fix:" in str(exc)

    def test_target_range_error(self):
        exc = target_range_error("poisson", "non-negative")
        assert isinstance(exc, DataError)
        assert "poisson" in str(exc)
        assert "non-negative" in str(exc)

    def test_xy_length_mismatch_error(self):
        exc = xy_length_mismatch_error(n_X=100, n_y=95)
        assert isinstance(exc, DataError)
        assert "100" in str(exc)
        assert "95" in str(exc)
        assert "Fix:" in str(exc)


class TestModelFactories:
    def test_not_fitted_error(self):
        exc = not_fitted_error("MambularClassifier", "predict")
        assert isinstance(exc, NotFittedError)
        assert "MambularClassifier" in str(exc)
        assert "predict" in str(exc)
        assert "fit(" in str(exc)

    def test_architecture_requirement_error(self):
        exc = architecture_requirement_error(
            arch="TabTransformer",
            requirement="requires categorical features",
            suggestion="use FTTransformer instead",
        )
        assert isinstance(exc, ArchitectureRequirementError)
        assert "TabTransformer" in str(exc)
        assert "requires categorical features" in str(exc)
        assert "FTTransformer" in str(exc)


class TestConfigFactories:
    def test_invalid_param_error_without_valid_values(self):
        exc = invalid_param_error("TrainerConfig", "lr", -0.01, "must be > 0")
        assert isinstance(exc, InvalidParamError)
        assert "TrainerConfig" in str(exc)
        assert "lr" in str(exc)
        assert "-0.01" in str(exc)
        assert "must be > 0" in str(exc)

    def test_invalid_param_error_with_valid_values(self):
        exc = invalid_param_error(
            "PreprocessingConfig",
            "scaling_strategy",
            "zscore",
            "must be a known strategy",
            ["minmax", "robust", "standardization"],
        )
        assert "zscore" in str(exc)
        assert "minmax" in str(exc)

    def test_incompatible_params_error(self):
        exc = incompatible_params_error("FTTransformerConfig", "d_model (64) must be divisible by n_heads (5).")
        assert isinstance(exc, IncompatibleParamsError)
        assert "FTTransformerConfig" in str(exc)
        assert "d_model" in str(exc)


class TestWarningHelpers:
    def test_warn_data_issues_data_warning(self):
        with pytest.warns(DataWarning, match="test data warning"):
            warn_data("test data warning", stacklevel=1)

    def test_warn_config_issues_config_warning(self):
        with pytest.warns(ConfigWarning, match="test config warning"):
            warn_config("test config warning", stacklevel=1)

    def test_warn_performance_issues_performance_warning(self):
        with pytest.warns(PerformanceWarning, match="test perf warning"):
            warn_performance("test perf warning", stacklevel=1)


# ===========================================================================
# 3 — PreprocessingConfig.__post_init__ validation
# ===========================================================================


class TestPreprocessingConfigValidation:
    from deeptab.configs import PreprocessingConfig

    def test_valid_numerical_preprocessing_values(self):
        from deeptab.configs import PreprocessingConfig

        for val in (
            "ple",
            "quantile",
            "standardization",
            "minmax",
            "robust",
            "splines",
            "box-cox",
            "yeo-johnson",
            None,
        ):
            cfg = PreprocessingConfig(numerical_preprocessing=val)
            assert cfg.numerical_preprocessing == val

    def test_invalid_numerical_preprocessing_raises(self):
        from deeptab.configs import PreprocessingConfig

        with pytest.raises(InvalidParamError, match="numerical_preprocessing"):
            PreprocessingConfig(numerical_preprocessing="zscore")

    def test_n_bins_zero_raises(self):
        from deeptab.configs import PreprocessingConfig

        with pytest.raises(InvalidParamError, match="n_bins"):
            PreprocessingConfig(n_bins=0)

    def test_n_bins_one_raises(self):
        from deeptab.configs import PreprocessingConfig

        with pytest.raises(InvalidParamError, match="n_bins"):
            PreprocessingConfig(n_bins=1)

    def test_n_bins_negative_raises(self):
        from deeptab.configs import PreprocessingConfig

        with pytest.raises(InvalidParamError, match="n_bins"):
            PreprocessingConfig(n_bins=-5)

    def test_n_bins_two_is_valid(self):
        from deeptab.configs import PreprocessingConfig

        cfg = PreprocessingConfig(n_bins=2)
        assert cfg.n_bins == 2

    def test_n_knots_one_raises(self):
        from deeptab.configs import PreprocessingConfig

        with pytest.raises(InvalidParamError, match="n_knots"):
            PreprocessingConfig(n_knots=1)

    def test_invalid_scaling_strategy_raises(self):
        from deeptab.configs import PreprocessingConfig

        with pytest.raises(InvalidParamError, match="scaling_strategy"):
            PreprocessingConfig(scaling_strategy="normalize")

    def test_valid_scaling_strategy_values(self):
        from deeptab.configs import PreprocessingConfig

        for val in ("minmax", "standardization", "robust", None):
            cfg = PreprocessingConfig(scaling_strategy=val)
            assert cfg.scaling_strategy == val

    def test_invalid_binning_strategy_raises(self):
        from deeptab.configs import PreprocessingConfig

        with pytest.raises(InvalidParamError, match="binning_strategy"):
            PreprocessingConfig(binning_strategy="entropy")

    def test_cat_cutoff_zero_raises(self):
        from deeptab.configs import PreprocessingConfig

        with pytest.raises(InvalidParamError, match="cat_cutoff"):
            PreprocessingConfig(cat_cutoff=0.0)

    def test_cat_cutoff_one_raises(self):
        from deeptab.configs import PreprocessingConfig

        with pytest.raises(InvalidParamError, match="cat_cutoff"):
            PreprocessingConfig(cat_cutoff=1.0)

    def test_cat_cutoff_valid(self):
        from deeptab.configs import PreprocessingConfig

        cfg = PreprocessingConfig(cat_cutoff=0.05)
        assert cfg.cat_cutoff == 0.05

    def test_degree_zero_raises(self):
        from deeptab.configs import PreprocessingConfig

        with pytest.raises(InvalidParamError, match="degree"):
            PreprocessingConfig(degree=0)

    def test_degree_one_is_valid(self):
        from deeptab.configs import PreprocessingConfig

        cfg = PreprocessingConfig(degree=1)
        assert cfg.degree == 1


# ===========================================================================
# 4 — TrainerConfig.__post_init__ validation
# ===========================================================================


class TestTrainerConfigValidation:
    def test_max_epochs_zero_raises(self):
        from deeptab.configs import TrainerConfig

        with pytest.raises(InvalidParamError, match="max_epochs"):
            TrainerConfig(max_epochs=0)

    def test_max_epochs_negative_raises(self):
        from deeptab.configs import TrainerConfig

        with pytest.raises(InvalidParamError, match="max_epochs"):
            TrainerConfig(max_epochs=-10)

    def test_batch_size_zero_raises(self):
        from deeptab.configs import TrainerConfig

        with pytest.raises(InvalidParamError, match="batch_size"):
            TrainerConfig(batch_size=0)

    def test_lr_zero_raises(self):
        from deeptab.configs import TrainerConfig

        with pytest.raises(InvalidParamError, match="lr"):
            TrainerConfig(lr=0.0)

    def test_lr_negative_raises(self):
        from deeptab.configs import TrainerConfig

        with pytest.raises(InvalidParamError, match="lr"):
            TrainerConfig(lr=-1e-3)

    def test_weight_decay_negative_raises(self):
        from deeptab.configs import TrainerConfig

        with pytest.raises(InvalidParamError, match="weight_decay"):
            TrainerConfig(weight_decay=-0.01)

    def test_val_size_zero_raises(self):
        from deeptab.configs import TrainerConfig

        with pytest.raises(InvalidParamError, match="val_size"):
            TrainerConfig(val_size=0.0)

    def test_val_size_one_raises(self):
        from deeptab.configs import TrainerConfig

        with pytest.raises(InvalidParamError, match="val_size"):
            TrainerConfig(val_size=1.0)

    def test_invalid_mode_raises(self):
        from deeptab.configs import TrainerConfig

        with pytest.raises(InvalidParamError, match="mode"):
            TrainerConfig(mode="maximum")

    def test_patience_ge_max_epochs_warns(self):
        from deeptab.configs import TrainerConfig

        with pytest.warns(ConfigWarning, match="patience"):
            TrainerConfig(max_epochs=5, patience=5)

    def test_patience_greater_than_max_epochs_warns(self):
        from deeptab.configs import TrainerConfig

        with pytest.warns(ConfigWarning, match="patience"):
            TrainerConfig(max_epochs=3, patience=10)

    def test_valid_config_no_warning(self):
        from deeptab.configs import TrainerConfig

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConfigWarning)
            cfg = TrainerConfig(max_epochs=100, patience=15)
        assert cfg.max_epochs == 100

    def test_lr_patience_zero_raises(self):
        from deeptab.configs import TrainerConfig

        with pytest.raises(InvalidParamError, match="lr_patience"):
            TrainerConfig(lr_patience=0)

    def test_lr_patience_negative_raises(self):
        from deeptab.configs import TrainerConfig

        with pytest.raises(InvalidParamError, match="lr_patience"):
            TrainerConfig(lr_patience=-5)

    def test_lr_factor_zero_raises(self):
        from deeptab.configs import TrainerConfig

        with pytest.raises(InvalidParamError, match="lr_factor"):
            TrainerConfig(lr_factor=0.0)

    def test_lr_factor_one_raises(self):
        from deeptab.configs import TrainerConfig

        with pytest.raises(InvalidParamError, match="lr_factor"):
            TrainerConfig(lr_factor=1.0)

    def test_lr_factor_negative_raises(self):
        from deeptab.configs import TrainerConfig

        with pytest.raises(InvalidParamError, match="lr_factor"):
            TrainerConfig(lr_factor=-0.5)

    def test_lr_factor_greater_than_one_raises(self):
        from deeptab.configs import TrainerConfig

        with pytest.raises(InvalidParamError, match="lr_factor"):
            TrainerConfig(lr_factor=1.5)

    def test_lr_patience_ge_max_epochs_warns(self):
        from deeptab.configs import TrainerConfig

        with pytest.warns(ConfigWarning, match="lr_patience"):
            TrainerConfig(max_epochs=5, lr_patience=5)

    def test_lr_patience_greater_than_max_epochs_warns(self):
        from deeptab.configs import TrainerConfig

        with pytest.warns(ConfigWarning, match="lr_patience"):
            TrainerConfig(max_epochs=3, lr_patience=10)

    def test_valid_lr_params_no_warning(self):
        from deeptab.configs import TrainerConfig

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConfigWarning)
            cfg = TrainerConfig(max_epochs=100, lr_patience=5, lr_factor=0.5)
        assert cfg.lr_patience == 5
        assert cfg.lr_factor == 0.5


# ===========================================================================
# 4b — Misplaced-config warning (estimator constructor)
# ===========================================================================


class TestMisplacedConfigWarning:
    """The estimator constructor warns when a config lands in the wrong slot."""

    def test_trainer_config_as_model_config_warns(self):
        from deeptab.configs import TrainerConfig
        from deeptab.models import MLPClassifier

        with pytest.warns(ConfigWarning, match="model_config.*expects a BaseModelConfig"):
            MLPClassifier(model_config=TrainerConfig())

    def test_model_config_as_preprocessing_config_warns(self):
        from deeptab.configs import MLPConfig
        from deeptab.models import MLPClassifier

        with pytest.warns(ConfigWarning, match="preprocessing_config.*expects a PreprocessingConfig"):
            MLPClassifier(preprocessing_config=MLPConfig())

    def test_preprocessing_config_as_trainer_config_warns(self):
        from deeptab.configs import PreprocessingConfig
        from deeptab.models import MLPClassifier

        with pytest.warns(ConfigWarning, match="trainer_config.*expects a TrainerConfig"):
            MLPClassifier(trainer_config=PreprocessingConfig())

    def test_split_config_in_wrong_slot_warns(self):
        from deeptab.configs import SplitConfig
        from deeptab.models import MLPClassifier

        with pytest.warns(ConfigWarning, match="trainer_config.*expects a TrainerConfig"):
            MLPClassifier(trainer_config=SplitConfig())

    def test_correct_slots_emit_no_misplacement_warning(self):
        from deeptab.configs import MLPConfig, PreprocessingConfig, TrainerConfig
        from deeptab.models import MLPClassifier

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConfigWarning)
            MLPClassifier(
                model_config=MLPConfig(),
                preprocessing_config=PreprocessingConfig(),
                trainer_config=TrainerConfig(max_epochs=100, patience=15),
            )

    def test_duck_typed_object_is_not_flagged(self):
        from deeptab.models import MLPClassifier

        class DuckConfig:
            def get_params(self, deep=True):
                return {}

        # An unknown duck-typed object must not trip the misplacement check.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConfigWarning)
            MLPClassifier(model_config=DuckConfig())


# ===========================================================================
# 5 — BaseModelConfig / per-model config validation
# ===========================================================================


class TestModelConfigValidation:
    def test_d_model_zero_raises(self):
        from deeptab.configs import MambularConfig

        with pytest.raises(InvalidParamError, match="d_model"):
            MambularConfig(d_model=0)

    def test_d_model_negative_raises(self):
        from deeptab.configs import FTTransformerConfig

        with pytest.raises(InvalidParamError, match="d_model"):
            FTTransformerConfig(d_model=-8)

    def test_n_layers_zero_raises(self):
        from deeptab.configs import MambularConfig

        with pytest.raises(InvalidParamError, match="n_layers"):
            MambularConfig(n_layers=0)

    def test_n_heads_zero_raises(self):
        from deeptab.configs import FTTransformerConfig

        with pytest.raises(InvalidParamError, match="n_heads"):
            FTTransformerConfig(n_heads=0)

    def test_d_model_not_divisible_by_n_heads_raises(self):
        from deeptab.configs import FTTransformerConfig

        with pytest.raises(IncompatibleParamsError, match="d_model"):
            FTTransformerConfig(d_model=64, n_heads=5)

    def test_dropout_negative_raises(self):
        from deeptab.configs import MambularConfig

        with pytest.raises(InvalidParamError, match="dropout"):
            MambularConfig(dropout=-0.1)

    def test_dropout_one_raises(self):
        from deeptab.configs import MambularConfig

        with pytest.raises(InvalidParamError, match="dropout"):
            MambularConfig(dropout=1.0)

    def test_attn_dropout_out_of_range_raises(self):
        from deeptab.configs import FTTransformerConfig

        with pytest.raises(InvalidParamError, match="attn_dropout"):
            FTTransformerConfig(attn_dropout=1.5)

    def test_head_dropout_out_of_range_raises(self):
        from deeptab.configs import MambularConfig

        with pytest.raises(InvalidParamError, match="head_dropout"):
            MambularConfig(head_dropout=-0.01)

    def test_valid_config_passes(self):
        from deeptab.configs import FTTransformerConfig

        cfg = FTTransformerConfig(d_model=128, n_heads=8)
        assert cfg.d_model == 128
        assert cfg.n_heads == 8

    def test_invalid_cat_encoding_raises(self):
        from deeptab.configs import MambularConfig

        with pytest.raises(InvalidParamError, match="cat_encoding"):
            MambularConfig(cat_encoding="embedding")

    def test_rnn_dropout_negative_raises(self):
        from deeptab.configs import TabulaRNNConfig

        with pytest.raises(InvalidParamError, match="rnn_dropout"):
            TabulaRNNConfig(rnn_dropout=-0.1)

    def test_rnn_dropout_one_raises(self):
        from deeptab.configs import TabulaRNNConfig

        with pytest.raises(InvalidParamError, match="rnn_dropout"):
            TabulaRNNConfig(rnn_dropout=1.0)

    def test_n_frequencies_zero_raises(self):
        from deeptab.configs import MambularConfig

        with pytest.raises(InvalidParamError, match="n_frequencies"):
            MambularConfig(n_frequencies=0)

    def test_n_frequencies_negative_raises(self):
        from deeptab.configs import MLPConfig

        with pytest.raises(InvalidParamError, match="n_frequencies"):
            MLPConfig(n_frequencies=-4)

    def test_frequencies_init_scale_zero_raises(self):
        from deeptab.configs import MambularConfig

        with pytest.raises(InvalidParamError, match="frequencies_init_scale"):
            MambularConfig(frequencies_init_scale=0.0)

    def test_frequencies_init_scale_negative_raises(self):
        from deeptab.configs import MLPConfig

        with pytest.raises(InvalidParamError, match="frequencies_init_scale"):
            MLPConfig(frequencies_init_scale=-1.0)

    def test_layer_norm_eps_zero_raises(self):
        from deeptab.configs import MambularConfig

        with pytest.raises(InvalidParamError, match="layer_norm_eps"):
            MambularConfig(layer_norm_eps=0.0)

    def test_layer_norm_eps_negative_raises(self):
        from deeptab.configs import MLPConfig

        with pytest.raises(InvalidParamError, match="layer_norm_eps"):
            MLPConfig(layer_norm_eps=-1e-5)

    def test_batch_norm_and_layer_norm_both_true_warns(self):
        from deeptab.configs import MambularConfig

        with pytest.warns(ConfigWarning, match="batch_norm"):
            MambularConfig(batch_norm=True, layer_norm=True)

    def test_expand_factor_zero_raises(self):
        from deeptab.configs import MambaTabConfig

        with pytest.raises(InvalidParamError, match="expand_factor"):
            MambaTabConfig(expand_factor=0)

    def test_d_conv_zero_raises(self):
        from deeptab.configs import MambaTabConfig

        with pytest.raises(InvalidParamError, match="d_conv"):
            MambaTabConfig(d_conv=0)

    def test_d_state_zero_raises(self):
        from deeptab.configs import MambaTabConfig

        with pytest.raises(InvalidParamError, match="d_state"):
            MambaTabConfig(d_state=0)

    def test_transformer_dim_feedforward_zero_raises(self):
        from deeptab.configs import FTTransformerConfig

        with pytest.raises(InvalidParamError, match="transformer_dim_feedforward"):
            FTTransformerConfig(transformer_dim_feedforward=0)

    def test_dim_feedforward_zero_raises(self):
        from deeptab.configs import TabulaRNNConfig

        with pytest.raises(InvalidParamError, match="dim_feedforward"):
            TabulaRNNConfig(dim_feedforward=0)


# ===========================================================================
# 6 — ensure_dataframe() guards
# ===========================================================================


class TestEnsureDataframe:
    def test_empty_rows_raises_empty_data_error(self):
        from deeptab.core.sklearn_compat import ensure_dataframe

        df = pd.DataFrame({"a": pd.Series([], dtype="float64")})
        with pytest.raises(EmptyDataError):
            ensure_dataframe(df)

    def test_empty_columns_raises_empty_data_error(self):
        from deeptab.core.sklearn_compat import ensure_dataframe

        df = pd.DataFrame(index=range(10))
        with pytest.raises(EmptyDataError):
            ensure_dataframe(df)

    def test_unsupported_dtype_raises_column_dtype_error(self):
        from deeptab.core.sklearn_compat import ensure_dataframe

        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "dt": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            }
        )
        with pytest.raises(ColumnDtypeError, match="dt"):
            ensure_dataframe(df)

    def test_bool_columns_auto_cast_to_int8(self):
        from deeptab.core.sklearn_compat import ensure_dataframe

        df = pd.DataFrame({"flag": [True, False, True], "val": [1.0, 2.0, 3.0]})
        result = ensure_dataframe(df)
        assert result["flag"].dtype == np.dtype("int8")

    def test_numeric_and_object_pass(self):
        from deeptab.core.sklearn_compat import ensure_dataframe

        df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]})
        result = ensure_dataframe(df)
        assert result.shape == (2, 2)

    def test_all_nan_column_warns(self):
        from deeptab.core.sklearn_compat import ensure_dataframe

        df = pd.DataFrame(
            {
                "good": [1.0, 2.0, 3.0],
                "all_nan": [np.nan, np.nan, np.nan],
            }
        )
        with pytest.warns(DataWarning, match="all_nan"):
            ensure_dataframe(df)

    def test_context_appears_in_empty_error_message(self):
        from deeptab.core.sklearn_compat import ensure_dataframe

        df = pd.DataFrame(index=range(10))
        with pytest.raises(EmptyDataError, match="predict"):
            ensure_dataframe(df, context="predict")


# ===========================================================================
# 7 — validate_input_features() guards
# ===========================================================================


class TestValidateInputFeatures:
    """Use a mock fitted estimator to test column validation."""

    def _make_estimator(self, n_features=3, feature_names=None):
        class FakeEstimator:  # pyright: ignore[reportGeneralTypeIssues]
            n_features_in_: int
            feature_names_in_: np.ndarray

        est = FakeEstimator()
        est.n_features_in_ = n_features  # type: ignore[assignment]
        if feature_names is not None:
            est.feature_names_in_ = np.array(feature_names, dtype=object)  # type: ignore[assignment]
        return est

    def test_column_count_mismatch_raises(self):
        from deeptab.core.sklearn_compat import validate_input_features

        est = self._make_estimator(n_features=3)
        X = pd.DataFrame({"a": [1], "b": [2]})  # 2 cols, expected 3
        with pytest.raises(ColumnCountError, match="3"):
            validate_input_features(est, X)

    def test_column_names_missing_raises(self):
        from deeptab.core.sklearn_compat import validate_input_features

        est = self._make_estimator(n_features=2, feature_names=["age", "income"])
        X = pd.DataFrame({"age": [25], "salary": [50000]})
        with pytest.raises(ColumnNameError, match="income"):
            validate_input_features(est, X)

    def test_column_names_extra_in_error_message(self):
        from deeptab.core.sklearn_compat import validate_input_features

        est = self._make_estimator(n_features=2, feature_names=["age", "income"])
        X = pd.DataFrame({"age": [25], "salary": [50000]})
        with pytest.raises(ColumnNameError, match="salary"):
            validate_input_features(est, X)

    def test_matching_columns_passes(self):
        from deeptab.core.sklearn_compat import validate_input_features

        est = self._make_estimator(n_features=2, feature_names=["age", "income"])
        X = pd.DataFrame({"age": [25], "income": [50000]})
        result = validate_input_features(est, X)
        assert result.shape == (1, 2)

    def test_no_feature_names_on_estimator_passes_count_check(self):
        from deeptab.core.sklearn_compat import validate_input_features

        est = self._make_estimator(n_features=2)
        X = pd.DataFrame({"x": [1], "y": [2]})
        result = validate_input_features(est, X)
        assert result.shape == (1, 2)


# ===========================================================================
# 8 — _validate_fit_inputs() guards
# ===========================================================================


class TestValidateFitInputs:
    from deeptab.models.base import _validate_fit_inputs

    def _X(self, n=50):
        return pd.DataFrame(np.random.randn(n, 3), columns=["a", "b", "c"])  # type: ignore[call-overload]

    def _y(self, n=50):
        return np.random.randn(n)

    def test_length_mismatch_raises(self):
        from deeptab.models.base import _validate_fit_inputs

        with pytest.raises(DataError, match="100"):
            _validate_fit_inputs(self._X(100), self._y(80), regression=True)

    def test_nan_in_y_float_raises(self):
        from deeptab.models.base import _validate_fit_inputs

        y = self._y(50)
        y[0] = np.nan
        with pytest.raises(DataError, match="NaN"):
            _validate_fit_inputs(self._X(50), y, regression=True)

    def test_integer_y_with_nan_does_not_raise(self):
        """Integer y cannot contain NaN; validation skips non-float arrays."""
        from deeptab.models.base import _validate_fit_inputs

        y = np.array([0, 1, 0, 1] * 10, dtype=int)
        _validate_fit_inputs(self._X(40), y, regression=False)  # no error

    def test_poisson_negative_y_raises(self):
        from deeptab.models.base import _validate_fit_inputs

        y = np.array([1.0, 2.0, -1.0] * 5)
        with pytest.raises(DataError, match="poisson"):
            _validate_fit_inputs(self._X(15), y, regression=True, family="poisson")

    def test_poisson_non_negative_y_passes(self):
        from deeptab.models.base import _validate_fit_inputs

        y = np.array([0.0, 1.0, 2.0, 3.0] * 10)
        _validate_fit_inputs(self._X(40), y, regression=True, family="poisson")

    def test_gamma_zero_y_raises(self):
        from deeptab.models.base import _validate_fit_inputs

        y = np.array([1.0, 0.0, 2.0] * 5)
        with pytest.raises(DataError, match="gamma"):
            _validate_fit_inputs(self._X(15), y, regression=True, family="gamma")

    def test_gamma_positive_y_passes(self):
        from deeptab.models.base import _validate_fit_inputs

        y = np.abs(np.random.randn(30)) + 0.01
        _validate_fit_inputs(self._X(30), y, regression=True, family="gamma")

    def test_binomial_non_binary_raises(self):
        from deeptab.models.base import _validate_fit_inputs

        y = np.array([0, 1, 2, 0] * 5)
        with pytest.raises(DataError, match="binomial"):
            _validate_fit_inputs(self._X(20), y, regression=False, family="binomial")

    def test_high_nan_columns_warns(self):
        from deeptab.models.base import _validate_fit_inputs

        X = self._X(40)
        X["a"] = np.nan  # 100 % NaN
        y = self._y(40)
        with pytest.warns(DataWarning, match="50%"):
            _validate_fit_inputs(X, y, regression=True)


# ===========================================================================
# 9 — Distribution registry: unknown family
# ===========================================================================


class TestDistributionRegistry:
    def test_unknown_family_raises_invalid_param_error(self):
        from deeptab.distributions import get_distribution

        with pytest.raises(InvalidParamError, match="family"):
            get_distribution("banana")

    def test_unknown_family_message_lists_valid_options(self):
        from deeptab.distributions import get_distribution

        with pytest.raises(InvalidParamError, match="normal"):
            get_distribution("xyz_unknown")

    def test_known_family_returns_distribution(self):
        from deeptab.distributions import get_distribution

        dist = get_distribution("normal")
        assert dist is not None


# ===========================================================================
# 10 — TabTransformer architecture requirement
# ===========================================================================


class TestTabTransformerArchitectureRequirement:
    def test_no_categorical_features_raises_architecture_error(self):
        from deeptab.architectures.tabtransformer import TabTransformer
        from deeptab.configs.models.tabtransformer_config import TabTransformerConfig

        num_info = {"f0": {"preprocessing": "ple", "dimension": 20, "categories": None}}
        cat_info = {}  # no categorical features
        emb_info = {}
        with pytest.raises(ArchitectureRequirementError, match="categorical"):
            TabTransformer(
                feature_information=(num_info, cat_info, emb_info),
                num_classes=2,
                config=TabTransformerConfig(),
            )

    def test_with_categorical_features_passes(self):
        from deeptab.architectures.tabtransformer import TabTransformer
        from deeptab.configs.models.tabtransformer_config import TabTransformerConfig

        num_info = {}
        cat_info = {"city": {"dimension": 1, "categories": ["NYC", "LA"]}}
        emb_info = {}
        # Should not raise — if it raises for other reasons (unrelated to the
        # requirement guard), that is a separate issue.
        try:
            TabTransformer(
                feature_information=(num_info, cat_info, emb_info),
                num_classes=2,
                config=TabTransformerConfig(),
            )
        except ArchitectureRequirementError:
            pytest.fail("ArchitectureRequirementError raised unexpectedly with categorical features")
        except Exception:  # noqa: S110
            pass


# ===========================================================================
# 11 — Public API exports
# ===========================================================================


class TestPublicAPIExports:
    def test_exceptions_exported_from_deeptab(self):
        """Only the catch-all base and NotFittedError (the one users legitimately handle) are exported."""
        for name in ("DeepTabError", "NotFittedError"):
            assert hasattr(deeptab, name), f"deeptab.{name} not exported"

    def test_internal_exceptions_not_in_deeptab_top_level(self):
        """Granular exception types live in deeptab.core.exceptions, not the top-level namespace."""
        for name in (
            "DataError",
            "ColumnDtypeError",
            "ColumnCountError",
            "EmptyDataError",
            "InvalidParamError",
            "ArchitectureRequirementError",
        ):
            assert not hasattr(deeptab, name), (
                f"deeptab.{name} should not be in the public top-level namespace "
                "(import from deeptab.core.exceptions instead)"
            )

    def test_warnings_exported_from_deeptab(self):
        for name in ("DeepTabWarning", "DataWarning", "ConfigWarning", "PerformanceWarning"):
            assert hasattr(deeptab, name), f"deeptab.{name} not exported"

    def test_exceptions_exported_from_deeptab_core(self):
        import deeptab.core as core

        for name in (
            "DeepTabError",
            "DataError",
            "ColumnDtypeError",
            "NotFittedError",
            "InvalidParamError",
            "ConfigWarning",
            "DataWarning",
        ):
            assert hasattr(core, name), f"deeptab.core.{name} not exported"

    def test_filterable_data_warning(self):
        """Users can filter DataWarning independently from other warnings."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_data("data issue", stacklevel=1)
            warn_config("config issue", stacklevel=1)
        data_warns = [
            w for w in caught if issubclass(w.category, DataWarning) and not issubclass(w.category, ConfigWarning)
        ]
        assert len(data_warns) == 1
        assert "data issue" in str(data_warns[0].message)
