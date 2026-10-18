"""Tests for config contracts."""

# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false

import dataclasses
import dataclasses as _dc

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from deeptab.configs import (
    AutoIntConfig,
    BaseModelConfig,
    FTTransformerConfig,
    MambaTabConfig,
    MambAttentionConfig,
    MambularConfig,
    MLPConfig,
    NDTFConfig,
    NODEConfig,
    PreprocessingConfig,
    ResNetConfig,
    SAINTConfig,
    TabMConfig,
    TabRConfig,
    TabTransformerConfig,
    TabulaRNNConfig,
    TrainerConfig,
)


class TestTrainerConfig:
    def test_instantiation_defaults(self):
        cfg = TrainerConfig()
        assert cfg.max_epochs == 100
        assert cfg.batch_size == 128
        assert cfg.val_size == 0.2
        assert cfg.shuffle is True
        assert cfg.stratify is True
        assert cfg.patience == 15
        assert cfg.monitor == "val_loss"
        assert cfg.mode == "min"
        assert cfg.lr == 1e-4
        assert cfg.lr_patience == 10
        assert cfg.lr_factor == 0.1
        assert cfg.weight_decay == 1e-6
        assert cfg.optimizer_type == "Adam"
        assert cfg.checkpoint_path == "model_checkpoints"

    def test_instantiation_custom(self):
        cfg = TrainerConfig(max_epochs=50, lr=1e-3, batch_size=256)
        assert cfg.max_epochs == 50
        assert cfg.lr == 1e-3
        assert cfg.batch_size == 256

    def test_does_not_contain_architecture_fields(self):
        """TrainerConfig must not carry model architecture fields."""
        cfg = TrainerConfig()
        architecture_fields = {"d_model", "n_layers", "n_heads", "dropout", "activation"}
        config_fields = {f.name for f in dataclasses.fields(cfg)}
        assert architecture_fields.isdisjoint(config_fields), (
            f"TrainerConfig unexpectedly contains architecture fields: {architecture_fields & config_fields}"
        )

    def test_does_not_contain_preprocessing_fields(self):
        """TrainerConfig must not carry preprocessing fields."""
        cfg = TrainerConfig()
        preprocessing_fields = {
            "numerical_preprocessing",
            "categorical_preprocessing",
            "n_bins",
            "scaling_strategy",
        }
        config_fields = {f.name for f in dataclasses.fields(cfg)}
        assert preprocessing_fields.isdisjoint(config_fields), (
            f"TrainerConfig unexpectedly contains preprocessing fields: {preprocessing_fields & config_fields}"
        )

    def test_get_params_returns_all_fields(self):
        cfg = TrainerConfig()
        params = cfg.get_params()
        expected_keys = {f.name for f in dataclasses.fields(TrainerConfig)}
        assert set(params.keys()) == expected_keys

    def test_get_params_reflects_custom_values(self):
        cfg = TrainerConfig(max_epochs=42, lr=5e-4)
        params = cfg.get_params()
        assert params["max_epochs"] == 42
        assert params["lr"] == 5e-4

    def test_set_params_updates_fields(self):
        cfg = TrainerConfig()
        cfg.set_params(max_epochs=200, patience=5)
        assert cfg.max_epochs == 200
        assert cfg.patience == 5

    def test_set_params_returns_self(self):
        cfg = TrainerConfig()
        result = cfg.set_params(max_epochs=50)
        assert result is cfg

    def test_sklearn_clone(self):
        cfg = TrainerConfig(max_epochs=50, lr=1e-3)
        cloned = clone(cfg)
        assert cloned is not cfg
        assert cloned.max_epochs == 50
        assert cloned.lr == 1e-3

    def test_sklearn_clone_independence(self):
        """Mutating the clone must not affect the original."""
        cfg = TrainerConfig(max_epochs=50)
        cloned = clone(cfg)
        cloned.set_params(max_epochs=999)
        assert cfg.max_epochs == 50


class TestPreprocessingConfig:
    def test_instantiation_defaults_all_none(self):
        cfg = PreprocessingConfig()
        for f in dataclasses.fields(cfg):
            assert getattr(cfg, f.name) is None, f"Expected {f.name} to default to None, got {getattr(cfg, f.name)}"

    def test_instantiation_custom(self):
        cfg = PreprocessingConfig(
            numerical_preprocessing="ple",
            categorical_preprocessing="int",
            n_bins=32,
        )
        assert cfg.numerical_preprocessing == "ple"
        assert cfg.categorical_preprocessing == "int"
        assert cfg.n_bins == 32

    def test_owns_preprocessing_fields(self):
        """All expected preprocessor arg names must be present."""
        expected = {
            "numerical_preprocessing",
            "categorical_preprocessing",
            "n_bins",
            "feature_preprocessing",
            "use_decision_tree_bins",
            "binning_strategy",
            "task",
            "cat_cutoff",
            "treat_all_integers_as_numerical",
            "degree",
            "scaling_strategy",
            "n_knots",
            "use_decision_tree_knots",
            "knots_strategy",
            "spline_implementation",
        }
        config_fields = {f.name for f in dataclasses.fields(PreprocessingConfig)}
        missing = expected - config_fields
        assert not missing, f"PreprocessingConfig is missing expected fields: {missing}"

    def test_does_not_contain_architecture_fields(self):
        cfg = PreprocessingConfig()
        architecture_fields = {"d_model", "n_layers", "activation", "dropout", "lr"}
        config_fields = {f.name for f in dataclasses.fields(cfg)}
        assert architecture_fields.isdisjoint(config_fields), (
            f"PreprocessingConfig unexpectedly contains non-preprocessing fields: {architecture_fields & config_fields}"
        )

    def test_get_params_returns_all_fields(self):
        cfg = PreprocessingConfig()
        params = cfg.get_params()
        expected_keys = {f.name for f in dataclasses.fields(PreprocessingConfig)}
        assert set(params.keys()) == expected_keys

    def test_get_params_reflects_custom_values(self):
        cfg = PreprocessingConfig(numerical_preprocessing="quantile", n_bins=64)
        params = cfg.get_params()
        assert params["numerical_preprocessing"] == "quantile"
        assert params["n_bins"] == 64

    def test_set_params_updates_fields(self):
        cfg = PreprocessingConfig()
        cfg.set_params(numerical_preprocessing="standard", n_bins=16)
        assert cfg.numerical_preprocessing == "standard"
        assert cfg.n_bins == 16

    def test_set_params_returns_self(self):
        cfg = PreprocessingConfig()
        result = cfg.set_params(n_bins=8)
        assert result is cfg

    def test_to_preprocessor_kwargs_excludes_none(self):
        cfg = PreprocessingConfig(numerical_preprocessing="ple", n_bins=32)
        kwargs = cfg.to_preprocessor_kwargs()
        # Legacy field names are resolved to their canonical PreTab 1.0 equivalents.
        assert kwargs["numerical_method"] == "ple"
        assert kwargs["output_dim"] == 32
        # Fields left as None must not appear
        assert "categorical_method" not in kwargs
        assert "scaling" not in kwargs

    def test_to_preprocessor_kwargs_empty_when_all_none(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            cfg = PreprocessingConfig()
        # output_dim resolves to DeepTab's explicit default of 7 even when unset.
        assert cfg.to_preprocessor_kwargs() == {"output_dim": 7}

    def test_sklearn_clone(self):
        cfg = PreprocessingConfig(numerical_preprocessing="ple", n_bins=32)
        cloned = clone(cfg)
        assert cloned is not cfg
        assert cloned.numerical_preprocessing == "ple"
        assert cloned.n_bins == 32

    def test_sklearn_clone_independence(self):
        cfg = PreprocessingConfig(n_bins=32)
        cloned = clone(cfg)
        cloned.set_params(n_bins=999)
        assert cfg.n_bins == 32


N = 120

RNG = np.random.default_rng(0)

X_cls = pd.DataFrame(RNG.standard_normal((N, 6)), columns=[f"f{i}" for i in range(6)])

y_cls = RNG.integers(0, 3, size=N)

X_reg = pd.DataFrame(RNG.standard_normal((N, 6)), columns=[f"f{i}" for i in range(6)])

y_reg = RNG.standard_normal(N)

_FAST_TRAINER = TrainerConfig(max_epochs=1, batch_size=64, patience=1)


class TestMLPConfig:
    def test_instantiation_defaults(self):
        cfg = MLPConfig()
        assert cfg.layer_sizes == [256, 128, 32]
        assert cfg.dropout == 0.2
        assert cfg.use_glu is False
        assert cfg.skip_connections is False

    def test_instantiation_custom(self):
        cfg = MLPConfig(layer_sizes=[128, 64], dropout=0.1)
        assert cfg.layer_sizes == [128, 64]
        assert cfg.dropout == 0.1

    def test_does_not_contain_dead_fields(self):
        """Fields the MLP neural network never reads must be absent from MLPConfig."""
        cfg_fields = {f.name for f in _dc.fields(MLPConfig)}
        # skip_layers is dead code in MLP: the network only reads skip_connections
        assert "skip_layers" not in cfg_fields, (
            "skip_layers is not read by the MLP network — it must not appear in MLPConfig"
        )

    def test_activation_not_redeclared(self):
        """activation must be inherited from BaseModelConfig, not re-declared in MLPConfig."""
        # The field must still be accessible (via inheritance)
        cfg = MLPConfig()
        assert hasattr(cfg, "activation")
        # But the redeclaration should be gone: its defining class must be BaseModelConfig
        for f in _dc.fields(MLPConfig):
            if f.name == "activation":
                # Verify position stays at the BaseModelConfig order (before layer_sizes)
                field_names = [fi.name for fi in _dc.fields(MLPConfig)]
                assert field_names.index("activation") < field_names.index("layer_sizes"), (
                    "activation should be inherited at the BaseModelConfig position, not after layer_sizes"
                )
                break

    def test_inherits_base_model_config(self):
        assert issubclass(MLPConfig, BaseModelConfig)

    def test_does_not_contain_training_fields(self):
        """MLPConfig must not carry any training/optimizer fields."""
        training_fields = {"lr", "lr_patience", "lr_factor", "weight_decay"}
        cfg_fields = {f.name for f in _dc.fields(MLPConfig)}
        assert training_fields.isdisjoint(cfg_fields), (
            f"MLPConfig unexpectedly contains training fields: {training_fields & cfg_fields}"
        )

    def test_contains_required_architecture_fields(self):
        """Fields that MLP neural network reads via self.hparams must be present."""
        required = {
            "layer_sizes",
            "dropout",
            "use_glu",
            "activation",
            "skip_connections",
            "use_embeddings",
            "d_model",
            "batch_norm",
            "layer_norm",
        }
        cfg_fields = {f.name for f in _dc.fields(MLPConfig)}
        missing = required - cfg_fields
        assert not missing, f"MLPConfig is missing required architecture fields: {missing}"

    def test_get_params_returns_all_fields(self):
        cfg = MLPConfig()
        params = cfg.get_params()
        expected = {f.name for f in _dc.fields(MLPConfig)}
        assert set(params.keys()) == expected

    def test_set_params_updates_fields(self):
        cfg = MLPConfig()
        cfg.set_params(layer_sizes=[64, 32], dropout=0.3)
        assert cfg.layer_sizes == [64, 32]
        assert cfg.dropout == 0.3

    def test_sklearn_clone(self):
        cfg = MLPConfig(layer_sizes=[64, 32], dropout=0.3)
        cloned = clone(cfg)
        assert cloned is not cfg
        assert cloned.layer_sizes == [64, 32]
        assert cloned.dropout == 0.3


_TRAINING_FIELDS = {"lr", "lr_patience", "lr_factor", "weight_decay"}

_PREPROCESSING_FIELDS = {
    "numerical_preprocessing",
    "categorical_preprocessing",
    "n_bins",
    "scaling_strategy",
}


def _config_field_names(cfg_class):
    return {f.name for f in _dc.fields(cfg_class)}


class TestConfigFieldSeparation:
    """Verify each new *Config: no training fields, no preprocessing fields."""

    @pytest.mark.parametrize(
        "cfg_class",
        [
            ResNetConfig,
            FTTransformerConfig,
            TabTransformerConfig,
            AutoIntConfig,
            SAINTConfig,
            NODEConfig,
            NDTFConfig,
            TabMConfig,
            TabRConfig,
            MambularConfig,
            MambaTabConfig,
            MambAttentionConfig,
            TabulaRNNConfig,
        ],
    )
    def test_no_training_fields(self, cfg_class):
        fields = _config_field_names(cfg_class)
        assert fields.isdisjoint(_TRAINING_FIELDS), (
            f"{cfg_class.__name__} contains training fields: {fields & _TRAINING_FIELDS}"
        )

    @pytest.mark.parametrize(
        "cfg_class",
        [
            ResNetConfig,
            FTTransformerConfig,
            TabTransformerConfig,
            AutoIntConfig,
            SAINTConfig,
            NODEConfig,
            NDTFConfig,
            TabMConfig,
            TabRConfig,
            MambularConfig,
            MambaTabConfig,
            MambAttentionConfig,
            TabulaRNNConfig,
        ],
    )
    def test_no_preprocessing_fields(self, cfg_class):
        fields = _config_field_names(cfg_class)
        assert fields.isdisjoint(_PREPROCESSING_FIELDS), (
            f"{cfg_class.__name__} contains preprocessing fields: {fields & _PREPROCESSING_FIELDS}"
        )

    @pytest.mark.parametrize(
        "cfg_class",
        [
            ResNetConfig,
            FTTransformerConfig,
            TabTransformerConfig,
            AutoIntConfig,
            SAINTConfig,
            NODEConfig,
            NDTFConfig,
            TabMConfig,
            TabRConfig,
            MambularConfig,
            MambaTabConfig,
            MambAttentionConfig,
            TabulaRNNConfig,
        ],
    )
    def test_get_params_set_params_clone(self, cfg_class):
        cfg = cfg_class()
        params = cfg.get_params()
        assert isinstance(params, dict)
        assert len(params) > 0
        # set_params returns self
        result = cfg.set_params(**{next(iter(params)): next(iter(params.values()))})
        assert result is cfg
        # clone produces a distinct object of the same type
        cloned = clone(cfg)
        assert cloned is not cfg
        assert type(cloned) is type(cfg)
        # Compare only non-Callable fields (nn.Module has no __eq__)
        from collections.abc import Callable as _Callable

        for fname, fval in params.items():
            if not callable(fval):
                assert cloned.get_params()[fname] == fval
