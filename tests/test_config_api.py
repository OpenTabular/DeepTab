"""Tests for the DeepTab split-config API: TrainerConfig, PreprocessingConfig, and per-model *Config classes."""

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
from deeptab.models.autoint import AutoIntClassifier, AutoIntRegressor
from deeptab.models.fttransformer import FTTransformerClassifier, FTTransformerRegressor
from deeptab.models.mambatab import MambaTabClassifier, MambaTabRegressor
from deeptab.models.mambattention import MambAttentionClassifier, MambAttentionRegressor
from deeptab.models.mambular import MambularClassifier, MambularRegressor
from deeptab.models.mlp import MLPClassifier, MLPRegressor
from deeptab.models.ndtf import NDTFClassifier, NDTFRegressor
from deeptab.models.node import NODEClassifier, NODERegressor
from deeptab.models.resnet import ResNetClassifier, ResNetRegressor
from deeptab.models.saint import SAINTClassifier, SAINTRegressor
from deeptab.models.tabm import TabMClassifier, TabMRegressor
from deeptab.models.tabr import TabRClassifier, TabRRegressor
from deeptab.models.tabtransformer import TabTransformerClassifier, TabTransformerRegressor
from deeptab.models.tabularnn import TabulaRNNClassifier, TabulaRNNRegressor

# ---------------------------------------------------------------------------
# TrainerConfig
# ---------------------------------------------------------------------------


class TestTrainerConfig:
    def test_instantiation_defaults(self):
        cfg = TrainerConfig()
        assert cfg.max_epochs == 100
        assert cfg.batch_size == 128
        assert cfg.val_size == 0.2
        assert cfg.shuffle is True
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


# ---------------------------------------------------------------------------
# PreprocessingConfig
# ---------------------------------------------------------------------------


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
        assert "numerical_preprocessing" in kwargs
        assert "n_bins" in kwargs
        # Fields left as None must not appear
        assert "categorical_preprocessing" not in kwargs
        assert "scaling_strategy" not in kwargs

    def test_to_preprocessor_kwargs_empty_when_all_none(self):
        cfg = PreprocessingConfig()
        assert cfg.to_preprocessor_kwargs() == {}

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


# ---------------------------------------------------------------------------
# Estimator-level tests — split-config API on SklearnBase
# ---------------------------------------------------------------------------


N = 120
RNG = np.random.default_rng(0)
X_cls = pd.DataFrame(RNG.standard_normal((N, 6)), columns=[f"f{i}" for i in range(6)])
y_cls = RNG.integers(0, 3, size=N)
X_reg = pd.DataFrame(RNG.standard_normal((N, 6)), columns=[f"f{i}" for i in range(6)])
y_reg = RNG.standard_normal(N)

# TrainerConfig with max_epochs=1 keeps CI fast
_FAST_TRAINER = TrainerConfig(max_epochs=1, batch_size=64, patience=1)


class TestEstimatorSplitConfigInit:
    def test_initializes_with_split_configs(self):
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[32, 16]),
            trainer_config=TrainerConfig(max_epochs=1),
        )
        assert model.model_config is not None
        assert model.trainer_config is not None
        assert model.preprocessing_config is not None  # defaults to empty PreprocessingConfig

    def test_initializes_with_only_trainer_config(self):
        model = MLPClassifier(trainer_config=_FAST_TRAINER)
        assert model.trainer_config is _FAST_TRAINER
        assert model.model_config is None
        assert model.config is not None  # default config created

    def test_initializes_with_random_state(self):
        model = MLPClassifier(
            model_config=MLPConfig(),
            trainer_config=_FAST_TRAINER,
            random_state=42,
        )
        assert model.random_state == 42

    def test_flat_kwargs_raise_error(self):
        """Flat kwargs must now raise TypeError with a helpful message (PR5)."""
        with pytest.raises(TypeError, match="no longer accepts flat"):
            MLPClassifier(layer_sizes=[32, 16])


class TestEstimatorGetParams:
    def test_get_params_returns_config_objects(self):
        mc = MLPConfig(layer_sizes=[32, 16])
        tc = TrainerConfig(max_epochs=1)
        pc = PreprocessingConfig(numerical_preprocessing="standard")
        model = MLPClassifier(model_config=mc, trainer_config=tc, preprocessing_config=pc)

        params = model.get_params(deep=False)
        assert params["model_config"] is mc
        assert params["trainer_config"] is tc
        assert params["preprocessing_config"] is pc

    def test_get_params_deep_exposes_nested_keys(self):
        mc = MLPConfig(layer_sizes=[32])
        tc = TrainerConfig(max_epochs=5, lr=1e-3)
        model = MLPClassifier(model_config=mc, trainer_config=tc)

        params = model.get_params(deep=True)
        assert "model_config__layer_sizes" in params
        assert "trainer_config__max_epochs" in params
        assert params["trainer_config__max_epochs"] == 5
        assert params["trainer_config__lr"] == 1e-3
        assert "preprocessing_config__numerical_preprocessing" in params

    def test_flat_kwargs_raise_type_error(self):
        """PR5: flat kwargs must now raise TypeError (legacy path removed)."""
        with pytest.raises(TypeError, match="no longer accepts flat"):
            MLPClassifier(layer_sizes=[32, 16])


class TestEstimatorSetParams:
    def test_set_params_nested_model_config(self):
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[64, 32]),
            trainer_config=_FAST_TRAINER,
        )
        model.set_params(model_config__layer_sizes=[128, 64])
        assert model.model_config.layer_sizes == [128, 64]

    def test_set_params_nested_trainer_config(self):
        model = MLPClassifier(
            model_config=MLPConfig(),
            trainer_config=TrainerConfig(max_epochs=10),
        )
        model.set_params(trainer_config__max_epochs=20, trainer_config__lr=5e-4)
        assert model.trainer_config.max_epochs == 20
        assert model.trainer_config.lr == 5e-4

    def test_set_params_nested_preprocessing_config(self):
        model = MLPClassifier(
            model_config=MLPConfig(),
            preprocessing_config=PreprocessingConfig(),
            trainer_config=_FAST_TRAINER,
        )
        model.set_params(preprocessing_config__numerical_preprocessing="quantile")
        assert model.preprocessing_config.numerical_preprocessing == "quantile"

    def test_set_params_replace_whole_config(self):
        model = MLPClassifier(
            model_config=MLPConfig(),
            trainer_config=TrainerConfig(max_epochs=10),
        )
        new_tc = TrainerConfig(max_epochs=99)
        model.set_params(trainer_config=new_tc)
        assert model.trainer_config is new_tc
        assert model.trainer_config.max_epochs == 99

    def test_set_params_returns_self(self):
        model = MLPClassifier(model_config=MLPConfig(), trainer_config=_FAST_TRAINER)
        result = model.set_params(trainer_config__lr=1e-5)
        assert result is model


class TestEstimatorSklearnClone:
    def test_clone_creates_new_object(self):
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[32]),
            trainer_config=TrainerConfig(max_epochs=1),
        )
        cloned = clone(model)
        assert cloned is not model

    def test_clone_preserves_config_values(self):
        mc = MLPConfig(layer_sizes=[32, 16])
        tc = TrainerConfig(max_epochs=3, lr=5e-4)
        model = MLPClassifier(model_config=mc, trainer_config=tc, random_state=7)
        cloned = clone(model)

        assert cloned.model_config.layer_sizes == [32, 16]
        assert cloned.trainer_config.max_epochs == 3
        assert cloned.trainer_config.lr == 5e-4
        assert cloned.random_state == 7

    def test_clone_independence(self):
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[32]),
            trainer_config=TrainerConfig(max_epochs=3),
        )
        cloned = clone(model)
        cloned.set_params(trainer_config__max_epochs=99)
        assert model.trainer_config.max_epochs == 3


class TestEstimatorFitPredict:
    """Functional smoke tests: fit → predict with the split-config API."""

    def test_classifier_fit_predict(self):
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[32, 16]),
            trainer_config=TrainerConfig(max_epochs=1, batch_size=64, patience=1),
        )
        model.fit(X_cls, y_cls)
        preds = model.predict(X_cls)
        assert len(preds) == N
        assert set(preds).issubset({0, 1, 2})

    def test_regressor_fit_predict(self):
        model = MLPRegressor(
            model_config=MLPConfig(layer_sizes=[32, 16]),
            trainer_config=TrainerConfig(max_epochs=1, batch_size=64, patience=1),
        )
        model.fit(X_reg, y_reg)
        preds = model.predict(X_reg)
        assert len(preds) == N
        assert np.isfinite(preds).all()

    def test_trainer_config_controls_max_epochs(self):
        """TrainerConfig.max_epochs must be used (not a hard-coded default)."""
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[16]),
            trainer_config=TrainerConfig(max_epochs=1, batch_size=64, patience=1),
        )
        model.fit(X_cls, y_cls)
        assert model.trainer.max_epochs == 1

    def test_random_state_is_honoured(self):
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[16]),
            trainer_config=TrainerConfig(max_epochs=1, batch_size=64, patience=1),
            random_state=42,
        )
        model.fit(X_cls, y_cls)
        assert model.random_state == 42


# ---------------------------------------------------------------------------
# PR 3 — MLPConfig (clean architecture-only config)
# ---------------------------------------------------------------------------


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


class TestMLPWithMLPConfig:
    """Functional smoke tests: full pipeline using the new MLPConfig."""

    _fast = TrainerConfig(max_epochs=1, batch_size=64, patience=1)

    def test_classifier_fit_predict_with_mlp_config(self):
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[32, 16]),
            trainer_config=self._fast,
        )
        model.fit(X_cls, y_cls)
        preds = model.predict(X_cls)
        assert len(preds) == N
        assert set(preds).issubset({0, 1, 2})

    def test_regressor_fit_predict_with_mlp_config(self):
        model = MLPRegressor(
            model_config=MLPConfig(layer_sizes=[32, 16]),
            trainer_config=self._fast,
        )
        model.fit(X_reg, y_reg)
        preds = model.predict(X_reg)
        assert len(preds) == N
        assert np.isfinite(preds).all()

    def test_predict_proba_with_mlp_config(self):
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[32, 16]),
            trainer_config=self._fast,
        )
        model.fit(X_cls, y_cls)
        proba = model.predict_proba(X_cls)
        assert proba.shape == (N, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_get_params_with_mlp_config(self):
        mc = MLPConfig(layer_sizes=[32])
        tc = TrainerConfig(max_epochs=2, lr=5e-4)
        model = MLPClassifier(model_config=mc, trainer_config=tc)

        params = model.get_params(deep=False)
        assert params["model_config"] is mc
        assert params["trainer_config"] is tc

        deep_params = model.get_params(deep=True)
        assert deep_params["model_config__layer_sizes"] == [32]
        assert deep_params["trainer_config__lr"] == 5e-4

    def test_set_params_with_mlp_config(self):
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[32]),
            trainer_config=TrainerConfig(max_epochs=2),
        )
        model.set_params(model_config__layer_sizes=[64, 32], trainer_config__lr=1e-5)
        assert model.model_config.layer_sizes == [64, 32]
        assert model.trainer_config.lr == 1e-5

    def test_sklearn_clone_with_mlp_config(self):
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[32, 16], dropout=0.1),
            trainer_config=TrainerConfig(max_epochs=2, lr=5e-4),
            random_state=13,
        )
        cloned = clone(model)
        assert cloned is not model
        assert cloned.model_config.layer_sizes == [32, 16]
        assert cloned.model_config.dropout == 0.1
        assert cloned.trainer_config.max_epochs == 2
        assert cloned.random_state == 13

    def test_clone_and_fit_independence(self):
        """Fitting the clone must not affect the original model object."""
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[16]),
            trainer_config=self._fast,
        )
        cloned = clone(model)
        cloned.fit(X_cls, y_cls)
        assert not getattr(model, "is_fitted_", False)

    def test_flat_kwargs_raise_error_after_pr5(self):
        """Flat kwargs must now raise TypeError (PR5)."""
        with pytest.raises(TypeError, match="no longer accepts flat"):
            MLPClassifier(layer_sizes=[32, 16])
        with pytest.raises(TypeError, match="no longer accepts flat"):
            MLPRegressor(layer_sizes=[32, 16])


# ===========================================================================
# PR 4: Tests for all 13 remaining new *Config classes
# ===========================================================================


_TRAINING_FIELDS = {"lr", "lr_patience", "lr_factor", "weight_decay"}
_PREPROCESSING_FIELDS = {
    "numerical_preprocessing",
    "categorical_preprocessing",
    "n_bins",
    "scaling_strategy",
}


def _config_field_names(cfg_class):
    return {f.name for f in _dc.fields(cfg_class)}


# ---------------------------------------------------------------------------
# Shared per-config assertions (no fit needed)
# ---------------------------------------------------------------------------


class TestPR4ConfigSanity:
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


# ---------------------------------------------------------------------------
# Per-model smoke tests (fit + predict with new config)
# ---------------------------------------------------------------------------


class TestResNetWithConfig:
    _fast = TrainerConfig(max_epochs=1, batch_size=64, patience=1)

    def test_classifier_fit_predict(self):
        model = ResNetClassifier(
            model_config=ResNetConfig(num_blocks=1, layer_sizes=[32]),
            trainer_config=self._fast,
        )
        model.fit(X_cls, y_cls)
        preds = model.predict(X_cls)
        assert len(preds) == N

    def test_regressor_fit_predict(self):
        model = ResNetRegressor(
            model_config=ResNetConfig(num_blocks=1, layer_sizes=[32]),
            trainer_config=self._fast,
        )
        model.fit(X_reg, y_reg)
        preds = model.predict(X_reg)
        assert len(preds) == N
        assert np.isfinite(preds).all()

    def test_get_params_set_params_clone_model(self):
        mc = ResNetConfig(num_blocks=2)
        model = ResNetClassifier(model_config=mc, trainer_config=self._fast)
        params = model.get_params(deep=True)
        assert "model_config__num_blocks" in params
        model.set_params(model_config__num_blocks=1)
        assert model.model_config.num_blocks == 1
        cloned = clone(model)
        assert cloned.model_config.num_blocks == 1

    def test_flat_kwargs_raise_error(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            ResNetClassifier(num_blocks=2, layer_sizes=[32])


class TestFTTransformerWithConfig:
    _fast = TrainerConfig(max_epochs=1, batch_size=64, patience=1)

    def test_classifier_fit_predict(self):
        model = FTTransformerClassifier(
            model_config=FTTransformerConfig(n_layers=2, d_model=32, n_heads=4),
            trainer_config=self._fast,
        )
        model.fit(X_cls, y_cls)
        preds = model.predict(X_cls)
        assert len(preds) == N

    def test_regressor_fit_predict(self):
        model = FTTransformerRegressor(
            model_config=FTTransformerConfig(n_layers=2, d_model=32, n_heads=4),
            trainer_config=self._fast,
        )
        model.fit(X_reg, y_reg)
        preds = model.predict(X_reg)
        assert len(preds) == N
        assert np.isfinite(preds).all()

    def test_get_params_set_params_clone_model(self):
        mc = FTTransformerConfig(n_layers=2)
        model = FTTransformerClassifier(model_config=mc, trainer_config=self._fast)
        params = model.get_params(deep=True)
        assert "model_config__n_layers" in params
        model.set_params(model_config__n_layers=3)
        assert model.model_config.n_layers == 3
        cloned = clone(model)
        assert cloned.model_config.n_layers == 3

    def test_flat_kwargs_raise_error(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            FTTransformerClassifier(n_layers=2, d_model=32)


class TestTabTransformerWithConfig:
    _fast = TrainerConfig(max_epochs=1, batch_size=64, patience=1)
    # TabTransformer requires at least one categorical feature
    _X_cls = X_cls.copy()
    _X_cls["cat_col"] = np.tile(["A", "B", "C"], N // 3 + 1)[:N]
    _X_reg = X_reg.copy()
    _X_reg["cat_col"] = np.tile(["A", "B", "C"], N // 3 + 1)[:N]

    def test_classifier_fit_predict(self):
        model = TabTransformerClassifier(
            model_config=TabTransformerConfig(n_layers=2, d_model=32, n_heads=4),
            trainer_config=self._fast,
        )
        model.fit(self._X_cls, y_cls)
        preds = model.predict(self._X_cls)
        assert len(preds) == N

    def test_regressor_fit_predict(self):
        model = TabTransformerRegressor(
            model_config=TabTransformerConfig(n_layers=2, d_model=32, n_heads=4),
            trainer_config=self._fast,
        )
        model.fit(self._X_reg, y_reg)
        preds = model.predict(self._X_reg)
        assert len(preds) == N
        assert np.isfinite(preds).all()

    def test_get_params_set_params_clone_model(self):
        mc = TabTransformerConfig(n_layers=2)
        model = TabTransformerClassifier(model_config=mc, trainer_config=self._fast)
        params = model.get_params(deep=True)
        assert "model_config__n_layers" in params
        model.set_params(model_config__n_layers=3)
        assert model.model_config.n_layers == 3
        cloned = clone(model)
        assert cloned.model_config.n_layers == 3

    def test_flat_kwargs_raise_error(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            TabTransformerClassifier(n_layers=2, d_model=32)


class TestAutoIntWithConfig:
    _fast = TrainerConfig(max_epochs=1, batch_size=64, patience=1)

    def test_classifier_fit_predict(self):
        model = AutoIntClassifier(
            model_config=AutoIntConfig(n_layers=2, d_model=32, n_heads=4),
            trainer_config=self._fast,
        )
        model.fit(X_cls, y_cls)
        preds = model.predict(X_cls)
        assert len(preds) == N

    def test_regressor_fit_predict(self):
        model = AutoIntRegressor(
            model_config=AutoIntConfig(n_layers=2, d_model=32, n_heads=4),
            trainer_config=self._fast,
        )
        model.fit(X_reg, y_reg)
        preds = model.predict(X_reg)
        assert len(preds) == N
        assert np.isfinite(preds).all()

    def test_get_params_set_params_clone_model(self):
        mc = AutoIntConfig(n_layers=2)
        model = AutoIntClassifier(model_config=mc, trainer_config=self._fast)
        params = model.get_params(deep=True)
        assert "model_config__n_layers" in params
        model.set_params(model_config__n_layers=3)
        assert model.model_config.n_layers == 3
        cloned = clone(model)
        assert cloned.model_config.n_layers == 3

    def test_flat_kwargs_raise_error(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            AutoIntClassifier(n_layers=2, d_model=32)


class TestSAINTWithConfig:
    _fast = TrainerConfig(max_epochs=1, batch_size=64, patience=1)

    def test_classifier_fit_predict(self):
        model = SAINTClassifier(
            model_config=SAINTConfig(n_layers=1, d_model=32, n_heads=4),
            trainer_config=self._fast,
        )
        model.fit(X_cls, y_cls)
        preds = model.predict(X_cls)
        assert len(preds) == N

    def test_regressor_fit_predict(self):
        model = SAINTRegressor(
            model_config=SAINTConfig(n_layers=1, d_model=32, n_heads=4),
            trainer_config=self._fast,
        )
        model.fit(X_reg, y_reg)
        preds = model.predict(X_reg)
        assert len(preds) == N
        assert np.isfinite(preds).all()

    def test_get_params_set_params_clone_model(self):
        mc = SAINTConfig(n_layers=1)
        model = SAINTClassifier(model_config=mc, trainer_config=self._fast)
        params = model.get_params(deep=True)
        assert "model_config__n_layers" in params
        model.set_params(model_config__n_layers=2)
        assert model.model_config.n_layers == 2
        cloned = clone(model)
        assert cloned.model_config.n_layers == 2

    def test_flat_kwargs_raise_error(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            SAINTClassifier(n_layers=1, d_model=32)


class TestNODEWithConfig:
    _fast = TrainerConfig(max_epochs=1, batch_size=64, patience=1)

    def test_classifier_fit_predict(self):
        model = NODEClassifier(
            model_config=NODEConfig(num_layers=2, layer_dim=64),
            trainer_config=self._fast,
        )
        model.fit(X_cls, y_cls)
        preds = model.predict(X_cls)
        assert len(preds) == N

    def test_regressor_fit_predict(self):
        model = NODERegressor(
            model_config=NODEConfig(num_layers=2, layer_dim=64),
            trainer_config=self._fast,
        )
        model.fit(X_reg, y_reg)
        preds = model.predict(X_reg)
        assert len(preds) == N
        assert np.isfinite(preds).all()

    def test_get_params_set_params_clone_model(self):
        mc = NODEConfig(num_layers=2)
        model = NODEClassifier(model_config=mc, trainer_config=self._fast)
        params = model.get_params(deep=True)
        assert "model_config__num_layers" in params
        model.set_params(model_config__num_layers=3)
        assert model.model_config.num_layers == 3
        cloned = clone(model)
        assert cloned.model_config.num_layers == 3

    def test_flat_kwargs_raise_error(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            NODEClassifier(num_layers=2)


class TestNDTFWithConfig:
    _fast = TrainerConfig(max_epochs=1, batch_size=64, patience=1)

    def test_classifier_fit_predict(self):
        model = NDTFClassifier(
            model_config=NDTFConfig(n_ensembles=4),
            trainer_config=self._fast,
        )
        model.fit(X_cls, y_cls)
        preds = model.predict(X_cls)
        assert len(preds) == N

    def test_regressor_fit_predict(self):
        model = NDTFRegressor(
            model_config=NDTFConfig(n_ensembles=4),
            trainer_config=self._fast,
        )
        model.fit(X_reg, y_reg)
        preds = model.predict(X_reg)
        assert len(preds) == N
        assert np.isfinite(preds).all()

    def test_get_params_set_params_clone_model(self):
        mc = NDTFConfig(n_ensembles=4)
        model = NDTFClassifier(model_config=mc, trainer_config=self._fast)
        params = model.get_params(deep=True)
        assert "model_config__n_ensembles" in params
        model.set_params(model_config__n_ensembles=6)
        assert model.model_config.n_ensembles == 6
        cloned = clone(model)
        assert cloned.model_config.n_ensembles == 6

    def test_flat_kwargs_raise_error(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            NDTFClassifier(n_ensembles=4)


class TestTabMWithConfig:
    _fast = TrainerConfig(max_epochs=1, batch_size=64, patience=1)

    def test_classifier_fit_predict(self):
        model = TabMClassifier(
            model_config=TabMConfig(layer_sizes=[32, 16], ensemble_size=4),
            trainer_config=self._fast,
        )
        model.fit(X_cls, y_cls)
        preds = model.predict(X_cls)
        assert len(preds) == N

    def test_regressor_fit_predict(self):
        model = TabMRegressor(
            model_config=TabMConfig(layer_sizes=[32, 16], ensemble_size=4),
            trainer_config=self._fast,
        )
        model.fit(X_reg, y_reg)
        preds = model.predict(X_reg)
        assert len(preds) == N
        assert np.isfinite(preds).all()

    def test_get_params_set_params_clone_model(self):
        mc = TabMConfig(ensemble_size=8)
        model = TabMClassifier(model_config=mc, trainer_config=self._fast)
        params = model.get_params(deep=True)
        assert "model_config__ensemble_size" in params
        model.set_params(model_config__ensemble_size=4)
        assert model.model_config.ensemble_size == 4
        cloned = clone(model)
        assert cloned.model_config.ensemble_size == 4

    def test_flat_kwargs_raise_error(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            TabMClassifier(ensemble_size=8)


class TestTabRWithConfig:
    _fast = TrainerConfig(max_epochs=1, batch_size=64, patience=1)

    @pytest.mark.skip(
        reason="TabR uses FAISS nearest-neighbour lookups that segfault on small datasets (pre-existing issue; TabR is also skipped in test_models.py)"
    )
    def test_classifier_fit_predict(self):
        model = TabRClassifier(
            model_config=TabRConfig(d_main=64, context_size=32),
            trainer_config=self._fast,
        )
        model.fit(X_cls, y_cls)
        preds = model.predict(X_cls)
        assert len(preds) == N

    @pytest.mark.skip(
        reason="TabR uses FAISS nearest-neighbour lookups that segfault on small datasets (pre-existing issue; TabR is also skipped in test_models.py)"
    )
    def test_regressor_fit_predict(self):
        model = TabRRegressor(
            model_config=TabRConfig(d_main=64, context_size=32),
            trainer_config=self._fast,
        )
        model.fit(X_reg, y_reg)
        preds = model.predict(X_reg)
        assert len(preds) == N
        assert np.isfinite(preds).all()

    def test_get_params_set_params_clone_model(self):
        mc = TabRConfig(d_main=64)
        model = TabRClassifier(model_config=mc, trainer_config=self._fast)
        params = model.get_params(deep=True)
        assert "model_config__d_main" in params
        model.set_params(model_config__d_main=128)
        assert model.model_config.d_main == 128
        cloned = clone(model)
        assert cloned.model_config.d_main == 128

    def test_flat_kwargs_raise_error(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            TabRClassifier(d_main=64)


class TestMambularWithConfig:
    _fast = TrainerConfig(max_epochs=1, batch_size=64, patience=1)

    def test_classifier_fit_predict(self):
        model = MambularClassifier(
            model_config=MambularConfig(d_model=32, n_layers=2),
            trainer_config=self._fast,
        )
        model.fit(X_cls, y_cls)
        preds = model.predict(X_cls)
        assert len(preds) == N

    def test_regressor_fit_predict(self):
        model = MambularRegressor(
            model_config=MambularConfig(d_model=32, n_layers=2),
            trainer_config=self._fast,
        )
        model.fit(X_reg, y_reg)
        preds = model.predict(X_reg)
        assert len(preds) == N
        assert np.isfinite(preds).all()

    def test_get_params_set_params_clone_model(self):
        mc = MambularConfig(n_layers=2)
        model = MambularClassifier(model_config=mc, trainer_config=self._fast)
        params = model.get_params(deep=True)
        assert "model_config__n_layers" in params
        model.set_params(model_config__n_layers=3)
        assert model.model_config.n_layers == 3
        cloned = clone(model)
        assert cloned.model_config.n_layers == 3

    def test_flat_kwargs_raise_error(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            MambularClassifier(n_layers=2)


class TestMambaTabWithConfig:
    _fast = TrainerConfig(max_epochs=1, batch_size=64, patience=1)

    def test_classifier_fit_predict(self):
        model = MambaTabClassifier(
            model_config=MambaTabConfig(d_model=32, n_layers=1),
            trainer_config=self._fast,
        )
        model.fit(X_cls, y_cls)
        preds = model.predict(X_cls)
        assert len(preds) == N

    def test_regressor_fit_predict(self):
        model = MambaTabRegressor(
            model_config=MambaTabConfig(d_model=32, n_layers=1),
            trainer_config=self._fast,
        )
        model.fit(X_reg, y_reg)
        preds = model.predict(X_reg)
        assert len(preds) == N
        assert np.isfinite(preds).all()

    def test_get_params_set_params_clone_model(self):
        mc = MambaTabConfig(n_layers=1)
        model = MambaTabClassifier(model_config=mc, trainer_config=self._fast)
        params = model.get_params(deep=True)
        assert "model_config__n_layers" in params
        model.set_params(model_config__n_layers=2)
        assert model.model_config.n_layers == 2
        cloned = clone(model)
        assert cloned.model_config.n_layers == 2

    def test_flat_kwargs_raise_error(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            MambaTabClassifier(n_layers=1)


class TestMambAttentionWithConfig:
    _fast = TrainerConfig(max_epochs=1, batch_size=64, patience=1)

    def test_classifier_fit_predict(self):
        model = MambAttentionClassifier(
            model_config=MambAttentionConfig(d_model=32, n_layers=2, n_heads=4),
            trainer_config=self._fast,
        )
        model.fit(X_cls, y_cls)
        preds = model.predict(X_cls)
        assert len(preds) == N

    def test_regressor_fit_predict(self):
        model = MambAttentionRegressor(
            model_config=MambAttentionConfig(d_model=32, n_layers=2, n_heads=4),
            trainer_config=self._fast,
        )
        model.fit(X_reg, y_reg)
        preds = model.predict(X_reg)
        assert len(preds) == N
        assert np.isfinite(preds).all()

    def test_get_params_set_params_clone_model(self):
        mc = MambAttentionConfig(n_layers=2)
        model = MambAttentionClassifier(model_config=mc, trainer_config=self._fast)
        params = model.get_params(deep=True)
        assert "model_config__n_layers" in params
        model.set_params(model_config__n_layers=3)
        assert model.model_config.n_layers == 3
        cloned = clone(model)
        assert cloned.model_config.n_layers == 3

    def test_flat_kwargs_raise_error(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            MambAttentionClassifier(n_layers=2)


class TestTabulaRNNWithConfig:
    _fast = TrainerConfig(max_epochs=1, batch_size=64, patience=1)

    def test_classifier_fit_predict(self):
        model = TabulaRNNClassifier(
            model_config=TabulaRNNConfig(d_model=32, n_layers=2),
            trainer_config=self._fast,
        )
        model.fit(X_cls, y_cls)
        preds = model.predict(X_cls)
        assert len(preds) == N

    def test_regressor_fit_predict(self):
        model = TabulaRNNRegressor(
            model_config=TabulaRNNConfig(d_model=32, n_layers=2),
            trainer_config=self._fast,
        )
        model.fit(X_reg, y_reg)
        preds = model.predict(X_reg)
        assert len(preds) == N
        assert np.isfinite(preds).all()

    def test_get_params_set_params_clone_model(self):
        mc = TabulaRNNConfig(n_layers=2)
        model = TabulaRNNClassifier(model_config=mc, trainer_config=self._fast)
        params = model.get_params(deep=True)
        assert "model_config__n_layers" in params
        model.set_params(model_config__n_layers=3)
        assert model.model_config.n_layers == 3
        cloned = clone(model)
        assert cloned.model_config.n_layers == 3

    def test_flat_kwargs_raise_error(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            TabulaRNNClassifier(n_layers=2)


# ===========================================================================
# PR 5: Reject legacy flat keyword arguments in Classifier / Regressor
# ===========================================================================


class TestPR5FlatParamRejection:
    """Verify that Classifier/Regressor raise TypeError for flat kwargs (PR5)."""

    # ---- MLP ----

    def test_mlp_classifier_rejects_flat_model_arch_param(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            MLPClassifier(layer_sizes=[32, 16])

    def test_mlp_regressor_rejects_flat_model_arch_param(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            MLPRegressor(dropout=0.3)

    def test_mlp_classifier_rejects_flat_trainer_param(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            MLPClassifier(max_epochs=50)

    def test_mlp_classifier_rejects_flat_preprocessing_param(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            MLPClassifier(numerical_preprocessing="standard")

    def test_mlp_classifier_rejects_multiple_flat_params(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            MLPClassifier(layer_sizes=[32], lr=1e-4, n_bins=20)

    # ---- Error message content ----

    def test_error_message_contains_param_names(self):
        with pytest.raises(TypeError) as exc_info:
            MLPClassifier(layer_sizes=[32], dropout=0.3)
        msg = str(exc_info.value)
        assert "dropout" in msg
        assert "layer_sizes" in msg

    def test_error_message_contains_config_class_hint(self):
        with pytest.raises(TypeError) as exc_info:
            MLPClassifier(layer_sizes=[32])
        assert "MLPConfig" in str(exc_info.value)

    def test_error_message_contains_trainer_config_hint(self):
        with pytest.raises(TypeError) as exc_info:
            MLPClassifier(layer_sizes=[32])
        assert "TrainerConfig" in str(exc_info.value)

    # ---- Other models ----

    def test_resnet_classifier_rejects_flat_params(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            ResNetClassifier(num_blocks=2)

    def test_fttransformer_regressor_rejects_flat_params(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            FTTransformerRegressor(n_layers=2)

    def test_tabm_classifier_rejects_flat_params(self):
        with pytest.raises(TypeError, match="no longer accepts flat"):
            TabMClassifier(ensemble_size=8)

    # ---- Split-config API still works (no error) ----

    def test_classifier_no_args_does_not_raise(self):
        """cls() with no args must NOT raise — defaults are still valid."""
        model = MLPClassifier()
        assert model is not None

    def test_regressor_no_args_does_not_raise(self):
        model = MLPRegressor()
        assert model is not None

    def test_classifier_with_split_configs_does_not_raise(self):
        from deeptab.configs import MLPConfig

        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[32]),
            trainer_config=TrainerConfig(max_epochs=1),
        )
        assert model.model_config is not None

    def test_resnet_with_split_config_does_not_raise(self):
        model = ResNetClassifier(
            model_config=ResNetConfig(num_blocks=1),
            trainer_config=TrainerConfig(max_epochs=1),
        )
        assert model.model_config is not None
