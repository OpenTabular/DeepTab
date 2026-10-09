"""Tests for configured estimators."""

# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

from deeptab.configs import (
    AutoIntConfig,
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
from deeptab.models.mlp import MLPLSS, MLPClassifier, MLPRegressor
from deeptab.models.ndtf import NDTFClassifier, NDTFRegressor
from deeptab.models.node import NODEClassifier, NODERegressor
from deeptab.models.resnet import ResNetClassifier, ResNetRegressor
from deeptab.models.saint import SAINTClassifier, SAINTRegressor
from deeptab.models.tabm import TabMClassifier, TabMRegressor
from deeptab.models.tabr import TabRClassifier, TabRRegressor
from deeptab.models.tabtransformer import TabTransformerClassifier, TabTransformerRegressor
from deeptab.models.tabularnn import TabulaRNNClassifier, TabulaRNNRegressor

N = 120

RNG = np.random.default_rng(0)

X_cls = pd.DataFrame(RNG.standard_normal((N, 6)), columns=[f"f{i}" for i in range(6)])

y_cls = RNG.integers(0, 3, size=N)

X_reg = pd.DataFrame(RNG.standard_normal((N, 6)), columns=[f"f{i}" for i in range(6)])

y_reg = RNG.standard_normal(N)

_FAST_TRAINER = TrainerConfig(max_epochs=1, batch_size=64, patience=1)


@pytest.mark.parametrize("voting", ["hard", "soft"])
def test_classifier_works_in_voting_classifier(voting, tmp_path):
    class CPUClassifier(MLPClassifier):
        def fit(self, X, y, **kwargs):
            return super().fit(X, y, accelerator="cpu", logger=False, enable_progress_bar=False, **kwargs)

    classifier = CPUClassifier(
        model_config=MLPConfig(layer_sizes=[16]),
        trainer_config=TrainerConfig(max_epochs=1, checkpoint_path=str(tmp_path / "checkpoints")),
        random_state=42,
    )
    ensemble = VotingClassifier(estimators=[("deeptab", classifier), ("linear", LogisticRegression())], voting=voting)
    ensemble.fit(X_cls, y_cls)
    assert ensemble.predict(X_cls).shape == (N,)
    if voting == "soft":
        assert ensemble.predict_proba(X_cls).shape == (N, 3)


@pytest.mark.parametrize("model_cls", [MLPClassifier, MLPRegressor, MLPLSS])
@pytest.mark.parametrize("sample_count", [50, 100])
def test_fit_with_drop_last_keeps_validation_and_prediction_rows(model_cls, sample_count, tmp_path):
    X, y = (X_cls, y_cls) if model_cls is MLPClassifier else (X_reg, y_reg)
    X, y = X.iloc[:sample_count], y[:sample_count]
    model = model_cls(
        model_config=MLPConfig(layer_sizes=[16]),
        trainer_config=TrainerConfig(max_epochs=1, batch_size=16, checkpoint_path=str(tmp_path)),
        random_state=42,
    )
    family_kwargs = {"family": "normal"} if model_cls is MLPLSS else {}
    model.fit(
        X,
        y,
        dataloader_kwargs={"drop_last": True},
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        **family_kwargs,
    )
    assert model._data_module.train_dataloader().drop_last is True
    assert model._data_module.val_dataloader().drop_last is False
    assert len(model._data_module.val_dataloader()) == (sample_count // 5 + 15) // 16
    assert "val_loss" in model._trainer.callback_metrics
    assert len(model.predict(X)) == sample_count


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
        assert model._trainer.max_epochs == 1

    def test_random_state_is_honoured(self):
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[16]),
            trainer_config=TrainerConfig(max_epochs=1, batch_size=64, patience=1),
            random_state=42,
        )
        model.fit(X_cls, y_cls)
        assert model.random_state == 42


class TestFitArgumentsTakePrecedenceOverTrainerConfig:
    """Regression tests for GH-442: explicit fit() args must win over TrainerConfig."""

    def test_fit_kwargs_override_trainer_config_on_classifier(self):
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[16]),
            trainer_config=TrainerConfig(max_epochs=1, batch_size=64, val_size=0.2, patience=1),
        )
        model.fit(X_cls, y_cls, max_epochs=2, batch_size=8, val_size=0.4, patience=1)
        assert model._trainer.max_epochs == 2
        assert model._data_module.batch_size == 8
        assert model._data_module.val_size == 0.4

    def test_fit_kwargs_override_trainer_config_on_regressor(self):
        model = MLPRegressor(
            model_config=MLPConfig(layer_sizes=[16]),
            trainer_config=TrainerConfig(max_epochs=1, batch_size=64, val_size=0.2, patience=1),
        )
        model.fit(X_reg, y_reg, max_epochs=2, batch_size=8, val_size=0.4, patience=1)
        assert model._trainer.max_epochs == 2
        assert model._data_module.batch_size == 8
        assert model._data_module.val_size == 0.4

    def test_model_config_only_does_not_force_trainer_defaults(self):
        """Passing only ``model_config`` must not silently create a TrainerConfig
        that then clobbers explicit fit() arguments."""
        model = MLPRegressor(model_config=MLPConfig(layer_sizes=[16]))
        assert model.trainer_config is not None  # a default TrainerConfig is created internally
        model.fit(X_reg, y_reg, max_epochs=2, batch_size=8, val_size=0.4, patience=1)
        assert model._trainer.max_epochs == 2
        assert model._data_module.batch_size == 8
        assert model._data_module.val_size == 0.4

    def test_omitted_fit_kwargs_still_fall_back_to_trainer_config(self):
        """When a fit() argument is left unset, the TrainerConfig value must still apply."""
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[16]),
            trainer_config=TrainerConfig(max_epochs=1, batch_size=32, patience=1),
        )
        model.fit(X_cls, y_cls, batch_size=8)
        assert model._trainer.max_epochs == 1  # from TrainerConfig, unset in fit()
        assert model._data_module.batch_size == 8  # explicit fit() arg wins


class TestFeaturePreprocessingMapping:
    """Regression test: a per-column dict must reach the preprocessor."""

    def test_dict_mapping_fits_end_to_end(self):
        model = MLPRegressor(
            model_config=MLPConfig(layer_sizes=[16]),
            preprocessing_config=PreprocessingConfig(feature_preprocessing={"f0": "quantile"}),
            trainer_config=TrainerConfig(max_epochs=1, batch_size=64, patience=1),
        )
        model.fit(X_reg, y_reg)
        preds = model.predict(X_reg)
        assert len(preds) == N


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

    def test_flat_constructor_kwargs_raise_error(self):
        """Flat constructor kwargs raise TypeError."""
        with pytest.raises(TypeError):
            MLPClassifier(layer_sizes=[32, 16])  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            MLPRegressor(layer_sizes=[32, 16])  # type: ignore[call-arg]


_TRAINING_FIELDS = {"lr", "lr_patience", "lr_factor", "weight_decay"}

_PREPROCESSING_FIELDS = {
    "numerical_preprocessing",
    "categorical_preprocessing",
    "n_bins",
    "scaling_strategy",
}


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
        with pytest.raises(TypeError):
            ResNetClassifier(num_blocks=2, layer_sizes=[32])  # type: ignore[call-arg]


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
        with pytest.raises(TypeError):
            FTTransformerClassifier(n_layers=2, d_model=32)  # type: ignore[call-arg]


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
        with pytest.raises(TypeError):
            TabTransformerClassifier(n_layers=2, d_model=32)  # type: ignore[call-arg]


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
        with pytest.raises(TypeError):
            AutoIntClassifier(n_layers=2, d_model=32)  # type: ignore[call-arg]


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
        with pytest.raises(TypeError):
            SAINTClassifier(n_layers=1, d_model=32)  # type: ignore[call-arg]


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
        with pytest.raises(TypeError):
            NODEClassifier(num_layers=2)  # type: ignore[call-arg]


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
        with pytest.raises(TypeError):
            NDTFClassifier(n_ensembles=4)  # type: ignore[call-arg]


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
        with pytest.raises(TypeError):
            TabMClassifier(ensemble_size=8)  # type: ignore[call-arg]


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
        with pytest.raises(TypeError):
            TabRClassifier(d_main=64)  # type: ignore[call-arg]


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
        with pytest.raises(TypeError):
            MambularClassifier(n_layers=2)  # type: ignore[call-arg]


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
        with pytest.raises(TypeError):
            MambaTabClassifier(n_layers=1)  # type: ignore[call-arg]


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
        with pytest.raises(TypeError):
            MambAttentionClassifier(n_layers=2)  # type: ignore[call-arg]


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
        with pytest.raises(TypeError):
            TabulaRNNClassifier(n_layers=2)  # type: ignore[call-arg]
