"""Tests for SklearnBaseLSS after Phase 5 (Option B) refactoring.

Verifies:
1. Inheritance — SklearnBaseLSS is a proper subclass of SklearnBase.
2. fit() / predict() end-to-end with a fast trainer config.
3. save() / load() round-trip preserves family, weights, and predictions.
4. get_params() / set_params() work correctly (inherited from SklearnBase).
5. LSS-specific methods (evaluate, score, get_default_metrics) are present.
6. optimize_hparams() correctly delegates regression=False to _HyperparameterMixin.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deeptab.configs import TrainerConfig
from deeptab.models.base import SklearnBase
from deeptab.models.fttransformer import FTTransformerClassifier, FTTransformerLSS, FTTransformerRegressor
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.mlp import MLPLSS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FAST_TRAINER = TrainerConfig(max_epochs=2, patience=2, lr_patience=2)

# Small regression dataset with strictly-positive targets (works for 'normal').
_RNG = np.random.default_rng(42)
_N = 80
_X = _RNG.standard_normal((_N, 8)).astype(np.float32)
_Y = _RNG.standard_normal(_N).astype(np.float32)  # normal family — unbounded


@pytest.fixture()
def fitted_mlplss():
    """Return a fitted MLPLSS instance using a minimal fast config."""
    model = MLPLSS(trainer_config=_FAST_TRAINER)
    model.fit(_X, _Y, family="normal")
    return model


@pytest.mark.parametrize("model_class", [FTTransformerClassifier, FTTransformerRegressor, FTTransformerLSS])
def test_encode_uses_eval_without_gradients_and_preserves_batches(model_class, monkeypatch):
    class EncodingBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_layer = torch.nn.Identity()
            self.dropout = torch.nn.Dropout(0.75)
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.batch_sizes = []

        def encode(self, data):
            assert not self.training
            assert not torch.is_grad_enabled()
            numerical, categorical, embeddings = data
            self.batch_sizes.append(len(numerical))
            return self.dropout(torch.stack([numerical, categorical, embeddings], dim=1) * self.weight)

    features = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    dataset = [(row, row + 100, row + 200) for row in features]
    model = model_class()
    task_model = torch.nn.Module()
    task_model.estimator = EncodingBackbone()
    monkeypatch.setattr(model, "_task_model", task_model)
    monkeypatch.setattr(model, "_data_module", SimpleNamespace(preprocess_new_data=lambda *args: dataset))

    first = model.encode(features.numpy(), batch_size=4)
    second = model.encode(features.numpy(), batch_size=4)

    assert not task_model.training
    assert task_model.estimator.batch_sizes == [4, 4, 2, 4, 4, 2]
    assert first.shape == (10, 3, 4)
    assert not first.requires_grad
    assert first.grad_fn is None
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, torch.stack([features, features + 100, features + 200], dim=1))


# ---------------------------------------------------------------------------
# 1. Inheritance
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_is_subclass_of_sklearn_base(self):
        assert issubclass(SklearnBaseLSS, SklearnBase)

    def test_mlplss_is_subclass_of_sklearn_base_lss(self):
        assert issubclass(MLPLSS, SklearnBaseLSS)

    def test_mro_contains_all_mixins(self):
        mro_names = [c.__name__ for c in SklearnBaseLSS.__mro__]
        for mixin in (
            "SklearnBase",
            "_ObservabilityMixin",
            "_FitMixin",
            "_PredictMixin",
            "_SerializationMixin",
            "_HyperparameterMixin",
            "InspectionMixin",
            "BaseEstimator",
        ):
            assert mixin in mro_names, f"{mixin} not in MRO: {mro_names}"

    def test_no_duplicate_init(self):
        """__init__ should be defined only on SklearnBase, not on SklearnBaseLSS."""
        assert "__init__" not in SklearnBaseLSS.__dict__, (
            "SklearnBaseLSS should not define __init__ after Phase 5 — it inherits SklearnBase.__init__"
        )

    def test_no_duplicate_get_params(self):
        assert "get_params" not in SklearnBaseLSS.__dict__

    def test_no_duplicate_set_params(self):
        assert "set_params" not in SklearnBaseLSS.__dict__

    def test_no_duplicate_get_number_of_params(self):
        assert "get_number_of_params" not in SklearnBaseLSS.__dict__


# ---------------------------------------------------------------------------
# 2. fit() / predict()
# ---------------------------------------------------------------------------


class TestFitPredict:
    def test_fit_returns_self(self):
        model = MLPLSS(trainer_config=_FAST_TRAINER)
        result = model.fit(_X, _Y, family="normal")
        assert result is model

    def test_predict_shape(self, fitted_mlplss):
        preds = fitted_mlplss.predict(_X)
        # normal distribution has 2 parameters (mean + variance), so shape is (N, 2)
        assert preds.shape[0] == _N

    def test_predict_no_nan(self, fitted_mlplss):
        preds = fitted_mlplss.predict(_X)
        assert not np.isnan(preds).any()

    def test_family_stored_after_fit(self, fitted_mlplss):
        assert fitted_mlplss.family_name == "normal"
        assert fitted_mlplss.family is not None

    def test_is_fitted_after_fit(self, fitted_mlplss):
        assert fitted_mlplss.__sklearn_is_fitted__()

    def test_predict_raises_before_fit(self):
        from deeptab.core.exceptions import NotFittedError

        model = MLPLSS(trainer_config=_FAST_TRAINER)
        with pytest.raises(NotFittedError):
            model.predict(_X)

    def test_fit_validates_family_range_for_gamma(self):
        """Gamma family requires strictly positive y; should raise on non-positive values."""
        from deeptab.core.exceptions import DataError

        model = MLPLSS(trainer_config=_FAST_TRAINER)
        y_bad = _Y.copy()
        y_bad[0] = -1.0
        with pytest.raises(DataError):
            model.fit(_X, y_bad, family="gamma")


# ---------------------------------------------------------------------------
# 3. save() / load() round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_creates_file(self, fitted_mlplss, tmp_path):
        path = str(tmp_path / "model.deeptab")
        fitted_mlplss.save(path)
        assert Path(path).exists()

    def test_load_returns_same_type(self, fitted_mlplss, tmp_path):
        path = str(tmp_path / "model.deeptab")
        fitted_mlplss.save(path)
        loaded = MLPLSS.load(path)
        assert type(loaded) is type(fitted_mlplss)

    def test_load_restores_family(self, fitted_mlplss, tmp_path):
        path = str(tmp_path / "model.deeptab")
        fitted_mlplss.save(path)
        loaded = MLPLSS.load(path)
        assert loaded.family_name == "normal"
        assert loaded.family is not None

    def test_load_predictions_match(self, fitted_mlplss, tmp_path):
        path = str(tmp_path / "model.deeptab")
        preds_before = fitted_mlplss.predict(_X)
        fitted_mlplss.save(path)
        loaded = MLPLSS.load(path)
        preds_after = loaded.predict(_X)
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-4)

    def test_load_restores_metadata_attributes(self, fitted_mlplss, tmp_path):
        path = str(tmp_path / "model.deeptab")
        fitted_mlplss.save(path)
        loaded = MLPLSS.load(path)
        assert hasattr(loaded, "input_columns_")
        assert hasattr(loaded, "versions_")


# ---------------------------------------------------------------------------
# 4. get_params / set_params (inherited from SklearnBase)
# ---------------------------------------------------------------------------


class TestParamInheritance:
    def test_get_params_returns_dict(self):
        model = MLPLSS(trainer_config=_FAST_TRAINER)
        params = model.get_params()
        assert isinstance(params, dict)

    def test_get_params_includes_trainer_config(self):
        model = MLPLSS(trainer_config=_FAST_TRAINER)
        params = model.get_params()
        assert "trainer_config" in params

    def test_set_params_returns_self(self):
        model = MLPLSS(trainer_config=_FAST_TRAINER)
        result = model.set_params(trainer_config=_FAST_TRAINER)
        assert result is model

    def test_get_params_round_trips_through_set_params(self):
        model = MLPLSS(trainer_config=_FAST_TRAINER)
        params = model.get_params(deep=False)
        cloned = MLPLSS(trainer_config=_FAST_TRAINER)
        cloned.set_params(**params)
        assert cloned.get_params(deep=False).keys() == params.keys()


# ---------------------------------------------------------------------------
# 5. LSS-specific methods
# ---------------------------------------------------------------------------


class TestLSSSpecificMethods:
    def test_evaluate_returns_dict(self, fitted_mlplss):
        scores = fitted_mlplss.evaluate(_X, _Y, distribution_family="normal")
        assert isinstance(scores, dict)
        assert len(scores) > 0

    def test_score_returns_value(self, fitted_mlplss):
        # score() delegates to task_model.family.evaluate_nll which returns a dict of metrics
        s = fitted_mlplss.score(_X, _Y)
        assert s is not None

    def test_get_default_metrics_returns_dict(self, fitted_mlplss):
        metrics = fitted_mlplss.get_default_metrics("normal")
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_get_number_of_params_inherited(self, fitted_mlplss):
        """get_number_of_params is inherited from _FitMixin, not defined on SklearnBaseLSS."""
        n = fitted_mlplss.get_number_of_params()
        assert isinstance(n, int)
        assert n > 0

    def test_encode_raises_for_model_without_embedding_layer(self, fitted_mlplss):
        """MLP does not have an embedding layer; encode should raise."""
        with pytest.raises(AttributeError):
            fitted_mlplss.encode(_X[:8])


def test_score_is_scalar_negative_nll_on_raw_parameters(fitted_mlplss):
    raw = fitted_mlplss.predict(_X, raw=True)
    expected = -float(fitted_mlplss.family.compute_loss(torch.tensor(raw), torch.tensor(_Y)))
    actual = fitted_mlplss.score(_X, _Y)
    assert isinstance(actual, float)
    assert actual == pytest.approx(expected)


def test_score_rejects_unknown_metric(fitted_mlplss):
    with pytest.raises(ValueError, match="Unsupported score metric"):
        fitted_mlplss.score(_X, _Y, metric="RMSE")


def test_score_single_column_targets_do_not_broadcast(fitted_mlplss):
    assert fitted_mlplss.score(_X[:8], _Y[:8, None]) == pytest.approx(fitted_mlplss.score(_X[:8], _Y[:8]))


def test_evaluate_uses_fitted_gamma_family(monkeypatch, fitted_mlplss):
    from deeptab.distributions import get_distribution

    model = fitted_mlplss
    model.family_name = "gamma"
    model.family = get_distribution("gamma")
    predictions = np.array([[2.0, 1.0], [6.0, 2.0]])
    monkeypatch.setattr(model, "predict", lambda *args, **kwargs: predictions)
    scores = model.evaluate(_X[:2], np.array([2.0, 3.0]))
    assert "gamma_deviance" in scores
    assert "crps" not in scores
    assert all(np.isfinite(value) for value in scores.values())


def test_default_quantile_metric_uses_fitted_quantile_configuration(fitted_mlplss):
    from deeptab.distributions import get_distribution

    model = fitted_mlplss
    model.family_name = "quantile"
    model.family = get_distribution("quantile", quantiles=[0.5, 0.9])
    metric = next(iter(model.get_default_metrics("quantile").values()))
    assert metric.col == 0
    assert metric.quantile == 0.5


def _small_lss(seed=42):
    from deeptab.configs import MLPConfig, PreprocessingConfig

    return MLPLSS(
        model_config=MLPConfig(d_model=16, dropout=0.0),
        preprocessing_config=PreprocessingConfig(numerical_method="standardization", output_dim=7),
        trainer_config=TrainerConfig(max_epochs=2, patience=1, lr_patience=1, batch_size=16),
        random_state=seed,
    )


def test_standalone_build_selects_family_and_can_fit_without_rebuilding(tmp_path):
    model = _small_lss()
    assert (
        model.build_model(
            _X[:24],
            _Y[:24],
            X_val=_X[24:32],
            y_val=_Y[24:32],
            family="quantile",
            distributional_kwargs={"quantiles": [0.1, 0.9]},
        )
        is model
    )
    assert model.family.param_count == 2
    task = model._task_model
    model.fit(
        _X[:24],
        _Y[:24],
        family="quantile",
        rebuild=False,
        max_epochs=1,
        accelerator="cpu",
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )
    assert model._task_model is task
    assert model.family.quantiles == [0.1, 0.9]
    assert model.predict(_X[:4]).shape == (4, 2)


def test_standalone_build_defaults_to_normal():
    model = _small_lss()
    model.build_model(_X[:24], _Y[:24], X_val=_X[24:32], y_val=_Y[24:32])
    assert model.family_name == "normal"
    assert model.family.param_count == 2


def test_lss_fit_seed_controls_initialization_and_training(tmp_path):
    predictions = []
    for global_seed in [1, 999]:
        torch.manual_seed(global_seed)
        model = _small_lss(seed=17)
        model.fit(
            _X[:24],
            _Y[:24],
            family="normal",
            X_val=_X[24:32],
            y_val=_Y[24:32],
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            default_root_dir=str(tmp_path),
        )
        predictions.append(model.predict(_X[:4], raw=True))
    np.testing.assert_allclose(predictions[0], predictions[1], rtol=0, atol=0)


@pytest.mark.parametrize("family", ["categorical", "dirichlet"])
def test_multivariate_families_build_and_fit_three_outputs(tmp_path, family):
    model = _small_lss()
    targets = np.tile(np.array([10, 20, 30]), 8) if family == "categorical" else np.tile([0.2, 0.3, 0.5], (24, 1))
    model.fit(
        _X[:24],
        targets,
        family=family,
        X_val=_X[24:30],
        y_val=targets[:6],
        max_epochs=1,
        accelerator="cpu",
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )
    predictions = model.predict(_X[:6])
    assert predictions.shape == (6, 3)
    assert np.isfinite(model.score(_X[:6], targets[:6]))
    if family == "categorical":
        np.testing.assert_allclose(predictions.sum(axis=1), 1, atol=1e-6)
        assert set(model.evaluate(_X[:6], targets[:6])) == {"accuracy", "log_loss"}
    else:
        assert (predictions > 0).all()
    path = str(tmp_path / f"{family}.deeptab")
    model.save(path)
    loaded = MLPLSS.load(path)
    assert loaded.family.param_count == 3
    np.testing.assert_allclose(loaded.predict(_X[:6]), predictions, atol=1e-6)
    assert loaded.score(_X[:6], targets[:6]) == pytest.approx(model.score(_X[:6], targets[:6]))


@pytest.mark.parametrize("model_name", ["TabM", "TabM-averaged", "Trompt"])
@pytest.mark.parametrize(
    "family,options",
    [
        ("poisson", {}),
        ("tweedie", {}),
        ("quantile", {"quantiles": [0.5]}),
        ("normal", {}),
        ("categorical", {}),
        ("dirichlet", {}),
    ],
)
def test_ensemble_lss_families_fit_predict_and_roundtrip(model_name, family, options, tmp_path):
    from deeptab.configs import PreprocessingConfig, TabMConfig
    from deeptab.configs.experimental.trompt_config import TromptConfig
    from deeptab.models import TabMLSS
    from deeptab.models.experimental import TromptLSS

    model_class = TabMLSS if model_name.startswith("TabM") else TromptLSS
    config = (
        TabMConfig(
            d_model=8,
            layer_sizes=[8, 4],
            ensemble_size=3,
            dropout=0.0,
            average_ensembles=model_name == "TabM-averaged",
        )
        if model_class is TabMLSS
        else TromptConfig(d_model=8, n_cycles=3, P=4)
    )
    model = model_class(
        model_config=config,
        preprocessing_config=PreprocessingConfig(numerical_method="standardization", output_dim=7),
        trainer_config=TrainerConfig(max_epochs=1, batch_size=8, checkpoint_path=str(tmp_path)),
        random_state=42,
    )
    targets = (
        np.tile([0.2, 0.3, 0.5], (32, 1))
        if family == "dirichlet"
        else np.tile([10, 20, 30, 10], 8)
        if family == "categorical"
        else np.arange(32) % 4
    )
    model.fit(
        _X[:24],
        targets[:24],
        family=family,
        distributional_kwargs=options,
        X_val=_X[24:32],
        y_val=targets[24:32],
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
    )
    raw = model.predict(_X[:5], raw=True)
    predictions = model.predict(_X[:5])
    assert raw.shape == predictions.shape == (5, model.family.param_count)
    assert model.predict(_X[:1]).shape == (1, model.family.param_count)
    assert np.isfinite(raw).all()
    assert np.isfinite(predictions).all()
    assert np.isfinite(model.score(_X[:5], targets[:5]))
    if family == "categorical":
        np.testing.assert_allclose(predictions.sum(axis=1), 1, atol=1e-6)
    elif family in {"poisson", "tweedie", "dirichlet"}:
        assert (predictions > 0).all()

    path = str(tmp_path / "ensemble.deeptab")
    model.save(path)
    loaded = model_class.load(path)
    assert loaded.family.param_count == model.family.param_count
    np.testing.assert_array_equal(loaded.predict(_X[:5], raw=True), raw)
    np.testing.assert_array_equal(loaded.predict(_X[:5]), predictions)


@pytest.mark.parametrize(
    "family,options,attribute",
    [
        ("quantile", {"quantiles": [0.1, 0.5, 0.9, 0.95]}, "quantiles"),
        ("tweedie", {"p": 1.7}, "p"),
        ("mog", {"n_components": 3}, "n_components"),
    ],
)
def test_custom_family_configuration_survives_artifact_roundtrip(tmp_path, family, options, attribute):
    model = _small_lss()
    targets = np.abs(_Y) + 0.5 if family == "tweedie" else _Y
    model.fit(
        _X[:24],
        targets[:24],
        family=family,
        distributional_kwargs=options,
        X_val=_X[24:32],
        y_val=targets[24:32],
        max_epochs=1,
        accelerator="cpu",
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )
    before = model.predict(_X[:4])
    score = model.score(_X[:4], targets[:4])
    path = str(tmp_path / f"{family}.deeptab")
    model.save(path)
    loaded = MLPLSS.load(path)
    assert getattr(loaded.family, attribute) == getattr(model.family, attribute)
    assert loaded.distributional_kwargs_ == options
    assert loaded.task_info_["distributional_kwargs"] == options
    np.testing.assert_allclose(loaded.predict(_X[:4]), before, atol=1e-6)
    assert loaded.score(_X[:4], targets[:4]) == pytest.approx(score)


def test_reused_build_rejects_changed_options_without_mutating_family():
    model = _small_lss()
    model.build_model(_X[:24], _Y[:24], family="normal")
    family = model.family
    with pytest.raises(ValueError, match="requires rebuild=True"):
        model.fit(_X[:24], _Y[:24], family="quantile", rebuild=False)
    assert model.family is family
    assert model.family_name == "normal"


def test_legacy_lss_bundle_loads_without_distributional_kwargs(tmp_path, fitted_mlplss):
    path = str(tmp_path / "legacy.deeptab")
    fitted_mlplss.save(path)
    bundle = torch.load(path, weights_only=False)
    del bundle["distributional_kwargs"]
    bundle["task_info"].pop("distributional_kwargs", None)
    torch.save(bundle, path)
    loaded = MLPLSS.load(path)
    assert loaded.distributional_kwargs_ == {}
    np.testing.assert_allclose(loaded.predict(_X[:4]), fitted_mlplss.predict(_X[:4]), atol=1e-6)
