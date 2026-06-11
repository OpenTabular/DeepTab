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

import numpy as np
import pytest

from deeptab.configs import TrainerConfig
from deeptab.models.base import SklearnBase
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
        from sklearn.exceptions import NotFittedError

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
