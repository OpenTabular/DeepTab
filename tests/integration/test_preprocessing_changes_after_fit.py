"""Changing preprocessing settings on a fitted estimator prepares the *next* fit().

The fitted model must keep working: saving it, and continuing to train it with
rebuild=False, both need the preprocessor the data was actually fitted with.
"""

import numpy as np
import pytest

from deeptab.configs import MLPConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MLPLSS, MLPRegressor

FIT_KWARGS = {"accelerator": "cpu", "logger": False, "enable_progress_bar": False, "enable_model_summary": False}


def _fitted(model_class, tmp_path, **fit_options):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(80, 3))
    model = model_class(
        model_config=MLPConfig(layer_sizes=[16]),
        preprocessing_config=PreprocessingConfig(),
        trainer_config=TrainerConfig(max_epochs=1, checkpoint_path=str(tmp_path)),
        random_state=0,
    )
    model.fit(features, features[:, 0] * 2, **fit_options, **FIT_KWARGS)
    return model, features


@pytest.mark.parametrize("model_class, fit_options", [(MLPRegressor, {}), (MLPLSS, {"family": "normal"})])
def test_save_after_set_params_keeps_the_fitted_preprocessor(model_class, fit_options, tmp_path):
    model, features = _fitted(model_class, tmp_path, **fit_options)
    expected = model.predict(features)
    model.set_params(preprocessing_config__cat_cutoff=0.2)
    path = str(tmp_path / "model.deeptab")
    model.save(path)
    np.testing.assert_allclose(model_class.load(path).predict(features), expected, rtol=1e-6)


@pytest.mark.parametrize("model_class, fit_options", [(MLPRegressor, {}), (MLPLSS, {"family": "normal"})])
def test_continuation_after_set_params_uses_the_fitted_preprocessor(model_class, fit_options, tmp_path):
    model, features = _fitted(model_class, tmp_path, **fit_options)
    model.set_params(preprocessing_config__cat_cutoff=0.2)
    model.fit(features, features[:, 0] * 2, rebuild=False, **fit_options, **FIT_KWARGS)
    assert model.is_fitted_
    assert np.isfinite(model.predict(features)).all()
