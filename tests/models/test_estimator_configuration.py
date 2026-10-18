"""Tests for estimator configuration."""

# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import StratifiedKFold, check_cv
from sklearn.utils import get_tags

from deeptab.configs import MLPConfig, PreprocessingConfig, ResNetConfig, TrainerConfig
from deeptab.models.fttransformer import FTTransformerRegressor
from deeptab.models.mlp import MLPClassifier, MLPRegressor
from deeptab.models.resnet import ResNetClassifier
from deeptab.models.tabm import TabMClassifier

N = 120

RNG = np.random.default_rng(0)

X_cls = pd.DataFrame(RNG.standard_normal((N, 6)), columns=[f"f{i}" for i in range(6)])

y_cls = RNG.integers(0, 3, size=N)

X_reg = pd.DataFrame(RNG.standard_normal((N, 6)), columns=[f"f{i}" for i in range(6)])

y_reg = RNG.standard_normal(N)

_FAST_TRAINER = TrainerConfig(max_epochs=1, batch_size=64, patience=1)


@pytest.mark.parametrize("model_cls", [MLPClassifier, ResNetClassifier, TabMClassifier])
def test_classifier_task_tags_and_default_cv(model_cls):
    model = model_cls()
    assert is_classifier(model)
    assert not is_regressor(model)
    tags = get_tags(model)
    assert tags.estimator_type == "classifier"
    assert tags.classifier_tags is not None
    assert tags.target_tags.required
    assert isinstance(check_cv(3, y_cls, classifier=is_classifier(model)), StratifiedKFold)


@pytest.mark.parametrize("model_cls", [MLPRegressor, FTTransformerRegressor])
def test_regressor_task_tags(model_cls):
    model = model_cls()
    assert is_regressor(model)
    assert not is_classifier(model)
    tags = get_tags(model)
    assert tags.estimator_type == "regressor"
    assert tags.regressor_tags is not None
    assert tags.target_tags.required


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
        """Flat constructor kwargs raise TypeError."""
        with pytest.raises(TypeError):
            MLPClassifier(layer_sizes=[32, 16])  # type: ignore[call-arg]


class TestEstimatorGetParams:
    def test_get_params_returns_config_objects(self):
        mc = MLPConfig(layer_sizes=[32, 16])
        tc = TrainerConfig(max_epochs=1)
        pc = PreprocessingConfig(numerical_preprocessing="standardization")
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
        """Flat constructor kwargs are not accepted."""
        with pytest.raises(TypeError):
            MLPClassifier(layer_sizes=[32, 16])  # type: ignore[call-arg]


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


class TestDefaultEstimatorSetParams:
    """No configs passed at construction: get_params/set_params/clone must
    reflect the estimator's real state and reject unknown keys."""

    def test_set_params_updates_the_actual_config(self):
        """set_params() must change what fit() would actually build, not just
        an inert bookkeeping dict."""
        model = MLPClassifier(random_state=42)
        model.set_params(layer_sizes=[8])
        assert model.config.layer_sizes == [8]

    def test_get_params_reports_overrides(self):
        model = MLPClassifier()
        model.set_params(layer_sizes=[8])
        assert model.get_params()["layer_sizes"] == [8]

    def test_set_params_updates_random_state(self):
        model = MLPClassifier(random_state=42)
        model.set_params(random_state=99)
        assert model.random_state == 99

    def test_set_params_invalid_key_raises(self):
        model = MLPClassifier()
        with pytest.raises(ValueError, match="abc"):
            model.set_params(abc=123)

    def test_clone_preserves_flat_overrides(self):
        model = MLPClassifier(random_state=42)
        model.set_params(layer_sizes=[8])
        cloned = clone(model)
        assert cloned.random_state == 42
        assert cloned.config.layer_sizes == [8]

    def test_gridsearchcv_style_set_params_changes_architecture(self):
        """Simulates what GridSearchCV does per candidate: clone the base
        estimator, then set_params() the candidate's hyperparameters."""
        base = MLPClassifier(random_state=0)

        small = clone(base)
        small.set_params(layer_sizes=[8])
        big = clone(base)
        big.set_params(layer_sizes=[256, 128, 32])

        assert small.config.layer_sizes != big.config.layer_sizes


class TestSplitConfigSetParams:
    def test_set_params_invalid_top_level_key_raises(self):
        model = MLPClassifier(model_config=MLPConfig(), trainer_config=_FAST_TRAINER)
        with pytest.raises(ValueError, match="abc"):
            model.set_params(abc=123)


_TRAINING_FIELDS = {"lr", "lr_patience", "lr_factor", "weight_decay"}

_PREPROCESSING_FIELDS = {
    "numerical_preprocessing",
    "categorical_preprocessing",
    "n_bins",
    "scaling_strategy",
}


class TestFlatConstructorParamRejection:
    """Classifiers and regressors reject flat constructor kwargs."""

    # ---- MLP ----

    def test_mlp_classifier_rejects_flat_model_arch_param(self):
        with pytest.raises(TypeError):
            MLPClassifier(layer_sizes=[32, 16])  # type: ignore[call-arg]

    def test_mlp_regressor_rejects_flat_model_arch_param(self):
        with pytest.raises(TypeError):
            MLPRegressor(dropout=0.3)  # type: ignore[call-arg]

    def test_mlp_classifier_rejects_flat_trainer_param(self):
        with pytest.raises(TypeError):
            MLPClassifier(max_epochs=50)  # type: ignore[call-arg]

    def test_mlp_classifier_rejects_flat_preprocessing_param(self):
        with pytest.raises(TypeError):
            MLPClassifier(numerical_preprocessing="standard")  # type: ignore[call-arg]

    def test_mlp_classifier_rejects_multiple_flat_params(self):
        with pytest.raises(TypeError):
            MLPClassifier(layer_sizes=[32], lr=1e-4, n_bins=20)  # type: ignore[call-arg]

    # ---- Error message content ----

    def test_error_message_contains_param_names(self):
        with pytest.raises(TypeError) as exc_info:
            MLPClassifier(layer_sizes=[32])  # type: ignore[call-arg]
        assert "layer_sizes" in str(exc_info.value)

    def test_error_message_contains_config_class_hint(self):
        with pytest.raises(TypeError) as exc_info:
            MLPClassifier(layer_sizes=[32])  # type: ignore[call-arg]
        assert "unexpected keyword argument" in str(exc_info.value)

    def test_error_message_contains_trainer_config_hint(self):
        with pytest.raises(TypeError) as exc_info:
            MLPClassifier(layer_sizes=[32])  # type: ignore[call-arg]
        assert "unexpected keyword argument" in str(exc_info.value)

    # ---- Other models ----

    def test_resnet_classifier_rejects_flat_params(self):
        with pytest.raises(TypeError):
            ResNetClassifier(num_blocks=2)  # type: ignore[call-arg]

    def test_fttransformer_regressor_rejects_flat_params(self):
        with pytest.raises(TypeError):
            FTTransformerRegressor(n_layers=2)  # type: ignore[call-arg]

    def test_tabm_classifier_rejects_flat_params(self):
        with pytest.raises(TypeError):
            TabMClassifier(ensemble_size=8)  # type: ignore[call-arg]

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
