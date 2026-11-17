"""Contract of the public ``pretrain`` method of classifiers and regressors."""

import numpy as np
import pytest

import deeptab.models._mixins.fit as fit_module
from deeptab.configs import MambularConfig, MLPConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MambularClassifier, MambularRegressor, MLPClassifier, MLPRegressor


def _data(classification):
    features = np.random.default_rng(17).normal(size=(48, 3)).astype(np.float32)
    targets = (features[:, 0] > 0).astype(int) if classification else features[:, 0].copy()
    return features, targets


@pytest.mark.parametrize(("estimator", "regression"), [(MambularClassifier, False), (MambularRegressor, True)])
def test_pretrain_forwards_its_options_and_the_task(monkeypatch, tmp_path, estimator, regression):
    features, targets = _data(classification=not regression)
    model = estimator(
        model_config=MambularConfig(d_model=8, n_layers=1, d_state=4),
        preprocessing_config=PreprocessingConfig(numerical_method="standardization", output_dim=7),
        trainer_config=TrainerConfig(max_epochs=1, batch_size=16, checkpoint_path=str(tmp_path)),
        random_state=17,
    )
    model.build_model(features[:32], targets[:32], X_val=features[32:], y_val=targets[32:])
    assert model._task_model is not None
    calls = []
    monkeypatch.setattr(fit_module, "pretrain_embeddings", lambda **kwargs: calls.append(kwargs))

    model.pretrain(
        pretrain_epochs=3,
        k_neighbors=4,
        temperature=0.5,
        save_path="warm.pth",
        lr=0.01,
        use_positive=False,
        use_negative=True,
        pool_sequence=False,
    )

    assert len(calls) == 1
    assert calls[0].pop("base_model") is model._task_model.estimator
    assert calls[0].pop("train_dataloader") is model._data_module
    assert calls[0] == {
        "pretrain_epochs": 3,
        "k_neighbors": 4,
        "temperature": 0.5,
        "save_path": "warm.pth",
        "regression": regression,
        "lr": 0.01,
        "use_positive": False,
        "use_negative": True,
        "pool_sequence": False,
    }
    assert model._is_pretrained is True


@pytest.mark.parametrize("estimator", [MLPClassifier, MLPRegressor])
def test_pretrain_needs_a_built_model_with_an_embedding_layer(tmp_path, estimator):
    features, targets = _data(classification=estimator is MLPClassifier)
    model = estimator(
        model_config=MLPConfig(layer_sizes=[8]),
        preprocessing_config=PreprocessingConfig(numerical_method="standardization", output_dim=7),
        trainer_config=TrainerConfig(max_epochs=1, batch_size=16, checkpoint_path=str(tmp_path)),
        random_state=17,
    )
    with pytest.raises(ValueError, match="has not been built"):
        model.pretrain()

    model.build_model(features[:32], targets[:32], X_val=features[32:], y_val=targets[32:])
    with pytest.raises(ValueError, match="does not have an embedding layer"):
        model.pretrain()
    assert model._is_pretrained is False
