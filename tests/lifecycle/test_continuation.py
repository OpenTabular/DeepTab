"""Weight continuation must refresh data without refitting preprocessing."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import SequentialSampler, WeightedRandomSampler

from deeptab.configs import MambularConfig, MLPConfig, PreprocessingConfig, TrainerConfig
from deeptab.configs.experimental.modernnca_config import ModernNCAConfig
from deeptab.core.exceptions import DataError, UnsupportedParamError
from deeptab.models import MLPLSS, MambularClassifier, MambularRegressor, MLPClassifier, MLPRegressor
from deeptab.models.experimental import ModernNCARegressor
from deeptab.training import TaskModel

pytestmark = pytest.mark.integration


class _ContinuationCapture(Callback):
    def on_train_start(self, trainer, pl_module):
        self.weights = {name: value.detach().clone() for name, value in pl_module.named_parameters()}
        datamodule = getattr(trainer, "datamodule", None)
        assert datamodule is not None
        self.datamodule = datamodule
        self.targets = self.datamodule.y_train.copy()
        self.validation_targets = self.datamodule.y_val.copy()
        self.batch = next(iter(self.datamodule.train_dataloader()))
        self.optimizer = trainer.optimizers[0]
        self.optimizer_state_size = len(self.optimizer.state)
        self.learning_rate = self.optimizer.param_groups[0]["lr"]
        self.weight_decay = self.optimizer.param_groups[0]["weight_decay"]
        self.max_epochs = trainer.max_epochs
        self.epochs = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.epochs.append(trainer.current_epoch)


@pytest.fixture(params=["regressor", "classifier", "normal", "categorical", "dirichlet"])
def fitted_case(request, tmp_path):
    features = np.random.default_rng(17).normal(size=(48, 3)).astype(np.float32)
    targets = features[:, 0].copy()
    estimator = MLPRegressor
    family_options = {}
    if request.param == "classifier":
        estimator = MLPClassifier
        targets = (np.arange(len(features)) % 3) * 10
    elif request.param in {"normal", "categorical", "dirichlet"}:
        estimator = MLPLSS
        family_options = {"family": request.param}
        if request.param == "categorical":
            targets = (np.arange(len(features)) % 3) * 10
        elif request.param == "dirichlet":
            targets = np.random.default_rng(19).dirichlet([2, 3, 4], size=len(features)).astype(np.float32)
    model = estimator(
        model_config=MLPConfig(layer_sizes=[8]),
        preprocessing_config=PreprocessingConfig(numerical_method="standardization", output_dim=7),
        trainer_config=TrainerConfig(max_epochs=1, batch_size=16, checkpoint_path=str(tmp_path)),
        random_state=17,
    )
    options = {
        "accelerator": "cpu",
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        **family_options,
    }
    features = pd.DataFrame(features, columns=pd.Index(["first", "second", "third"]))
    model.fit(features[:32], targets[:32], X_val=features[32:], y_val=targets[32:], **options)
    return model, features, targets, options


def _updated_targets(targets):
    if targets.ndim == 2:
        return np.roll(targets, 1, axis=1)
    if np.issubdtype(targets.dtype, np.integer):
        return np.roll(targets, 1)
    return targets + 1000


def test_repeated_continuation_refreshes_all_task_families_and_optimizer(fitted_case, monkeypatch):
    model, features, targets, options = fitted_case
    preprocessor = model._preprocessor
    task = model._task_model
    previous_optimizer = model._trainer.optimizers[0]
    assert previous_optimizer.state

    def forbid_refit(*args, **kwargs):
        raise AssertionError("Continuation refitted preprocessing")

    monkeypatch.setattr(type(preprocessor), "fit", forbid_refit)
    for offset, batch_size in [(10, 7), (20, 9)]:
        updated = _updated_targets(targets)
        weights = {name: value.detach().clone() for name, value in task.named_parameters()}
        capture = _ContinuationCapture()
        model.fit(
            features[:32] + offset,
            updated[:32],
            X_val=features[32:] + offset,
            y_val=updated[32:],
            rebuild=False,
            batch_size=batch_size,
            max_epochs=2,
            lr=0.005,
            weight_decay=0.02,
            lr_patience=3,
            lr_factor=0.4,
            shuffle=False,
            dataloader_kwargs={"drop_last": True},
            callbacks=[capture],
            **options,
        )
        expected = updated[:32] // 10 if np.issubdtype(targets.dtype, np.integer) else updated[:32]
        np.testing.assert_array_equal(capture.targets, expected)
        assert capture.batch[1].shape[0] == batch_size
        assert isinstance(capture.datamodule.train_dataloader().sampler, SequentialSampler)
        assert capture.datamodule.train_dataloader().drop_last is True
        assert capture.datamodule.val_dataloader().drop_last is False
        assert capture.optimizer is not previous_optimizer
        assert capture.optimizer_state_size == 0
        assert capture.learning_rate == 0.005
        assert capture.weight_decay == 0.02
        assert capture.max_epochs == 2
        assert capture.epochs == [0, 1]
        assert task.lr_patience == 3
        assert task.lr_factor == 0.4
        assert model._task_model is task
        assert model._preprocessor is preprocessor
        for name, previous in weights.items():
            torch.testing.assert_close(capture.weights[name], previous, rtol=0, atol=0)
        previous_optimizer = capture.optimizer


@pytest.mark.parametrize(
    "invalid", ["width", "column_order", "metrics", "train_metrics", "batch_size", "loader", "target_shape"]
)
def test_incompatible_continuation_preserves_fitted_state(fitted_case, invalid):
    model, features, targets, options = fitted_case
    before = model.predict(features[32:])
    attributes = vars(model).copy()
    supplied = features[:32]
    extra = {}
    if invalid == "width":
        supplied = supplied.iloc[:, :-1]
    elif invalid == "column_order":
        supplied = supplied.iloc[:, ::-1]
    elif invalid == "train_metrics":
        extra = {"train_metrics": {}}
    elif invalid == "batch_size":
        extra = {"batch_size": 0}
    elif invalid == "loader":
        extra = {"dataloader_kwargs": {"num_workers": -1}}
    elif invalid == "target_shape":
        targets = np.column_stack([targets] * 2)
    else:
        extra = {"val_metrics": {}}
    with pytest.raises((DataError, ValueError, UnsupportedParamError)):
        model.fit(supplied, targets[:32], rebuild=False, **extra, **options)
    assert model.is_fitted_ is True
    assert all(vars(model)[name] is value for name, value in attributes.items())
    np.testing.assert_array_equal(model.predict(features[32:]), before)


@pytest.mark.parametrize("fitted_case", ["classifier", "categorical"], indirect=True)
def test_changed_labels_preserve_continuation_state(fitted_case):
    model, features, targets, options = fitted_case
    predictions = model.predict(features[32:])
    attributes = vars(model).copy()
    with pytest.raises(DataError, match="rebuild=True"):
        model.fit(features[:32], targets[:32] + 100, rebuild=False, **options)
    assert all(vars(model)[name] is value for name, value in attributes.items())
    np.testing.assert_array_equal(model.predict(features[32:]), predictions)


def test_loaded_estimator_continuation_uses_new_data(fitted_case, tmp_path):
    model, features, targets, options = fitted_case
    path = tmp_path / "continuation.deeptab"
    model.save(str(path))
    loaded = type(model).load(str(path))
    preprocessor = loaded._preprocessor
    task = loaded._task_model
    weights = {name: value.detach().clone() for name, value in task.named_parameters()}
    capture = _ContinuationCapture()
    updated = _updated_targets(targets)
    loaded.fit(
        features[:32] + 10,
        updated[:32],
        X_val=features[32:] + 10,
        y_val=updated[32:],
        rebuild=False,
        batch_size=7,
        callbacks=[capture],
        **options,
    )
    expected = updated[:32] // 10 if np.issubdtype(targets.dtype, np.integer) else updated[:32]
    np.testing.assert_array_equal(capture.targets, expected)
    assert loaded._preprocessor is preprocessor
    assert loaded._task_model is task
    for name, previous in weights.items():
        torch.testing.assert_close(capture.weights[name], previous, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["valid", "missing", "width", "rows", "validation", "nonfinite"])
def test_continuation_embedding_compatibility(tmp_path, case):
    features = np.random.default_rng(17).normal(size=(48, 3)).astype(np.float32)
    targets = features[:, 0].copy()
    embeddings = np.random.default_rng(19).normal(size=(48, 4)).astype(np.float32)
    model = MLPRegressor(
        model_config=MLPConfig(layer_sizes=[8]),
        preprocessing_config=PreprocessingConfig(numerical_method="standardization", output_dim=7),
        trainer_config=TrainerConfig(max_epochs=1, batch_size=16, checkpoint_path=str(tmp_path)),
        random_state=17,
    )
    options: dict[str, Any] = {"accelerator": "cpu", "logger": False, "enable_progress_bar": False}
    model.fit(
        features[:32],
        targets[:32],
        X_val=features[32:],
        y_val=targets[32:],
        embeddings=embeddings[:32],
        embeddings_val=embeddings[32:],
        **options,
    )
    attributes = vars(model).copy()
    task = model._task_model
    assert isinstance(task, TaskModel)
    weights = {name: value.detach().clone() for name, value in task.named_parameters()}
    supplied = embeddings[:32].copy() + 10
    supplied_validation = embeddings[32:] + 10
    if case == "missing":
        supplied = None
    elif case == "width":
        supplied = supplied[:, :-1]
    elif case == "rows":
        supplied = supplied[:-1]
    elif case == "validation":
        supplied_validation = None
    elif case == "nonfinite":
        supplied[0, 0] = np.nan
    capture = _ContinuationCapture()
    kwargs = {
        "X_val": features[32:],
        "y_val": targets[32:] + 1000,
        "embeddings": supplied,
        "embeddings_val": supplied_validation,
        "rebuild": False,
        "batch_size": 7,
        "shuffle": False,
        "callbacks": [capture],
        **options,
    }
    if case == "valid":
        assert supplied is not None
        model.fit(features[:32], targets[:32] + 1000, **kwargs)
        torch.testing.assert_close(capture.batch[0][2][0], torch.tensor(supplied[:7]))
        for name, previous in weights.items():
            torch.testing.assert_close(capture.weights[name], previous, rtol=0, atol=0)
    else:
        with pytest.raises(DataError, match="embedding"):
            model.fit(features[:32], targets[:32] + 1000, **kwargs)
        assert all(vars(model)[name] is value for name, value in attributes.items())
        for name, previous in weights.items():
            torch.testing.assert_close(dict(task.named_parameters())[name], previous, rtol=0, atol=0)


@pytest.mark.parametrize("estimator", [MambularClassifier, MambularRegressor])
def test_pretrained_continuation_preserves_warm_weights_and_refreshes_data(tmp_path, estimator):
    torch.manual_seed(17)
    features = np.random.default_rng(17).normal(size=(48, 3)).astype(np.float32)
    targets = (features[:, 0] > 0).astype(int) if estimator is MambularClassifier else features[:, 0].copy()
    model = estimator(
        model_config=MambularConfig(d_model=8, n_layers=1, d_state=4),
        preprocessing_config=PreprocessingConfig(numerical_method="standardization", output_dim=7),
        trainer_config=TrainerConfig(max_epochs=1, batch_size=16, checkpoint_path=str(tmp_path)),
        random_state=17,
    )
    model.build_model(features[:32], targets[:32], X_val=features[32:], y_val=targets[32:])
    task = model._task_model
    assert isinstance(task, TaskModel)
    embedding_layer = task.estimator.embedding_layer
    assert isinstance(embedding_layer, torch.nn.Module)
    embedding_before = next(embedding_layer.parameters()).detach().clone()
    model.pretrain(pretrain_epochs=1, k_neighbors=2, save_path=str(tmp_path / "warm.pth"))
    embedding_after = next(embedding_layer.parameters()).detach().clone()
    assert not torch.equal(embedding_before, embedding_after)
    assert model._is_pretrained is True
    preprocessor = model._preprocessor
    weights = {name: value.detach().clone() for name, value in task.named_parameters()}
    updated = 1 - targets if estimator is MambularClassifier else targets + 1000
    capture = _ContinuationCapture()
    model.fit(
        features[:32] + 10,
        updated[:32],
        X_val=features[32:] + 10,
        y_val=updated[32:],
        batch_size=7,
        callbacks=[capture],
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
    )
    np.testing.assert_array_equal(capture.targets, updated[:32])
    assert model._is_pretrained is False
    assert model._task_model is task
    assert model._preprocessor is preprocessor
    for name, previous in weights.items():
        torch.testing.assert_close(capture.weights[name], previous, rtol=0, atol=0)


@pytest.mark.parametrize("load_first", [False, True])
def test_modernnca_continuation_refreshes_candidate_bank(tmp_path, load_first):
    features = np.random.default_rng(17).normal(size=(48, 3)).astype(np.float32)
    targets = features[:, 0].copy()
    model = ModernNCARegressor(
        model_config=ModernNCAConfig(dim=8, d_block=16, n_blocks=0, n_frequencies=4, sample_rate=1.0),
        preprocessing_config=PreprocessingConfig(numerical_method="standardization", output_dim=7),
        trainer_config=TrainerConfig(max_epochs=1, batch_size=16, checkpoint_path=str(tmp_path)),
        random_state=17,
    )
    options: dict[str, Any] = {"accelerator": "cpu", "logger": False, "enable_progress_bar": False}
    model.fit(features[:32], targets[:32], X_val=features[32:], y_val=targets[32:], **options)
    if load_first:
        path = tmp_path / "modernnca.deeptab"
        model.save(str(path))
        model = ModernNCARegressor.load(str(path))
    task = model._task_model
    assert isinstance(task, TaskModel)
    old_bank = task.train_targets
    weights = {name: value.detach().clone() for name, value in task.named_parameters()}
    capture = _ContinuationCapture()
    model.fit(
        features[:24] + 10,
        targets[:24] + 1000,
        X_val=features[32:] + 10,
        y_val=targets[32:] + 1000,
        rebuild=False,
        batch_size=7,
        callbacks=[capture],
        **options,
    )
    assert task.train_targets is not old_bank
    assert task.train_targets is not None
    assert task.train_features is not None
    torch.testing.assert_close(task.train_targets.flatten(), torch.tensor(targets[:24] + 1000))
    assert task.train_features[0][0].shape[0] == 24
    datamodule = model._data_module
    assert datamodule is not None and datamodule.num_feature_info is not None
    first_feature = next(iter(datamodule.num_feature_info))
    transformed = model._preprocessor.transform(getattr(datamodule, "X_train", None))
    np.testing.assert_array_equal(task.train_features[0][0].cpu().numpy(), transformed[f"num_{first_feature}"])
    for name, previous in weights.items():
        torch.testing.assert_close(capture.weights[name], previous, rtol=0, atol=0)
    assert np.isfinite(model.predict(features[32:])).all()


def test_continuation_creates_new_automatic_validation_split(fitted_case):
    model, features, targets, options = fitted_case
    updated = _updated_targets(targets)
    capture = _ContinuationCapture()
    model.fit(
        features + 10,
        updated,
        rebuild=False,
        val_size=0.25,
        random_state=23,
        batch_size=7,
        callbacks=[capture],
        **options,
    )
    assert len(capture.targets) == 36
    assert len(capture.validation_targets) == 12
    expected = updated // 10 if np.issubdtype(targets.dtype, np.integer) else updated
    combined = np.concatenate([capture.targets, capture.validation_targets])
    if targets.ndim == 2:
        np.testing.assert_array_equal(np.sort(combined, axis=0), np.sort(expected, axis=0))
    else:
        np.testing.assert_array_equal(np.sort(combined), np.sort(expected))
    assert capture.datamodule.random_state == 23
    assert capture.datamodule.train_dataloader().generator.initial_seed() == 23


def test_row_specific_sampling_weights_require_rebuild(tmp_path):
    features = np.random.default_rng(17).normal(size=(48, 3)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(int)
    model = MLPClassifier(
        model_config=MLPConfig(layer_sizes=[8]),
        trainer_config=TrainerConfig(max_epochs=1, checkpoint_path=str(tmp_path)),
        random_state=17,
    )
    options: dict[str, Any] = {"accelerator": "cpu", "logger": False, "enable_progress_bar": False}
    model.fit(
        features[:32], labels[:32], X_val=features[32:], y_val=labels[32:], sample_weight=np.arange(1, 33), **options
    )
    predictions = model.predict(features[32:])
    attributes = vars(model).copy()
    with pytest.raises(UnsupportedParamError, match="rebuild=True"):
        model.fit(features[:32] + 10, labels[:32], rebuild=False, **options)
    assert all(vars(model)[name] is value for name, value in attributes.items())
    np.testing.assert_array_equal(model.predict(features[32:]), predictions)


def test_balanced_sampling_uses_new_labels(tmp_path):
    features = np.random.default_rng(17).normal(size=(48, 3)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(int)
    model = MLPClassifier(
        model_config=MLPConfig(layer_sizes=[8]),
        trainer_config=TrainerConfig(max_epochs=1, checkpoint_path=str(tmp_path)),
        random_state=17,
    )
    options: dict[str, Any] = {"accelerator": "cpu", "logger": False, "enable_progress_bar": False}
    model.fit(features[:32], labels[:32], X_val=features[32:], y_val=labels[32:], balanced_sampler=True, **options)
    capture = _ContinuationCapture()
    model.fit(
        features[:32],
        1 - labels[:32],
        X_val=features[32:],
        y_val=1 - labels[32:],
        rebuild=False,
        batch_size=7,
        callbacks=[capture],
        **options,
    )
    loader = capture.datamodule.train_dataloader()
    assert isinstance(loader.sampler, WeightedRandomSampler)
    np.testing.assert_array_equal(capture.targets, 1 - labels[:32])
    counts = np.bincount(1 - labels[:32])
    np.testing.assert_allclose(loader.sampler.weights.numpy(), 1 / counts[1 - labels[:32]])


def test_continuation_refreshes_targets_and_batch_size_without_replacing_weights(tmp_path):
    features = np.random.default_rng(17).normal(size=(48, 3)).astype(np.float32)
    targets = features[:, 0].copy()
    model = MLPRegressor(
        model_config=MLPConfig(layer_sizes=[8]),
        preprocessing_config=PreprocessingConfig(numerical_method="standardization", output_dim=7),
        trainer_config=TrainerConfig(max_epochs=1, batch_size=16, checkpoint_path=str(tmp_path)),
        random_state=17,
    )
    options: dict[str, Any] = {
        "accelerator": "cpu",
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    model.fit(features[:32], targets[:32], X_val=features[32:], y_val=targets[32:], **options)
    preprocessor = model._preprocessor
    datamodule = model._data_module
    task = model._task_model
    assert isinstance(task, TaskModel)
    weights = {name: value.detach().clone() for name, value in task.named_parameters()}
    capture = _ContinuationCapture()

    model.fit(
        features[:32] + 10,
        targets[:32] + 1000,
        X_val=features[32:] + 10,
        y_val=targets[32:] + 1000,
        rebuild=False,
        batch_size=7,
        shuffle=False,
        callbacks=[capture],
        **options,
    )

    np.testing.assert_array_equal(capture.targets, targets[:32] + 1000)
    np.testing.assert_array_equal(capture.validation_targets, targets[32:] + 1000)
    assert capture.batch[1].shape[0] == 7
    torch.testing.assert_close(capture.batch[1].flatten(), torch.tensor(targets[:7] + 1000))
    assert capture.datamodule is not datamodule
    assert model._preprocessor is preprocessor
    assert model._task_model is task
    for name, previous in weights.items():
        torch.testing.assert_close(capture.weights[name], previous, rtol=0, atol=0)
