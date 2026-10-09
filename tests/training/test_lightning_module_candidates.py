"""Training-pool identity and candidate-aware evaluation contracts."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import lightning as pl
import pytest
import torch
import torch.nn as nn

from deeptab.configs import MLPConfig, ModernNCAConfig
from deeptab.data import TabularDataset
from deeptab.training.lightning_module import TaskModel


class _CandidateEstimator(nn.Module):
    uses_candidates = True

    def __init__(self, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))
        self.received = None

    def train_with_candidates(self, *data, targets, candidate_x, candidate_y, query_indices):
        self.received = candidate_x, candidate_y, query_indices
        return self.scale.expand(len(targets), 1)

    def predict_with_candidates(self, *data, candidate_x, candidate_y):
        self.received = candidate_x, candidate_y
        return self.scale.expand(len(data[0][0]), 1)


def _setup_task(
    monkeypatch, model_class: type[nn.Module] = _CandidateEstimator, config: Any = None, num_classes=1
) -> tuple[Any, TabularDataset]:
    labels = torch.arange(6, dtype=torch.float32).unsqueeze(-1)
    dataset = TabularDataset(
        [torch.arange(6)],
        [torch.ones(6, 2)],
        [torch.arange(6, dtype=torch.float32).unsqueeze(-1)],
        labels,
    )
    task = TaskModel(
        model_class=model_class,
        config=config or MLPConfig(),
        feature_information=({"feature": {"dimension": 2}}, {}, {}),
        num_classes=num_classes,
    )
    cast(Any, task)._trainer = SimpleNamespace(datamodule=SimpleNamespace(train_dataset=dataset))
    monkeypatch.setattr(task, "log", Mock())
    task.setup("fit")
    return task, dataset


def test_setup_keeps_complete_dataset_order_without_training_sampler(monkeypatch):
    task, dataset = _setup_task(monkeypatch)
    assert dataset.return_indices is True
    torch.testing.assert_close(task.train_targets, dataset.labels)
    torch.testing.assert_close(task.train_features[1][0], torch.arange(6))


@pytest.mark.parametrize("step", ["validation_step", "test_step", "predict_step"])
@pytest.mark.parametrize("restore_weights", [False, True])
def test_candidate_evaluation_without_prior_fit_has_initialized_guards(monkeypatch, step, restore_weights):
    class StandaloneCandidateEstimator(_CandidateEstimator):
        def forward(self, *data):
            return self.scale.expand(len(data[0][0]), 1)

        def validate_with_candidates(self, *data, candidate_x, candidate_y):
            return self.predict_with_candidates(*data, candidate_x=candidate_x, candidate_y=candidate_y)

    kwargs = {
        "model_class": StandaloneCandidateEstimator,
        "config": MLPConfig(),
        "feature_information": ({"feature": {"dimension": 2}}, {}, {}),
    }
    task = TaskModel(**kwargs)
    if restore_weights:
        restored = TaskModel(**kwargs)
        restored.load_state_dict(task.state_dict())
        task = restored
    assert task.train_features is None
    assert task.train_targets is None
    monkeypatch.setattr(task, "log", Mock())
    data = ([torch.ones(2, 2)], [], [])
    batch = data if step == "predict_step" else (data, torch.zeros(2, 1))
    assert torch.isfinite(getattr(task, step)(batch, 0)).all()
    assert task.estimator.received is None


def test_training_excludes_row_ids_and_preserves_other_identical_features(monkeypatch):
    task, dataset = _setup_task(monkeypatch)
    data, labels, indices = next(iter(torch.utils.data.DataLoader(dataset, batch_size=3, sampler=[4, 1, 4])))
    loss = task.training_step((data, labels, indices), 0)
    assert torch.isfinite(loss)
    candidate_x, candidate_y, query_ids = task.estimator.received
    assert query_ids.tolist() == [4, 1, 4]
    assert candidate_y.flatten().tolist() == [0, 2, 3, 5]
    assert candidate_x[1][0].tolist() == [0, 2, 3, 5]
    assert candidate_x[2][0].flatten().tolist() == [0, 2, 3, 5]
    torch.testing.assert_close(candidate_x[0][0], torch.ones(4, 2))


def test_candidate_training_rejects_batches_without_row_identity(monkeypatch):
    task, dataset = _setup_task(monkeypatch)
    data, labels, _ = cast(tuple, dataset[0])
    with pytest.raises(ValueError, match="row indices"):
        task.training_step((data, labels), 0)


def test_test_step_uses_singular_candidate_keywords_and_full_pool(monkeypatch):
    task, dataset = _setup_task(monkeypatch)
    data, labels, _ = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    assert torch.isfinite(task.test_step((data, labels), 0))
    candidate_x, candidate_y = task.estimator.received
    assert candidate_x is task.train_features
    assert candidate_y is task.train_targets


def test_modernnca_trainer_prediction_does_not_depend_on_own_pool_label(monkeypatch):
    from deeptab.architectures.experimental.modern_nca import ModernNCA

    task, _dataset = _setup_task(
        monkeypatch,
        model_class=ModernNCA,
        config=ModernNCAConfig(use_embeddings=False, dim=4, n_blocks=0, sample_rate=1.0),
        num_classes=3,
    )
    task.train_features = ([torch.arange(6, dtype=torch.float32).unsqueeze(-1).repeat(1, 2)], [], [])
    task.train_targets = torch.full((6,), 2, dtype=torch.long)
    task.eval()
    predictions = []
    original = task.estimator.train_with_candidates

    def record(*args, **kwargs):
        output = original(*args, **kwargs)
        predictions.append(output.detach().clone())
        return output

    monkeypatch.setattr(task.estimator, "train_with_candidates", record)
    data = ([task.train_features[0][0][[4]]], [], [])
    for label in (0, 1):
        task.train_targets[4] = label
        task.training_step((data, torch.tensor([label]), torch.tensor([4])), 0)
    torch.testing.assert_close(predictions[0], predictions[1])
    assert predictions[0].argmax(dim=-1).item() == 2


@pytest.mark.parametrize("missing_bank", ["train_features", "train_targets"])
def test_candidate_training_requires_a_complete_bank(monkeypatch, missing_bank):
    task, _ = _setup_task(monkeypatch)
    setattr(task, missing_bank, None)
    data = ([torch.ones(2, 2)], [], [])
    with pytest.raises(RuntimeError, match="training bank"):
        task.training_step((data, torch.ones(2), torch.tensor([0, 1])), 0)


def test_lightning_fit_and_test_use_candidate_aware_paths(tmp_path):
    from deeptab.architectures.experimental.modern_nca import ModernNCA

    class CandidateDataModule(pl.LightningDataModule):
        def __init__(self):
            super().__init__()
            features = [torch.arange(12, dtype=torch.float32).reshape(6, 2)]
            labels = torch.tensor([0, 1, 2, 0, 1, 2])
            self.train_dataset = TabularDataset([], features, [], labels)
            self.test_dataset = TabularDataset([], [torch.ones(2, 2)], [], torch.tensor([0, 1]))

        def train_dataloader(self):
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=2, sampler=[4, 1, 4, 0, 2, 3])

        def val_dataloader(self):
            return torch.utils.data.DataLoader(self.test_dataset, batch_size=2)

        def test_dataloader(self):
            return torch.utils.data.DataLoader(self.test_dataset, batch_size=2)

    datamodule = CandidateDataModule()
    task = TaskModel(
        model_class=ModernNCA,
        config=ModernNCAConfig(use_embeddings=False, dim=4, n_blocks=0, sample_rate=1.0),
        feature_information=({"feature": {"dimension": 2}}, {}, {}),
        num_classes=3,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(task, datamodule=datamodule)
    results = trainer.test(task, datamodule=datamodule)
    assert torch.isfinite(torch.tensor(results[0]["test_loss_epoch"]))
    assert task.train_targets is not None
    assert len(task.train_targets) == 6
