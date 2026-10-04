"""Validation history and pruning tests for TaskModel."""

from types import SimpleNamespace
from typing import Any, cast

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn
from torch.utils.data import DataLoader

from deeptab.configs import MLPConfig
from deeptab.data.dataset import TabularDataset
from deeptab.training.lightning_module import TaskModel


class _PruningEstimator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, numericals, categoricals, embeddings):
        return self.linear(torch.cat(numericals, dim=-1))


def _task():
    return TaskModel(
        model_class=_PruningEstimator,
        config=MLPConfig(),
        feature_information=({}, {}, {}),
        num_classes=1,
        scheduler_type=None,
    )


def _trainer(task, *, epoch=0, loss=2.0, fitting=True, sanity_checking=False):
    trainer = SimpleNamespace(
        state=SimpleNamespace(fn=TrainerFn.FITTING if fitting else TrainerFn.VALIDATING),
        sanity_checking=sanity_checking,
        current_epoch=epoch,
        callback_metrics={"val_loss": torch.tensor(loss)},
        should_stop=False,
    )
    cast(Any, task)._trainer = trainer
    return trainer


@pytest.mark.parametrize("fitting,sanity_checking", [(True, True), (False, False)])
def test_nontraining_validation_does_not_record_or_prune(fitting, sanity_checking):
    task = _task()
    task.early_pruning_threshold = 1.0
    task.pruning_epoch = 0
    trainer = _trainer(task, fitting=fitting, sanity_checking=sanity_checking)
    task.on_validation_epoch_end()
    assert task.val_losses == []
    assert not trainer.should_stop


def test_validation_history_uses_actual_zero_based_epoch():
    task = _task()
    trainer = _trainer(task, epoch=2, loss=3.0)
    task.on_validation_epoch_end()
    assert task.epoch_val_loss_at(0) == float("inf")
    assert task.epoch_val_loss_at(1) == float("inf")
    assert task.epoch_val_loss_at(2) == 3.0
    trainer.callback_metrics["val_loss"] = torch.tensor(2.5)
    task.on_validation_epoch_end()
    assert task.val_losses == [float("inf"), float("inf"), 2.5]
    assert task.epoch_val_loss_at(-1) == float("inf")
    assert task.epoch_val_loss_at(3) == float("inf")


def test_pruning_starts_at_requested_training_epoch():
    task = _task()
    task.early_pruning_threshold = -1.0
    task.pruning_epoch = 1
    trainer = _trainer(task, epoch=0, loss=0.0)
    task.on_validation_epoch_end()
    assert not trainer.should_stop
    trainer.current_epoch = 1
    task.on_validation_epoch_end()
    assert trainer.should_stop


def test_real_fit_records_only_training_validation_epochs():
    task = _task()
    features = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
    dataset = TabularDataset(
        cat_features_list=[],
        num_features_list=[features],
        embeddings_list=[],
        labels=features.sum(dim=1, keepdim=True),
    )
    loader = DataLoader(dataset, batch_size=4)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        num_sanity_val_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(task, train_dataloaders=loader, val_dataloaders=loader)
    assert len(task.val_losses) == 2
    assert task.epoch_val_loss_at(0) == task.val_losses[0]
    assert task.epoch_val_loss_at(1) == task.val_losses[1]
    recorded = list(task.val_losses)
    trainer.validate(task, dataloaders=loader, verbose=False)
    assert task.val_losses == recorded
