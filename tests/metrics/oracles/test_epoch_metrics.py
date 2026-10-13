"""Global epoch metric oracles through Lightning's actual lifecycle."""

from typing import Any

import lightning as pl
import numpy as np
import pytest
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch import nn
from torch.multiprocessing.spawn import ProcessRaisedException
from torch.utils.data import DataLoader, TensorDataset, default_collate
from torchmetrics.classification import BinaryAUROC, BinaryF1Score
from torchmetrics.regression import MeanSquaredError
from torchmetrics.regression import R2Score as TorchR2Score

from deeptab.configs import MLPConfig
from deeptab.metrics import AUROC, F1Score, R2Score, RootMeanSquaredError
from deeptab.training import TaskModel

pytestmark = pytest.mark.integration


class _EpochEcho(nn.Module):
    def __init__(self, config, feature_information, num_classes, lss):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, numerical, categorical, embeddings):
        return numerical[0] + self.anchor * 0


def _collate_epoch_rows(rows):
    features, targets = default_collate(rows)
    return ([features], [], []), targets


def _loader(scores, targets, batch_size=2):
    dataset = TensorDataset(
        torch.tensor(scores, dtype=torch.float32).reshape(-1, 1),
        torch.tensor(targets, dtype=torch.float32).reshape(-1, 1),
    )
    return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_epoch_rows)


def _task(metric, classes=2, train=False, **kwargs):
    return TaskModel(
        model_class=_EpochEcho,
        config=MLPConfig(layer_sizes=[8]),
        feature_information=({}, {}, {}),
        num_classes=classes,
        train_metrics={"oracle": metric} if train else None,
        val_metrics={"oracle": metric},
        lr=0,
        **kwargs,
    )


def _trainer(**kwargs):
    options: dict[str, Any] = {
        "accelerator": "cpu",
        "max_epochs": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    return pl.Trainer(**{**options, **kwargs})


@pytest.mark.parametrize("metric_type", [AUROC, BinaryAUROC])
def test_repeated_validation_resets_metric_state(metric_type):
    metric = metric_type()
    task = _task(metric)
    trainer = _trainer()
    for scores, expected in [([0.1, 0.2, 0.3, 0.4], 0.75), ([0.4, 0.3, 0.2, 0.1], 0.25)]:
        result = trainer.validate(task, dataloaders=_loader(scores, [0, 1, 0, 1]), verbose=False)
        assert result[0]["val_oracle"] == pytest.approx(expected)
        assert task._metric_buffers["val"] == {} and task._metric_buffer_bytes["val"] == 0
        if isinstance(metric, BinaryAUROC):
            assert metric.update_called is False


@pytest.mark.parametrize("metric_type", [F1Score, BinaryF1Score])
def test_global_f1_with_one_row_final_batch(metric_type):
    task = _task(metric_type())
    result = _trainer().validate(
        task,
        dataloaders=_loader([-3, 2, -1, 1, 3], [0, 1, 1, 0, 1]),
        verbose=False,
    )
    assert result[0]["val_oracle"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "metric,expected",
    [
        pytest.param(R2Score(), 0.4, id="numpy-r2"),
        pytest.param(TorchR2Score(), 0.4, id="torch-r2"),
        pytest.param(RootMeanSquaredError(), np.sqrt(6 / 5), id="numpy-rmse"),
        pytest.param(MeanSquaredError(squared=False), np.sqrt(6 / 5), id="torch-rmse"),
    ],
)
def test_global_regression_metrics_and_row_weighted_loss(metric, expected):
    task = _task(metric, classes=1)
    result = _trainer().validate(task, dataloaders=_loader([0, 2, 2, 5, 3], [0, 1, 2, 3, 4]), verbose=False)
    assert result[0]["val_oracle"] == pytest.approx(expected)
    assert result[0]["val_loss"] == pytest.approx(6 / 5)


class _LifecycleCapture(Callback):
    def __init__(self, validation_loader):
        self.validation_loader = validation_loader
        self.sanity_scores = []
        self.epoch_scores = []

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 1:
            dataset = self.validation_loader.dataset
            assert isinstance(dataset, TensorDataset)
            dataset.tensors[0].copy_(torch.tensor([[0.4], [0.3], [0.2], [0.1]]))

    def on_validation_end(self, trainer, pl_module):
        value = float(trainer.callback_metrics["val_oracle"])
        if trainer.sanity_checking:
            self.sanity_scores.append(value)
        else:
            self.epoch_scores.append(value)


def test_sanity_stages_consecutive_epochs_and_checkpoint_use_completed_metrics(tmp_path):
    shared = BinaryAUROC()
    task = _task(shared, train=True)
    assert task.train_metrics["oracle"] is shared
    assert task.val_metrics["oracle"] is not shared
    validation_loader = _loader([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    capture = _LifecycleCapture(validation_loader)
    checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="val_oracle", mode="max", save_top_k=1)
    trainer = _trainer(
        max_epochs=2,
        num_sanity_val_steps=1,
        enable_checkpointing=True,
        callbacks=[capture, checkpoint],
    )
    trainer.fit(task, train_dataloaders=_loader([4, 3, 2, 1], [0, 1, 0, 1]), val_dataloaders=validation_loader)
    assert capture.sanity_scores == pytest.approx([1.0])
    assert capture.epoch_scores == pytest.approx([0.75, 0.25])
    assert float(trainer.callback_metrics["train_oracle"]) == pytest.approx(0.25)
    assert checkpoint.best_model_score is not None
    assert float(checkpoint.best_model_score) == pytest.approx(0.75)
    assert len(task.val_losses) == 2
    assert task._metric_buffers == {"train": {}, "val": {}}


def test_callable_buffer_is_detached_bounded_and_released_on_overflow():
    task = _task(AUROC(), metric_buffer_limit_bytes=24)
    predictions = torch.tensor([[-2.0], [2.0]], requires_grad=True)
    targets = torch.tensor([[0.0], [1.0]])
    task._update_epoch_metrics("val", predictions, targets)
    buffered = task._metric_buffers["val"]["oracle"][0]
    assert all(value.device.type == "cpu" and value.grad_fn is None and not value.requires_grad for value in buffered)
    assert task._metric_buffer_bytes["val"] == 24
    with pytest.raises(MemoryError, match="metric_buffer_limit_bytes"):
        task._update_epoch_metrics("val", predictions, targets)
    assert task._metric_buffers["val"] == {} and task._metric_buffer_bytes["val"] == 0


@pytest.mark.parametrize("limit", [0, -1, True, float("nan")])
def test_invalid_callable_buffer_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="metric_buffer_limit_bytes"):
        _task(AUROC(), metric_buffer_limit_bytes=limit)


@pytest.mark.parametrize("metric", [R2Score(), TorchR2Score(), RootMeanSquaredError(), MeanSquaredError(squared=False)])
def test_training_metrics_use_complete_uneven_epoch(metric):
    task = _task(metric, classes=1, train=True)
    loader = _loader([0, 2, 2, 5, 3], [0, 1, 2, 3, 4])
    trainer = _trainer(num_sanity_val_steps=0)
    trainer.fit(task, train_dataloaders=loader, val_dataloaders=loader)
    expected = 0.4 if isinstance(metric, (R2Score, TorchR2Score)) else np.sqrt(6 / 5)
    assert float(trainer.callback_metrics["train_oracle"]) == pytest.approx(expected)
    assert float(trainer.callback_metrics["train_loss_epoch"]) == pytest.approx(6 / 5)


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="Requires the CPU Gloo distributed backend.",
)
@pytest.mark.parametrize("metric_type", [AUROC, BinaryAUROC])
@pytest.mark.parametrize("rows", [4, 6])
def test_two_real_cpu_ranks_compute_global_auroc(metric_type, rows):
    task = _task(metric_type())
    trainer = _trainer(devices=2, strategy="ddp_spawn")
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6][:rows]
    labels = [0, 1, 0, 1, 0, 1][:rows]
    result = trainer.validate(task, dataloaders=_loader(scores, labels), verbose=False)
    expected_auc = 0.75 if rows == 4 else 2 / 3
    assert result[0]["val_oracle"] == pytest.approx(expected_auc)
    logits = torch.tensor(scores)
    targets = torch.tensor(labels, dtype=torch.float32)
    expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert result[0]["val_loss"] == pytest.approx(float(expected_loss))


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="Requires the CPU Gloo distributed backend.",
)
def test_real_cpu_ranks_enforce_global_callable_budget():
    task = _task(AUROC(), metric_buffer_limit_bytes=32)
    with pytest.raises(ProcessRaisedException, match="Global callable metric inputs exceed"):
        _trainer(devices=2, strategy="ddp_spawn").validate(
            task,
            dataloaders=_loader([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1]),
            verbose=False,
        )


class _EpochCallable:
    def __init__(self, fail=False):
        self.shapes = []
        self.fail = fail

    def __call__(self, targets, predictions):
        self.shapes.append((targets.shape, predictions.shape))
        if self.fail:
            raise ValueError("Callable failure")
        return float(np.mean(predictions))


def test_callable_is_evaluated_once_on_concatenated_epoch():
    metric = _EpochCallable()
    task = _task(metric, classes=1)
    result = _trainer().validate(task, dataloaders=_loader([1, 2, 3, 4, 5], [0, 1, 2, 3, 4]), verbose=False)
    assert result[0]["val_oracle"] == pytest.approx(3)
    assert metric.shapes == [((5, 1), (5, 1))]


def test_callable_failure_releases_epoch_buffers():
    task = _task(_EpochCallable(fail=True), classes=1)
    with pytest.raises(ValueError, match="Callable failure"):
        _trainer().validate(task, dataloaders=_loader([1, 2, 3], [0, 1, 2]), verbose=False)
    assert task._metric_buffers["val"] == {} and task._metric_buffer_bytes["val"] == 0


def test_callable_buffer_budget_is_total_across_metric_names():
    task = _task(AUROC(), metric_buffer_limit_bytes=32)
    task.val_metrics["second"] = AUROC()
    with pytest.raises(MemoryError, match="metric_buffer_limit_bytes"):
        task._update_epoch_metrics("val", torch.ones(2, 1), torch.zeros(2, 1))
    assert task._metric_buffers["val"] == {} and task._metric_buffer_bytes["val"] == 0


def test_duplicate_native_names_are_isolated_without_mutating_caller_dict():
    shared = BinaryAUROC()
    metrics = {"first": shared, "second": shared}
    task = TaskModel(
        model_class=_EpochEcho,
        config=MLPConfig(layer_sizes=[8]),
        feature_information=({}, {}, {}),
        num_classes=2,
        val_metrics=metrics,
    )
    assert metrics["first"] is metrics["second"] is shared
    assert task.val_metrics["first"] is not task.val_metrics["second"]
    result = _trainer().validate(task, dataloaders=_loader([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1]), verbose=False)
    assert result[0]["val_first"] == pytest.approx(0.75)
    assert result[0]["val_second"] == pytest.approx(0.75)


@pytest.mark.parametrize("metric_type", [AUROC, BinaryAUROC])
def test_validation_auroc_is_global_not_average_of_batch_scores(metric_type):
    task = TaskModel(
        model_class=_EpochEcho,
        config=MLPConfig(layer_sizes=[8]),
        feature_information=({}, {}, {}),
        num_classes=2,
        val_metrics={"auc": metric_type()},
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    result = trainer.validate(task, dataloaders=_loader([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1]), verbose=False)
    assert result[0]["val_auc"] == pytest.approx(0.75)
