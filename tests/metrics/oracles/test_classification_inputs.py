"""Known-answer classification metric inputs through task steps and adaptation."""

import numpy as np
import pytest
import torch
from torch import nn
from torchmetrics import Metric as TorchMetric
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, MulticlassAccuracy, MulticlassAUROC

from deeptab.configs import MLPConfig
from deeptab.distributions.normal import NormalDistribution
from deeptab.metrics import AUPRC, AUROC, Accuracy, BrierScore, ExpectedCalibrationError, F1Score, LogLoss
from deeptab.training import TaskModel

pytestmark = pytest.mark.integration


class _FixedLogits(nn.Module):
    returns_ensemble = False

    def __init__(self, config, feature_information, num_classes, lss):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, numerical, categorical, embeddings):
        return numerical[0] + self.anchor * 0


class _FixedEnsembleLogits(_FixedLogits):
    returns_ensemble = True


class _InputProbe:
    def __init__(self, input_space=None, needs_raw=False):
        self.input_space = input_space
        self.needs_raw = needs_raw
        self.calls = []

    def __call__(self, targets, predictions):
        self.calls.append((targets, predictions))
        return 0.0


class _TorchInputProbe(TorchMetric):
    def __init__(self, input_space=None):
        super().__init__()
        self.input_space = input_space
        self.calls = []

    def update(self, predictions, targets):
        self.calls.append((predictions.detach().clone(), targets.detach().clone()))

    def compute(self):
        return self.calls[-1][0].new_zeros(())


def _task(metric, stage="val", classes=2, ensemble=False, **kwargs):
    return TaskModel(
        model_class=_FixedEnsembleLogits if ensemble else _FixedLogits,
        config=MLPConfig(layer_sizes=[8]),
        feature_information=({}, {}, {}),
        num_classes=classes,
        train_metrics={"oracle": metric} if stage == "train" else None,
        val_metrics={"oracle": metric} if stage == "val" else None,
        **kwargs,
    )


def _run_step(task, stage, logits, targets, monkeypatch):
    recorded = {}
    monkeypatch.setattr(task, "log", lambda name, value, **kwargs: recorded.update({name: value}))
    batch = (([logits], [], []), targets)
    if stage == "train":
        task.training_step(batch, 0)
    else:
        task.validation_step(batch, 0)
    return float(recorded[f"{stage}_oracle"])


@pytest.mark.parametrize("stage", ["train", "val"])
@pytest.mark.parametrize(
    "metric_type", [Accuracy, F1Score, AUROC, AUPRC, LogLoss, BrierScore, ExpectedCalibrationError]
)
def test_binary_logits_reach_metrics_as_probabilities(stage, metric_type, monkeypatch):
    metric = metric_type()
    task = _task(metric, stage)
    logits = torch.tensor([[-2.0], [2.0]])
    targets = torch.tensor([[0.0], [1.0]])
    wrong_probability = 1 / (1 + np.exp(2))
    expected = {
        Accuracy: 1.0,
        F1Score: 1.0,
        AUROC: 1.0,
        AUPRC: 1.0,
        LogLoss: -np.log(1 - wrong_probability),
        BrierScore: wrong_probability**2,
        ExpectedCalibrationError: wrong_probability,
    }
    assert _run_step(task, stage, logits, targets, monkeypatch) == pytest.approx(expected[metric_type])


@pytest.mark.parametrize("stage", ["train", "val"])
@pytest.mark.parametrize("metric_type,expected", [(BinaryAccuracy, 0.5), (BinaryAUROC, 1.0)])
def test_native_binary_metrics_always_receive_probabilities(stage, metric_type, expected, monkeypatch):
    task = _task(metric_type(), stage)
    logits = torch.tensor([[0.2], [0.8]])
    targets = torch.tensor([[0.0], [1.0]])
    assert _run_step(task, stage, logits, targets, monkeypatch) == pytest.approx(expected)


@pytest.mark.parametrize("stage", ["train", "val"])
@pytest.mark.parametrize("classes", [2, 3])
def test_class_probability_layout_and_column_order(stage, classes, monkeypatch):
    logits = torch.eye(classes) * 2 - 1
    targets = torch.arange(classes)
    probe = _InputProbe("probabilities")
    task = _task(probe, stage, classes, loss_fct=nn.CrossEntropyLoss())
    _run_step(task, stage, logits, targets, monkeypatch)
    received_targets, received = probe.calls[0]
    exp_logits = np.exp(logits.numpy())
    expected = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(received, expected, rtol=1e-6)
    np.testing.assert_array_equal(received_targets.ravel(), np.arange(classes))
    np.testing.assert_array_equal(received.argmax(axis=1), np.arange(classes))


@pytest.mark.parametrize("stage", ["train", "val"])
@pytest.mark.parametrize("metric_type", [MulticlassAccuracy, MulticlassAUROC])
def test_native_multiclass_metrics_receive_flat_integer_targets(stage, metric_type, monkeypatch):
    metric = metric_type(num_classes=3)
    task = _task(metric, stage, classes=3)
    logits = torch.eye(3) * 3
    targets = torch.arange(3)
    assert _run_step(task, stage, logits, targets, monkeypatch) == pytest.approx(1.0)
    _, received_targets = task._metric_inputs(logits, targets.reshape(-1, 1), metric)
    assert received_targets.shape == (3,) and received_targets.dtype == torch.long


@pytest.mark.parametrize("stage", ["train", "val"])
@pytest.mark.parametrize("metric_type", [Accuracy, F1Score, AUROC, LogLoss, ExpectedCalibrationError])
def test_builtin_multiclass_metrics_match_independent_oracles(stage, metric_type, monkeypatch):
    metric = metric_type(average="macro") if metric_type is F1Score else metric_type()
    task = _task(metric, stage, classes=3)
    logits = torch.eye(3) * 2
    targets = torch.arange(3)
    correct_probability = np.exp(2) / (np.exp(2) + 2)
    expected = 1.0
    if metric_type is LogLoss:
        expected = -np.log(correct_probability)
    elif metric_type is ExpectedCalibrationError:
        expected = 1 - correct_probability
    assert _run_step(task, stage, logits, targets, monkeypatch) == pytest.approx(expected)


@pytest.mark.parametrize("classes", [2, 3])
def test_ensembles_reduce_declared_member_axis_before_activation(classes):
    if classes == 2:
        logits = torch.tensor([[-8.0, 2.0, 2.0], [8.0, -2.0, -2.0]])
    else:
        logits = torch.tensor([[[8.0, 1.0, -2.0], [-2.0, 1.0, 4.0]], [[0.0, 6.0, 1.0], [2.0, -1.0, 4.0]]])
    task = _task(_InputProbe("probabilities"), classes=classes, ensemble=True)
    received, _ = task._metric_inputs(logits, torch.zeros(2, 1), _InputProbe("probabilities"))
    average = logits.numpy().mean(axis=1)
    if classes == 2:
        positive = 1 / (1 + np.exp(-average))
        expected = np.column_stack([1 - positive, positive])
        member_average = (1 / (1 + np.exp(-logits.numpy()))).mean(axis=1)
        assert not np.allclose(expected[:, 1], member_average)
    else:
        exponentials = np.exp(average)
        expected = exponentials / exponentials.sum(axis=1, keepdims=True)
        member_exponentials = np.exp(logits.numpy())
        member_average = (member_exponentials / member_exponentials.sum(axis=2, keepdims=True)).mean(axis=1)
        assert not np.allclose(expected, member_average)
    np.testing.assert_allclose(received.numpy(), expected, rtol=1e-6)
    assert received.shape == (2, classes)


@pytest.mark.parametrize("stage", ["train", "val"])
@pytest.mark.parametrize("classes", [2, 3])
def test_actual_ensemble_steps_use_reduced_probability_columns(stage, classes, monkeypatch):
    task = _task(Accuracy(), stage, classes=classes, ensemble=True)
    if classes == 2:
        logits = torch.tensor([[-8.0, 2.0, 2.0], [8.0, -2.0, -2.0]])
        targets = torch.tensor([[0.0], [1.0]])
    else:
        logits = torch.tensor([[[8.0, 1.0, -2.0], [-2.0, 1.0, 4.0]], [[0.0, 6.0, 1.0], [2.0, -1.0, 5.0]]])
        targets = torch.tensor([0, 2])
    assert _run_step(task, stage, logits, targets, monkeypatch) == pytest.approx(1.0)


@pytest.mark.parametrize("stage", ["train", "val"])
@pytest.mark.parametrize("space", [None, "logits", "probabilities", "labels"])
def test_custom_callable_inputs_only_change_when_declared(stage, space, monkeypatch):
    probe = _InputProbe(space)
    task = _task(probe, stage)
    logits = torch.tensor([[-2.0], [2.0]], requires_grad=True)
    targets = torch.tensor([[0.0], [1.0]])
    _run_step(task, stage, logits, targets, monkeypatch)
    received_targets, received = probe.calls[0]
    assert isinstance(received, np.ndarray) and isinstance(received_targets, np.ndarray)
    np.testing.assert_array_equal(received_targets, targets.numpy())
    if space in {None, "logits"}:
        np.testing.assert_array_equal(received, logits.detach().numpy())
    elif space == "labels":
        np.testing.assert_array_equal(received, [0, 1])
    else:
        positive = 1 / (1 + np.exp(-logits.detach().numpy()))
        np.testing.assert_allclose(received, np.concatenate([1 - positive, positive], axis=1))


def test_raw_opt_out_and_regression_ensemble_axes_are_unchanged():
    logits = torch.arange(12.0).reshape(2, 3, 2)
    targets = torch.zeros(2, 1)
    task = _task(_InputProbe(), classes=1, ensemble=True)
    for metric in [_InputProbe(), _InputProbe("probabilities", needs_raw=True)]:
        received, received_targets = task._metric_inputs(logits, targets, metric)
        assert received is logits and received_targets is targets


@pytest.mark.parametrize("raw", [False, True])
def test_lss_parameter_axes_and_raw_contract_are_unchanged(raw):
    family = NormalDistribution()
    metric = _InputProbe(needs_raw=raw)
    task = _task(metric, ensemble=True, lss=True, family=family)
    logits = torch.arange(-6.0, 6.0).reshape(2, 3, 2)
    targets = torch.zeros(2, 1)
    received, received_targets = task._metric_inputs(logits, targets, metric)
    assert received_targets is targets
    if raw:
        assert received is logits
    else:
        assert received.shape == (2, 2)
        torch.testing.assert_close(received[:, 0], logits.mean(dim=1)[:, 0], rtol=0, atol=0)
        assert torch.all(received[:, 1] > 0)


@pytest.mark.parametrize("stage", ["train", "val"])
def test_actual_lss_ensemble_steps_preserve_parameter_columns(stage, monkeypatch):
    probe = _InputProbe()
    task = _task(probe, stage, ensemble=True, lss=True, family=NormalDistribution())
    logits = torch.arange(-6.0, 6.0).reshape(2, 3, 2)
    targets = torch.zeros(2, 1)
    assert _run_step(task, stage, logits, targets, monkeypatch) == 0.0
    received = probe.calls[0][1]
    assert received.shape == (2, 2)
    np.testing.assert_allclose(received[:, 0], logits.numpy().mean(axis=1)[:, 0])


@pytest.mark.parametrize("stage", ["train", "val"])
@pytest.mark.parametrize("space", [None, "logits", "probabilities", "labels"])
def test_custom_torchmetrics_keep_tensor_order_and_explicit_contract(stage, space, monkeypatch):
    metric = _TorchInputProbe(space)
    task = _task(metric, stage)
    logits = torch.tensor([[-2.0], [2.0]], requires_grad=True)
    targets = torch.tensor([[0.0], [1.0]])
    _run_step(task, stage, logits, targets, monkeypatch)
    received, received_targets = metric.calls[0]
    assert received.device == logits.device and received_targets.device == targets.device
    if space in {None, "logits"}:
        torch.testing.assert_close(received, logits)
        torch.testing.assert_close(received_targets, targets)
    elif space == "labels":
        torch.testing.assert_close(received, torch.tensor([0, 1]))
        assert received_targets.shape == (2,) and received_targets.dtype == torch.long
    else:
        positive = 1 / (1 + np.exp(-logits.detach().numpy()))
        np.testing.assert_allclose(received.numpy(), np.concatenate([1 - positive, positive], axis=1))
        assert received_targets.shape == (2,) and received_targets.dtype == torch.long


def test_invalid_classifier_shapes_and_binary_metric_tasks_fail_clearly():
    task = _task(_InputProbe())
    with pytest.raises(ValueError, match="per-row logits"):
        task._metric_inputs(torch.zeros(2, 3, 4), torch.zeros(2), _InputProbe("probabilities"))
    with pytest.raises(ValueError, match="exactly two"):
        task._metric_inputs(torch.zeros(2, 3), torch.zeros(2), BinaryAccuracy())


def test_invalid_input_contract_rejects_instead_of_guessing():
    task = _task(_InputProbe())
    with pytest.raises(ValueError, match="input_space"):
        task._metric_inputs(torch.zeros(2, 1), torch.zeros(2, 1), _InputProbe("guess"))


@pytest.mark.parametrize("metric_type", [AUPRC, BrierScore])
def test_binary_only_metrics_reject_multiclass_columns(metric_type):
    with pytest.raises(ValueError, match="binary"):
        metric_type()(np.array([0, 1]), np.full((2, 3), 1 / 3))
