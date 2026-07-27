"""Regression test: candidate-model step methods must use the right kwarg names.

``predict_with_candidates`` is defined as ``(self, *data, candidate_x,
candidate_y)`` in TabR and ModernNCA, but ``test_step`` called it with
``candidates_x``/``candidates_y``, so ``trainer.test()`` on those models always
raised TypeError. ``predict_step`` already used the singular names.
"""

import torch
import torch.nn as nn

from deeptab.configs import MLPConfig
from deeptab.training.lightning_module import TaskModel


class _CandidateEstimator(nn.Module):
    """Minimal stand-in with TabR's ``predict_with_candidates`` signature."""

    uses_candidates = True
    returns_ensemble = False

    def __init__(self, config=None, feature_information=None, num_classes=1, lss=False, **kwargs):
        super().__init__()
        self.linear = nn.Linear(4, num_classes)
        self.seen = {}

    def forward(self, *data):
        return self.linear(data[0][0])

    def predict_with_candidates(self, *data, candidate_x, candidate_y):
        self.seen["candidate_x"] = candidate_x
        self.seen["candidate_y"] = candidate_y
        return self.forward(*data)


def _task_model():
    task = TaskModel(
        model_class=_CandidateEstimator,
        config=MLPConfig(),
        feature_information=({}, {}, {}),
        num_classes=1,
    )
    task.train_features = ([torch.randn(8, 4)], [], None)
    task.train_targets = torch.randn(8, 1)
    return task


def _batch(batch_size=4):
    return (([torch.randn(batch_size, 4)], [], None), torch.randn(batch_size, 1))


def test_test_step_passes_candidate_kwargs():
    task = _task_model()
    loss = task.test_step(_batch(), 0)
    assert loss.isfinite()
    assert task.estimator.seen["candidate_x"] is task.train_features
    assert task.estimator.seen["candidate_y"] is task.train_targets


def test_predict_step_passes_candidate_kwargs():
    task = _task_model()
    # predict_step unpacks the batch as the data tuple itself (no labels).
    preds = task.predict_step(_batch()[0], 0)
    assert preds.shape == (4, 1)
    assert task.estimator.seen["candidate_x"] is task.train_features
