"""Seeded easy-problem improvement over training-only dummy baselines."""

import numpy as np
import pytest
import torch
from lightning.pytorch.callbacks import Callback
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, r2_score

from deeptab.core.registry import MODEL_REGISTRY

pytestmark = pytest.mark.integration
STABLE_ARCHITECTURES = sorted(name for name, info in MODEL_REGISTRY.items() if info.status == "stable")


def _assert_learning(task, score, baseline, initial_loss=None):
    if task == "classification":
        assert score >= 0.85 and score >= baseline + 0.4, (score, baseline)
    elif task == "regression":
        assert score >= 0.75 and score >= baseline + 0.6, (score, baseline)
    else:
        assert initial_loss is not None
        assert score <= baseline - 0.2 and score <= initial_loss - 0.2, (score, baseline, initial_loss)


@pytest.mark.parametrize(
    "task, score, baseline",
    [("classification", 1 / 3, 1 / 3), ("regression", 0.0, 0.0), ("regression", -100.0, 0.0), ("lss", 1.5, 1.5)],
)
def test_dummy_or_unimproved_scores_fail_learning_contract(task, score, baseline):
    with pytest.raises(AssertionError):
        _assert_learning(task, score, baseline, initial_loss=baseline)


class LearningTrace(Callback):
    def __init__(self):
        self.initial = {}
        self.initial_loss = None

    def on_train_start(self, trainer, pl_module):
        self.initial = {
            name: value.detach().cpu().clone() for name, value in pl_module.named_parameters() if value.requires_grad
        }

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            self.initial_loss = float(trainer.callback_metrics["val_loss"])


@pytest.mark.parametrize("architecture", STABLE_ARCHITECTURES)
@pytest.mark.parametrize("task", ["classification", "regression", "lss"])
def test_stable_model_improves_dummy_baseline(architecture, task, stable_model_factory, easy_problem, record_property):
    train, validation, train_classes, validation_classes, train_values, validation_values = easy_problem
    targets, held_out = (
        (train_classes, validation_classes) if task == "classification" else (train_values, validation_values)
    )
    model = stable_model_factory(architecture, task)
    trace = LearningTrace()
    result = model.fit(
        train,
        targets,
        X_val=validation,
        y_val=held_out,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=-1,
        callbacks=[trace],
        **({"family": "normal"} if task == "lss" else {}),
    )
    assert result is model
    predictions = model.predict(validation)
    assert np.isfinite(predictions).all()
    assert model._task_model is not None
    changed = [
        name
        for name, parameter in model._task_model.named_parameters()
        if name in trace.initial and not torch.equal(parameter.detach().cpu(), trace.initial[name])
    ]
    assert changed, "Training must update parameters, not only return finite predictions."
    assert any("head" not in name for name in changed), "The backbone must participate in learning."
    if task == "classification":
        baseline = DummyClassifier(strategy="most_frequent").fit(train, targets).score(validation, held_out)
        score = accuracy_score(held_out, predictions)
    elif task == "regression":
        baseline = DummyRegressor().fit(train, targets).score(validation, held_out)
        score = r2_score(held_out, predictions)
    else:
        assert predictions.shape == (len(validation), 2)
        assert (predictions[:, 1] > 0).all()
        mean, scale = float(np.mean(targets)), float(np.std(targets))
        baseline = float(-torch.distributions.Normal(mean, scale).log_prob(torch.as_tensor(held_out)).mean())
        score = float(
            -torch.distributions.Normal(torch.as_tensor(predictions[:, 0]), torch.as_tensor(predictions[:, 1]))
            .log_prob(torch.as_tensor(held_out))
            .mean()
        )
    _assert_learning(task, score, baseline, initial_loss=trace.initial_loss)
    for name, value in {
        "architecture": architecture,
        "task": task,
        "seed": 17,
        "epochs": 40,
        "baseline": float(baseline),
        "score": float(score),
        "initial_loss": trace.initial_loss,
        "changed_parameters": len(changed),
    }.items():
        record_property(name, value)
