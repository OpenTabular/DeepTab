"""Regression tests for the 2.0.x bug-hunt fixes that span several modules."""

import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from deeptab.models import MambaTabRegressor, MLPRegressor, SAINTRegressor

FIT_KW = {"max_epochs": 1, "batch_size": 16, "accelerator": "cpu"}


def _make_data(n=60, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"num1": rng.randn(n), "num2": rng.rand(n)})
    y = rng.randn(n)
    return X, y


def _backward_one_batch(model):
    """Run one manual forward/backward on the fitted task model."""
    task = model._task_model
    task.zero_grad()
    (num, cat, emb), labels = next(iter(model._data_module.train_dataloader()))
    preds = task(num, cat, emb)
    task.compute_loss(preds, labels).backward()
    return task


class TestBackbonesReceiveGradients:
    def test_mambatab_trains_its_mamba_block(self):
        """Regression test: MambaTab.forward previously never called self.mamba."""
        X, y = _make_data()
        model = MambaTabRegressor()
        model.fit(X, y, **FIT_KW)
        task = _backward_one_batch(model)
        mamba_params = [(n, p) for n, p in task.named_parameters() if ".mamba." in n]
        assert mamba_params, "expected mamba parameters on the estimator"
        assert all(p.grad is not None for _, p in mamba_params)

    def test_saint_attention_layernorms_receive_gradients(self):
        """Regression test: RowColTransformer skipped its pre-attention norms."""
        X, y = _make_data()
        model = SAINTRegressor()
        model.fit(X, y, **FIT_KW)
        task = _backward_one_batch(model)
        norm_params = [
            (n, p)
            for n, p in task.named_parameters()
            if "layers" in n and (".0.0." in n or ".2.0." in n) and "weight" in n
        ]
        assert norm_params, "expected attention LayerNorm parameters"
        assert all(p.grad is not None for _, p in norm_params)


class TestContrastivePairing:
    def test_aligned_anchors_give_zero_loss(self):
        """Regression test: anchors were paired with other anchors' neighbors.

        With orthonormal embeddings, self-positives (cos=1) and orthogonal
        negatives (cos=0, margin 0) give exactly zero CosineEmbeddingLoss --
        but only if every pair is attributed to the right anchor.
        """
        from deeptab.training.pretraining import ContrastivePretrainer

        pt = ContrastivePretrainer.__new__(ContrastivePretrainer)
        torch.nn.Module.__init__(pt)  # minimal init; full __init__ needs an estimator
        pt.pool_sequence = True
        pt.use_positive = True
        pt.use_negative = True
        pt.loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.0)

        embeddings = torch.eye(4)
        knn_indices = torch.arange(4).view(4, 1)  # each anchor's positive is itself
        neg_indices = (torch.arange(4).view(4, 1) + 1) % 4  # orthogonal negative

        # `device` is a Lightning property; contrastive_loss only uses it to
        # place the label tensors, so run on CPU tensors directly.
        ContrastivePretrainer.device = property(lambda self: torch.device("cpu"))
        try:
            loss = pt.contrastive_loss(embeddings, knn_indices, neg_indices)
        finally:
            del ContrastivePretrainer.device
        assert float(loss) == pytest.approx(0.0, abs=1e-6)


class TestSmallFixes:
    def test_getstate_excludes_lightning_module(self):
        X, y = _make_data()
        model = MLPRegressor()
        model.fit(X, y, **FIT_KW)
        state = model.__getstate__()
        assert state["_task_model"] is None
        assert "task_model" not in state or state.get("task_model") is None

    def test_fit_accepts_plain_lists(self):
        X = [[1.0, 2.0], [2.0, 1.0], [0.5, 0.1], [1.5, 0.7]] * 10
        y = list(np.random.RandomState(0).randn(40))
        model = MLPRegressor()
        model.fit(X, y, **FIT_KW)
        assert model.predict(X).shape == (40,)

    def test_sparsemax_does_not_mutate_input(self):
        from deeptab.nn.blocks.common import sparsemax

        logits = torch.randn(4, 8)
        original = logits.clone()
        sparsemax(logits, dim=-1)
        assert torch.equal(logits, original)

    def test_validate_fit_inputs_positive_families(self):
        from deeptab.models.base import _validate_fit_inputs

        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y_nonpositive = np.array([1.0, 0.0, 2.0])
        for family in ("gamma", "inversegamma", "lognormal"):
            with pytest.raises(Exception, match="positive"):
                _validate_fit_inputs(X, y_nonpositive, regression=True, family=family)

    def test_taskmodel_candidate_pool_defaults_to_none(self):
        from deeptab.configs import MLPConfig
        from deeptab.training.lightning_module import TaskModel

        class _Dummy(torch.nn.Module):
            def __init__(self, config=None, feature_information=None, num_classes=1, lss=False, **kw):
                super().__init__()
                self.linear = torch.nn.Linear(2, num_classes)

        task = TaskModel(model_class=_Dummy, config=MLPConfig(), feature_information=({}, {}, {}))
        assert task.train_features is None
        assert task.train_targets is None
