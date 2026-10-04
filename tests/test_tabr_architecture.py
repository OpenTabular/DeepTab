"""Candidate alignment and memory-efficient retrieval tests for TabR."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from deeptab.architectures.tabr import TabR
from deeptab.configs import TabRConfig


class _ExactL2Index:
    def __init__(self, dim):
        self.keys = torch.empty(0, dim)

    def reset(self):
        self.keys = self.keys[:0]

    def add(self, keys):
        self.keys = keys

    def search(self, queries, count):
        return torch.cdist(queries, self.keys).square().topk(count, largest=False)


class _Lambda(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, inputs):
        return self.function(inputs)


@pytest.fixture
def tabr_factory(monkeypatch):
    monkeypatch.setattr(
        TabR,
        "delu",
        SimpleNamespace(iter_batches=lambda rows, size: rows.split(size), nn=SimpleNamespace(Lambda=_Lambda)),
    )
    monkeypatch.setattr(TabR, "faiss", SimpleNamespace(IndexFlatL2=_ExactL2Index))
    monkeypatch.setattr(TabR, "faiss_torch_utils", object())

    def build(num_classes=1, **overrides):
        options: dict[str, Any] = {
            "use_embeddings": False,
            "d_main": 4,
            "context_size": 1,
            "candidate_encoding_batch_size": 2,
            "dropout0": 0.0,
            "dropout1": 0.0,
            "context_dropout": 0.0,
        }
        options.update(overrides)
        model = TabR(
            ({"feature": {"dimension": 2, "categories": None}}, {}, {}),
            num_classes=num_classes,
            config=TabRConfig(**options),
        )
        with torch.no_grad():
            model.linear.weight.zero_()
            model.linear.weight[:2, :2].copy_(torch.eye(2))
            model.linear.bias.zero_()
            model.K.weight.copy_(torch.eye(4))
            model.K.bias.zero_()
        return model.eval()

    return build


def _features(rows):
    return ([torch.tensor(rows, dtype=torch.float32)], [], [])


@pytest.mark.parametrize("memory_efficient", [False, True])
def test_nearest_external_labels_are_not_removed_by_batch_positions(tabr_factory, memory_efficient):
    model = tabr_factory(memory_efficient=memory_efficient)
    seen = []
    model.label_encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    output = model.train_with_candidates(
        *_features([[4, 0], [1, 0]]),
        targets=torch.tensor([3.0, 7.0]),
        candidate_x=_features([[4.1, 0], [1.1, 0], [99, 0]]),
        candidate_y=torch.tensor([10.0, 20.0, 30.0]),
        query_indices=torch.tensor([4, 1]),
    )
    torch.testing.assert_close(seen[0].squeeze(-1), torch.tensor([[10.0], [20.0]]))
    assert output.shape == (2, 1)
    output.sum().backward()
    assert model.linear.weight.grad is not None
    assert torch.isfinite(model.linear.weight.grad).all()


def test_other_query_labels_are_available_but_own_label_is_not(tabr_factory):
    model = tabr_factory()
    seen = []
    model.label_encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    model.train_with_candidates(
        *_features([[4, 0], [4.1, 0]]),
        targets=torch.tensor([3.0, 7.0]),
        candidate_x=_features([[99, 0]]),
        candidate_y=torch.tensor([30.0]),
    )
    torch.testing.assert_close(seen[0].squeeze(-1), torch.tensor([[7.0], [3.0]]))


@pytest.mark.parametrize("memory_efficient", [False, True])
def test_repeated_query_identity_cannot_retrieve_its_other_copy(tabr_factory, memory_efficient):
    model = tabr_factory(memory_efficient=memory_efficient)
    inputs = _features([[4, 0], [4, 0]])
    kwargs = {
        "candidate_x": _features([[4.1, 0]]),
        "candidate_y": torch.tensor([10.0]),
        "query_indices": torch.tensor([4, 4]),
    }
    original = model.train_with_candidates(*inputs, targets=torch.tensor([3.0, 3.0]), **kwargs)
    changed = model.train_with_candidates(*inputs, targets=torch.tensor([100.0, 100.0]), **kwargs)
    torch.testing.assert_close(original, changed)


def test_small_candidate_pool_clamps_context_for_all_paths(tabr_factory):
    model = tabr_factory(context_size=96)
    inputs = _features([[4, 0]])
    kwargs = {"candidate_x": _features([[4.1, 0]]), "candidate_y": torch.tensor([10.0])}
    for method in (model.validate_with_candidates, model.predict_with_candidates):
        assert torch.isfinite(method(*inputs, **kwargs)).all()
    assert torch.isfinite(model.train_with_candidates(*inputs, targets=torch.tensor([3.0]), **kwargs)).all()


def test_no_distinct_training_neighbor_is_reported(tabr_factory):
    model = tabr_factory()
    with pytest.raises(ValueError, match="different row identity"):
        model.train_with_candidates(
            *_features([[4, 0]]),
            targets=torch.tensor([3.0]),
            candidate_x=([torch.empty(0, 2)], [], []),
            candidate_y=torch.empty(0),
        )


@pytest.mark.parametrize("use_embeddings", [False, True])
@pytest.mark.parametrize("num_classes", [1, 3])
def test_memory_efficient_predictions_and_gradients_match_standard_path(tabr_factory, use_embeddings, num_classes):
    standard = tabr_factory(num_classes=num_classes, use_embeddings=use_embeddings, embedding_type="linear", d_model=8)
    efficient = tabr_factory(
        num_classes=num_classes,
        use_embeddings=use_embeddings,
        embedding_type="linear",
        d_model=8,
        memory_efficient=True,
    )
    efficient.load_state_dict(standard.state_dict())
    data = _features([[4, 0], [1, 0]])
    kwargs = {
        "targets": torch.tensor([0, 1]) if num_classes > 1 else torch.tensor([3.0, 7.0]),
        "candidate_x": _features([[4.1, 0], [1.1, 0], [99, 0]]),
        "candidate_y": torch.tensor([0, 1, 2]) if num_classes > 1 else torch.tensor([10.0, 20.0, 30.0]),
    }
    expected = standard.train_with_candidates(*data, **kwargs)
    actual = efficient.train_with_candidates(*data, **kwargs)
    torch.testing.assert_close(actual, expected)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(efficient.linear.weight.grad, standard.linear.weight.grad)
    if use_embeddings:
        assert any(parameter.grad is not None for parameter in efficient.embedding_layer.parameters())
