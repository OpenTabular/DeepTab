"""Tests for deeptab/training/pretraining.py — Phase 7c."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from deeptab.training.pretraining import ContrastivePretrainer, _validate_pretrainable_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal model stub that satisfies ContrastivePretrainer's interface."""

    embedding_layer = object()

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def encode(self, x, grad=False):
        n = x.shape[0] if hasattr(x, "shape") else 4
        return torch.randn(n, 8)

    def pool_sequence(self, x):
        return x

    def get_embedding_state_dict(self):
        return {}

    def parameters(self):
        return iter([torch.zeros(4, requires_grad=True)])


def _make_pretrainer(**kwargs) -> ContrastivePretrainer:
    defaults = {
        "base_model": _FakeModel(),
        "k_neighbors": 2,
        "regression": False,
        "pool_sequence": True,
    }
    defaults.update(kwargs)
    return ContrastivePretrainer(**defaults)


# ---------------------------------------------------------------------------
# _validate_pretrainable_model
# ---------------------------------------------------------------------------


def test_validate_ok():
    """Model with all required attributes passes without error."""
    _validate_pretrainable_model(_FakeModel(), pool_sequence=True, save_embeddings=True)


def test_validate_missing_encode():
    from deeptab.core.exceptions import ArchitectureRequirementError

    class NoEncode:
        embedding_layer = object()

    with pytest.raises(ArchitectureRequirementError, match="encode"):
        _validate_pretrainable_model(NoEncode(), pool_sequence=False, save_embeddings=False)


def test_validate_missing_embedding_layer():
    from deeptab.core.exceptions import ArchitectureRequirementError

    class NoLayer:
        def encode(self, x, grad=False):
            return x

    with pytest.raises(ArchitectureRequirementError, match="embedding_layer"):
        _validate_pretrainable_model(NoLayer(), pool_sequence=False, save_embeddings=False)


def test_validate_missing_pool_sequence_when_required():
    from deeptab.core.exceptions import ArchitectureRequirementError

    class NoPool:
        embedding_layer = object()

        def encode(self, x, grad=False):
            return x

    with pytest.raises(ArchitectureRequirementError, match="pool_sequence"):
        _validate_pretrainable_model(NoPool(), pool_sequence=True, save_embeddings=False)


def test_validate_pool_sequence_not_required_when_false():
    """pool_sequence=False must not require pool_sequence() method."""

    class NoPool:
        embedding_layer = object()

        def encode(self, x, grad=False):
            return x

        def get_embedding_state_dict(self):
            return {}

    _validate_pretrainable_model(NoPool(), pool_sequence=False, save_embeddings=True)


def test_validate_missing_get_embedding_state_dict_when_required():
    from deeptab.core.exceptions import ArchitectureRequirementError

    class NoStateDict:
        embedding_layer = object()

        def encode(self, x, grad=False):
            return x

    with pytest.raises(ArchitectureRequirementError, match="get_embedding_state_dict"):
        _validate_pretrainable_model(NoStateDict(), pool_sequence=False, save_embeddings=True)


def test_validate_multiple_missing_reported():
    from deeptab.core.exceptions import ArchitectureRequirementError

    class Empty:
        pass

    with pytest.raises(ArchitectureRequirementError) as exc_info:
        _validate_pretrainable_model(Empty(), pool_sequence=True, save_embeddings=True)

    msg = str(exc_info.value)
    assert "embedding_layer" in msg
    assert "encode" in msg
    assert "pool_sequence" in msg
    assert "get_embedding_state_dict" in msg


# ---------------------------------------------------------------------------
# _sample_indices
# ---------------------------------------------------------------------------


def test_sample_indices_normal():
    pt = _make_pretrainer()
    indices = torch.tensor([1, 2, 3, 4, 5])
    result = pt._sample_indices(indices, 3)
    assert result.shape == (3,)
    assert all(r.item() in [1, 2, 3, 4, 5] for r in result)


def test_sample_indices_exact_k():
    pt = _make_pretrainer()
    indices = torch.tensor([10, 20, 30])
    result = pt._sample_indices(indices, 3)
    assert result.shape == (3,)
    assert set(result.tolist()).issubset({10, 20, 30})


def test_sample_indices_with_replacement():
    """When fewer indices than k, the result is filled with replacement."""
    pt = _make_pretrainer()
    indices = torch.tensor([1, 2])
    result = pt._sample_indices(indices, 5)
    assert result.shape == (5,)
    assert all(r.item() in [1, 2] for r in result)


def test_sample_indices_empty_returns_empty():
    pt = _make_pretrainer()
    indices = torch.tensor([], dtype=torch.long)
    result = pt._sample_indices(indices, 3)
    assert result.numel() == 0


def test_sample_indices_k_equals_one():
    pt = _make_pretrainer()
    indices = torch.tensor([7, 8, 9])
    result = pt._sample_indices(indices, 1)
    assert result.shape == (1,)
    assert result.item() in [7, 8, 9]


# ---------------------------------------------------------------------------
# temperature deprecation warning
# ---------------------------------------------------------------------------


def test_temperature_default_no_warning():
    """Default temperature=0.1 must not emit a FutureWarning about temperature."""
    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _make_pretrainer(temperature=0.1)

    temp_warnings = [
        w for w in record if issubclass(w.category, FutureWarning) and "temperature" in str(w.message).lower()
    ]
    assert len(temp_warnings) == 0


def test_temperature_nondefault_warns():
    """Non-default temperature emits a FutureWarning."""
    with pytest.warns(FutureWarning, match="temperature"):
        _make_pretrainer(temperature=0.5)


# ---------------------------------------------------------------------------
# get_knn
# ---------------------------------------------------------------------------


def test_get_knn_regression_shapes():
    pt = _make_pretrainer(regression=True, k_neighbors=2)
    labels = torch.randn(8, 1)
    knn, neg = pt.get_knn(labels)
    assert knn.shape == (8, 2)
    assert neg.shape == (8, 2)


def test_get_knn_classification_shapes():
    # 4 samples, 2 classes → each sample has ≥1 same-class and ≥1 different-class neighbor
    pt = _make_pretrainer(regression=False, k_neighbors=1)
    labels = torch.tensor([0, 0, 1, 1])
    knn, neg = pt.get_knn(labels)
    # shapes: (valid_samples, k_neighbors)
    assert knn.shape[1] == 1
    assert neg.shape[1] == 1


def test_get_knn_classification_all_same_class_raises():
    """Single-class batch must raise ValueError."""
    pt = _make_pretrainer(regression=False, k_neighbors=1)
    labels = torch.tensor([0, 0, 0, 0])
    with pytest.raises(ValueError, match=r"no.*same-class or no.*different-class"):
        pt.get_knn(labels)


# ---------------------------------------------------------------------------
# ContrastivePretrainer init
# ---------------------------------------------------------------------------


def test_constructor_stores_attributes():
    pt = _make_pretrainer(k_neighbors=3, regression=True, margin=0.3)
    assert pt.k_neighbors == 3
    assert pt.regression is True
    assert pt.margin == 0.3
    assert isinstance(pt.loss_fn, nn.CosineEmbeddingLoss)
