"""ModernNCA training retrieval excludes each query's row identity."""

import pytest
import torch

from deeptab.architectures.experimental.modern_nca import ModernNCA
from deeptab.configs import ModernNCAConfig


def _model(sample_rate=1.0, lss=False):
    return ModernNCA(
        ({"feature": {"dimension": 2}}, {}, {}),
        num_classes=3,
        lss=lss,
        config=ModernNCAConfig(use_embeddings=False, dim=4, n_blocks=0, sample_rate=sample_rate),
    ).eval()


@pytest.mark.parametrize("sample_rate", [0.5, 1.0])
def test_repeated_query_labels_do_not_affect_predictions(sample_rate):
    model = _model(sample_rate)
    data = ([torch.ones(2, 2)], [], [])
    kwargs = {
        "candidate_x": ([torch.tensor([[2.0, 3.0], [3.0, 4.0]])], [], []),
        "candidate_y": torch.tensor([2, 2]),
        "query_indices": torch.tensor([7, 7]),
    }
    first = model.train_with_candidates(*data, targets=torch.tensor([0, 0]), **kwargs)
    second = model.train_with_candidates(*data, targets=torch.tensor([1, 1]), **kwargs)
    torch.testing.assert_close(first, second)
    assert first.argmax(dim=1).tolist() == [2, 2]
    first.sum().backward()
    assert model.encoder.weight.grad is not None
    assert torch.isfinite(model.encoder.weight.grad).all()


def test_distinct_query_rows_can_use_each_others_labels_without_external_pool():
    model = _model()
    output = model.train_with_candidates(
        [torch.tensor([[1.0, 2.0], [2.0, 3.0]])],
        [],
        [],
        targets=torch.tensor([0, 1]),
        candidate_x=([torch.empty(0, 2)], [], []),
        candidate_y=torch.empty(0, dtype=torch.long),
    )
    assert output.argmax(dim=1).tolist() == [1, 0]


@pytest.mark.parametrize("sample_rate", [0.0, 0.5])
def test_no_eligible_neighbor_raises_instead_of_returning_nan(sample_rate):
    model = _model(sample_rate)
    with pytest.raises(ValueError, match="different row identity"):
        model.train_with_candidates(
            [torch.ones(1, 2)],
            [],
            [],
            targets=torch.tensor([0]),
            candidate_x=([torch.empty(0, 2)], [], []),
            candidate_y=torch.empty(0, dtype=torch.long),
        )


def test_lss_path_masks_repeated_query_representations():
    model = _model(lss=True)
    data = ([torch.ones(2, 2)], [], [])
    kwargs = {
        "targets": torch.zeros(2),
        "candidate_x": ([torch.tensor([[2.0, 3.0]])], [], []),
        "candidate_y": torch.ones(1),
        "query_indices": torch.tensor([7, 7]),
    }
    seen = []
    model.tabular_head.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    output = model.train_with_candidates(*data, **kwargs)
    expected = model.encoder(kwargs["candidate_x"][0][0]).expand(2, -1)
    torch.testing.assert_close(seen[0], expected)
    assert output.shape == (2, 3)
