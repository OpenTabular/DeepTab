"""Regression tests for three independent numerical/robustness defects."""

from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch

from deeptab.distributions import MultinomialDistribution
from deeptab.nn.blocks.common import sparsemax

QUIET = {
    "accelerator": "cpu",
    "devices": 1,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "logger": False,
}


class TestMultinomialProbabilities:
    def test_forward_returns_probabilities_not_logits(self):
        """predict(raw=False) must apply the softmax the docstring promises."""
        family = MultinomialDistribution(num_classes=3)
        logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])

        probs = family(logits)
        assert torch.allclose(probs, torch.softmax(logits, dim=-1), atol=1e-6)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)
        assert (probs >= 0).all()


class TestSparsemaxGradients:
    def test_gradient_unaffected_by_a_leading_size_one_axis(self):
        """A bare squeeze() in backward dropped unrelated size-1 axes."""
        data = torch.tensor([[1.0, 0.5, -0.2, 2.0, 0.1]])

        flat = data.clone().requires_grad_(True)
        cast(torch.Tensor, sparsemax(flat, dim=-1)).sum().backward()

        nested = data.clone().unsqueeze(1).requires_grad_(True)  # shape (1, 1, 5)
        cast(torch.Tensor, sparsemax(nested, dim=-1)).sum().backward()

        assert flat.grad is not None and nested.grad is not None
        torch.testing.assert_close(nested.grad.squeeze(1), flat.grad)

    def test_gradient_matches_batched_reference(self):
        rows = torch.tensor([[1.0, 0.5, -0.2, 2.0, 0.1], [0.3, 0.3, 0.9, -1.0, 0.2]])

        batched = rows.clone().requires_grad_(True)
        cast(torch.Tensor, sparsemax(batched, dim=-1)).sum().backward()

        single = rows[:1].clone().requires_grad_(True)
        cast(torch.Tensor, sparsemax(single, dim=-1)).sum().backward()

        assert batched.grad is not None and single.grad is not None
        torch.testing.assert_close(single.grad[0], batched.grad[0])


class TestTangosSingleRowBatch:
    def test_fit_with_a_trailing_single_row_batch(self):
        """An odd-sized dataset leaves a batch of one; that must not crash."""
        pytest.importorskip("torch.func")
        from deeptab.models.experimental import TangosRegressor

        rng = np.random.default_rng(0)
        # 42 rows with the default 0.2 split leaves 33 training rows, so the
        # final batch of 16 holds exactly one row.
        n = 42
        X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        y = rng.normal(size=n)

        model = TangosRegressor()
        model.fit(X, y, max_epochs=1, batch_size=16, random_state=101, **QUIET)
        assert np.isfinite(model.predict(X)).all()
