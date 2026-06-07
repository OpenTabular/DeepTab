"""
Tests for the deeptab.distributions public API.

Verifies that all distribution classes are importable from ``deeptab.distributions``,
that ``__all__`` is complete, and that concrete classes have a working
``parameter_count`` / ``name`` interface (inherited from BaseDistribution).
"""

import pytest

EXPECTED_DISTRIBUTIONS = [
    "BaseDistribution",
    "BetaDistribution",
    "CategoricalDistribution",
    "DirichletDistribution",
    "GammaDistribution",
    "InverseGammaDistribution",
    "JohnsonSuDistribution",
    "LogNormalDistribution",
    "MixtureOfGaussiansDistribution",
    "MultinomialDistribution",
    "NegativeBinomialDistribution",
    "NormalDistribution",
    "PoissonDistribution",
    "Quantile",
    "StudentTDistribution",
    "TweedieDistribution",
    "ZeroInflatedPoissonDistribution",
]

# Concrete (instantiable-with-no-args) classes and their expected parameter counts
CONCRETE_NO_ARGS = [
    ("NormalDistribution", 2),
    ("LogNormalDistribution", 2),
    ("PoissonDistribution", 1),
    ("ZeroInflatedPoissonDistribution", 2),
    ("GammaDistribution", 2),
    ("InverseGammaDistribution", 2),
    ("BetaDistribution", 2),
    ("DirichletDistribution", 1),
    ("StudentTDistribution", 3),
    ("JohnsonSuDistribution", 4),
    ("NegativeBinomialDistribution", 2),
    ("CategoricalDistribution", 1),
    ("MultinomialDistribution", 2),
    ("Quantile", 3),
    ("TweedieDistribution", 1),
    ("MixtureOfGaussiansDistribution", 9),
]


# ---------------------------------------------------------------------------
# Importability / __all__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("class_name", EXPECTED_DISTRIBUTIONS)
def test_distribution_importable(class_name: str):
    """Every distribution class is importable from deeptab.distributions."""
    import importlib

    mod = importlib.import_module("deeptab.distributions")
    assert hasattr(mod, class_name), f"{class_name!r} not found in deeptab.distributions"


def test_distributions_all_complete():
    """deeptab.distributions.__all__ contains every expected class."""
    import deeptab.distributions as d

    for name in EXPECTED_DISTRIBUTIONS:
        assert name in d.__all__, f"{name!r} missing from deeptab.distributions.__all__"


# ---------------------------------------------------------------------------
# Interface checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("class_name,expected_param_count", CONCRETE_NO_ARGS)
def test_distribution_parameter_count(class_name: str, expected_param_count: int):
    """Concrete distributions report the correct number of parameters."""
    import importlib

    mod = importlib.import_module("deeptab.distributions")
    cls = getattr(mod, class_name)
    obj = cls()
    assert obj.parameter_count == expected_param_count


@pytest.mark.parametrize("class_name,_", CONCRETE_NO_ARGS)
def test_distribution_has_name(class_name: str, _):
    """Concrete distributions expose a non-empty name string."""
    import importlib

    mod = importlib.import_module("deeptab.distributions")
    cls = getattr(mod, class_name)
    obj = cls()
    assert isinstance(obj.name, str) and obj.name


def test_distribution_is_nn_module():
    """BaseDistribution and its subclasses are torch.nn.Module instances."""
    import torch.nn as nn

    from deeptab.distributions import NormalDistribution

    obj = NormalDistribution()
    assert isinstance(obj, nn.Module)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_all_families():
    from deeptab.distributions import DISTRIBUTION_REGISTRY

    expected_keys = {
        "normal",
        "lognormal",
        "poisson",
        "zip",
        "gamma",
        "inversegamma",
        "beta",
        "dirichlet",
        "studentt",
        "johnsonsu",
        "negativebinom",
        "categorical",
        "multinomial",
        "quantile",
        "tweedie",
        "mog",
    }
    assert expected_keys == set(DISTRIBUTION_REGISTRY.keys())


def test_get_distribution_unknown_raises():
    from deeptab.distributions import get_distribution

    with pytest.raises(ValueError, match="Unknown distribution family"):
        get_distribution("not_a_family")


# ---------------------------------------------------------------------------
# LogNormal
# ---------------------------------------------------------------------------


class TestLogNormalDistribution:
    def setup_method(self):
        import torch

        self.torch = torch
        from deeptab.distributions import LogNormalDistribution

        self.dist = LogNormalDistribution()
        self.B = 16
        # targets must be strictly positive for log-normal
        self.y = torch.abs(torch.randn(self.B)) + 0.1
        self.preds = torch.randn(self.B, 2)

    def test_param_count(self):
        assert self.dist.parameter_count == 2

    def test_name(self):
        assert self.dist.name == "LogNormal"

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0
        assert loss.item() == loss.item()  # not NaN

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        loss = self.dist.compute_loss(preds, self.y)
        loss.backward()
        assert preds.grad is not None

    def test_evaluate_nll_keys(self):
        metrics = self.dist.evaluate_nll(self.y.numpy(), self.preds.detach().numpy())
        for key in ("NLL", "mse", "mae", "rmse"):
            assert key in metrics


# ---------------------------------------------------------------------------
# ZeroInflatedPoisson
# ---------------------------------------------------------------------------


class TestZeroInflatedPoissonDistribution:
    def setup_method(self):
        import torch

        self.torch = torch
        from deeptab.distributions import ZeroInflatedPoissonDistribution

        self.dist = ZeroInflatedPoissonDistribution()
        self.B = 16
        # count data with some zeros
        self.y = torch.randint(0, 6, (self.B,)).float()
        self.preds = torch.randn(self.B, 2)

    def test_param_count(self):
        assert self.dist.parameter_count == 2

    def test_name(self):
        assert self.dist.name == "ZeroInflatedPoisson"

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0
        assert loss.item() == loss.item()

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        loss = self.dist.compute_loss(preds, self.y)
        loss.backward()
        assert preds.grad is not None

    def test_all_zeros_target(self):
        """Loss must be finite even when all targets are zero."""
        y_zeros = self.torch.zeros(self.B)
        loss = self.dist.compute_loss(self.preds, y_zeros)
        assert loss.isfinite()

    def test_evaluate_nll_keys(self):
        metrics = self.dist.evaluate_nll(self.y.numpy(), self.preds.detach().numpy())
        for key in ("NLL", "mse", "mae", "rmse"):
            assert key in metrics


# ---------------------------------------------------------------------------
# Tweedie
# ---------------------------------------------------------------------------


class TestTweedieDistribution:
    def setup_method(self):
        import torch

        self.torch = torch
        from deeptab.distributions import TweedieDistribution

        self.dist = TweedieDistribution(p=1.5)
        self.B = 16
        # Tweedie targets are non-negative (mix of zeros and positives)
        self.y = torch.abs(torch.randn(self.B))
        self.preds = torch.randn(self.B, 1)

    def test_param_count(self):
        assert self.dist.parameter_count == 1

    def test_name(self):
        assert self.dist.name == "Tweedie"

    def test_invalid_p_raises(self):
        from deeptab.distributions import TweedieDistribution

        with pytest.raises(ValueError, match="power p must be in"):
            TweedieDistribution(p=0.5)
        with pytest.raises(ValueError, match="power p must be in"):
            TweedieDistribution(p=2.0)

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0
        assert loss.isfinite()

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        loss = self.dist.compute_loss(preds, self.y)
        loss.backward()
        assert preds.grad is not None

    def test_evaluate_nll_keys(self):
        metrics = self.dist.evaluate_nll(self.y.numpy(), self.preds.detach().numpy())
        for key in ("NLL", "mse", "mae", "rmse", "tweedie_deviance"):
            assert key in metrics

    @pytest.mark.parametrize("p", [1.1, 1.5, 1.9])
    def test_various_p_values(self, p):
        from deeptab.distributions import TweedieDistribution

        d = TweedieDistribution(p=p)
        loss = d.compute_loss(self.preds, self.y)
        assert loss.isfinite()


# ---------------------------------------------------------------------------
# Multinomial
# ---------------------------------------------------------------------------


class TestMultinomialDistribution:
    def setup_method(self):
        import torch

        self.torch = torch
        from deeptab.distributions import MultinomialDistribution

        self.K = 3
        self.dist = MultinomialDistribution(num_classes=self.K)
        self.B = 16
        # one-hot vectors that sum to total_count=1
        idx = torch.randint(0, self.K, (self.B,))
        self.y = torch.zeros(self.B, self.K)
        self.y[torch.arange(self.B), idx] = 1.0
        self.preds = torch.randn(self.B, self.K)

    def test_param_count(self):
        assert self.dist.parameter_count == self.K

    def test_name(self):
        assert self.dist.name == "Multinomial"

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0
        assert loss.isfinite()

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        loss = self.dist.compute_loss(preds, self.y)
        loss.backward()
        assert preds.grad is not None

    def test_param_count_scales_with_num_classes(self):
        from deeptab.distributions import MultinomialDistribution

        for K in (2, 5, 10):
            d = MultinomialDistribution(num_classes=K)
            assert d.parameter_count == K


# ---------------------------------------------------------------------------
# MixtureOfGaussians
# ---------------------------------------------------------------------------


class TestMixtureOfGaussiansDistribution:
    def setup_method(self):
        import torch

        self.torch = torch
        from deeptab.distributions import MixtureOfGaussiansDistribution

        self.K = 3
        self.dist = MixtureOfGaussiansDistribution(n_components=self.K)
        self.B = 16
        self.y = torch.randn(self.B)
        self.preds = torch.randn(self.B, 3 * self.K)

    def test_param_count(self):
        assert self.dist.parameter_count == 3 * self.K

    def test_name(self):
        assert self.dist.name == "MixtureOfGaussians"

    def test_invalid_n_components_raises(self):
        from deeptab.distributions import MixtureOfGaussiansDistribution

        with pytest.raises(ValueError, match="n_components must be"):
            MixtureOfGaussiansDistribution(n_components=0)

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0
        assert loss.isfinite()

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        loss = self.dist.compute_loss(preds, self.y)
        loss.backward()
        assert preds.grad is not None

    def test_evaluate_nll_keys(self):
        metrics = self.dist.evaluate_nll(self.y.numpy(), self.preds.detach().numpy())
        for key in ("NLL", "mse", "mae", "rmse"):
            assert key in metrics

    @pytest.mark.parametrize("K", [1, 2, 5])
    def test_various_component_counts(self, K):
        from deeptab.distributions import MixtureOfGaussiansDistribution

        d = MixtureOfGaussiansDistribution(n_components=K)
        assert d.parameter_count == 3 * K
        loss = d.compute_loss(self.torch.randn(self.B, 3 * K), self.y)
        assert loss.isfinite()


# ---------------------------------------------------------------------------
# NormalDistribution
# ---------------------------------------------------------------------------


class TestNormalDistribution:
    def setup_method(self):
        import torch

        from deeptab.distributions import NormalDistribution

        self.dist = NormalDistribution()
        self.B = 16
        self.y = torch.randn(self.B)
        self.preds = torch.randn(self.B, 2)

    def test_param_count(self):
        assert self.dist.parameter_count == 2

    def test_name(self):
        assert self.dist.name == "Normal"

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0 and loss.isfinite()

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        self.dist.compute_loss(preds, self.y).backward()
        assert preds.grad is not None

    def test_evaluate_nll_keys(self):
        m = self.dist.evaluate_nll(self.y.numpy(), self.preds.detach().numpy())
        for k in ("NLL", "mse", "mae", "rmse"):
            assert k in m


# ---------------------------------------------------------------------------
# PoissonDistribution
# ---------------------------------------------------------------------------


class TestPoissonDistribution:
    def setup_method(self):
        import torch

        from deeptab.distributions import PoissonDistribution

        self.dist = PoissonDistribution()
        self.B = 16
        self.y = torch.randint(0, 10, (self.B,)).float()
        self.preds = torch.randn(self.B, 1)

    def test_param_count(self):
        assert self.dist.parameter_count == 1

    def test_name(self):
        assert self.dist.name == "Poisson"

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0 and loss.isfinite()

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        self.dist.compute_loss(preds, self.y).backward()
        assert preds.grad is not None

    def test_evaluate_nll_keys(self):
        m = self.dist.evaluate_nll(self.y.numpy(), self.preds.detach().numpy())
        for k in ("NLL", "mse", "mae", "rmse", "poisson_deviance"):
            assert k in m


# ---------------------------------------------------------------------------
# GammaDistribution
# ---------------------------------------------------------------------------


class TestGammaDistribution:
    def setup_method(self):
        import torch

        from deeptab.distributions import GammaDistribution

        self.dist = GammaDistribution()
        self.B = 16
        self.y = torch.abs(torch.randn(self.B)) + 0.1  # strictly positive
        self.preds = torch.randn(self.B, 2)

    def test_param_count(self):
        assert self.dist.parameter_count == 2

    def test_name(self):
        assert self.dist.name == "Gamma"

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0 and loss.isfinite()

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        self.dist.compute_loss(preds, self.y).backward()
        assert preds.grad is not None

    def test_evaluate_nll_returns_nll(self):
        m = self.dist.evaluate_nll(self.y.numpy(), self.preds.detach().numpy())
        assert "NLL" in m


# ---------------------------------------------------------------------------
# InverseGammaDistribution
# ---------------------------------------------------------------------------


class TestInverseGammaDistribution:
    def setup_method(self):
        import torch

        from deeptab.distributions import InverseGammaDistribution

        self.dist = InverseGammaDistribution()
        self.B = 16
        self.y = torch.abs(torch.randn(self.B)) + 0.1
        self.preds = torch.randn(self.B, 2)

    def test_param_count(self):
        assert self.dist.parameter_count == 2

    def test_name(self):
        assert self.dist.name == "InverseGamma"

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0 and loss.isfinite()

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        self.dist.compute_loss(preds, self.y).backward()
        assert preds.grad is not None


# ---------------------------------------------------------------------------
# BetaDistribution
# ---------------------------------------------------------------------------


class TestBetaDistribution:
    def setup_method(self):
        import torch

        from deeptab.distributions import BetaDistribution

        self.dist = BetaDistribution()
        self.B = 16
        # targets must be strictly in (0, 1)
        self.y = torch.sigmoid(torch.randn(self.B)).clamp(1e-3, 1 - 1e-3)
        self.preds = torch.randn(self.B, 2)

    def test_param_count(self):
        assert self.dist.parameter_count == 2

    def test_name(self):
        assert self.dist.name == "Beta"

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0 and loss.isfinite()

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        self.dist.compute_loss(preds, self.y).backward()
        assert preds.grad is not None


# ---------------------------------------------------------------------------
# DirichletDistribution
# ---------------------------------------------------------------------------


class TestDirichletDistribution:
    def setup_method(self):
        import torch

        from deeptab.distributions import DirichletDistribution

        self.K = 3
        self.dist = DirichletDistribution()
        self.B = 16
        # targets must lie on the K-simplex (rows sum to 1, all > 0)
        self.y = torch.softmax(torch.randn(self.B, self.K), dim=-1)
        self.preds = torch.randn(self.B, self.K)

    def test_param_count(self):
        assert self.dist.parameter_count == 1

    def test_name(self):
        assert self.dist.name == "Dirichlet"

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0 and loss.isfinite()

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        self.dist.compute_loss(preds, self.y).backward()
        assert preds.grad is not None


# ---------------------------------------------------------------------------
# NegativeBinomialDistribution
# ---------------------------------------------------------------------------


class TestNegativeBinomialDistribution:
    def setup_method(self):
        import torch

        from deeptab.distributions import NegativeBinomialDistribution

        self.dist = NegativeBinomialDistribution()
        self.B = 16
        self.y = torch.randint(0, 10, (self.B,)).float()
        self.preds = torch.randn(self.B, 2)

    def test_param_count(self):
        assert self.dist.parameter_count == 2

    def test_name(self):
        assert self.dist.name == "NegativeBinomial"

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0 and loss.isfinite()

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        self.dist.compute_loss(preds, self.y).backward()
        assert preds.grad is not None


# ---------------------------------------------------------------------------
# StudentTDistribution
# ---------------------------------------------------------------------------


class TestStudentTDistribution:
    def setup_method(self):
        import torch

        from deeptab.distributions import StudentTDistribution

        self.dist = StudentTDistribution()
        self.B = 16
        self.y = torch.randn(self.B)
        self.preds = torch.randn(self.B, 3)

    def test_param_count(self):
        assert self.dist.parameter_count == 3

    def test_name(self):
        assert self.dist.name == "StudentT"

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0 and loss.isfinite()

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        self.dist.compute_loss(preds, self.y).backward()
        assert preds.grad is not None

    def test_evaluate_nll_keys(self):
        m = self.dist.evaluate_nll(self.y.numpy(), self.preds.detach().numpy())
        for k in ("NLL", "mse", "mae", "rmse"):
            assert k in m


# ---------------------------------------------------------------------------
# JohnsonSuDistribution
# ---------------------------------------------------------------------------


class TestJohnsonSuDistribution:
    def setup_method(self):
        import torch

        from deeptab.distributions import JohnsonSuDistribution

        self.dist = JohnsonSuDistribution()
        self.B = 16
        self.y = torch.randn(self.B)
        self.preds = torch.randn(self.B, 4)

    def test_param_count(self):
        assert self.dist.parameter_count == 4

    def test_name(self):
        assert self.dist.name == "JohnsonSu"

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0 and loss.isfinite()

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        self.dist.compute_loss(preds, self.y).backward()
        assert preds.grad is not None

    def test_evaluate_nll_keys(self):
        m = self.dist.evaluate_nll(self.y.numpy(), self.preds.detach().numpy())
        for k in ("NLL", "mse", "mae", "rmse"):
            assert k in m


# ---------------------------------------------------------------------------
# CategoricalDistribution
# ---------------------------------------------------------------------------


class TestCategoricalDistribution:
    def setup_method(self):
        import torch

        from deeptab.distributions import CategoricalDistribution

        self.K = 4
        self.dist = CategoricalDistribution()
        self.B = 16
        self.y = torch.randint(0, self.K, (self.B,))  # integer class indices
        self.preds = torch.randn(self.B, self.K)

    def test_param_count(self):
        assert self.dist.parameter_count == 1

    def test_name(self):
        assert self.dist.name == "Categorical"

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0 and loss.isfinite()

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        self.dist.compute_loss(preds, self.y).backward()
        assert preds.grad is not None


# ---------------------------------------------------------------------------
# Quantile
# ---------------------------------------------------------------------------


class TestQuantile:
    def setup_method(self):
        import torch

        from deeptab.distributions import Quantile

        self.quantiles = [0.1, 0.5, 0.9]
        self.dist = Quantile(quantiles=self.quantiles)
        self.B = 16
        self.y = torch.randn(self.B)
        self.preds = torch.randn(self.B, len(self.quantiles))

    def test_param_count(self):
        assert self.dist.parameter_count == len(self.quantiles)

    def test_default_param_count(self):
        from deeptab.distributions import Quantile

        assert Quantile().parameter_count == 3  # default [0.25, 0.5, 0.75]

    def test_name(self):
        assert self.dist.name == "Quantile"

    def test_compute_loss_scalar(self):
        loss = self.dist.compute_loss(self.preds, self.y)
        assert loss.ndim == 0 and loss.isfinite()

    def test_loss_requires_grad(self):
        preds = self.preds.requires_grad_(True)
        self.dist.compute_loss(preds, self.y).backward()
        assert preds.grad is not None

    def test_y_true_requires_grad_raises(self):
        import torch

        y_grad = torch.randn(self.B, requires_grad=True)
        with pytest.raises(ValueError, match="y_true should not require"):
            self.dist.compute_loss(self.preds, y_grad)

    def test_batch_size_mismatch_raises(self):
        import torch

        with pytest.raises(ValueError, match="Batch size"):
            self.dist.compute_loss(self.preds, torch.randn(self.B + 1))
