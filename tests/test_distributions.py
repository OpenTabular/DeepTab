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
    "NegativeBinomialDistribution",
    "NormalDistribution",
    "PoissonDistribution",
    "Quantile",
    "StudentTDistribution",
]

# Concrete (instantiable-with-no-args) classes and their expected parameter counts
CONCRETE_NO_ARGS = [
    ("NormalDistribution", 2),
    ("BetaDistribution", 2),
    ("PoissonDistribution", 1),
    ("GammaDistribution", 2),
    ("StudentTDistribution", 3),
    ("NegativeBinomialDistribution", 2),
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


def test_quantile_parameter_count():
    """Quantile distribution reports parameter_count == len(quantiles)."""
    from deeptab.distributions import Quantile

    q = Quantile(quantiles=[0.1, 0.5, 0.9])
    assert q.parameter_count == 3


def test_distribution_is_nn_module():
    """BaseDistribution and its subclasses are torch.nn.Module instances."""
    import torch.nn as nn

    from deeptab.distributions import NormalDistribution

    obj = NormalDistribution()
    assert isinstance(obj, nn.Module)
