"""Distribution registry: maps family name strings to distribution classes."""

from __future__ import annotations

from .base import BaseDistribution
from .beta import BetaDistribution, DirichletDistribution
from .categorical import CategoricalDistribution, MultinomialDistribution, Quantile
from .gamma import GammaDistribution, InverseGammaDistribution
from .mixture import MixtureOfGaussiansDistribution
from .negative_binomial import NegativeBinomialDistribution
from .normal import LogNormalDistribution, NormalDistribution
from .poisson import PoissonDistribution, ZeroInflatedPoissonDistribution
from .student_t import JohnsonSuDistribution, StudentTDistribution
from .tweedie import TweedieDistribution

DISTRIBUTION_REGISTRY: dict[str, type[BaseDistribution]] = {
    "normal": NormalDistribution,
    "lognormal": LogNormalDistribution,
    "poisson": PoissonDistribution,
    "zip": ZeroInflatedPoissonDistribution,
    "gamma": GammaDistribution,
    "inversegamma": InverseGammaDistribution,
    "beta": BetaDistribution,
    "dirichlet": DirichletDistribution,
    "studentt": StudentTDistribution,
    "johnsonsu": JohnsonSuDistribution,
    "negativebinom": NegativeBinomialDistribution,
    "categorical": CategoricalDistribution,
    "multinomial": MultinomialDistribution,
    "quantile": Quantile,
    "tweedie": TweedieDistribution,
    "mog": MixtureOfGaussiansDistribution,
}


def get_distribution(family: str, **kwargs: object) -> BaseDistribution:
    """Instantiate a distribution by its registry name.

    Parameters
    ----------
    family : str
        The distribution family key (e.g. ``"normal"``, ``"gamma"``).
    **kwargs
        Extra keyword arguments forwarded to the distribution constructor
        (e.g. ``quantiles=[0.1, 0.5, 0.9]`` for ``"quantile"``).

    Returns
    -------
    BaseDistribution
        A ready-to-use distribution instance.

    Raises
    ------
    ValueError
        If *family* is not a registered key.
    """
    if family not in DISTRIBUTION_REGISTRY:
        available = sorted(DISTRIBUTION_REGISTRY)
        raise ValueError(f"Unknown distribution family '{family}'. Available families: {available}")
    return DISTRIBUTION_REGISTRY[family](**kwargs)  # type: ignore[call-arg]
