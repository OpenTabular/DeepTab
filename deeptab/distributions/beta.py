"""Beta and Dirichlet distributions for bounded / compositional LSS models."""

import torch
import torch.distributions as dist

from .base import BaseDistribution


class BetaDistribution(BaseDistribution):
    """
    Represents a Beta distribution, a continuous distribution defined on the interval [0, 1], commonly used
    in Bayesian statistics for modeling probabilities. This class extends BaseDistribution and includes parameter
    transformation and loss computation specific to the Beta distribution.

    Parameters
    ----------
        name (str): The name of the distribution, defaulted to "Beta".
        shape_transform (str or callable): Transformation for the alpha (shape) parameter to ensure
        it remains positive.
        scale_transform (str or callable): Transformation for the beta (scale) parameter to ensure
        it remains positive.
    """

    def __init__(
        self,
        name="Beta",
        shape_transform="positive",
        scale_transform="positive",
    ):
        param_names = [
            "alpha",
            "beta",
        ]
        super().__init__(name, param_names)

        self.alpha_transform = self.get_transform(shape_transform)
        self.beta_transform = self.get_transform(scale_transform)

    def compute_loss(self, predictions, y_true):
        alpha = self.alpha_transform(predictions[:, self.param_names.index("alpha")])
        beta = self.beta_transform(predictions[:, self.param_names.index("beta")])

        beta_dist = dist.Beta(alpha, beta)
        nll = -beta_dist.log_prob(y_true).mean()
        return nll


class DirichletDistribution(BaseDistribution):
    """
    Represents a Dirichlet distribution, a multivariate generalization of the Beta distribution. It is commonly
    used in Bayesian statistics for modeling multinomial distribution probabilities. This class extends
    BaseDistribution and includes parameter transformation and loss computation
    specific to the Dirichlet distribution.

    Parameters
    ----------
        name (str): The name of the distribution, defaulted to "Dirichlet".
        concentration_transform (str or callable): Transformation to apply to
        concentration parameters to ensure they remain positive.
    """

    def __init__(self, name="Dirichlet", concentration_transform="positive"):
        param_names = ["concentration"]
        super().__init__(name, param_names)
        self.concentration_transform = self.get_transform(concentration_transform)

    def compute_loss(self, predictions, y_true):
        concentration = self.concentration_transform(predictions)

        dirichlet_dist = dist.Dirichlet(concentration)

        nll = -dirichlet_dist.log_prob(y_true).mean()
        return nll
