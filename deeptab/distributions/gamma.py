"""Gamma and Inverse-Gamma distributions for positive continuous LSS models."""

import torch
import torch.distributions as dist

from .base import BaseDistribution


class GammaDistribution(BaseDistribution):
    """
    Represents a Gamma distribution, a two-parameter family of continuous probability distributions. It's
    widely used in various fields of science for modeling a wide range of phenomena. This class extends
    BaseDistribution and includes parameter transformation and loss computation specific to
    the Gamma distribution.

    Parameters
    ----------
        name (str): The name of the distribution, defaulted to "Gamma".
        shape_transform (str or callable): Transformation for the shape parameter to ensure it remains positive.
        rate_transform (str or callable): Transformation for the rate parameter to ensure it remains positive.
    """

    def __init__(self, name="Gamma", shape_transform="positive", rate_transform="positive"):
        param_names = ["shape", "rate"]
        super().__init__(name, param_names)

        self.shape_transform = self.get_transform(shape_transform)
        self.rate_transform = self.get_transform(rate_transform)

    def compute_loss(self, predictions, y_true):
        shape = self.shape_transform(predictions[:, self.param_names.index("shape")])
        rate = self.rate_transform(predictions[:, self.param_names.index("rate")])

        gamma_dist = dist.Gamma(shape, rate)

        nll = -gamma_dist.log_prob(y_true).mean()
        return nll


class InverseGammaDistribution(BaseDistribution):
    """
    Represents an Inverse Gamma distribution, often used as a prior distribution in Bayesian statistics,
    especially for scale parameters in other distributions. This class extends BaseDistribution and includes
    parameter transformation and loss computation specific to the Inverse Gamma distribution.

    Parameters
    ----------
        name (str): The name of the distribution, defaulted to "InverseGamma".
        shape_transform (str or callable): Transformation for the shape parameter to
        ensure it remains positive.
        scale_transform (str or callable): Transformation for the scale parameter to
        ensure it remains positive.
    """

    def __init__(
        self,
        name="InverseGamma",
        shape_transform="positive",
        scale_transform="positive",
    ):
        param_names = [
            "shape",
            "scale",
        ]
        super().__init__(name, param_names)

        self.shape_transform = self.get_transform(shape_transform)
        self.scale_transform = self.get_transform(scale_transform)

    def compute_loss(self, predictions, y_true):
        shape = self.shape_transform(predictions[:, self.param_names.index("shape")])
        scale = self.scale_transform(predictions[:, self.param_names.index("scale")])

        inverse_gamma_dist = dist.InverseGamma(shape, scale)
        nll = -inverse_gamma_dist.log_prob(y_true).mean()
        return nll
