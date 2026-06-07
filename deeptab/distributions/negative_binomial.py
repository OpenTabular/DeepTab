"""Negative Binomial distribution for overdispersed count LSS models."""

import torch
import torch.distributions as dist

from .base import BaseDistribution


class NegativeBinomialDistribution(BaseDistribution):
    """
    Represents a Negative Binomial distribution, often used for count data and modeling the number
    of failures before a specified number of successes occurs in a series of Bernoulli trials.
    This class extends BaseDistribution and includes parameter transformation and loss computation
    specific to the Negative Binomial distribution.

    Parameters
    ----------
        name (str): The name of the distribution, defaulted to "NegativeBinomial".
        mean_transform (str or callable): Transformation for the mean parameter to ensure it remains positive.
        dispersion_transform (str or callable): Transformation for the dispersion parameter to
        ensure it remains positive.
    """

    def __init__(
        self,
        name="NegativeBinomial",
        mean_transform="positive",
        dispersion_transform="positive",
    ):
        param_names = ["mean", "dispersion"]
        super().__init__(name, param_names)

        self.mean_transform = self.get_transform(mean_transform)
        self.dispersion_transform = self.get_transform(dispersion_transform)

    def compute_loss(self, predictions, y_true):
        mean = self.mean_transform(predictions[:, self.param_names.index("mean")])
        dispersion = self.dispersion_transform(predictions[:, self.param_names.index("dispersion")])

        # variance = mean + mean^2 / dispersion
        r = torch.tensor(1.0) / dispersion  # type: ignore[operator]
        p = r / (r + mean)

        negative_binomial_dist = dist.NegativeBinomial(total_count=r, probs=p)

        nll = -negative_binomial_dist.log_prob(y_true).mean()
        return nll
