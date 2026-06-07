"""Poisson distribution for count data LSS models."""

import numpy as np
import torch
import torch.distributions as dist

from .base import BaseDistribution


class PoissonDistribution(BaseDistribution):
    """
    Represents a Poisson distribution, typically used for modeling count data or the number of events
    occurring within a fixed interval of time or space. This class extends the BaseDistribution and
    includes parameter transformation and loss computation specific to the Poisson distribution.

    Parameters
    ----------
        name (str): The name of the distribution, defaulted to "Poisson".
        rate_transform (str or callable): Transformation to apply to the rate parameter
        to ensure it remains positive.
    """

    def __init__(self, name="Poisson", rate_transform="positive"):
        param_names = ["rate"]
        super().__init__(name, param_names)
        self.rate_transform = self.get_transform(rate_transform)

    def compute_loss(self, predictions, y_true):
        rate = self.rate_transform(predictions[:, self.param_names.index("rate")])

        poisson_dist = dist.Poisson(rate)

        nll = -poisson_dist.log_prob(y_true).mean()
        return nll

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)

        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
        rate = self.rate_transform(y_pred_tensor[:, self.param_names.index("rate")])

        mse_loss = torch.nn.functional.mse_loss(y_true_tensor, rate)  # type: ignore
        rmse = np.sqrt(mse_loss.detach().numpy())
        mae = (
            torch.nn.functional.l1_loss(y_true_tensor, rate)  # type: ignore
            .detach()
            .numpy()  # type: ignore
        )  # type: ignore
        poisson_deviance = 2 * torch.sum(y_true_tensor * torch.log(y_true_tensor / rate) - (y_true_tensor - rate))  # type: ignore[operator]

        metrics["mse"] = mse_loss.detach().numpy()
        metrics["mae"] = mae
        metrics["rmse"] = rmse
        metrics["poisson_deviance"] = poisson_deviance.detach().numpy()

        return metrics
