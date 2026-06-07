"""Normal (Gaussian) distribution for LSS models."""

from collections.abc import Callable

import numpy as np
import torch
import torch.distributions as dist

from .base import BaseDistribution


class NormalDistribution(BaseDistribution):
    """
    Represents a Normal (Gaussian) distribution with parameters for mean and variance,
    including functionality for transforming these parameters and computing the loss.

    Inherits from BaseDistribution.

    Parameters
    ----------
        name (str): The name of the distribution. Defaults to "Normal".
        mean_transform (str or callable): The transformation for the mean parameter.
        Defaults to "none".
        var_transform (str or callable): The transformation for the variance parameter.
        Defaults to "positive".
    """

    def __init__(self, name="Normal", mean_transform="none", var_transform="positive"):
        param_names = [
            "mean",
            "variance",
        ]
        super().__init__(name, param_names)

        self.mean_transform = self.get_transform(mean_transform)
        self.variance_transform = self.get_transform(var_transform)

    def compute_loss(self, predictions, y_true):
        mean = self.mean_transform(predictions[:, self.param_names.index("mean")])
        variance = self.variance_transform(predictions[:, self.param_names.index("variance")])

        normal_dist = dist.Normal(mean, variance)

        nll = -normal_dist.log_prob(y_true).mean()
        return nll

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)

        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        mse_loss = torch.nn.functional.mse_loss(y_true_tensor, y_pred_tensor[:, self.param_names.index("mean")])
        rmse = np.sqrt(mse_loss.detach().numpy())
        mae = (
            torch.nn.functional.l1_loss(y_true_tensor, y_pred_tensor[:, self.param_names.index("mean")])
            .detach()
            .numpy()
        )

        metrics["mse"] = mse_loss.detach().numpy()
        metrics["mae"] = mae
        metrics["rmse"] = rmse

        return metrics
