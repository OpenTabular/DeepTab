"""Normal (Gaussian) and Log-Normal distributions for LSS models."""

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


class LogNormalDistribution(BaseDistribution):
    """
    Represents a Log-Normal distribution for right-skewed positive continuous targets
    such as wages, prices, latencies, and insurance claim amounts.

    The neural network predicts the mean (``loc``) and standard deviation (``scale``) of
    the underlying normal distribution in log-space.  The median of the outcome is
    ``exp(loc)`` and the mean is ``exp(loc + scale²/2)``.

    Parameters
    ----------
        name (str): The name of the distribution. Defaults to ``"LogNormal"``.
        loc_transform (str or callable): Transform for the log-space mean. Defaults to
            ``"none"`` (identity — mean in log-space can be any real number).
        scale_transform (str or callable): Transform for the log-space standard deviation.
            Defaults to ``"positive"`` (softplus, ensures sigma > 0).
    """

    def __init__(self, name="LogNormal", loc_transform="none", scale_transform="positive"):
        param_names = ["loc", "scale"]
        super().__init__(name, param_names)

        self.loc_transform = self.get_transform(loc_transform)
        self.scale_transform = self.get_transform(scale_transform)

    def compute_loss(self, predictions, y_true):
        loc = self.loc_transform(predictions[:, self.param_names.index("loc")])
        scale = self.scale_transform(predictions[:, self.param_names.index("scale")])

        lognormal_dist = dist.LogNormal(loc, scale)
        nll = -lognormal_dist.log_prob(y_true).mean()
        return nll

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)

        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        # Median prediction = exp(loc) — a natural point estimate for log-normal
        loc = self.loc_transform(y_pred_tensor[:, self.param_names.index("loc")])
        median_pred = torch.exp(loc)

        mse_loss = torch.nn.functional.mse_loss(y_true_tensor, median_pred)
        rmse = np.sqrt(mse_loss.detach().numpy())
        mae = torch.nn.functional.l1_loss(y_true_tensor, median_pred).detach().numpy()

        metrics["mse"] = mse_loss.detach().numpy()
        metrics["mae"] = mae
        metrics["rmse"] = rmse

        return metrics
