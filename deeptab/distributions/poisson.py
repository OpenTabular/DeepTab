"""Poisson and Zero-Inflated Poisson distributions for count data LSS models."""

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


class ZeroInflatedPoissonDistribution(BaseDistribution):
    """
    Represents a Zero-Inflated Poisson (ZIP) distribution for count data with
    excess zeros (e.g. number of insurance claims, rare-event counts).

    The model outputs two parameters:

    * **pi** — zero-inflation probability π ∈ (0, 1).  Extra zeros arise with
      probability pi; with probability (1 - pi) the count follows Poisson(rate).
    * **rate** — Poisson rate λ > 0.

    The mixture probability mass function is:

    .. math::

        P(Y = 0)   &= \\pi + (1 - \\pi)\\,e^{-\\lambda} \\\\
        P(Y = k>0) &= (1 - \\pi)\\,\\text{Poisson}(k;\\,\\lambda)

    Parameters
    ----------
        name (str): Defaults to ``"ZeroInflatedPoisson"``.
        pi_transform (str or callable): Transform for the inflation probability.
            Defaults to ``"sigmoid"`` to map logits → (0, 1).
        rate_transform (str or callable): Transform for the Poisson rate.
            Defaults to ``"positive"`` (softplus).
    """

    def __init__(
        self,
        name="ZeroInflatedPoisson",
        pi_transform="sigmoid",
        rate_transform="positive",
    ):
        param_names = ["pi", "rate"]
        super().__init__(name, param_names)

        self.pi_transform = self.get_transform(pi_transform)
        self.rate_transform = self.get_transform(rate_transform)

    def compute_loss(self, predictions, y_true):
        pi = self.pi_transform(predictions[:, self.param_names.index("pi")])
        rate = self.rate_transform(predictions[:, self.param_names.index("rate")])

        # log P(Y=0) = log(pi + (1-pi)*exp(-rate))
        log_zero = torch.log(pi + (1.0 - pi) * torch.exp(-rate) + 1e-8)
        # log P(Y=k>0) = log(1-pi) + Poisson log-prob
        log_nonzero = torch.log(1.0 - pi + 1e-8) + dist.Poisson(rate).log_prob(y_true)

        log_prob = torch.where(y_true == 0, log_zero, log_nonzero)
        nll = -log_prob.mean()
        return nll

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)

        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        pi = self.pi_transform(y_pred_tensor[:, self.param_names.index("pi")])
        rate = self.rate_transform(y_pred_tensor[:, self.param_names.index("rate")])

        # E[Y] = (1 - pi) * rate
        mean_pred = (1.0 - pi) * rate

        mse_loss = torch.nn.functional.mse_loss(y_true_tensor, mean_pred)
        rmse = np.sqrt(mse_loss.detach().numpy())
        mae = torch.nn.functional.l1_loss(y_true_tensor, mean_pred).detach().numpy()

        metrics["mse"] = mse_loss.detach().numpy()
        metrics["mae"] = mae
        metrics["rmse"] = rmse

        return metrics
