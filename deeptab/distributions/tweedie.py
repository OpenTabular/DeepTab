"""Tweedie distribution for zero-plus-positive compound targets (insurance, rainfall)."""

import numpy as np
import torch

from .base import BaseDistribution


class TweedieDistribution(BaseDistribution):
    """
    Represents a Tweedie distribution for targets that are a mixture of zeros and
    positive continuous values — common in insurance claims, rainfall totals, and
    sales volumes.

    The Tweedie family unifies several distributions through a single *power* parameter
    ``p``:

    * ``p = 0`` — Normal
    * ``p = 1`` — Poisson (integer counts)
    * ``1 < p < 2`` — compound Poisson-Gamma (**this class targets this range**)
    * ``p = 2`` — Gamma

    The neural network predicts only the mean ``mu > 0``.  The power ``p`` and
    dispersion ``phi`` are fixed hyperparameters set at construction time.

    The loss is the **Tweedie log-likelihood** (terms not depending on ``mu`` are
    dropped), which is equivalent to minimising the Tweedie deviance:

    .. math::

        \\mathcal{L} = \\frac{\\mu^{2-p}}{2-p} - \\frac{y \\cdot \\mu^{1-p}}{1-p}

    Parameters
    ----------
        name (str): Defaults to ``"Tweedie"``.
        p (float): Tweedie power parameter.  Must satisfy ``1 < p < 2``.
            Defaults to ``1.5`` (midpoint of the compound Poisson-Gamma range).
        mu_transform (str or callable): Transform for the mean prediction to ensure
            ``mu > 0``.  Defaults to ``"positive"`` (softplus).
    """

    def __init__(
        self,
        name="Tweedie",
        p: float = 1.5,
        mu_transform="positive",
    ):
        if not (1.0 < p < 2.0):
            raise ValueError(
                f"Tweedie power p must be in the open interval (1, 2) for the compound Poisson-Gamma family, got p={p}."
            )
        param_names = ["mu"]
        super().__init__(name, param_names)

        self.p = p
        self.mu_transform = self.get_transform(mu_transform)

    def compute_loss(self, predictions, y_true):
        mu = self.mu_transform(predictions[:, self.param_names.index("mu")])
        p = self.p

        # Tweedie log-likelihood (ignoring terms that do not depend on mu)
        # L = mu^(2-p)/(2-p) - y * mu^(1-p)/(1-p)
        term_a = torch.pow(mu, 2.0 - p) / (2.0 - p)
        term_b = y_true * torch.pow(mu, 1.0 - p) / (1.0 - p)
        loss = torch.mean(term_a - term_b)
        return loss

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)

        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        mu = self.mu_transform(y_pred_tensor[:, self.param_names.index("mu")])

        # Tweedie deviance: D(y, mu) = 2 * [y^(2-p)/((1-p)(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p)]
        p = self.p
        d = 2.0 * (
            torch.pow(y_true_tensor.clamp(min=1e-8), 2.0 - p) / ((1.0 - p) * (2.0 - p))
            - y_true_tensor * torch.pow(mu, 1.0 - p) / (1.0 - p)
            + torch.pow(mu, 2.0 - p) / (2.0 - p)
        )
        metrics["tweedie_deviance"] = d.mean().detach().numpy()

        mse_loss = torch.nn.functional.mse_loss(y_true_tensor, mu)
        rmse = np.sqrt(mse_loss.detach().numpy())
        mae = torch.nn.functional.l1_loss(y_true_tensor, mu).detach().numpy()

        metrics["mse"] = mse_loss.detach().numpy()
        metrics["mae"] = mae
        metrics["rmse"] = rmse

        return metrics
