"""Mixture of Gaussians distribution for multimodal continuous targets."""

import numpy as np
import torch
import torch.distributions as dist

from .base import BaseDistribution


class MixtureOfGaussiansDistribution(BaseDistribution):
    """
    Represents a Mixture of Gaussians (MoG) distribution for multimodal continuous
    targets (e.g. bimodal price distributions, multi-cluster outcomes).

    The neural network outputs ``3 * n_components`` values:

    * **n_components mixing logits** → softmax → weights ``w_k``
    * **n_components means** (``mu_k``, unconstrained)
    * **n_components log-scales** → softplus → standard deviations ``sigma_k``

    The log-likelihood uses the log-sum-exp trick for numerical stability:

    .. math::

        \\log p(y) = \\text{logsumexp}_k\\bigl[\\log w_k +
                     \\log \\mathcal{N}(y;\\,\\mu_k,\\,\\sigma_k)\\bigr]

    Parameters
    ----------
        name (str): Defaults to ``"MixtureOfGaussians"``.
        n_components (int): Number of Gaussian components ``K``.  Defaults to ``3``.
            Sets ``param_count = 3 * K``.
    """

    def __init__(self, name="MixtureOfGaussians", n_components: int = 3):
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}.")
        self.n_components = n_components
        K = n_components
        # Layout: [w_0..w_{K-1}, mu_0..mu_{K-1}, sigma_0..sigma_{K-1}]
        param_names = [f"w_{k}" for k in range(K)] + [f"mu_{k}" for k in range(K)] + [f"sigma_{k}" for k in range(K)]
        super().__init__(name, param_names)

    def _split(self, predictions):
        """Split raw predictions into (log_weights, means, log_scales)."""
        K = self.n_components
        w_logits = predictions[:, :K]  # (B, K) — mixing logits
        means = predictions[:, K : 2 * K]  # (B, K) — component means
        log_scales = predictions[:, 2 * K :]  # (B, K) — log-scale logits
        return w_logits, means, log_scales

    def compute_loss(self, predictions, y_true):
        w_logits, means, log_scales = self._split(predictions)

        log_weights = torch.log_softmax(w_logits, dim=-1)  # (B, K)
        sigmas = torch.nn.functional.softplus(log_scales)  # (B, K) > 0

        # Expand y_true to (B, K) for vectorised component log-probs
        y_expanded = y_true.unsqueeze(-1).expand_as(means)  # (B, K)
        component_log_probs = dist.Normal(means, sigmas).log_prob(y_expanded)  # (B, K)

        # log p(y) = logsumexp_k [log w_k + log N(y; mu_k, sigma_k)]
        log_prob = torch.logsumexp(log_weights + component_log_probs, dim=-1)  # (B,)
        nll = -log_prob.mean()
        return nll

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)

        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        w_logits, means, _log_scales = self._split(y_pred_tensor)

        weights = torch.softmax(w_logits, dim=-1)  # (B, K)

        # E[Y] = sum_k w_k * mu_k
        mean_pred = (weights * means).sum(dim=-1)  # (B,)

        mse_loss = torch.nn.functional.mse_loss(y_true_tensor, mean_pred)
        rmse = np.sqrt(mse_loss.detach().numpy())
        mae = torch.nn.functional.l1_loss(y_true_tensor, mean_pred).detach().numpy()

        metrics["mse"] = mse_loss.detach().numpy()
        metrics["mae"] = mae
        metrics["rmse"] = rmse

        return metrics
