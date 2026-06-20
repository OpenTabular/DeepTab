"""Categorical, Quantile, and Multinomial distributions for multi-class / distribution-free LSS models."""

import torch
import torch.distributions as dist

from .base import BaseDistribution


class CategoricalDistribution(BaseDistribution):
    """
    Represents a Categorical distribution, a discrete distribution that describes the possible results of a
    random variable that can take on one of K possible categories, with the probability of each category
    separately specified. This class extends BaseDistribution and includes parameter transformation and loss
    computation specific to the Categorical distribution.

    Parameters
    ----------
        name (str): The name of the distribution, defaulted to "Categorical".
        prob_transform (str or callable): Transformation for the probabilities to ensure
        they remain valid (i.e., non-negative and sum to 1).
    """

    def __init__(self, name="Categorical", prob_transform="probabilities"):
        param_names = ["probs"]
        super().__init__(name, param_names)
        self.probs_transform = self.get_transform(prob_transform)

    def compute_loss(self, predictions, y_true):
        probs = self.probs_transform(predictions)

        cat_dist = dist.Categorical(probs=probs)

        nll = -cat_dist.log_prob(y_true).mean()
        return nll


class Quantile(BaseDistribution):
    """
    Quantile Regression Loss class.

    This class computes the quantile loss (also known as pinball loss) for a set of quantiles.
    It is used to handle quantile regression tasks where we aim to predict a given quantile of the target distribution.

    Parameters
    ----------
    name : str, optional
        The name of the distribution, by default "Quantile".
    quantiles : list of float, optional
        A list of quantiles to be used for computing the loss, by default [0.25, 0.5, 0.75].

    Attributes
    ----------
    quantiles : list of float
        List of quantiles for which the pinball loss is computed.

    Methods
    -------
    compute_loss(predictions, y_true)
        Computes the quantile regression loss between the predictions and true values.
    """

    def __init__(self, name="Quantile", quantiles=[0.25, 0.5, 0.75]):
        param_names = [f"q_{q}" for q in quantiles]
        super().__init__(name, param_names)
        self.quantiles = quantiles

    def compute_loss(self, predictions, y_true):
        if y_true.requires_grad:
            raise ValueError("y_true should not require gradients")
        if predictions.size(0) != y_true.size(0):
            raise ValueError("Batch size of predictions and y_true must match")

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = y_true - predictions[:, i]
            quantile_loss = torch.max((q - 1) * errors, q * errors)
            losses.append(quantile_loss)

        loss = torch.mean(torch.stack(losses, dim=1).sum(dim=1))
        return loss


class MultinomialDistribution(BaseDistribution):
    """
    Represents a Multinomial distribution for modelling count vectors that sum to a
    known total (e.g. word counts per document, allele frequencies, multi-label counts
    where total responses per sample is fixed).

    The neural network outputs ``num_classes`` logits which are converted to probabilities
    via softmax.  ``total_count`` is a fixed constructor argument, not a predicted
    parameter.

    Parameters
    ----------
        name (str): Defaults to ``"Multinomial"``.
        num_classes (int): Number of categories K.  Sets ``param_count = K``.
            Defaults to ``2``.
        total_count (int): Total number of trials n (e.g. 1 makes this equivalent
            to Categorical).  Defaults to ``1``.
        prob_transform (str or callable): Transform for the class logits.
            Defaults to ``"probabilities"`` (softmax).
    """

    def __init__(
        self,
        name="Multinomial",
        num_classes=2,
        total_count=1,
        prob_transform="probabilities",
    ):
        param_names = [f"p_{k}" for k in range(num_classes)]
        super().__init__(name, param_names)

        self.total_count = total_count
        self.probs_transform = self.get_transform(prob_transform)

    def compute_loss(self, predictions, y_true):
        probs = self.probs_transform(predictions)

        multinomial_dist = dist.Multinomial(total_count=self.total_count, probs=probs)
        nll = -multinomial_dist.log_prob(y_true).mean()
        return nll
