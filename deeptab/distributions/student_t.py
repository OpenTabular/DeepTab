"""Student-t and Johnson SU distributions for heavy-tailed / skewed LSS models."""

import numpy as np
import torch
import torch.distributions as dist

from .base import BaseDistribution


class StudentTDistribution(BaseDistribution):
    """
    Represents a Student's t-distribution, a family of continuous probability distributions that arise when
    estimating the mean of a normally distributed population in situations where the sample size is small.
    This class extends BaseDistribution and includes parameter transformation and loss computation specific
    to the Student's t-distribution.

    Parameters
    ----------
        name (str): The name of the distribution, defaulted to "StudentT".
        df_transform (str or callable): Transformation for the degrees of freedom parameter
        to ensure it remains positive.
        loc_transform (str or callable): Transformation for the location parameter.
        scale_transform (str or callable): Transformation for the scale parameter
        to ensure it remains positive.
    """

    def __init__(
        self,
        name="StudentT",
        df_transform="positive",
        loc_transform="none",
        scale_transform="positive",
    ):
        param_names = ["df", "loc", "scale"]
        super().__init__(name, param_names)

        self.df_transform = self.get_transform(df_transform)
        self.loc_transform = self.get_transform(loc_transform)
        self.scale_transform = self.get_transform(scale_transform)

    def compute_loss(self, predictions, y_true):
        df = self.df_transform(predictions[:, self.param_names.index("df")])
        loc = self.loc_transform(predictions[:, self.param_names.index("loc")])
        scale = self.scale_transform(predictions[:, self.param_names.index("scale")])

        student_t_dist = dist.StudentT(df, loc, scale)  # type: ignore

        nll = -student_t_dist.log_prob(y_true).mean()
        return nll

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)

        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        mse_loss = torch.nn.functional.mse_loss(y_true_tensor, y_pred_tensor[:, self.param_names.index("loc")])
        rmse = np.sqrt(mse_loss.detach().numpy())
        mae = (
            torch.nn.functional.l1_loss(y_true_tensor, y_pred_tensor[:, self.param_names.index("loc")]).detach().numpy()
        )

        metrics["mse"] = mse_loss.detach().numpy()
        metrics["mae"] = mae
        metrics["rmse"] = rmse

        return metrics


class JohnsonSuDistribution(BaseDistribution):
    """
    Represents a Johnson's SU distribution with parameters for skewness, shape, location, and scale.

    Parameters
    ----------
        name (str): The name of the distribution. Defaults to "JohnsonSu".
        skew_transform (str or callable): The transformation for the skewness parameter. Defaults to "none".
        shape_transform (str or callable): The transformation for the shape parameter. Defaults to "positive".
        loc_transform (str or callable): The transformation for the location parameter. Defaults to "none".
        scale_transform (str or callable): The transformation for the scale parameter. Defaults to "positive".
    """

    def __init__(
        self,
        name="JohnsonSu",
        skew_transform="none",
        shape_transform="positive",
        loc_transform="none",
        scale_transform="positive",
    ):
        param_names = ["skew", "shape", "location", "scale"]
        super().__init__(name, param_names)

        self.skew_transform = self.get_transform(skew_transform)
        self.shape_transform = self.get_transform(shape_transform)
        self.loc_transform = self.get_transform(loc_transform)
        self.scale_transform = self.get_transform(scale_transform)

    def log_prob(self, x, skew, shape, loc, scale):
        """Compute the log probability density of the Johnson's SU distribution."""
        z = skew + shape * torch.asinh((x - loc) / scale)
        log_pdf = (
            torch.log(shape / (scale * np.sqrt(2 * np.pi))) - 0.5 * z**2 - 0.5 * torch.log(1 + ((x - loc) / scale) ** 2)
        )
        return log_pdf

    def compute_loss(self, predictions, y_true):
        skew = self.skew_transform(predictions[:, self.param_names.index("skew")])
        shape = self.shape_transform(predictions[:, self.param_names.index("shape")])
        loc = self.loc_transform(predictions[:, self.param_names.index("location")])
        scale = self.scale_transform(predictions[:, self.param_names.index("scale")])

        log_probs = self.log_prob(y_true, skew, shape, loc, scale)
        nll = -log_probs.mean()
        return nll

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)

        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        mse_loss = torch.nn.functional.mse_loss(y_true_tensor, y_pred_tensor[:, self.param_names.index("location")])
        rmse = np.sqrt(mse_loss.detach().numpy())
        mae = (
            torch.nn.functional.l1_loss(y_true_tensor, y_pred_tensor[:, self.param_names.index("location")])
            .detach()
            .numpy()
        )

        metrics.update({"mse": mse_loss.detach().numpy(), "mae": mae, "rmse": rmse})

        return metrics
