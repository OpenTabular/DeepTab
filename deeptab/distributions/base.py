"""Base class for all DeepTab distribution families."""

from collections.abc import Callable

import torch


class BaseDistribution(torch.nn.Module):
    """
    The base class for various statistical distributions, providing a common interface and utilities.

    This class defines the basic structure and methods that are inherited by specific distribution
    classes, allowing for the implementation of custom distributions with specific parameter transformations
    and loss computations.

    Attributes
    ----------
        _name (str): The name of the distribution.
        param_names (list of str): A list of names for the parameters of the distribution.
        param_count (int): The number of parameters for the distribution.
        predefined_transforms (dict): A dictionary of predefined transformation functions for parameters.

    Parameters
    ----------
        name (str): The name of the distribution.
        param_names (list of str): A list of names for the parameters of the distribution.
    """

    def __init__(self, name, param_names):
        super().__init__()

        self._name = name
        self.param_names = param_names
        self.param_count = len(param_names)
        # Predefined transformation functions accessible to all subclasses
        self.predefined_transforms: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            "positive": torch.nn.functional.softplus,
            "none": lambda x: x,
            "square": lambda x: x**2,
            "exp": torch.exp,
            "sqrt": torch.sqrt,
            "probabilities": lambda x: torch.softmax(x, dim=-1),
            # Adding a small constant for numerical stability
            "log": lambda x: torch.log(x + 1e-6),
        }

    @property
    def name(self):
        return self._name

    @property
    def parameter_count(self):
        return self.param_count

    def get_transform(
        self, transform_name: str | Callable[[torch.Tensor], torch.Tensor]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Retrieve a transformation function by name, or return the function if it's custom.
        """
        if callable(transform_name):
            # Custom transformation function provided
            return transform_name
        # Default to 'none'
        return self.predefined_transforms.get(transform_name, lambda x: x)

    def compute_loss(self, predictions, y_true):
        """
        Computes the loss (e.g., negative log likelihood) for the distribution given
        predictions and true values.

        This method must be implemented by subclasses.

        Parameters
        ----------
            predictions (torch.Tensor): The predicted parameters of the distribution.
            y_true (torch.Tensor): The true values.

        Raises
        ------
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def evaluate_nll(self, y_true, y_pred):
        """
        Evaluates the negative log likelihood (NLL) for given true values and predictions.

        Parameters
        ----------
            y_true (array-like): The true values.
            y_pred (array-like): The predicted values.

        Returns
        -------
            dict: A dictionary containing the NLL value.
        """

        # Convert numpy arrays to torch tensors
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        # Compute NLL using the provided loss function
        nll_loss_tensor = self.compute_loss(y_pred_tensor, y_true_tensor)

        # Convert the NLL loss tensor back to a numpy array and return
        return {
            "NLL": nll_loss_tensor.detach().numpy(),
        }

    def forward(self, predictions):
        """
        Apply the appropriate transformations to the predicted parameters.

        Parameters:
            predictions (torch.Tensor): The predicted parameters of the distribution.

        Returns:
            torch.Tensor: A tensor with transformed parameters.
        """
        transformed_params = []
        for idx, param_name in enumerate(self.param_names):
            transform_func = self.get_transform(getattr(self, f"{param_name}_transform", "none"))
            transformed_params.append(
                transform_func(predictions[:, idx]).unsqueeze(  # type: ignore
                    1
                )  # type: ignore
            )
        return torch.cat(transformed_params, dim=1)
