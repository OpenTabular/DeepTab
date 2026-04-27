import numpy as np
import torch
import torch.nn as nn

from ..arch_utils.neural_decision_tree import NeuralDecisionTree
from ..configs.ndtf_config import DefaultNDTFConfig
from ..utils.get_feature_dimensions import get_feature_dimensions
from .utils.basemodel import BaseModel


class NDTF(BaseModel):
    """A Neural Decision Tree Forest (NDTF) model for tabular data, composed of an ensemble of neural decision trees
    with convolutional feature interactions, capable of producing predictions and penalty-based regularization.

    Parameters
    ----------
    cat_feature_info : dict
        Dictionary containing information about categorical features, including their names and dimensions.
    num_feature_info : dict
        Dictionary containing information about numerical features, including their names and dimensions.
    num_classes : int, optional
        The number of output classes or target dimensions for regression, by default 1.
    config : DefaultNDTFConfig, optional
        Configuration object containing model hyperparameters such as the number of ensembles,
        tree depth, penalty factor,
        sampling settings, and temperature, by default DefaultNDTFConfig().
    **kwargs : dict
        Additional keyword arguments for the BaseModel class.

    Attributes
    ----------
    cat_feature_info : dict
        Stores categorical feature information.
    num_feature_info : dict
        Stores numerical feature information.
    penalty_factor : float
        Scaling factor for the penalty applied during training, specified in the self.hparams.
    input_dimensions : list of int
        List of input dimensions for each tree in the ensemble, with random sampling.
    trees : nn.ModuleList
        List of neural decision trees used in the ensemble.
    conv_layer : nn.Conv1d
        Convolutional layer for feature interactions before passing inputs to trees.
    tree_weights : nn.Parameter
        Learnable parameter to weight each tree's output in the ensemble.

    Methods
    -------
    forward(num_features, cat_features) -> torch.Tensor
        Perform a forward pass through the model, producing predictions based on an ensemble of neural decision trees.
    penalty_forward(num_features, cat_features) -> tuple of torch.Tensor
        Perform a forward pass with penalty regularization, returning predictions and the calculated penalty term.
    """

    def __init__(
        self,
        feature_information: tuple,  # Expecting (num_feature_info, cat_feature_info, embedding_feature_info)
        num_classes: int = 1,
        config: DefaultNDTFConfig = DefaultNDTFConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["feature_information"])

        self.returns_ensemble = False

        input_dim = get_feature_dimensions(*feature_information)

        self.input_dimensions = [input_dim]

        for _ in range(self.hparams.n_ensembles - 1):
            self.input_dimensions.append(np.random.randint(1, input_dim))

        self.trees = nn.ModuleList(
            [
                NeuralDecisionTree(
                    input_dim=self.input_dimensions[idx],
                    depth=np.random.randint(self.hparams.min_depth, self.hparams.max_depth),
                    output_dim=num_classes,
                    lamda=self.hparams.lamda,
                    temperature=self.hparams.temperature + np.abs(np.random.normal(0, 0.1)),
                    node_sampling=self.hparams.node_sampling,
                )
                for idx in range(self.hparams.n_ensembles)
            ]
        )

        self.conv_layer = nn.Conv1d(
            in_channels=self.input_dimensions[0],
            out_channels=1,  # Single channel output if one feature interaction is desired
            # Choose appropriate kernel size
            kernel_size=self.input_dimensions[0],
            # To keep output size the same as input_dim if desired
            padding=self.input_dimensions[0] - 1,
            bias=True,
        )

        self.tree_weights = nn.Parameter(
            torch.full((self.hparams.n_ensembles, 1), 1.0 / self.hparams.n_ensembles),
            requires_grad=True,
        )

    def forward(self, *data) -> torch.Tensor:
        """Forward pass of the NDTF model.

        Parameters
        ----------
        data : tuple
            Input tuple of tensors of num_features, cat_features, embeddings.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = torch.cat([t for tensors in data for t in tensors], dim=1)
        x = self.conv_layer(x.unsqueeze(2))
        x = x.transpose(1, 2).squeeze(-1)

        preds = []

        for idx, tree in enumerate(self.trees):
            tree_input = x[:, : self.input_dimensions[idx]]
            preds.append(tree(tree_input, return_penalty=False))

        preds = torch.stack(preds, dim=1).squeeze(-1)

        return preds @ self.tree_weights

    def penalty_forward(self, *data) -> torch.Tensor:
        """Forward pass of the NDTF model.

        Parameters
        ----------
        data : tuple
            Input tuple of tensors of num_features, cat_features, embeddings.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = torch.cat([t for tensors in data for t in tensors], dim=1)
        x = self.conv_layer(x.unsqueeze(2))
        x = x.transpose(1, 2).squeeze(-1)

        penalty = 0.0
        preds = []

        # Iterate over trees and collect predictions and penalties
        for idx, tree in enumerate(self.trees):
            # Select subset of features for the current tree
            tree_input = x[:, : self.input_dimensions[idx]]

            # Get prediction and penalty from the current tree
            pred, pen = tree(tree_input, return_penalty=True)
            preds.append(pred)
            penalty += pen

        # Stack predictions and calculate mean across trees
        preds = torch.stack(preds, dim=1).squeeze(-1)
        return preds @ self.tree_weights, self.hparams.penalty_factor * penalty  # type: ignore
