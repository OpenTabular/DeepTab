import numpy as np
import torch
import torch.nn as nn

from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..configs.tangos_config import DefaultTangosConfig
from ..utils.get_feature_dimensions import get_feature_dimensions
from .utils.basemodel import BaseModel


class Tangos(BaseModel):
    """
    A Multi-Layer Perceptron (MLP) model with optional GLU activation, batch normalization, layer normalization, and dropout.  # noqa: W505
    It includes a penalty term for specialization and orthogonality.

    Parameters
    ----------
    feature_information : tuple
        A tuple containing feature information for numerical and categorical features.
    num_classes : int, optional (default=1)
        The number of output classes.
    config : DefaultTangosConfig, optional (default=DefaultTangosConfig())
        Configuration object defining model hyperparameters.
    **kwargs : dict
        Additional arguments for the base model.

    Attributes
    ----------
    returns_ensemble : bool
        Whether the model returns an ensemble of predictions.
    lamda1 : float
        Regularization weight for the specialization loss.
    lamda2 : float
        Regularization weight for the orthogonality loss.
    subsample : float
        Proportion of neuron pairs to use for orthogonality loss calculation.
    embedding_layer : EmbeddingLayer or None
        Optional embedding layer for categorical features.
    layers : nn.ModuleList
        The main MLP layers including linear, normalization, and activation layers.
    head : nn.Linear
        The final output layer.
    """

    def __init__(
        self,
        feature_information: tuple,
        num_classes=1,
        config: DefaultTangosConfig = DefaultTangosConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["feature_information"])
        self.returns_ensemble = False

        self.lamda1 = config.lamda1
        self.lamda2 = config.lamda2
        self.subsample = config.subsample

        input_dim = get_feature_dimensions(*feature_information)

        # Initialize layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, self.hparams.layer_sizes[0]))
        if self.hparams.batch_norm:
            self.layers.append(nn.BatchNorm1d(self.hparams.layer_sizes[0]))

        if self.hparams.use_glu:
            self.layers.append(nn.GLU())
        else:
            self.layers.append(self.hparams.activation)
        if self.hparams.dropout > 0.0:
            self.layers.append(nn.Dropout(self.hparams.dropout))

        # Hidden layers
        for i in range(1, len(self.hparams.layer_sizes)):
            self.layers.append(nn.Linear(self.hparams.layer_sizes[i - 1], self.hparams.layer_sizes[i]))
            if self.hparams.batch_norm:
                self.layers.append(nn.BatchNorm1d(self.hparams.layer_sizes[i]))
            if self.hparams.layer_norm:
                self.layers.append(nn.LayerNorm(self.hparams.layer_sizes[i]))
            if self.hparams.use_glu:
                self.layers.append(nn.GLU())
            else:
                self.layers.append(self.hparams.activation)
            if self.hparams.dropout > 0.0:
                self.layers.append(nn.Dropout(self.hparams.dropout))

        # Output layer
        self.head = nn.Linear(self.hparams.layer_sizes[-1], num_classes)

    def repr_forward(self, x) -> torch.Tensor:
        """
        Computes the forward pass for feature representations.

        This method processes the input through the MLP layers, optionally using
        skip connections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, feature_dim).

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the representation layers.
        """

        x = x.unsqueeze(0)

        for i in range(len(self.layers)):
            if isinstance(self.layers[i], nn.Linear):
                out = self.layers[i](x)
                if self.hparams.skip_connections and x.shape == out.shape:
                    x = x + out
                else:
                    x = out
            else:
                x = self.layers[i](x)

        return x

    def forward(self, *data) -> torch.Tensor:
        """
        Performs a forward pass of the MLP model.

        This method concatenates all input tensors before applying MLP layers.

        Parameters
        ----------
        data : tuple
            A tuple containing lists of numerical, categorical, and embedded feature tensors.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (batch_size, num_classes).
        """

        x = torch.cat([t for tensors in data for t in tensors], dim=1)

        for i in range(len(self.layers)):
            if isinstance(self.layers[i], nn.Linear):
                out = self.layers[i](x)
                if self.hparams.skip_connections and x.shape == out.shape:
                    x = x + out
                else:
                    x = out
            else:
                x = self.layers[i](x)
        x = self.head(x)
        return x

    def penalty_forward(self, *data):
        """
        Computes both the model predictions and a penalty term.

        The penalty term includes:
        - **Specialization loss**: Measures feature importance concentration.
        - **Orthogonality loss**: Encourages diversity among learned features.

        The method uses `jacrev` to compute the Jacobian of the representation function.

        Parameters
        ----------
        data : tuple
            A tuple containing lists of numerical, categorical, and embedded feature tensors.

        Returns
        -------
        tuple
            - predictions : torch.Tensor
                Model predictions of shape (batch_size, num_classes).
            - penalty : torch.Tensor
                The computed penalty term for regularization.
        """

        x = torch.cat([t for tensors in data for t in tensors], dim=1)
        batch_size = x.shape[0]
        subsample = np.int32(self.subsample * batch_size)

        # Flatten before passing to jacrev
        flat_data = torch.cat([t for tensors in data for t in tensors], dim=1)

        # Compute Jacobian
        jacobian = torch.func.vmap(torch.func.jacrev(self.repr_forward), randomness="different")(flat_data)
        jacobian = jacobian.squeeze()

        neuron_attr = jacobian.swapaxes(0, 1)
        h_dim = neuron_attr.shape[0]
        if len(neuron_attr.shape) > 3:
            # h_dim x batch_size x features
            neuron_attr = neuron_attr.flatten(start_dim=2)

        # calculate specialization loss component
        spec_loss = torch.norm(neuron_attr, p=1) / (batch_size * h_dim * neuron_attr.shape[2])
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        orth_loss = torch.tensor(0.0, requires_grad=True).to(x.device)
        # apply subsampling routine for orthogonalization loss
        if self.subsample > 0 and self.subsample < h_dim * (h_dim - 1) / 2:
            tensor_pairs = [list(np.random.choice(h_dim, size=(2), replace=False)) for i in range(subsample)]
            for tensor_pair in tensor_pairs:
                pairwise_corr = cos(neuron_attr[tensor_pair[0], :, :], neuron_attr[tensor_pair[1], :, :]).norm(p=1)
                orth_loss = orth_loss + pairwise_corr

            orth_loss = orth_loss / (batch_size * self.subsample)
        else:
            for neuron_i in range(1, h_dim):
                for neuron_j in range(0, neuron_i):
                    pairwise_corr = cos(neuron_attr[neuron_i, :, :], neuron_attr[neuron_j, :, :]).norm(p=1)
                    orth_loss = orth_loss + pairwise_corr
            num_pairs = h_dim * (h_dim - 1) / 2
            orth_loss = orth_loss / (batch_size * num_pairs)

        penalty = self.lamda1 * spec_loss + self.lamda2 * orth_loss
        predictions = self.forward(*data)

        return predictions, penalty
