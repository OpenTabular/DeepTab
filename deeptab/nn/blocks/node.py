# ruff: noqa: E402
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralDecisionTree(nn.Module):
    def __init__(
        self,
        input_dim,
        depth,
        output_dim=1,
        lamda=1e-3,
        temperature=0.0,
        node_sampling=0.3,
    ):
        """Initialize the neural decision tree with a neural network at each leaf.

        Parameters:
        -----------
        input_dim: int
            The number of input features.
        depth: int
            The depth of the tree. The number of leaves will be 2^depth.
        output_dim: int
            The number of output classes (default is 1 for regression tasks).
        lamda: float
            Regularization parameter.
        """
        super().__init__()
        self.internal_node_num_ = 2**depth - 1
        self.leaf_node_num_ = 2**depth
        self.lamda = lamda
        self.depth = depth
        self.temperature = temperature
        self.node_sampling = node_sampling

        # Different penalty coefficients for nodes in different layers
        self.penalty_list = [self.lamda * (2 ** (-d)) for d in range(0, depth)]

        # Initialize internal nodes with linear layers followed by hard thresholds
        self.inner_nodes = nn.Sequential(
            nn.Linear(input_dim + 1, self.internal_node_num_, bias=False),
        )

        self.leaf_nodes = nn.Linear(self.leaf_node_num_, output_dim, bias=False)

    def forward(self, X, return_penalty=False):
        if return_penalty:
            _mu, _penalty = self._penalty_forward(X)
        else:
            _mu = self._forward(X)
        y_pred = self.leaf_nodes(_mu)
        if return_penalty:
            return y_pred, _penalty  # type: ignore
        else:
            return y_pred

    def _penalty_forward(self, X):
        """Implementation of the forward pass with hard decision boundaries."""
        batch_size = X.size()[0]
        X = self._data_augment(X)

        # Get the decision boundaries for the internal nodes
        decision_boundaries = self.inner_nodes(X)

        # Apply hard thresholding to simulate binary decisions
        if self.temperature > 0.0:
            # Replace sigmoid with Gumbel-Softmax for path_prob calculation
            logits = decision_boundaries / self.temperature
            path_prob = (logits > 0).float() + logits.sigmoid() - logits.sigmoid().detach()
        else:
            path_prob = (decision_boundaries > 0).float()

        # Prepare for routing at the internal nodes
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        _penalty = torch.tensor(0.0)

        # Iterate through internal odes in each layer to compute the final path
        # probabilities and the regularization term.
        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            # Extract internal nodes in the current layer to compute the
            # regularization term
            _penalty = _penalty + self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)

            _mu = _mu * _path_prob  # update path probabilities

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(batch_size, self.leaf_node_num_)

        return mu, _penalty

    def _forward(self, X):
        """Implementation of the forward pass with hard decision boundaries."""
        batch_size = X.size()[0]
        X = self._data_augment(X)

        # Get the decision boundaries for the internal nodes
        decision_boundaries = self.inner_nodes(X)

        # Apply hard thresholding to simulate binary decisions
        if self.temperature > 0.0:
            # Replace sigmoid with Gumbel-Softmax for path_prob calculation
            logits = decision_boundaries / self.temperature
            path_prob = (logits > 0).float() + logits.sigmoid() - logits.sigmoid().detach()
        else:
            path_prob = (decision_boundaries > 0).float()

        # Prepare for routing at the internal nodes
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)

        # Iterate through internal nodes in each layer to compute the final path
        # probabilities and the regularization term.
        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)

            _mu = _mu * _path_prob  # update path probabilities

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(batch_size, self.leaf_node_num_)

        return mu

    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        """Calculate the regularization penalty by sampling a fraction of nodes with safeguards against NaNs."""
        batch_size = _mu.size(0)

        # Reshape _mu and _path_prob for broadcasting
        _mu = _mu.view(batch_size, 2**layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))

        # Determine sample size
        num_nodes = _path_prob.size(1)
        sample_size = max(1, int(self.node_sampling * num_nodes))

        # Randomly sample nodes for penalty calculation
        indices = torch.randperm(num_nodes)[:sample_size]
        sampled_path_prob = _path_prob[:, indices]
        sampled_mu = _mu[:, indices // 2]

        # Calculate alpha in a batched manner
        epsilon = 1e-6  # Small constant to prevent division by zero
        alpha = torch.sum(sampled_path_prob * sampled_mu, dim=0) / (torch.sum(sampled_mu, dim=0) + epsilon)

        # Clip alpha to avoid NaNs in log calculation
        alpha = alpha.clamp(epsilon, 1 - epsilon)

        # Calculate penalty with broadcasting
        coeff = self.penalty_list[layer_idx]
        penalty = -0.5 * coeff * (torch.log(alpha) + torch.log(1 - alpha)).sum()

        return penalty

    def _data_augment(self, X):
        return F.pad(X, (1, 0), value=1)


# Source: https://github.com/Qwicen/node
from warnings import warn

import numpy as np
import torch.nn as nn

from deeptab.core.utils import check_numpy
from deeptab.nn.blocks.common import sparsemax, sparsemoid
from deeptab.nn.initialization import ModuleWithInit


class ODST(ModuleWithInit):
    def __init__(
        self,
        in_features,
        num_trees,
        depth=6,
        tree_dim=1,
        flatten_output=True,
        choice_function=sparsemax,
        bin_function=sparsemoid,
        initialize_response_=nn.init.normal_,
        initialize_selection_logits_=nn.init.uniform_,
        threshold_init_beta=1.0,
        threshold_init_cutoff=1.0,
    ):
        """Oblivious Differentiable Sparsemax Trees (ODST).

        ODST is a differentiable module for decision tree-based models, where each tree
        is trained using sparsemax to compute feature weights and sparsemoid to compute
        binary leaf weights. This class is designed as a drop-in replacement for `nn.Linear` layers.

        Parameters
        ----------
        in_features : int
            Number of features in the input tensor.
        num_trees : int
            Number of trees in this layer.
        depth : int, optional
            Number of splits (depth) in each tree. Default is 6.
        tree_dim : int, optional
            Number of output channels for each tree's response. Default is 1.
        flatten_output : bool, optional
            If True, returns output in a flattened shape of [..., num_trees * tree_dim];
            otherwise returns [..., num_trees, tree_dim]. Default is True.
        choice_function : callable, optional
            Function that computes feature weights as a simplex, such that
            `choice_function(tensor, dim).sum(dim) == 1`. Default is `sparsemax`.
        bin_function : callable, optional
            Function that computes tree leaf weights as values in the range [0, 1].
            Default is `sparsemoid`.
        initialize_response_ : callable, optional
            In-place initializer for the response tensor in each tree. Default is `nn.init.normal_`.
        initialize_selection_logits_ : callable, optional
            In-place initializer for the feature selection logits. Default is `nn.init.uniform_`.
        threshold_init_beta : float, optional
            Initializes thresholds based on quantiles of the data using a Beta distribution.
            Controls the initial threshold distribution; values > 1 make thresholds closer to the median.
            Default is 1.0.
        threshold_init_cutoff : float, optional
            Initializer for log-temperatures, with values > 1.0 adding margin between data points
            and sparse-sigmoid cutoffs. Default is 1.0.

        Attributes
        ----------
        response : torch.nn.Parameter
            Parameter for tree responses.
        feature_selection_logits : torch.nn.Parameter
            Logits that select features for the trees.
        feature_thresholds : torch.nn.Parameter
            Threshold values for feature splits in the trees.
        log_temperatures : torch.nn.Parameter
            Log-temperatures for threshold adjustments.
        bin_codes_1hot : torch.nn.Parameter
            One-hot encoded binary codes for leaf mapping.

        Methods
        -------
        forward(input)
            Forward pass through the ODST model.
        initialize(input, eps=1e-6)
            Data-aware initialization of thresholds and log-temperatures based on input data.
        """

        super().__init__()
        self.depth, self.num_trees, self.tree_dim, self.flatten_output = (
            depth,
            num_trees,
            tree_dim,
            flatten_output,
        )
        self.choice_function, self.bin_function = choice_function, bin_function
        self.threshold_init_beta, self.threshold_init_cutoff = (
            threshold_init_beta,
            threshold_init_cutoff,
        )

        self.response = nn.Parameter(torch.zeros([num_trees, tree_dim, 2**depth]), requires_grad=True)
        initialize_response_(self.response)

        self.feature_selection_logits = nn.Parameter(torch.zeros([in_features, num_trees, depth]), requires_grad=True)
        initialize_selection_logits_(self.feature_selection_logits)

        self.feature_thresholds = nn.Parameter(
            torch.full([num_trees, depth], float("nan"), dtype=torch.float32),
            requires_grad=True,
        )  # nan values will be initialized on first batch (data-aware init)

        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, depth], float("nan"), dtype=torch.float32),
            requires_grad=True,
        )

        # binary codes for mapping between 1-hot vectors and bin indices
        with torch.no_grad():
            indices = torch.arange(2**self.depth)
            offsets = 2 ** torch.arange(self.depth)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)
            # ^-- [depth, 2 ** depth, 2]

    def forward(self, x):  # type: ignore
        """Forward pass through ODST model.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape [batch_size, in_features] or higher dimensions.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, num_trees * tree_dim] if `flatten_output` is True,
            otherwise [batch_size, num_trees, tree_dim].
        """
        if len(x.shape) < 2:
            raise ValueError("Input tensor must have at least 2 dimensions")
        if len(x.shape) > 2:
            return self.forward(x.view(-1, x.shape[-1])).view(*x.shape[:-1], -1)
        # new input shape: [batch_size, in_features]

        feature_logits = self.feature_selection_logits
        feature_selectors = self.choice_function(feature_logits, dim=0)
        # ^--[in_features, num_trees, depth]

        feature_values = torch.einsum("bi,ind->bnd", x, feature_selectors)
        # ^--[batch_size, num_trees, depth]

        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(-self.log_temperatures)

        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        # ^--[batch_size, num_trees, depth, 2]

        bins = self.bin_function(threshold_logits)
        # ^--[batch_size, num_trees, depth, 2], approximately binary

        bin_matches = torch.einsum("btds,dcs->btdc", bins, self.bin_codes_1hot)
        # ^--[batch_size, num_trees, depth, 2 ** depth]

        response_weights = torch.prod(bin_matches, dim=-2)
        # ^-- [batch_size, num_trees, 2 ** depth]

        response = torch.einsum("bnd,ncd->bnc", response_weights, self.response)
        # ^-- [batch_size, num_trees, tree_dim]

        return response.flatten(1, 2) if self.flatten_output else response

    def initialize(self, x, eps=1e-6):
        """Data-aware initialization of thresholds and log-temperatures based on input data.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape [batch_size, in_features] used for threshold initialization.
        eps : float, optional
            Small value added to avoid log(0) errors in temperature initialization. Default is 1e-6.
        """
        # data-aware initializer
        if len(x.shape) != 2:
            raise ValueError("Input tensor must have 2 dimensions")
        if x.shape[0] < 1000:
            warn(  # noqa
                "Data-aware initialization is performed on less than 1000 data points. This may cause instability."
                "To avoid potential problems, run this model on a data batch with at least 1000 data samples."
                "You can do so manually before training. Use with torch.no_grad() for memory efficiency."
            )
        with torch.no_grad():
            feature_selectors = self.choice_function(self.feature_selection_logits, dim=0)
            # ^--[in_features, num_trees, depth]

            feature_values = torch.einsum("bi,ind->bnd", x, feature_selectors)
            # ^--[batch_size, num_trees, depth]

            # initialize thresholds: sample random percentiles of data
            percentiles_q = 100 * np.random.beta(
                self.threshold_init_beta,
                self.threshold_init_beta,
                size=[self.num_trees, self.depth],
            )
            self.feature_thresholds.data[...] = torch.as_tensor(
                list(
                    map(
                        np.percentile,
                        check_numpy(feature_values.flatten(1, 2).t()),
                        percentiles_q.flatten(),
                    )
                ),
                dtype=feature_values.dtype,
                device=feature_values.device,
            ).view(self.num_trees, self.depth)

            # init temperatures: make sure enough data points are in the linear region of sparse-sigmoid
            temperatures = np.percentile(
                check_numpy(abs(feature_values - self.feature_thresholds)),
                q=100 * min(1.0, self.threshold_init_cutoff),
                axis=0,
            )

            # if threshold_init_cutoff > 1, scale everything down by it
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(torch.as_tensor(temperatures) + eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.feature_selection_logits.shape[0]}, \
            num_trees={self.num_trees}, depth={self.depth}, tree_dim={self.tree_dim}, \
            flatten_output={self.flatten_output})"


class DenseBlock(nn.Sequential):
    """DenseBlock is a multi-layer module that sequentially stacks instances of `Module`,
    typically decision tree models like `ODST`. Each layer in the block produces additional features,
    enabling the model to learn complex representations.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    layer_dim : int
        Dimensionality of each layer in the block.
    num_layers : int
        Number of layers to stack in the block.
    tree_dim : int, optional
        Dimensionality of the output channels from each tree. Default is 1.
    max_features : int, optional
        Maximum dimensionality for feature expansion. If None, feature expansion is unrestricted.
        Default is None.
    input_dropout : float, optional
        Dropout rate applied to the input features of each layer during training. Default is 0.0.
    flatten_output : bool, optional
        If True, flattens the output along the tree dimension. Default is True.
    Module : nn.Module, optional
        Module class to use for each layer in the block, typically a decision tree model.
        Default is `ODST`.
    **kwargs : dict
        Additional keyword arguments for the `Module` instances.

    Attributes
    ----------
    num_layers : int
        Number of layers in the block.
    layer_dim : int
        Dimensionality of each layer.
    tree_dim : int
        Dimensionality of each tree's output in the layer.
    max_features : int or None
        Maximum feature dimensionality allowed for expansion.
    flatten_output : bool
        Determines whether to flatten the output.
    input_dropout : float
        Dropout rate applied to each layer's input.

    Methods
    -------
    forward(x)
        Performs the forward pass through the block, producing feature-expanded outputs.
    """

    def __init__(
        self,
        input_dim,
        layer_dim,
        num_layers,
        tree_dim=1,
        max_features=None,
        input_dropout=0.0,
        flatten_output=True,
        Module=ODST,
        **kwargs,
    ):
        layers = []
        for i in range(num_layers):
            oddt = Module(input_dim, layer_dim, tree_dim=tree_dim, flatten_output=True, **kwargs)
            input_dim = min(input_dim + layer_dim * tree_dim, max_features or float("inf"))
            layers.append(oddt)

        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.tree_dim = num_layers, layer_dim, tree_dim
        self.max_features, self.flatten_output = max_features, flatten_output
        self.input_dropout = input_dropout

    def forward(self, x):  # type: ignore
        """Forward pass through the DenseBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, input_dim] or higher dimensions.

        Returns
        -------
        torch.Tensor
            Output tensor with expanded features, where shape depends on `flatten_output`.
            If `flatten_output` is True, returns tensor of shape
            [..., num_layers * layer_dim * tree_dim].
            Otherwise, returns [..., num_layers * layer_dim, tree_dim].
        """
        initial_features = x.shape[-1]
        for layer in self:
            layer_inp = x
            if self.max_features is not None:
                tail_features = min(self.max_features, layer_inp.shape[-1]) - initial_features
                if tail_features != 0:
                    layer_inp = torch.cat(
                        [
                            layer_inp[..., :initial_features],
                            layer_inp[..., -tail_features:],
                        ],
                        dim=-1,
                    )
            if self.training and self.input_dropout:
                layer_inp = F.dropout(layer_inp, self.input_dropout)
            h = layer(layer_inp)
            x = torch.cat([x, h], dim=-1)

        outputs = x[..., initial_features:]
        if not self.flatten_output:
            outputs = outputs.view(*outputs.shape[:-1], self.num_layers * self.layer_dim, self.tree_dim)
        return outputs


import torch.nn as nn

from deeptab.nn.blocks.common import sparsemax, sparsemoid
from deeptab.nn.initialization import ModuleWithInit


class ODSTE(ModuleWithInit):
    def __init__(
        self,
        in_features,  # J (number of features)
        num_trees,
        embed_dim,  # D (embedding dimension per feature)
        depth=6,
        tree_dim=1,
        flatten_output=True,
        choice_function=sparsemax,
        bin_function=sparsemoid,
        initialize_response_=nn.init.normal_,
        initialize_selection_logits_=nn.init.uniform_,
        threshold_init_beta=1.0,
        threshold_init_cutoff=1.0,
    ):
        """Oblivious Differentiable Sparsemax Trees (ODST) with Feature & Embedding Splitting."""
        super().__init__()
        self.depth, self.num_trees, self.tree_dim, self.flatten_output = (
            depth,
            num_trees,
            tree_dim,
            flatten_output,
        )
        self.choice_function, self.bin_function = choice_function, bin_function
        self.in_features, self.embed_dim = in_features, embed_dim
        self.threshold_init_beta, self.threshold_init_cutoff = (
            threshold_init_beta,
            threshold_init_cutoff,
        )

        # Response values for each leaf
        self.response = nn.Parameter(torch.zeros([num_trees, tree_dim, embed_dim, 2**depth]), requires_grad=True)

        initialize_response_(self.response)

        # Feature selection logits (choose J)
        self.feature_selection_logits = nn.Parameter(torch.zeros([num_trees, depth, in_features]), requires_grad=True)
        initialize_selection_logits_(self.feature_selection_logits)

        # Embedding selection logits (choose D within J)
        self.embedding_selection_logits = nn.Parameter(torch.randn([num_trees, depth, in_features, embed_dim]))

        # Thresholds & temperatures (random initialization)
        self.feature_thresholds = nn.Parameter(torch.randn([num_trees, depth]))
        self.log_temperatures = nn.Parameter(torch.randn([num_trees, depth]))

        # Binary code mappings
        with torch.no_grad():
            indices = torch.arange(2**self.depth)
            offsets = 2 ** torch.arange(self.depth)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)

    def initialize(self, x, eps=1e-6):
        """Data-aware initialization of thresholds and log-temperatures based on input data.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, in_features, embed_dim] used for threshold initialization.
        eps : float, optional
            Small value added to avoid log(0) errors in temperature initialization. Default is 1e-6.
        """
        if len(x.shape) != 3:
            raise ValueError("Input tensor must have shape (batch_size, J, D)")

        if x.shape[0] < 1000:
            warn(  # noqa: B028
                "Data-aware initialization is performed on less than 1000 data points. This may cause instability."
                "To avoid potential problems, run this model on a data batch with at least 1000 data samples."
                "You can do so manually before training. Use with torch.no_grad() for memory efficiency."
            )

        with torch.no_grad():
            # Select features (J)
            feature_selectors = self.choice_function(self.feature_selection_logits, dim=-1)
            # feature_selectors shape: (num_trees, depth, J)

            selected_features = torch.einsum("bjd,ntj->bntd", x, feature_selectors)
            # selected_features shape: (B, num_trees, depth, D)

            # Select embeddings (D)
            embedding_selectors = self.choice_function(self.embedding_selection_logits, dim=-1)
            # embedding_selectors shape: (num_trees, depth, J, D)

            selected_embeddings = torch.einsum("bntd,ntjd->bntd", selected_features, embedding_selectors)
            # selected_embeddings shape: (B, num_trees, depth, D)

            # Initialize thresholds using percentiles from the data
            percentiles_q = 100 * np.random.beta(
                self.threshold_init_beta,
                self.threshold_init_beta,
                size=[self.num_trees, self.depth],
            )

            reshaped_embeddings = selected_embeddings.permute(1, 2, 0, 3).reshape(self.num_trees * self.depth, -1)
            self.feature_thresholds.data[...] = torch.as_tensor(
                list(
                    map(
                        np.percentile,
                        check_numpy(reshaped_embeddings),  # Now correctly 2D
                        percentiles_q.flatten(),
                    )
                ),
                dtype=selected_embeddings.dtype,
                device=selected_embeddings.device,
            ).view(self.num_trees, self.depth)

            # Initialize temperatures based on the threshold differences
            temperatures = np.percentile(
                check_numpy(abs(selected_embeddings - self.feature_thresholds.unsqueeze(-1))),
                q=100 * min(1.0, self.threshold_init_cutoff),
                axis=0,
            )

            # Scale temperatures based on the cutoff
            temperatures /= max(1.0, self.threshold_init_cutoff)

            self.log_temperatures.data[...] = torch.log(
                torch.as_tensor(
                    temperatures.mean(-1),
                    dtype=selected_embeddings.dtype,
                    device=selected_embeddings.device,
                )
                + eps
            )

    def forward(self, x):
        if len(x.shape) != 3:
            raise ValueError("Input tensor must have shape (batch_size, J, D)")

        # Select feature (J) and embedding dimension (D) separately
        feature_selectors = self.choice_function(self.feature_selection_logits, dim=-1)  # [num_trees, depth, J]

        embedding_selectors = self.choice_function(self.embedding_selection_logits, dim=-1)  # [num_trees, depth, J, D]

        # Select features (J) first
        selected_features = torch.einsum("bjd,ntj->bntd", x, feature_selectors)

        # Select embeddings (D) within selected features
        selected_embeddings = torch.einsum("bntd,ntjd->bntd", selected_features, embedding_selectors)

        # Compute threshold logits
        threshold_logits = (selected_embeddings - self.feature_thresholds.unsqueeze(0).unsqueeze(-1)) * torch.exp(
            -self.log_temperatures.unsqueeze(0).unsqueeze(-1)
        )

        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)

        # Compute binary decisions
        bins = self.bin_function(threshold_logits)

        bin_matches = torch.einsum("bntds,tcs->bntdc", bins, self.bin_codes_1hot)

        response_weights = torch.prod(bin_matches, dim=2)

        # Compute final response
        response = torch.einsum("bnds,ncds->bnd", response_weights, self.response)
        return response

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, embed_dim={self.embed_dim}, num_trees={self.num_trees}, depth={self.depth}, tree_dim={self.tree_dim}, flatten_output={self.flatten_output})"


class ENODEDenseBlock(nn.Module):
    """ENODEDenseBlock that sequentially stacks attention layers and `Module` layers (e.g., ODSTE)
    with feature and embedding-aware splits.

    Parameters
    ----------
    input_dim : int
        Number of features (J) in the input.
    embed_dim : int
        Embedding dimension per feature (D).
    layer_dim : int
        Dimensionality of each ODSTE layer.
    num_layers : int
        Number of layers to stack in the block.
    tree_dim : int, optional
        Number of output channels from each tree. Default is 1.
    max_features : int, optional
        Maximum number of features for expansion. Default is None.
    input_dropout : float, optional
        Dropout rate applied to inputs during training. Default is 0.0.
    flatten_output : bool, optional
        If True, flattens the output along the tree dimension. Default is True.
    Module : nn.Module, optional
        Module class to use for each layer in the block. Default is `ODSTE`.
    **kwargs : dict
        Additional keyword arguments for `Module` instances.
    """

    def __init__(
        self,
        input_dim,
        embed_dim,
        layer_dim,
        num_layers,
        tree_dim=1,
        max_features=None,
        input_dropout=0.0,
        flatten_output=True,
        Module=ODSTE,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layer_dim = layer_dim
        self.tree_dim = tree_dim
        self.max_features = max_features
        self.input_dropout = input_dropout
        self.flatten_output = flatten_output

        self.attention_layers = nn.ModuleList()
        self.odste_layers = nn.ModuleList()

        for _ in range(num_layers):
            # self.attention_layers.append(
            #    nn.MultiheadAttention(
            #        embed_dim=embed_dim, num_heads=1, batch_first=True
            #    )
            # )
            self.odste_layers.append(
                Module(
                    in_features=input_dim,
                    embed_dim=embed_dim,
                    num_trees=layer_dim,
                    tree_dim=tree_dim,
                    flatten_output=True,
                    **kwargs,
                )
            )
            input_dim = min(input_dim + layer_dim * tree_dim, max_features or float("inf"))

    def forward(self, x):
        """Forward pass through the ENODEDenseBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, J, D].

        Returns
        -------
        torch.Tensor
            Output tensor with expanded features.
        """
        initial_features = x.shape[1]  # J (num features)

        for odste_layer in self.odste_layers:
            # x, _ = attn_layer(x, x, x)  # Apply attention

            if self.max_features is not None:
                tail_features = min(self.max_features, x.shape[1]) - initial_features
                if tail_features > 0:
                    x = torch.cat([x[:, :initial_features, :], x[:, -tail_features:, :]], dim=1)

            if self.training and self.input_dropout:
                x = F.dropout(x, self.input_dropout)

            h = odste_layer(x)  # Apply ODSTE layer
            x = torch.cat([x, h], dim=1)  # Concatenate new features

        return x
