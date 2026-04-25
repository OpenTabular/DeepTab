from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeptab.arch_utils.layer_utils.sparsemax import sparsemax, sparsemoid

from .data_aware_initialization import ModuleWithInit
from .numpy_utils import check_numpy


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


class DenseBlock(nn.Module):
    """DenseBlock that sequentially stacks attention layers and `Module` layers (e.g., ODSTE)
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
        """Forward pass through the DenseBlock.

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
