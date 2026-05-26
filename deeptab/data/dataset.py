import torch
from torch.utils.data import Dataset

from deeptab.data.schema import TabularBatch


class TabularDataset(Dataset):
    """Custom dataset for handling structured tabular data with separate categorical
    and numerical features.

    This dataset is task-agnostic and simply stores and retrieves features and labels
    without any task-specific preprocessing. Label dtype conversion should be handled
    externally by the DataModule or training logic.

    Parameters
    ----------
    cat_features_list : list of Tensors
        A list of tensors representing the categorical features.
    num_features_list : list of Tensors
        A list of tensors representing the numerical features.
    embeddings_list : list of Tensors, optional
        A list of tensors representing the embeddings.
    labels : Tensor, optional
        A tensor of labels. If None, the dataset is used for prediction.
    return_batch_object : bool, default=False
        If True, returns a TabularBatch object instead of a tuple. For backward
        compatibility, defaults to False.
    """

    def __init__(
        self,
        cat_features_list,
        num_features_list,
        embeddings_list=None,
        labels=None,
        return_batch_object=False,
    ):
        assert cat_features_list or num_features_list  # noqa: S101

        self.cat_features_list = cat_features_list  # Categorical features tensors
        self.num_features_list = num_features_list  # Numerical features tensors
        self.embeddings_list = embeddings_list  # Embeddings tensors (optional)
        self.labels = labels  # Labels (optional, None in prediction mode)
        self.return_batch_object = return_batch_object

    def __len__(self):
        _feats = self.num_features_list if self.num_features_list else self.cat_features_list
        return len(_feats[0])

    def __getitem__(self, idx):
        """Retrieves the features and label for a given index.

        Parameters
        ----------
        idx : int
            The index of the data point.

        Returns
        -------
        tuple or TabularBatch
            If return_batch_object is False (default), returns a tuple containing
            lists of tensors for numerical features, categorical features, embeddings
            (if available), and a label (if available).
            If return_batch_object is True, returns a TabularBatch object.
        """
        cat_features = [feature_tensor[idx] for feature_tensor in self.cat_features_list]
        num_features = [
            torch.as_tensor(feature_tensor[idx]).clone().detach().to(torch.float32)
            for feature_tensor in self.num_features_list
        ]

        if self.embeddings_list is not None:
            embeddings = [
                torch.as_tensor(embed_tensor[idx]).clone().detach().to(torch.float32)
                for embed_tensor in self.embeddings_list
            ]
        else:
            embeddings = None

        label = self.labels[idx] if self.labels is not None else None

        if self.return_batch_object:
            return TabularBatch(
                numerical_features=num_features,
                categorical_features=cat_features,
                embeddings=embeddings,
                labels=label,
            )
        else:
            # Legacy tuple format
            if label is not None:
                return (num_features, cat_features, embeddings), label
            else:
                return (num_features, cat_features, embeddings)
