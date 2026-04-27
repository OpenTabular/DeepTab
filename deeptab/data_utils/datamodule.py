import lightning as pl
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import MambularDataset


class MambularDataModule(pl.LightningDataModule):
    """A PyTorch Lightning data module for managing training and validation data loaders in a structured way.

    This class simplifies the process of batch-wise data loading for training and validation datasets during
    the training loop, and is particularly useful when working with PyTorch Lightning's training framework.

    Parameters:
        preprocessor: object
            An instance of your preprocessor class.
        batch_size: int
            Size of batches for the DataLoader.
        shuffle: bool
            Whether to shuffle the training data in the DataLoader.
        X_val: DataFrame or None, optional
            Validation features. If None, uses train-test split.
        y_val: array-like or None, optional
            Validation labels. If None, uses train-test split.
        val_size: float, optional
            Proportion of data to include in the validation split if `X_val` and `y_val` are None.
        random_state: int, optional
            Random seed for reproducibility in data splitting.
        regression: bool, optional
            Whether the problem is regression (True) or classification (False).
    """

    def __init__(
        self,
        preprocessor,
        batch_size,
        shuffle,
        regression,
        X_val=None,
        y_val=None,
        val_size=0.2,
        random_state=101,
        **dataloader_kwargs,
    ):
        """Initialize the data module with the specified preprocessor, batch size, shuffle option, and optional
        validation data settings.

        Args:
            preprocessor (object): An instance of the preprocessor class for data preprocessing.
            batch_size (int): Size of batches for the DataLoader.
            shuffle (bool): Whether to shuffle the training data in the DataLoader.
            X_val (DataFrame or None, optional): Validation features. If None, uses train-test split.
            y_val (array-like or None, optional): Validation labels. If None, uses train-test split.
            val_size (float, optional): Proportion of data to include in the validation split
            if `X_val` and `y_val` are None.
            random_state (int, optional): Random seed for reproducibility in data splitting.
            regression (bool, optional): Whether the problem is regression (True) or classification (False).
        """
        super().__init__()
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cat_feature_info = None
        self.num_feature_info = None
        self.X_val = X_val
        self.y_val = y_val
        self.val_size = val_size
        self.random_state = random_state
        self.regression = regression
        if self.regression:
            self.labels_dtype = torch.float32
        else:
            self.labels_dtype = torch.long

        # Initialize placeholders for data
        self.X_train = None
        self.y_train = None
        self.embeddings_train = None
        self.embeddings_val = None
        self.test_preprocessor_fitted = False
        self.dataloader_kwargs = dataloader_kwargs

    def preprocess_data(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        embeddings_train=None,
        embeddings_val=None,
        val_size=0.2,
        random_state=101,
    ):
        """Preprocesses the training and validation data.

        Parameters
        ----------
        X_train : DataFrame or array-like, shape (n_samples_train, n_features)
            Training feature set.
        y_train : array-like, shape (n_samples_train,)
            Training target values.
        embeddings_train : array-like or list of array-like, optional
            Training embeddings if available.
        X_val : DataFrame or array-like, shape (n_samples_val, n_features), optional
            Validation feature set. If None, a validation set will be created from `X_train`.
        y_val : array-like, shape (n_samples_val,), optional
            Validation target values. If None, a validation set will be created from `y_train`.
        embeddings_val : array-like or list of array-like, optional
            Validation embeddings if available.
        val_size : float, optional
            Proportion of data to include in the validation split if `X_val` and `y_val` are None.
        random_state : int, optional
            Random seed for reproducibility in data splitting.

        Returns
        -------
        None
        """

        if X_val is None or y_val is None:
            split_data = [X_train, y_train]

            if embeddings_train is not None:
                if not isinstance(embeddings_train, list):
                    embeddings_train = [embeddings_train]
                if embeddings_val is not None and not isinstance(embeddings_val, list):
                    embeddings_val = [embeddings_val]

                split_data += embeddings_train
                split_result = train_test_split(*split_data, test_size=val_size, random_state=random_state)

                self.X_train, self.X_val, self.y_train, self.y_val = split_result[:4]
                self.embeddings_train = split_result[4::2]
                self.embeddings_val = split_result[5::2]
            else:
                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                    *split_data, test_size=val_size, random_state=random_state
                )
                self.embeddings_train = None
                self.embeddings_val = None
        else:
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val

            if embeddings_train is not None and embeddings_val is not None:
                if not isinstance(embeddings_train, list):
                    embeddings_train = [embeddings_train]
                if not isinstance(embeddings_val, list):
                    embeddings_val = [embeddings_val]
                self.embeddings_train = embeddings_train
                self.embeddings_val = embeddings_val
            else:
                self.embeddings_train = None
                self.embeddings_val = None

        # Fit the preprocessor on the combined training and validation data
        combined_X = pd.concat([self.X_train, self.X_val], axis=0).reset_index(drop=True)  # type: ignore[arg-type]
        combined_y = np.concatenate((self.y_train, self.y_val), axis=0)

        if self.embeddings_train is not None and self.embeddings_val is not None:
            combined_embeddings = [
                np.concatenate((emb_train, emb_val), axis=0)
                for emb_train, emb_val in zip(self.embeddings_train, self.embeddings_val, strict=False)
            ]
        else:
            combined_embeddings = None

        self.preprocessor.fit(combined_X, combined_y, combined_embeddings)

        # Update feature info based on the actual processed data
        (
            self.num_feature_info,
            self.cat_feature_info,
            self.embedding_feature_info,
        ) = self.preprocessor.get_feature_info()

    def setup(self, stage: str):
        """Transform the data and create DataLoaders."""
        if stage == "fit":
            train_preprocessed_data = self.preprocessor.transform(self.X_train, self.embeddings_train)
            val_preprocessed_data = self.preprocessor.transform(self.X_val, self.embeddings_val)

            # Initialize lists for tensors
            train_cat_tensors = []
            train_num_tensors = []
            train_emb_tensors = []
            val_cat_tensors = []
            val_num_tensors = []
            val_emb_tensors = []

            # Populate tensors for categorical features, if present in processed data
            for key in self.cat_feature_info:  # type: ignore
                dtype = (
                    torch.float32
                    if any(x in self.cat_feature_info[key]["preprocessing"] for x in ["onehot", "pretrained"])  # type: ignore
                    else torch.long
                )

                cat_key = "cat_" + str(key)  # Assuming categorical keys are prefixed with 'cat_'
                if cat_key in train_preprocessed_data:
                    train_cat_tensors.append(torch.tensor(train_preprocessed_data[cat_key], dtype=dtype))
                if cat_key in val_preprocessed_data:
                    val_cat_tensors.append(torch.tensor(val_preprocessed_data[cat_key], dtype=dtype))

                binned_key = "num_" + str(key)  # for binned features
                if binned_key in train_preprocessed_data:
                    train_cat_tensors.append(torch.tensor(train_preprocessed_data[binned_key], dtype=dtype))

                if binned_key in val_preprocessed_data:
                    val_cat_tensors.append(torch.tensor(val_preprocessed_data[binned_key], dtype=dtype))

            # Populate tensors for numerical features, if present in processed data
            for key in self.num_feature_info:  # type: ignore
                num_key = "num_" + str(key)  # Assuming numerical keys are prefixed with 'num_'
                if num_key in train_preprocessed_data:
                    train_num_tensors.append(torch.tensor(train_preprocessed_data[num_key], dtype=torch.float32))
                if num_key in val_preprocessed_data:
                    val_num_tensors.append(torch.tensor(val_preprocessed_data[num_key], dtype=torch.float32))

            if self.embedding_feature_info is not None:
                for key in self.embedding_feature_info:
                    if key in train_preprocessed_data:
                        train_emb_tensors.append(torch.tensor(train_preprocessed_data[key], dtype=torch.float32))
                    if key in val_preprocessed_data:
                        val_emb_tensors.append(torch.tensor(val_preprocessed_data[key], dtype=torch.float32))

            train_labels = torch.tensor(self.y_train, dtype=self.labels_dtype).unsqueeze(dim=1)
            val_labels = torch.tensor(self.y_val, dtype=self.labels_dtype).unsqueeze(dim=1)

            self.train_dataset = MambularDataset(
                train_cat_tensors,
                train_num_tensors,
                train_emb_tensors,
                train_labels,
                regression=self.regression,
            )
            self.val_dataset = MambularDataset(
                val_cat_tensors,
                val_num_tensors,
                val_emb_tensors,
                val_labels,
                regression=self.regression,
            )

    def preprocess_new_data(self, X, embeddings=None):
        cat_tensors = []
        num_tensors = []
        emb_tensors = []
        preprocessed_data = self.preprocessor.transform(X, embeddings)

        # Populate tensors for categorical features, if present in processed data
        for key in self.cat_feature_info:  # type: ignore
            dtype = (
                torch.float32
                if any(x in self.cat_feature_info[key]["preprocessing"] for x in ["onehot", "pretrained"])  # type: ignore
                else torch.long
            )
            cat_key = "cat_" + str(key)  # Assuming categorical keys are prefixed with 'cat_'
            if cat_key in preprocessed_data:
                cat_tensors.append(torch.tensor(preprocessed_data[cat_key], dtype=dtype))

            binned_key = "num_" + str(key)  # for binned features
            if binned_key in preprocessed_data:
                cat_tensors.append(torch.tensor(preprocessed_data[binned_key], dtype=dtype))

        # Populate tensors for numerical features, if present in processed data
        for key in self.num_feature_info:  # type: ignore
            num_key = "num_" + str(key)  # Assuming numerical keys are prefixed with 'num_'
            if num_key in preprocessed_data:
                num_tensors.append(torch.tensor(preprocessed_data[num_key], dtype=torch.float32))

        if self.embedding_feature_info is not None:
            for key in self.embedding_feature_info:
                if key in preprocessed_data:
                    emb_tensors.append(torch.tensor(preprocessed_data[key], dtype=torch.float32))

        return MambularDataset(
            cat_tensors,
            num_tensors,
            emb_tensors,
            labels=None,
            regression=self.regression,
        )

    def assign_predict_dataset(self, X, embeddings=None):
        self.predict_dataset = self.preprocess_new_data(X, embeddings)

    def assign_test_dataset(self, X, embeddings=None):
        self.test_dataset = self.preprocess_new_data(X, embeddings)

    def train_dataloader(self):
        """Returns the training dataloader.

        Returns:
            DataLoader: DataLoader instance for the training dataset.
        """
        if hasattr(self, "train_dataset"):
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                **self.dataloader_kwargs,
            )
        else:
            raise ValueError("No training dataset provided!")

    def val_dataloader(self):
        """Returns the validation dataloader.

        Returns:
            DataLoader: DataLoader instance for the validation dataset.
        """
        if hasattr(self, "val_dataset"):
            return DataLoader(self.val_dataset, batch_size=self.batch_size, **self.dataloader_kwargs)
        else:
            raise ValueError("No validation dataset provided!")

    def test_dataloader(self):
        """Returns the test dataloader.

        Returns:
            DataLoader: DataLoader instance for the test dataset.
        """
        if hasattr(self, "test_dataset"):
            return DataLoader(self.test_dataset, batch_size=self.batch_size, **self.dataloader_kwargs)
        else:
            raise ValueError("No test dataset provided!")

    def predict_dataloader(self):
        if hasattr(self, "predict_dataset"):
            return DataLoader(
                self.predict_dataset,
                batch_size=self.batch_size,
                **self.dataloader_kwargs,
            )
        else:
            raise ValueError("No predict dataset provided!")
