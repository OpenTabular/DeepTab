import lightning as pl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from deeptab.data.dataset import TabularDataset
from deeptab.data.schema import FeatureSchema


class TabularDataModule(pl.LightningDataModule):
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
        sampler=None,
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
        self.embedding_feature_info = None
        self.X_val = X_val
        self.y_val = y_val
        self.val_size = val_size
        self.random_state = random_state
        self.regression = regression
        self.sampler = sampler
        self._train_sample_weights = None
        if self.regression:
            self.labels_dtype = torch.float32
        else:
            self.labels_dtype = torch.long

        # Initialize placeholders for data
        self.input_columns_: list[str] | None = None
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

            # Determine stratify parameter for classification tasks
            stratify = y_train if not self.regression else None

            if embeddings_train is not None:
                if not isinstance(embeddings_train, list):
                    embeddings_train = [embeddings_train]
                if embeddings_val is not None and not isinstance(embeddings_val, list):
                    embeddings_val = [embeddings_val]

                split_data += embeddings_train
                split_result = train_test_split(
                    *split_data, test_size=val_size, random_state=random_state, stratify=stratify
                )

                self.X_train, self.X_val, self.y_train, self.y_val = split_result[:4]
                self.embeddings_train = split_result[4::2]
                self.embeddings_val = split_result[5::2]
            else:
                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                    *split_data, test_size=val_size, random_state=random_state, stratify=stratify
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

        self.preprocessor.fit(self.X_train, self.y_train, self.embeddings_train)

        # Align explicit per-row sampling weights with the (possibly auto-split) train set.
        self._train_sample_weights = self._resolve_train_sample_weights(
            y_train if (X_val is None or y_val is None) else None,
            val_size=val_size,
            random_state=random_state,
        )

        # Update feature info based on the actual processed data
        (
            self.num_feature_info,
            self.cat_feature_info,
            self.embedding_feature_info,
        ) = self.preprocessor.get_feature_info()

    def _resolve_train_sample_weights(self, y_full, val_size, random_state):
        """Resolve explicit per-row sampling weights, splitting them to match the train set.

        Returns the per-row weights aligned with ``self.y_train`` when ``self.sampler``
        is an explicit array of weights, otherwise ``None`` (the ``"balanced"`` case is
        computed lazily from the training labels in :meth:`train_dataloader`).
        """
        sampler = self.sampler
        if sampler is None or isinstance(sampler, bool | str):
            return None

        weights = np.asarray(sampler, dtype=np.float64)
        if y_full is None:
            # Explicit validation set was provided -> no split, weights map 1:1 onto X_train.
            if len(weights) != len(self.y_train):  # type: ignore[arg-type]
                raise ValueError(
                    f"sample_weight has length {len(weights)} but the training set has {len(self.y_train)} rows."  # type: ignore[arg-type]
                )
            return weights

        if len(weights) != len(y_full):
            raise ValueError(f"sample_weight has length {len(weights)} but X has {len(y_full)} rows.")
        # Same random_state + stratify + test_size reproduce the X/y partition exactly.
        stratify = y_full if not self.regression else None
        train_weights, _ = train_test_split(weights, test_size=val_size, random_state=random_state, stratify=stratify)
        return train_weights

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

            # Prepare labels with appropriate shape and dtype based on task
            if self.regression:
                # Regression: float32, shape (batch_size, 1)
                train_labels = torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(dim=1)
                val_labels = torch.tensor(self.y_val, dtype=torch.float32).unsqueeze(dim=1)
            else:
                # Classification: determine if binary or multiclass
                num_classes = len(np.unique(self.y_train))  # type: ignore[arg-type]
                if num_classes > 2:
                    # Multiclass: long dtype, shape (batch_size,) - no unsqueeze
                    train_labels = torch.tensor(self.y_train, dtype=torch.long).view(-1)
                    val_labels = torch.tensor(self.y_val, dtype=torch.long).view(-1)
                else:
                    # Binary: float32, shape (batch_size, 1)
                    train_labels = torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(dim=1)
                    val_labels = torch.tensor(self.y_val, dtype=torch.float32).unsqueeze(dim=1)

            self.train_dataset = TabularDataset(
                train_cat_tensors,
                train_num_tensors,
                train_emb_tensors,
                train_labels,
            )
            self.val_dataset = TabularDataset(
                val_cat_tensors,
                val_num_tensors,
                val_emb_tensors,
                val_labels,
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

        return TabularDataset(
            cat_tensors,
            num_tensors,
            emb_tensors,
            labels=None,
        )

    def assign_predict_dataset(self, X, embeddings=None):
        self.predict_dataset = self.preprocess_new_data(X, embeddings)

    def assign_test_dataset(self, X, embeddings=None):
        self.test_dataset = self.preprocess_new_data(X, embeddings)

    def _build_train_sampler(self):
        """Build a :class:`WeightedRandomSampler` for the training set, if requested.

        Returns ``None`` when no weighted sampling is configured, in which case the
        DataLoader falls back to plain ``shuffle``.
        """
        spec = self.sampler
        if spec is None or spec is False:
            return None

        if self._train_sample_weights is not None:
            weights = np.asarray(self._train_sample_weights, dtype=np.float64)
        elif spec is True or spec == "balanced":
            y = np.asarray(self.y_train)
            classes, counts = np.unique(y, return_counts=True)
            inv_freq = {cls: 1.0 / count for cls, count in zip(classes, counts, strict=False)}
            weights = np.array([inv_freq[label] for label in y], dtype=np.float64)
        elif isinstance(spec, str):
            raise ValueError(f"Unsupported sampler {spec!r}; expected 'balanced', True, or an array of weights.")
        else:
            return None

        return WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),  # type: ignore[arg-type]
            num_samples=len(weights),
            replacement=True,
        )

    def train_dataloader(self):
        """Returns the training dataloader.

        Returns:
            DataLoader: DataLoader instance for the training dataset.
        """
        if hasattr(self, "train_dataset"):
            sampler = self._build_train_sampler()
            if sampler is not None:
                # A sampler and shuffle are mutually exclusive; the sampler randomises order.
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    sampler=sampler,
                    **self.dataloader_kwargs,
                )
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

    @property
    def schema(self) -> FeatureSchema | None:
        """Get the feature schema after preprocessing.

        Returns
        -------
        FeatureSchema or None
            Feature schema with metadata about categorical, numerical, and
            embedding features, or None if preprocessing hasn't been done yet.
        """
        if self.num_feature_info is None or self.cat_feature_info is None:
            return None

        return FeatureSchema.from_preprocessor_info(
            self.num_feature_info,
            self.cat_feature_info,
            self.embedding_feature_info,
        )
