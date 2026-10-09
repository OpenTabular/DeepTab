"""Prepare tabular features, labels, and data loaders for Lightning workflows."""

import lightning as pl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from deeptab.core.exceptions import multi_output_regression_error
from deeptab.core.preprocessing import fit_preprocessor
from deeptab.data.dataset import TabularDataset
from deeptab.data.schema import FeatureSchema


def _prepare_regression_labels(y, *, vector_targets: bool = False) -> torch.Tensor:
    """Build float labels without flattening the sample-to-target alignment.

    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values. A one-dimensional input becomes a single-column matrix.
    vector_targets : bool, default=False
        Allow multiple target columns for multivariate distribution families.
        When false, only a vector or single-column matrix is accepted.

    Returns
    -------
    torch.Tensor
        Float32 labels of shape ``(n_samples, 1)`` for scalar targets, or
        ``(n_samples, n_targets)`` when vector targets are enabled.

    Raises
    ------
    DataError
        If the target shape is unsupported, including multiple columns when
        ``vector_targets`` is false.
    """
    y_arr = np.asarray(y)
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]
    elif not (y_arr.ndim == 2 and (y_arr.shape[1] == 1 or vector_targets)):
        raise multi_output_regression_error(y_arr.shape)
    return torch.as_tensor(y_arr, dtype=torch.float32)


class TabularDataModule(pl.LightningDataModule):
    """Manage tabular preprocessing, datasets, and Lightning data loaders.

    Training and validation features are transformed with a shared fitted
    preprocessor. Labels are converted to tensors according to the task, and
    optional weighted sampling controls how training rows are drawn.

    Parameters
    ----------
    preprocessor : object
        Preprocessor implementing ``fit``, ``transform``, and
        ``get_feature_info`` for tabular features and optional embeddings.
    batch_size : int
        Number of samples per data-loader batch.
    shuffle : bool
        Shuffle training rows when no weighted sampler is configured.
    regression : bool
        Use floating-point regression labels when true. Otherwise, use
        classification label shapes and dtypes.
    X_val : pandas.DataFrame or array-like, optional
        Validation features initially stored on the module. Training and
        validation data are assigned by :meth:`preprocess_data`.
    y_val : array-like, optional
        Validation targets initially stored on the module.
    val_size : float, default=0.2
        Validation fraction stored on the module. Pass the desired split
        fraction to :meth:`preprocess_data` when preparing data.
    random_state : int or None, default=101
        Seed for training-loader ordering and weighted sampling. The split
        seed is supplied separately to :meth:`preprocess_data`. ``None`` uses
        the generators' unseeded behavior.
    stratify : bool, default=True
        Preserve class proportions in automatic validation splits. Ignored
        for regression and when explicit validation data are supplied.
    sampler : bool, str or array-like, optional
        ``True`` or ``"balanced"`` samples with inverse class frequencies.
        An array supplies one sampling weight per original training row.
        ``None`` or ``False`` disables weighted sampling.
    vector_targets : bool, default=False
        Preserve multiple target columns for multivariate regression
        distributions. Ignored for classification.
    **dataloader_kwargs : dict
        Additional keyword arguments forwarded to PyTorch ``DataLoader``,
        such as ``num_workers`` or ``pin_memory``. ``drop_last`` applies only
        to training; validation, test, and prediction retain partial batches.

    Notes
    -----
    Call :meth:`preprocess_data` to fit preprocessing and assign the data,
    then ``setup("fit")`` to create training and validation datasets.
    Prediction and test datasets are assigned separately using
    :meth:`assign_predict_dataset` and :meth:`assign_test_dataset`.
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
        random_state: int | None = 101,
        stratify=True,
        sampler=None,
        vector_targets=False,
        **dataloader_kwargs,
    ):
        """Initialize data handling with the options documented on the class."""
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
        self.stratify = stratify
        self.sampler = sampler
        self.vector_targets = vector_targets
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
        random_state: int | None = 101,
    ):
        """Split data, fit preprocessing, and record feature metadata.

        Parameters
        ----------
        X_train : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Training features before an optional validation split.
        y_train : array-like of shape (n_samples,) or (n_samples, n_targets)
            Targets aligned with training rows. Multiple target columns require
            ``vector_targets=True`` when regression datasets are created.
        X_val : pandas.DataFrame or array-like, optional
            Explicit validation features. If either ``X_val`` or ``y_val`` is
            missing, both validation arrays are created from the training data.
        y_val : array-like, optional
            Explicit validation targets aligned with ``X_val``.
        embeddings_train : array-like or list of array-like, optional
            One or more embedding matrices aligned with training rows. They
            are split alongside the features and targets when needed.
        embeddings_val : array-like or list of array-like, optional
            Embedding matrices aligned with explicit validation rows. With an
            explicit validation set, embeddings are retained only when both
            training and validation embeddings are supplied.
        val_size : float, default=0.2
            Fraction of training rows reserved for automatic validation.
        random_state : int or None, default=101
            Seed for the automatic split and matching sampling-weight split.
            This argument does not change the module's loader seed.

        Notes
        -----
        The preprocessor is fitted only on the resulting training partition.
        This method records data and feature metadata but does not create
        datasets; call ``setup("fit")`` afterward.
        """

        if X_val is None or y_val is None:
            split_data = [X_train, y_train]

            # Stratify classification splits on the labels when enabled; a
            # continuous regression target cannot be stratified.
            stratify = y_train if (self.stratify and not self.regression) else None

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

        self.preprocessor = fit_preprocessor(self.preprocessor, self.X_train, self.y_train, self.embeddings_train)

        # Align explicit per-row sampling weights with the (possibly auto-split) train set.
        self._train_sample_weights = self._resolve_train_sample_weights(
            y_train if (X_val is None or y_val is None) else None,
            val_size=val_size,
            random_state=random_state,
        )

        # Update feature info based on the actual processed data.
        # `get_feature_info` has its own independent `verbose=True` default
        # that would log the per-feature table a second time here; the
        # preprocessor's own `fit()` already logs it once when its `verbose`
        # level calls for it, so this call is only used for its return value.
        (
            self.num_feature_info,
            self.cat_feature_info,
            self.embedding_feature_info,
        ) = self.preprocessor.get_feature_info(verbose=False)

    def _resolve_train_sample_weights(self, y_full, val_size, random_state):
        """Align explicit sampling weights with the training partition.

        Parameters
        ----------
        y_full : array-like or None
            Targets before an automatic split. ``None`` indicates an explicit
            validation set, so weights already correspond to training rows.
        val_size : float
            Validation fraction used when splitting the original targets.
        random_state : int or None
            Seed used for the feature and target split.

        Returns
        -------
        numpy.ndarray or None
            Float64 sampling weights aligned with ``self.y_train``, or ``None``
            when explicit weights are not configured. Balanced weights are
            computed later from the training labels.

        Raises
        ------
        ValueError
            If the number of weights does not match the corresponding targets.
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
        stratify = y_full if (self.stratify and not self.regression) else None
        train_weights, _ = train_test_split(weights, test_size=val_size, random_state=random_state, stratify=stratify)
        return train_weights

    def setup(self, stage: str):
        """Transform training and validation data into tensor-backed datasets.

        Parameters
        ----------
        stage : str
            Lightning lifecycle stage. Only ``"fit"`` creates datasets; other
            stages leave the module unchanged.

        Raises
        ------
        DataError
            If regression targets have an unsupported shape.

        Notes
        -----
        Requires data and feature metadata prepared by :meth:`preprocess_data`.
        Scalar regression labels have shape ``(n_samples, 1)`` and dtype
        float32; vector regression labels retain their columns. Classification
        with more than two training classes uses int64 labels of shape
        ``(n_samples,)``. Otherwise, classification labels use float32 and
        shape ``(n_samples, 1)``. Data loaders are created by the loader methods,
        not by this method.
        """
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

            # Prepare labels with appropriate shape and dtype based on task.
            if self.regression:
                # Regression: float32, shape (batch_size, 1)
                train_labels = _prepare_regression_labels(self.y_train, vector_targets=self.vector_targets)
                val_labels = _prepare_regression_labels(self.y_val, vector_targets=self.vector_targets)
            else:
                # Classification: determine if binary or multiclass
                num_classes = len(np.unique(self.y_train))  # type: ignore[arg-type]
                if num_classes > 2:
                    # Multiclass: long dtype, shape (batch_size,) - no unsqueeze
                    train_labels = torch.as_tensor(np.asarray(self.y_train).reshape(-1), dtype=torch.long)
                    val_labels = torch.as_tensor(np.asarray(self.y_val).reshape(-1), dtype=torch.long)
                else:
                    # Binary: float32, shape (batch_size, 1)
                    train_labels = torch.as_tensor(np.asarray(self.y_train).reshape(-1, 1), dtype=torch.float32)
                    val_labels = torch.as_tensor(np.asarray(self.y_val).reshape(-1, 1), dtype=torch.float32)

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
        """Transform new features into an unlabeled dataset.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Features compatible with the fitted preprocessor.
        embeddings : array-like or list of array-like, optional
            Embedding matrices aligned with the feature rows.

        Returns
        -------
        TabularDataset
            Dataset containing transformed categorical, numerical, and
            embedding tensors without labels.

        Notes
        -----
        Reuses the fitted preprocessor and feature metadata without refitting.
        """
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
        """Prepare and store the unlabeled prediction dataset.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Features compatible with the fitted preprocessor.
        embeddings : array-like or list of array-like, optional
            Embedding matrices aligned with the feature rows.
        """
        self.predict_dataset = self.preprocess_new_data(X, embeddings)

    def assign_test_dataset(self, X, embeddings=None):
        """Prepare and store the unlabeled test dataset.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Features compatible with the fitted preprocessor.
        embeddings : array-like or list of array-like, optional
            Embedding matrices aligned with the feature rows.
        """
        self.test_dataset = self.preprocess_new_data(X, embeddings)

    def _build_train_sampler(self):
        """Build a weighted training sampler when configured.

        Returns
        -------
        WeightedRandomSampler or None
            Sampler drawing as many rows as the training partition, with
            replacement, or ``None`` to use the loader's shuffle setting.

        Raises
        ------
        ValueError
            If a string sampler specification is not ``"balanced"``.

        Notes
        -----
        Explicit weights must be aligned by :meth:`preprocess_data` first.
        Balanced sampling uses inverse frequencies from the training labels.
        A non-null ``random_state`` seeds the sampler's generator.
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

        generator = None
        if self.random_state is not None:
            generator = torch.Generator()
            generator.manual_seed(self.random_state)
        return WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),  # type: ignore[arg-type]
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )

    def train_dataloader(self):
        """Create a training loader with optional weighted sampling.

        Returns
        -------
        DataLoader
            Batches from the training dataset. Weighted sampling replaces
            shuffling when a sampler is configured.

        Raises
        ------
        ValueError
            If ``setup("fit")`` has not created the training dataset.

        Notes
        -----
        A non-null ``random_state`` seeds the loader's generator. Additional
        loader options come from ``dataloader_kwargs``.
        """
        if hasattr(self, "train_dataset"):
            sampler = self._build_train_sampler()
            # Build a seeded Generator for worker-process batch ordering when
            # num_workers > 0; falls back to None (global RNG) otherwise.
            generator = None
            if self.random_state is not None:
                generator = torch.Generator()
                generator.manual_seed(self.random_state)
            if sampler is not None:
                # A sampler and shuffle are mutually exclusive; the sampler randomises order.
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    sampler=sampler,
                    generator=generator,
                    **self.dataloader_kwargs,
                )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                generator=generator,
                **self.dataloader_kwargs,
            )
        else:
            raise ValueError("No training dataset provided!")

    def val_dataloader(self):
        """Create a loader for the validation dataset.

        Returns
        -------
        DataLoader
            Validation batches in dataset order, unless overridden by
            ``dataloader_kwargs``.

        Raises
        ------
        ValueError
            If ``setup("fit")`` has not created the validation dataset.
        """
        if hasattr(self, "val_dataset"):
            return DataLoader(
                self.val_dataset, batch_size=self.batch_size, **{**self.dataloader_kwargs, "drop_last": False}
            )
        else:
            raise ValueError("No validation dataset provided!")

    def test_dataloader(self):
        """Create a loader for the unlabeled test dataset.

        Returns
        -------
        DataLoader
            Test batches using the configured batch size and loader options.

        Raises
        ------
        ValueError
            If :meth:`assign_test_dataset` has not assigned the test dataset.
        """
        if hasattr(self, "test_dataset"):
            return DataLoader(
                self.test_dataset, batch_size=self.batch_size, **{**self.dataloader_kwargs, "drop_last": False}
            )
        else:
            raise ValueError("No test dataset provided!")

    def predict_dataloader(self):
        """Create a loader for the unlabeled prediction dataset.

        Returns
        -------
        DataLoader
            Prediction batches using the configured batch size and loader
            options.

        Raises
        ------
        ValueError
            If :meth:`assign_predict_dataset` has not assigned a dataset.
        """
        if hasattr(self, "predict_dataset"):
            return DataLoader(
                self.predict_dataset,
                batch_size=self.batch_size,
                **{**self.dataloader_kwargs, "drop_last": False},
            )
        else:
            raise ValueError("No predict dataset provided!")

    @property
    def schema(self) -> FeatureSchema | None:
        """Build a feature schema from the recorded preprocessing metadata.

        Returns
        -------
        FeatureSchema or None
            Schema describing categorical, numerical, and embedding features,
            or ``None`` if numerical or categorical metadata is unavailable.
        """
        if self.num_feature_info is None or self.cat_feature_info is None:
            return None

        return FeatureSchema.from_preprocessor_info(
            self.num_feature_info,
            self.cat_feature_info,
            self.embedding_feature_info,
        )
