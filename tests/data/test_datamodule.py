"""Tests for datamodule."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import make_classification, make_regression

from deeptab.data import FeatureSchema, TabularDataModule, TabularDataset


@pytest.fixture
def regression_data():
    """Generate synthetic regression dataset."""
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)  # type: ignore[misc]
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])  # type: ignore[arg-type]
    return X_df, y


@pytest.fixture
def classification_data():
    """Generate synthetic classification dataset with imbalanced classes."""
    X, y = make_classification(  # type: ignore[misc]
        n_samples=200,
        n_features=10,
        n_classes=3,
        n_informative=8,
        n_redundant=2,
        weights=[0.6, 0.3, 0.1],
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])  # type: ignore[arg-type]
    return X_df, y


@pytest.fixture
def binary_classification_data():
    """Generate synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_classes=2,
        n_informative=8,
        weights=[0.8, 0.2],
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])  # type: ignore[arg-type]
    return X_df, y


class TestTabularDataModuleContract:
    """Test the contract and interface of TabularDataModule."""

    def test_datamodule_initialization(self):
        """Test datamodule can be initialized with required parameters."""
        from pretab.preprocessor import Preprocessor

        preprocessor = Preprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=True,
            regression=True,
        )

        assert datamodule.batch_size == 32
        assert datamodule.shuffle is True
        assert datamodule.regression is True

    def test_datamodule_preprocess_data_creates_splits(self, regression_data):
        """Test preprocess_data creates train/val splits."""
        from pretab.preprocessor import Preprocessor

        X, y = regression_data
        preprocessor = Preprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=True,
            regression=True,
        )

        datamodule.preprocess_data(X, y)

        assert datamodule.X_train is not None
        assert datamodule.X_val is not None
        assert datamodule.y_train is not None
        assert datamodule.y_val is not None
        # Default split is 80/20
        assert len(datamodule.X_train) == 160
        assert len(datamodule.X_val) == 40

    def test_datamodule_accepts_external_validation_set(self, regression_data):
        """Test datamodule accepts pre-split validation data."""
        from pretab.preprocessor import Preprocessor

        X, y = regression_data
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]

        preprocessor = Preprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=True,
            regression=True,
        )

        datamodule.preprocess_data(X_train, y_train, X_val, y_val)

        assert len(datamodule.X_train) == 150  # type: ignore[arg-type]
        assert len(datamodule.X_val) == 50  # type: ignore[arg-type]

    def test_datamodule_fits_preprocessor_on_training_split_only(self, regression_data):
        """Test validation data is transformed only and not used to fit preprocessing."""

        class RecordingPreprocessor:
            def fit(self, X, y, embeddings=None):
                self.fit_rows = len(X)
                self.fit_index = list(X.index)
                self.fit_y_rows = len(y)
                self.fit_embeddings = embeddings
                return self

            def get_feature_info(self, verbose=True):
                return {}, {}, None

        X, y = regression_data
        X_train, X_val = X.iloc[:150], X.iloc[150:]
        y_train, y_val = y[:150], y[150:]
        preprocessor = RecordingPreprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=True,
            regression=True,
        )

        datamodule.preprocess_data(X_train, y_train, X_val, y_val)

        assert preprocessor.fit_rows == len(X_train)
        assert preprocessor.fit_y_rows == len(y_train)
        assert preprocessor.fit_index == list(X_train.index)

    def test_datamodule_stratified_split_for_classification(self, classification_data):
        """Test datamodule uses stratified split for classification."""
        from pretab.preprocessor import Preprocessor

        X, y = classification_data
        preprocessor = Preprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=True,
            regression=False,
        )

        datamodule.preprocess_data(X, y)

        # Check class distribution is preserved
        train_dist = np.bincount(datamodule.y_train.astype(int)) / len(datamodule.y_train)  # type: ignore[union-attr, arg-type]
        val_dist = np.bincount(datamodule.y_val.astype(int)) / len(datamodule.y_val)  # type: ignore[union-attr, arg-type]
        overall_dist = np.bincount(y.astype(int)) / len(y)

        # Allow 5% tolerance for distribution preservation
        np.testing.assert_allclose(train_dist, overall_dist, atol=0.05)
        np.testing.assert_allclose(val_dist, overall_dist, atol=0.05)

    def test_datamodule_no_stratification_for_regression(self, regression_data):
        """Test datamodule doesn't stratify for regression."""
        from pretab.preprocessor import Preprocessor

        X, y = regression_data
        preprocessor = Preprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=True,
            regression=True,
        )

        # Should not raise error
        datamodule.preprocess_data(X, y)
        assert datamodule.X_train is not None

    def test_datamodule_stratify_defaults_to_true(self):
        """The stratify flag defaults to True and is stored on the datamodule."""
        from pretab.preprocessor import Preprocessor

        preprocessor = Preprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=True,
            regression=False,
        )

        assert datamodule.stratify is True

    def test_datamodule_stratify_false_allows_singleton_class(self):
        """With stratify=False a class with a single member no longer blocks the split.

        Stratified splitting raises when the least-populated class has fewer
        members than the number of splits, so a singleton class is a clean way
        to prove the flag actually switches stratification off.
        """
        from pretab.preprocessor import Preprocessor

        # 20 rows: class 0 (x10), class 1 (x9), class 2 (x1 -> singleton).
        X = pd.DataFrame({"f": list(range(20))})
        y = np.array([0] * 10 + [1] * 9 + [2])

        stratified = TabularDataModule(
            preprocessor=Preprocessor(),
            batch_size=4,
            shuffle=True,
            regression=False,
            stratify=True,
        )
        with pytest.raises(ValueError):
            stratified.preprocess_data(X, y)

        unstratified = TabularDataModule(
            preprocessor=Preprocessor(),
            batch_size=4,
            shuffle=True,
            regression=False,
            stratify=False,
        )
        unstratified.preprocess_data(X, y)
        assert unstratified.X_train is not None
        assert unstratified.X_val is not None

    def test_datamodule_setup_creates_datasets(self, regression_data):
        """Test setup() creates train and val datasets."""
        from pretab.preprocessor import Preprocessor

        X, y = regression_data
        preprocessor = Preprocessor(output_structure="blocks")
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=True,
            regression=True,
        )

        datamodule.preprocess_data(X, y)
        datamodule.setup("fit")

        assert hasattr(datamodule, "train_dataset")
        assert hasattr(datamodule, "val_dataset")
        assert isinstance(datamodule.train_dataset, TabularDataset)
        assert isinstance(datamodule.val_dataset, TabularDataset)

    def test_datamodule_dataloaders_work(self, regression_data):
        """Test datamodule creates working dataloaders."""
        from pretab.preprocessor import Preprocessor

        X, y = regression_data
        preprocessor = Preprocessor(output_structure="blocks")
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=True,
            regression=True,
        )

        datamodule.preprocess_data(X, y)
        datamodule.setup("fit")

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None

        # Check batch can be retrieved
        batch = next(iter(train_loader))
        assert batch is not None

    def test_datamodule_schema_property(self, regression_data):
        """Test schema property returns FeatureSchema after preprocessing."""
        from pretab.preprocessor import Preprocessor

        X, y = regression_data
        preprocessor = Preprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=True,
            regression=True,
        )

        # Before preprocessing, schema should be None
        assert datamodule.schema is None

        datamodule.preprocess_data(X, y)

        # After preprocessing, schema should be available
        schema = datamodule.schema
        assert schema is not None
        assert isinstance(schema, FeatureSchema)
        assert schema.num_numerical_features > 0

    def test_datamodule_handles_embeddings(self, regression_data):
        """Test datamodule handles embedding features."""
        from pretab.preprocessor import Preprocessor

        X, y = regression_data
        embeddings_train = np.random.randn(200, 16)
        embeddings_val = None

        preprocessor = Preprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=True,
            regression=True,
        )

        datamodule.preprocess_data(X, y, embeddings_train=embeddings_train)

        assert datamodule.embeddings_train is not None
        assert datamodule.embeddings_val is not None

    def test_datamodule_multiclass_label_shape(self, classification_data):
        """Test multiclass labels have correct shape (batch_size,) not (batch_size, 1)."""
        from pretab.preprocessor import Preprocessor

        X, y = classification_data
        preprocessor = Preprocessor(output_structure="blocks")
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=False,
            regression=False,
        )

        datamodule.preprocess_data(X, y)
        datamodule.setup("fit")

        # Get a batch
        batch = next(iter(datamodule.train_dataloader()))
        _features, labels = batch

        # Multiclass labels should be (batch_size,) shape
        assert labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1)
        if labels.ndim == 1:
            assert labels.shape[0] <= 32
        assert labels.dtype == torch.long

    def test_datamodule_binary_label_shape(self, binary_classification_data):
        """Test binary classification labels have correct shape (batch_size, 1)."""
        from pretab.preprocessor import Preprocessor

        X, y = binary_classification_data
        preprocessor = Preprocessor(output_structure="blocks")
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=False,
            regression=False,
        )

        datamodule.preprocess_data(X, y)
        datamodule.setup("fit")

        # Get a batch
        batch = next(iter(datamodule.train_dataloader()))
        _features, labels = batch

        # Binary labels should be (batch_size, 1) shape
        assert labels.shape[1] == 1
        assert labels.dtype == torch.float32

    def test_datamodule_regression_label_shape(self, regression_data):
        """Test regression labels have correct shape (batch_size, 1)."""
        from pretab.preprocessor import Preprocessor

        X, y = regression_data
        preprocessor = Preprocessor(output_structure="blocks")
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=False,
            regression=True,
        )

        datamodule.preprocess_data(X, y)
        datamodule.setup("fit")

        # Get a batch
        batch = next(iter(datamodule.train_dataloader()))
        _features, labels = batch

        # Regression labels should be (batch_size, 1) shape
        assert labels.shape[1] == 1

    def test_datamodule_regression_column_vector_y_matches_1d(self, regression_data):
        """A (n,1) column-vector y must train against the same (batch,1) labels as a 1-D y."""
        from pretab.preprocessor import Preprocessor

        X, y = regression_data

        dm_1d = TabularDataModule(
            preprocessor=Preprocessor(output_structure="blocks"), batch_size=32, shuffle=False, regression=True
        )
        dm_1d.preprocess_data(X, y, random_state=101)
        dm_1d.setup("fit")
        _, labels_1d = next(iter(dm_1d.train_dataloader()))

        dm_2d = TabularDataModule(
            preprocessor=Preprocessor(output_structure="blocks"), batch_size=32, shuffle=False, regression=True
        )
        dm_2d.preprocess_data(X, y.reshape(-1, 1), random_state=101)
        dm_2d.setup("fit")
        _, labels_2d = next(iter(dm_2d.train_dataloader()))

        assert labels_2d.shape == labels_1d.shape == (labels_1d.shape[0], 1)
        assert torch.equal(labels_2d, labels_1d)

    def test_datamodule_binary_column_vector_y_does_not_crash(self, binary_classification_data):
        """A (n,1) column-vector y must not be broadcast into a (batch,1,1) label tensor."""
        from pretab.preprocessor import Preprocessor

        X, y = binary_classification_data
        datamodule = TabularDataModule(
            preprocessor=Preprocessor(output_structure="blocks"), batch_size=32, shuffle=False, regression=False
        )

        datamodule.preprocess_data(X, y.reshape(-1, 1))
        datamodule.setup("fit")

        _features, labels = next(iter(datamodule.train_dataloader()))
        assert labels.shape == (labels.shape[0], 1)
        assert labels.dtype == torch.float32

    def test_datamodule_regression_multi_target_y_raises(self, regression_data):
        """A genuinely multi-column y (n, k>1) must raise instead of being silently reshaped."""
        from deeptab.core.exceptions import DataError
        from deeptab.data.datamodule import _prepare_regression_labels

        _, y = regression_data
        y_multi = np.column_stack([y, y])

        with pytest.raises(DataError, match="multi-output regression"):
            _prepare_regression_labels(y_multi)


class TestValidationLeakage:
    """Regression tests that guard against data leakage from val into train preprocessing."""

    # ------------------------------------------------------------------
    # 1. Index disjointness after automatic split
    # ------------------------------------------------------------------

    def test_auto_split_train_val_indices_are_disjoint(self, regression_data):
        """Rows in the auto-generated train split must not appear in val."""
        from pretab.preprocessor import Preprocessor

        X, y = regression_data
        preprocessor = Preprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=False,
            regression=True,
            val_size=0.2,
            random_state=0,
        )
        datamodule.preprocess_data(X, y)

        train_idx = set(datamodule.X_train.index.tolist())  # type: ignore[union-attr]
        val_idx = set(datamodule.X_val.index.tolist())  # type: ignore[union-attr]

        assert train_idx.isdisjoint(val_idx), "Leakage detected: some row indices appear in both train and val splits."
        assert len(train_idx) + len(val_idx) == len(X), "Train + val sizes must equal the full dataset size."

    # ------------------------------------------------------------------
    # 2. Explicit val set is never fed to the preprocessor fit
    # ------------------------------------------------------------------

    def test_explicit_val_set_not_used_in_preprocessor_fit(self, regression_data):
        """When X_val/y_val are passed explicitly, the preprocessor must only see training rows."""

        fit_index_seen: list[list] = []

        class IndexTrackingPreprocessor:
            def fit(self, X, y, embeddings=None):
                fit_index_seen.append(list(X.index))
                return self

            def get_feature_info(self, verbose=True):
                return {}, {}, None

        X, y = regression_data
        X_train, X_val = X.iloc[:160], X.iloc[160:]
        y_train, y_val = y[:160], y[160:]

        preprocessor = IndexTrackingPreprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=False,
            regression=True,
        )
        datamodule.preprocess_data(X_train, y_train, X_val=X_val, y_val=y_val)

        assert len(fit_index_seen) == 1, "Preprocessor.fit should be called exactly once."
        assert fit_index_seen[0] == list(X_train.index), (
            "Preprocessor was fit on rows other than the training set — validation leakage detected."
        )
        val_idx = set(X_val.index.tolist())
        assert val_idx.isdisjoint(set(fit_index_seen[0])), "Validation row indices were seen during preprocessor fit."

    # ------------------------------------------------------------------
    # 3. Preprocessing fit called exactly once (no re-fit on val)
    # ------------------------------------------------------------------

    def test_preprocessor_fit_called_exactly_once(self, regression_data):
        """Preprocessor.fit must be called exactly once regardless of whether val is explicit."""
        fit_call_count = [0]

        class CountingPreprocessor:
            def fit(self, X, y, embeddings=None):
                fit_call_count[0] += 1
                return self

            def get_feature_info(self, verbose=True):
                return {}, {}, None

        X, y = regression_data
        X_train, X_val = X.iloc[:160], X.iloc[160:]
        y_train, y_val = y[:160], y[160:]

        preprocessor = CountingPreprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=False,
            regression=True,
        )
        datamodule.preprocess_data(X_train, y_train, X_val=X_val, y_val=y_val)

        assert fit_call_count[0] == 1, f"Preprocessor.fit was called {fit_call_count[0]} times; expected exactly 1."

    # ------------------------------------------------------------------
    # 4. Val split size respects requested val_size
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("val_size", [0.1, 0.2, 0.3])
    def test_val_split_size_is_correct(self, regression_data, val_size):
        """The validation split must contain approximately N * val_size rows."""
        import math

        from pretab.preprocessor import Preprocessor

        X, y = regression_data
        n = len(X)
        preprocessor = Preprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=False,
            regression=True,
        )
        datamodule.preprocess_data(X, y, val_size=val_size, random_state=0)

        expected_val = math.ceil(n * val_size)
        actual_val = len(datamodule.X_val)  # type: ignore[arg-type]
        # Allow ±1 row for rounding differences across sklearn versions
        assert abs(actual_val - expected_val) <= 1, (
            f"val_size={val_size}: expected ~{expected_val} val rows, got {actual_val}."
        )

    # ------------------------------------------------------------------
    # 5. Explicit val set passed through unchanged (no extra rows)
    # ------------------------------------------------------------------

    def test_explicit_val_set_size_preserved(self, regression_data):
        """When X_val is supplied, the datamodule must not modify its length."""
        from pretab.preprocessor import Preprocessor

        X, y = regression_data
        X_train, X_val = X.iloc[:150], X.iloc[150:]
        y_train, y_val = y[:150], y[150:]

        preprocessor = Preprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=False,
            regression=True,
        )
        datamodule.preprocess_data(X_train, y_train, X_val=X_val, y_val=y_val)

        assert len(datamodule.X_val) == len(X_val), (  # type: ignore[arg-type]
            "Explicit val set size was changed during preprocessing — unexpected re-split."
        )
        assert len(datamodule.X_train) == len(X_train), (  # type: ignore[arg-type]
            "Training set size was changed when an explicit val set was provided."
        )


class TestDataLoaderGeneratorSeeding:
    """Test that random_state seeds the torch.Generator passed to DataLoader and WeightedRandomSampler."""

    def _make_datamodule(self, regression_data, random_state, sampler=None, shuffle=True):
        from pretab.preprocessor import Preprocessor

        X, y = regression_data
        preprocessor = Preprocessor(output_structure="blocks")
        dm = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=shuffle,
            regression=regression_data is not None and True,
            random_state=random_state,
            sampler=sampler,
        )
        dm.preprocess_data(X, y)
        dm.setup("fit")
        return dm

    def test_train_dataloader_has_generator_when_random_state_set(self, regression_data):
        """DataLoader must carry a seeded Generator when random_state is provided."""
        dm = self._make_datamodule(regression_data, random_state=42)
        loader = dm.train_dataloader()
        assert loader.generator is not None

    def test_train_dataloader_generator_is_none_when_no_random_state(self, regression_data):
        """DataLoader must not inject a Generator when random_state=None."""
        dm = self._make_datamodule(regression_data, random_state=None)
        loader = dm.train_dataloader()
        assert loader.generator is None

    def test_train_dataloader_generator_seed_matches_random_state(self, regression_data):
        """Two DataLoaders built with the same random_state must carry generators with equal initial_seed."""
        dm1 = self._make_datamodule(regression_data, random_state=7)
        dm2 = self._make_datamodule(regression_data, random_state=7)
        seed1 = dm1.train_dataloader().generator.initial_seed()  # type: ignore[union-attr]
        seed2 = dm2.train_dataloader().generator.initial_seed()  # type: ignore[union-attr]
        assert seed1 == seed2

    def test_train_dataloader_different_seeds_differ(self, regression_data):
        """DataLoaders with different random_states must carry generators with different seeds."""
        dm1 = self._make_datamodule(regression_data, random_state=1)
        dm2 = self._make_datamodule(regression_data, random_state=2)
        seed1 = dm1.train_dataloader().generator.initial_seed()  # type: ignore[union-attr]
        seed2 = dm2.train_dataloader().generator.initial_seed()  # type: ignore[union-attr]
        assert seed1 != seed2

    def test_weighted_sampler_has_generator_when_random_state_set(self, classification_data):
        """WeightedRandomSampler must carry a seeded Generator when random_state is provided."""
        from pretab.preprocessor import Preprocessor
        from torch.utils.data import WeightedRandomSampler

        X, y = classification_data
        preprocessor = Preprocessor(output_structure="blocks")
        dm = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=True,
            regression=False,
            random_state=99,
            sampler="balanced",
        )
        dm.preprocess_data(X, y)
        dm.setup("fit")

        sampler = dm._build_train_sampler()
        assert isinstance(sampler, WeightedRandomSampler)
        assert sampler.generator is not None

    def test_weighted_sampler_generator_is_none_when_no_random_state(self, classification_data):
        """WeightedRandomSampler must not inject a Generator when random_state=None."""
        from pretab.preprocessor import Preprocessor
        from torch.utils.data import WeightedRandomSampler

        X, y = classification_data
        preprocessor = Preprocessor(output_structure="blocks")
        dm = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=True,
            regression=False,
            random_state=None,  # type: ignore[arg-type]
            sampler="balanced",
        )
        dm.preprocess_data(X, y)
        dm.setup("fit")

        sampler = dm._build_train_sampler()
        assert isinstance(sampler, WeightedRandomSampler)
        assert sampler.generator is None
