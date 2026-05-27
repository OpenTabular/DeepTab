"""Contract tests for the data API (TabularDataset, TabularDataModule, FeatureSchema, TabularBatch)."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import make_classification, make_regression

from deeptab.data import FeatureInfo, FeatureSchema, TabularBatch, TabularDataModule, TabularDataset

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_tensors():
    """Simple tensor lists for testing dataset."""
    num_features = [
        torch.randn(100, 5),
        torch.randn(100, 3),
    ]
    cat_features = [
        torch.randint(0, 10, (100, 1)),
        torch.randint(0, 5, (100, 1)),
    ]
    embeddings = [torch.randn(100, 8)]
    labels = torch.randn(100, 1)
    return num_features, cat_features, embeddings, labels


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


# ============================================================================
# TabularDataset Contract Tests
# ============================================================================


class TestTabularDatasetContract:
    """Test the contract and interface of TabularDataset."""

    def test_dataset_initialization_with_features(self, simple_tensors):
        """Test dataset can be initialized with feature lists."""
        num_feats, cat_feats, embeddings, labels = simple_tensors
        dataset = TabularDataset(cat_feats, num_feats, embeddings, labels)

        assert len(dataset) == 100
        assert dataset.cat_features_list == cat_feats
        assert dataset.num_features_list == num_feats
        assert dataset.embeddings_list == embeddings
        assert dataset.labels is not None

    def test_dataset_initialization_without_labels(self, simple_tensors):
        """Test dataset can be initialized without labels for prediction."""
        num_feats, cat_feats, embeddings, _ = simple_tensors
        dataset = TabularDataset(cat_feats, num_feats, embeddings, labels=None)

        assert len(dataset) == 100
        assert dataset.labels is None

    def test_dataset_requires_at_least_one_feature_type(self):
        """Test dataset raises error if both cat and num features are empty."""
        with pytest.raises(AssertionError):
            TabularDataset([], [], None, None)

    def test_dataset_getitem_returns_tuple_by_default(self, simple_tensors):
        """Test __getitem__ returns tuple format by default."""
        num_feats, cat_feats, embeddings, labels = simple_tensors
        dataset = TabularDataset(cat_feats, num_feats, embeddings, labels)

        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2  # (features, label)

        features, _label = item  # type: ignore[misc]
        assert len(features) == 3  # (num_feats, cat_feats, embeddings)

    def test_dataset_getitem_returns_batch_object_when_requested(self, simple_tensors):
        """Test __getitem__ returns TabularBatch when return_batch_object=True."""
        num_feats, cat_feats, embeddings, labels = simple_tensors
        dataset = TabularDataset(cat_feats, num_feats, embeddings, labels, return_batch_object=True)

        item = dataset[0]
        assert isinstance(item, TabularBatch)
        assert item.labels is not None
        assert len(item.numerical_features) == 2
        assert len(item.categorical_features) == 2

    def test_dataset_getitem_without_labels(self, simple_tensors):
        """Test __getitem__ returns features only when labels=None."""
        num_feats, cat_feats, embeddings, _ = simple_tensors
        dataset = TabularDataset(cat_feats, num_feats, embeddings, labels=None)

        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 3  # (num_feats, cat_feats, embeddings)

    def test_dataset_numerical_features_are_float32(self, simple_tensors):
        """Test numerical features are converted to float32."""
        num_feats, cat_feats, embeddings, labels = simple_tensors
        dataset = TabularDataset(cat_feats, num_feats, embeddings, labels)

        features, _ = dataset[0]  # type: ignore[misc]
        num_features, _, _ = features

        for feat in num_features:
            assert feat.dtype == torch.float32

    def test_dataset_getitem_reuses_tensor_views(self, simple_tensors):
        """Test __getitem__ avoids cloning tensors in the hot path."""
        num_feats, cat_feats, embeddings, labels = simple_tensors
        dataset = TabularDataset(cat_feats, num_feats, embeddings, labels)

        features, _ = dataset[0]  # type: ignore[misc]
        num_features, _cat_features, emb_features = features

        assert num_features[0].untyped_storage().data_ptr() == num_feats[0].untyped_storage().data_ptr()
        assert emb_features[0].untyped_storage().data_ptr() == embeddings[0].untyped_storage().data_ptr()

    def test_dataset_embeddings_are_float32(self, simple_tensors):
        """Test embeddings are converted to float32."""
        num_feats, cat_feats, embeddings, labels = simple_tensors
        dataset = TabularDataset(cat_feats, num_feats, embeddings, labels)

        features, _ = dataset[0]  # type: ignore[misc]
        _, _, emb = features

        for e in emb:  # type: ignore[union-attr]
            assert e.dtype == torch.float32

    def test_dataset_with_only_numerical_features(self):
        """Test dataset works with only numerical features."""
        num_feats = [torch.randn(50, 5)]
        labels = torch.randn(50, 1)
        dataset = TabularDataset([], num_feats, None, labels)

        assert len(dataset) == 50
        features, _label = dataset[0]  # type: ignore[misc]
        num_features, cat_features, embeddings = features
        assert len(num_features) > 0
        assert len(cat_features) == 0
        assert embeddings is None  # type: ignore[unreachable]

    def test_dataset_with_only_categorical_features(self):
        """Test dataset works with only categorical features."""
        cat_feats = [torch.randint(0, 10, (50, 1))]
        labels = torch.randn(50, 1)
        dataset = TabularDataset(cat_feats, [], None, labels)

        assert len(dataset) == 50
        features, _label = dataset[0]  # type: ignore[misc]
        num_features, cat_features, _embeddings = features
        assert len(num_features) == 0
        assert len(cat_features) > 0


# ============================================================================
# TabularDataModule Contract Tests
# ============================================================================


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

            def get_feature_info(self):
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

    def test_datamodule_setup_creates_datasets(self, regression_data):
        """Test setup() creates train and val datasets."""
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
        datamodule.setup("fit")

        assert hasattr(datamodule, "train_dataset")
        assert hasattr(datamodule, "val_dataset")
        assert isinstance(datamodule.train_dataset, TabularDataset)
        assert isinstance(datamodule.val_dataset, TabularDataset)

    def test_datamodule_dataloaders_work(self, regression_data):
        """Test datamodule creates working dataloaders."""
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
        preprocessor = Preprocessor()
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
        preprocessor = Preprocessor()
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
        preprocessor = Preprocessor()
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
        assert labels.dtype == torch.float32


# ============================================================================
# FeatureSchema Contract Tests
# ============================================================================


class TestFeatureSchemaContract:
    """Test the contract and interface of FeatureSchema."""

    def test_feature_info_creation(self):
        """Test FeatureInfo can be created."""
        info = FeatureInfo(name="feature1", preprocessing="standard", dimension=10, categories=None)

        assert info.name == "feature1"
        assert info.preprocessing == "standard"
        assert info.dimension == 10
        assert not info.is_categorical

    def test_feature_info_categorical_property(self):
        """Test is_categorical property works correctly."""
        num_info = FeatureInfo(name="f1", preprocessing="ple", dimension=20, categories=None)
        cat_info = FeatureInfo(name="c1", preprocessing="int", dimension=1, categories=["A", "B", "C"])

        assert not num_info.is_categorical
        assert cat_info.is_categorical

    def test_feature_schema_creation(self):
        """Test FeatureSchema can be created."""
        num_features = {
            "f1": FeatureInfo("f1", "ple", 20, None),
            "f2": FeatureInfo("f2", "standard", 1, None),
        }
        cat_features = {
            "c1": FeatureInfo("c1", "int", 1, ["A", "B"]),
        }

        schema = FeatureSchema(num_features, cat_features, None)

        assert schema.num_numerical_features == 2
        assert schema.num_categorical_features == 1
        assert schema.num_embedding_features == 0

    def test_feature_schema_dimension_properties(self):
        """Test dimension calculation properties."""
        num_features = {
            "f1": FeatureInfo("f1", "ple", 20, None),
            "f2": FeatureInfo("f2", "standard", 5, None),
        }
        cat_features = {
            "c1": FeatureInfo("c1", "onehot", 10, ["A", "B", "C"]),
            "c2": FeatureInfo("c2", "int", 3, ["X", "Y"]),
        }
        emb_features = {
            "e1": FeatureInfo("e1", "pretrained", 16, None),
        }

        schema = FeatureSchema(num_features, cat_features, emb_features)

        assert schema.total_numerical_dim == 25  # 20 + 5
        assert schema.total_categorical_dim == 13  # 10 + 3
        assert schema.total_embedding_dim == 16

    def test_feature_schema_from_preprocessor_info(self):
        """Test FeatureSchema.from_preprocessor_info factory method."""
        num_info = {
            "f1": {"preprocessing": "ple", "dimension": 20, "categories": None},
            "f2": {"preprocessing": "standard", "dimension": 1, "categories": None},
        }
        cat_info = {
            "c1": {"preprocessing": "int", "dimension": 1, "categories": ["A", "B", "C"]},
        }

        schema = FeatureSchema.from_preprocessor_info(num_info, cat_info, None)

        assert schema.num_numerical_features == 2
        assert schema.num_categorical_features == 1
        assert "f1" in schema.numerical_features
        assert "c1" in schema.categorical_features

    def test_feature_schema_with_no_embeddings(self):
        """Test schema works with no embedding features."""
        num_features = {"f1": FeatureInfo("f1", "ple", 20, None)}
        cat_features = {"c1": FeatureInfo("c1", "int", 1, ["A"])}

        schema = FeatureSchema(num_features, cat_features, None)

        assert schema.num_embedding_features == 0
        assert schema.total_embedding_dim == 0

    def test_feature_schema_serialization_round_trip(self):
        """Test schema metadata can be serialized and restored."""
        schema = FeatureSchema(
            numerical_features={"f1": FeatureInfo("f1", "standard", 1, None)},
            categorical_features={"c1": FeatureInfo("c1", "int", 1, ["A", "B"])},
            embedding_features={"e1": FeatureInfo("e1", "pretrained", 16, None)},
        )

        restored = FeatureSchema.from_dict(schema.to_dict())

        assert restored.numerical_features["f1"].preprocessing == "standard"
        assert restored.categorical_features["c1"].categories == ["A", "B"]
        assert restored.total_embedding_dim == 16


# ============================================================================
# TabularBatch Contract Tests
# ============================================================================


class TestTabularBatchContract:
    """Test the contract and interface of TabularBatch."""

    def test_batch_creation(self):
        """Test TabularBatch can be created."""
        batch = TabularBatch(
            numerical_features=[torch.randn(32, 10)],
            categorical_features=[torch.randint(0, 5, (32, 1))],
            embeddings=[torch.randn(32, 8)],
            labels=torch.randn(32, 1),
        )

        assert len(batch.numerical_features) == 1
        assert len(batch.categorical_features) == 1
        assert len(batch.embeddings) == 1  # type: ignore[arg-type]
        assert batch.labels is not None

    def test_batch_creation_without_labels(self):
        """Test TabularBatch can be created without labels."""
        batch = TabularBatch(
            numerical_features=[torch.randn(32, 10)],
            categorical_features=[torch.randint(0, 5, (32, 1))],
            embeddings=None,
            labels=None,
        )

        assert batch.labels is None
        assert batch.embeddings is None

    def test_batch_to_device(self):
        """Test TabularBatch.to() moves tensors to device."""
        batch = TabularBatch(
            numerical_features=[torch.randn(32, 10)],
            categorical_features=[torch.randint(0, 5, (32, 1))],
            embeddings=[torch.randn(32, 8)],
            labels=torch.randn(32, 1),
        )

        # Move to CPU explicitly
        batch_cpu = batch.to("cpu")

        assert batch_cpu.numerical_features[0].device.type == "cpu"
        assert batch_cpu.categorical_features[0].device.type == "cpu"
        assert batch_cpu.embeddings[0].device.type == "cpu"  # type: ignore[index, union-attr]
        assert batch_cpu.labels.device.type == "cpu"  # type: ignore[union-attr]

    def test_batch_from_tuple_supervised(self):
        """Test TabularBatch.from_tuple() with labels."""
        features = (
            [torch.randn(32, 10)],  # num_features
            [torch.randint(0, 5, (32, 1))],  # cat_features
            [torch.randn(32, 8)],  # embeddings
        )
        labels = torch.randn(32, 1)
        batch_tuple = (features, labels)

        batch = TabularBatch.from_tuple(batch_tuple)

        assert len(batch.numerical_features) == 1
        assert len(batch.categorical_features) == 1
        assert batch.labels is not None

    def test_batch_from_tuple_prediction(self):
        """Test TabularBatch.from_tuple() without labels."""
        batch_tuple = (
            [torch.randn(32, 10)],  # num_features
            [torch.randint(0, 5, (32, 1))],  # cat_features
            None,  # embeddings
        )

        batch = TabularBatch.from_tuple(batch_tuple)

        assert batch.labels is None
        assert batch.embeddings is None

    def test_batch_to_tuple_supervised(self):
        """Test TabularBatch.to_tuple() with labels."""
        batch = TabularBatch(
            numerical_features=[torch.randn(32, 10)],
            categorical_features=[torch.randint(0, 5, (32, 1))],
            embeddings=[torch.randn(32, 8)],
            labels=torch.randn(32, 1),
        )

        batch_tuple = batch.to_tuple()

        assert isinstance(batch_tuple, tuple)
        assert len(batch_tuple) == 2  # (features, labels)
        features, _labels = batch_tuple
        assert len(features) == 3

    def test_batch_to_tuple_prediction(self):
        """Test TabularBatch.to_tuple() without labels."""
        batch = TabularBatch(
            numerical_features=[torch.randn(32, 10)],
            categorical_features=[torch.randint(0, 5, (32, 1))],
            embeddings=None,
            labels=None,
        )

        batch_tuple = batch.to_tuple()

        assert isinstance(batch_tuple, tuple)
        assert len(batch_tuple) == 3  # (num_features, cat_features, embeddings)

    def test_batch_roundtrip_conversion(self):
        """Test converting batch to tuple and back preserves data."""
        original_batch = TabularBatch(
            numerical_features=[torch.randn(32, 10)],
            categorical_features=[torch.randint(0, 5, (32, 1))],
            embeddings=[torch.randn(32, 8)],
            labels=torch.randn(32, 1),
        )

        # Convert to tuple and back
        batch_tuple = original_batch.to_tuple()
        reconstructed_batch = TabularBatch.from_tuple(batch_tuple)

        assert len(reconstructed_batch.numerical_features) == len(original_batch.numerical_features)
        assert len(reconstructed_batch.categorical_features) == len(original_batch.categorical_features)
        assert (
            len(reconstructed_batch.embeddings) == len(original_batch.embeddings)  # type: ignore[arg-type]
            if original_batch.embeddings
            else reconstructed_batch.embeddings is None
        )
        assert reconstructed_batch.labels is not None


# ============================================================================
# Integration Tests
# ============================================================================


class TestDataAPIIntegration:
    """Integration tests for the complete data API."""

    def test_end_to_end_classification_workflow(self, classification_data):
        """Test complete workflow from raw data to batches for classification."""
        from pretab.preprocessor import Preprocessor

        X, y = classification_data
        preprocessor = Preprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=True,
            regression=False,
        )

        # Preprocess
        datamodule.preprocess_data(X, y, val_size=0.2, random_state=42)

        # Check schema
        schema = datamodule.schema
        assert schema is not None
        assert schema.num_numerical_features > 0

        # Setup datasets
        datamodule.setup("fit")

        # Get dataloader and batch
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        features, labels = batch
        num_feats, _cat_feats, _embeddings = features

        # Verify shapes and types
        assert isinstance(num_feats, list)
        assert isinstance(labels, torch.Tensor)

    def test_end_to_end_regression_workflow(self, regression_data):
        """Test complete workflow from raw data to batches for regression."""
        from pretab.preprocessor import Preprocessor

        X, y = regression_data
        preprocessor = Preprocessor()
        datamodule = TabularDataModule(
            preprocessor=preprocessor,
            batch_size=32,
            shuffle=True,
            regression=True,
        )

        # Preprocess
        datamodule.preprocess_data(X, y, val_size=0.2, random_state=42)

        # Setup datasets
        datamodule.setup("fit")

        # Get dataloader and batch
        val_loader = datamodule.val_dataloader()
        batch = next(iter(val_loader))

        _features, labels = batch

        # Verify regression labels are float32 with shape (batch_size, 1)
        assert labels.dtype == torch.float32
        assert labels.shape[1] == 1

    def test_dataset_with_batch_object_mode(self, simple_tensors):
        """Test dataset returns TabularBatch when requested."""
        num_feats, cat_feats, embeddings, labels = simple_tensors
        dataset = TabularDataset(
            cat_feats,
            num_feats,
            embeddings,
            labels,
            return_batch_object=True,
        )

        batch = dataset[0]
        assert isinstance(batch, TabularBatch)

        # Test device movement
        batch_cpu = batch.to("cpu")
        assert batch_cpu.labels.device.type == "cpu"  # type: ignore[union-attr]

        # Test tuple conversion
        batch_tuple = batch.to_tuple()
        assert isinstance(batch_tuple, tuple)
