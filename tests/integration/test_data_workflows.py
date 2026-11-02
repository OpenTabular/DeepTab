"""Tests for data workflows."""

import pandas as pd
import pytest
import torch
from sklearn.datasets import make_classification, make_regression

from deeptab.data import TabularBatch, TabularDataModule, TabularDataset


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


class TestDataAPIIntegration:
    """Integration tests for the complete data API."""

    def test_end_to_end_classification_workflow(self, classification_data):
        """Test complete workflow from raw data to batches for classification."""
        from pretab.preprocessor import Preprocessor

        X, y = classification_data
        preprocessor = Preprocessor(output_structure="blocks")
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
        preprocessor = Preprocessor(output_structure="blocks")
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


@pytest.mark.parametrize("missing_column", ["num", "cat"])
def test_single_row_with_missing_value_matches_batched_prediction(missing_column):
    """A missing value is imputed by the fitted preprocessor, whatever the batch size."""
    import numpy as np

    from deeptab.configs import TrainerConfig
    from deeptab.core.inference import InferenceModel
    from deeptab.models import MLPRegressor

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(50, 10, 120),
            "cat": rng.choice(["a", "b", "c"], 120).astype(object),
            "other": rng.normal(size=120),
        }
    )
    X.loc[::10, "num"] = np.nan
    X.loc[5::10, "cat"] = np.nan
    y = X["other"].to_numpy() + rng.normal(size=120)
    model = MLPRegressor(trainer_config=TrainerConfig(max_epochs=1, batch_size=32), random_state=0)
    model.fit(X, y, accelerator="cpu", logger=False, enable_progress_bar=False, enable_model_summary=False)

    row = X.iloc[[1]].copy()
    row[missing_column] = np.nan
    batch = pd.concat([row, X.iloc[[2]]], ignore_index=True)
    np.testing.assert_allclose(model.predict(row), model.predict(batch)[:1], rtol=1e-5)

    validated = InferenceModel.from_estimator(model).validate_input(row)
    assert bool(validated[missing_column].isna().all())
