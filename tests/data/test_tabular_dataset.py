"""Tests for tabular dataset."""

import pytest
import torch

from deeptab.data import TabularBatch, TabularDataset


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


class TestTabularDatasetContract:
    """Test the contract and interface of TabularDataset."""

    def test_indexed_tuple_tracks_rows_with_replacement_sampling(self):
        features = torch.arange(5, dtype=torch.float32).unsqueeze(-1)
        dataset = TabularDataset([], [features], [], features, return_indices=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=3, sampler=[4, 1, 4])
        data, labels, indices = next(iter(loader))
        assert indices.tolist() == [4, 1, 4]
        torch.testing.assert_close(data[0][0], features[indices])
        torch.testing.assert_close(labels, features[indices])

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
        assert emb_features[0].untyped_storage().data_ptr() == embeddings[0].untyped_storage().data_ptr()  # type: ignore[index]

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
