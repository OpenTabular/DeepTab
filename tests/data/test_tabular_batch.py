"""Tests for tabular batch."""

import torch

from deeptab.data import TabularBatch


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
