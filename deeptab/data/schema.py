"""Schema definitions for tabular data structures.

Provides typed containers and metadata for tabular datasets.

New in v2.0.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FeatureInfo:
    """Information about a single feature in the tabular dataset.

    Parameters
    ----------
    name : str
        Feature name or identifier.
    preprocessing : str
        Preprocessing strategy applied to this feature.
    dimension : int
        Output dimension after preprocessing (e.g., embedding size).
    categories : list or None
        List of categories for categorical features, None for numerical.
    """

    name: str
    preprocessing: str
    dimension: int
    categories: list[Any] | None = None

    @property
    def is_categorical(self) -> bool:
        """Check if this feature is categorical."""
        return self.categories is not None


@dataclass
class FeatureSchema:
    """Schema describing the structure of tabular input features.

    Tracks categorical, numerical, and embedding features with their
    preprocessing metadata and dimensions.

    Parameters
    ----------
    numerical_features : dict[str, FeatureInfo]
        Dictionary mapping numerical feature names to their metadata.
    categorical_features : dict[str, FeatureInfo]
        Dictionary mapping categorical feature names to their metadata.
    embedding_features : dict[str, FeatureInfo] | None
        Dictionary mapping embedding feature names to their metadata.
    """

    numerical_features: dict[str, FeatureInfo]
    categorical_features: dict[str, FeatureInfo]
    embedding_features: dict[str, FeatureInfo] | None = None

    @property
    def num_numerical_features(self) -> int:
        """Total number of numerical features."""
        return len(self.numerical_features)

    @property
    def num_categorical_features(self) -> int:
        """Total number of categorical features."""
        return len(self.categorical_features)

    @property
    def num_embedding_features(self) -> int:
        """Total number of embedding features."""
        return len(self.embedding_features) if self.embedding_features else 0

    @property
    def total_numerical_dim(self) -> int:
        """Total dimension across all numerical features."""
        return sum(f.dimension for f in self.numerical_features.values())

    @property
    def total_categorical_dim(self) -> int:
        """Total dimension across all categorical features."""
        return sum(f.dimension for f in self.categorical_features.values())

    @property
    def total_embedding_dim(self) -> int:
        """Total dimension across all embedding features."""
        if not self.embedding_features:
            return 0
        return sum(f.dimension for f in self.embedding_features.values())

    @classmethod
    def from_preprocessor_info(
        cls,
        num_feature_info: dict | None,
        cat_feature_info: dict | None,
        embedding_feature_info: dict | None = None,
    ) -> FeatureSchema:
        """Create a FeatureSchema from preprocessor feature info dictionaries.

        Parameters
        ----------
        num_feature_info : dict or None
            Numerical feature information from preprocessor.
        cat_feature_info : dict or None
            Categorical feature information from preprocessor.
        embedding_feature_info : dict or None
            Embedding feature information from preprocessor.

        Returns
        -------
        FeatureSchema
            Constructed feature schema.
        """
        numerical_features = {}
        if num_feature_info:
            for name, info in num_feature_info.items():
                numerical_features[str(name)] = FeatureInfo(
                    name=str(name),
                    preprocessing=info.get("preprocessing", "unknown"),
                    dimension=info.get("dimension", 1),
                    categories=None,
                )

        categorical_features = {}
        if cat_feature_info:
            for name, info in cat_feature_info.items():
                categorical_features[str(name)] = FeatureInfo(
                    name=str(name),
                    preprocessing=info.get("preprocessing", "unknown"),
                    dimension=info.get("dimension", 1),
                    categories=info.get("categories"),
                )

        embedding_features = None
        if embedding_feature_info:
            embedding_features = {}
            for name, info in embedding_feature_info.items():
                embedding_features[str(name)] = FeatureInfo(
                    name=str(name),
                    preprocessing=info.get("preprocessing", "unknown"),
                    dimension=info.get("dimension", 1),
                    categories=None,
                )

        return cls(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            embedding_features=embedding_features,
        )


@dataclass
class TabularBatch:
    """Typed container for a batch of tabular data.

    Provides a structured interface for accessing different feature types
    and labels in a batch, replacing raw tuples.

    Parameters
    ----------
    numerical_features : list[torch.Tensor]
        List of tensors for numerical features.
    categorical_features : list[torch.Tensor]
        List of tensors for categorical features.
    embeddings : list[torch.Tensor] | None
        List of tensors for precomputed embeddings, if any.
    labels : torch.Tensor | None
        Labels for supervised learning, None for prediction mode.
    """

    numerical_features: list[torch.Tensor]
    categorical_features: list[torch.Tensor]
    embeddings: list[torch.Tensor] | None = None
    labels: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> TabularBatch:
        """Move all tensors in the batch to the specified device.

        Parameters
        ----------
        device : torch.device or str
            Target device (e.g., 'cuda', 'cpu', 'mps').

        Returns
        -------
        TabularBatch
            A new batch with all tensors moved to the device.
        """
        return TabularBatch(
            numerical_features=[t.to(device) for t in self.numerical_features],
            categorical_features=[t.to(device) for t in self.categorical_features],
            embeddings=[t.to(device) for t in self.embeddings] if self.embeddings else None,
            labels=self.labels.to(device) if self.labels is not None else None,
        )

    @classmethod
    def from_tuple(cls, batch_tuple: tuple) -> TabularBatch:
        """Create a TabularBatch from the legacy tuple format.

        Parameters
        ----------
        batch_tuple : tuple
            Either ((num_feats, cat_feats, embeddings), labels) or
            (num_feats, cat_feats, embeddings).

        Returns
        -------
        TabularBatch
            Typed batch container.
        """
        if len(batch_tuple) == 2:
            # Supervised mode: (features, labels)
            features, labels = batch_tuple
            num_feats, cat_feats, embeddings = features
            return cls(
                numerical_features=num_feats,
                categorical_features=cat_feats,
                embeddings=embeddings,
                labels=labels,
            )
        else:
            # Prediction mode: just features
            num_feats, cat_feats, embeddings = batch_tuple
            return cls(
                numerical_features=num_feats,
                categorical_features=cat_feats,
                embeddings=embeddings,
                labels=None,
            )

    def to_tuple(self) -> tuple:
        """Convert back to legacy tuple format for backward compatibility.

        Returns
        -------
        tuple
            Either ((num_feats, cat_feats, embeddings), labels) or
            (num_feats, cat_feats, embeddings).
        """
        features = (self.numerical_features, self.categorical_features, self.embeddings)
        if self.labels is not None:
            return (features, self.labels)
        return features
