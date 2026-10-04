"""Tests for feature schema."""

from deeptab.data import FeatureInfo, FeatureSchema


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
