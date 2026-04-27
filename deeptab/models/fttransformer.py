from ..base_models.ft_transformer import FTTransformer
from ..configs.fttransformer_config import DefaultFTTransformerConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class FTTransformerRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultFTTransformerConfig,
        model_description="""
        FTTransformer regressor. This class extends the SklearnBaseRegressor
        class and uses the FTTransformer model with the default FTTransformer
        configuration.
        """,
        examples="""
        >>> from deeptab.models import FTTransformerRegressor
        >>> model = FTTransformerRegressor(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=FTTransformer, config=DefaultFTTransformerConfig, **kwargs)


class FTTransformerClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultFTTransformerConfig,
        """FTTransformer Classifier. This class extends the SklearnBaseClassifier class
        and uses the FTTransformer model with the default FTTransformer configuration.""",
        examples="""
        >>> from deeptab.models import FTTransformerClassifier
        >>> model = FTTransformerClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=FTTransformer, config=DefaultFTTransformerConfig, **kwargs)


class FTTransformerLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultFTTransformerConfig,
        """FTTransformer for distributional regression.
        This class extends the SklearnBaseLSS class and uses the
        FTTransformer model with the default FTTransformer configuration.""",
        examples="""
        >>> from deeptab.models import FTTransformerLSS
        >>> model = FTTransformerLSS(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train, family="normal")
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=FTTransformer, config=DefaultFTTransformerConfig, **kwargs)
