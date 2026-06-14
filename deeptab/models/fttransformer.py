from deeptab.architectures.ft_transformer import FTTransformer
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.models.fttransformer_config import FTTransformerConfig
from ._docstring import generate_docstring


class FTTransformerRegressor(SklearnBaseRegressor):
    _model_cls = FTTransformer
    _config_cls = FTTransformerConfig

    __doc__ = generate_docstring(
        FTTransformerConfig,
        model_description="""
        FTTransformer regressor. This class extends the SklearnBaseRegressor
        class and uses the FTTransformer model with the default FTTransformer
        configuration.
        """,
        examples="""
        >>> from deeptab.models import FTTransformerRegressor
        >>> from deeptab.configs import FTTransformerConfig
        >>> model = FTTransformerRegressor(model_config=FTTransformerConfig(d_model=64, n_layers=8))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class FTTransformerClassifier(SklearnBaseClassifier):
    _model_cls = FTTransformer
    _config_cls = FTTransformerConfig

    __doc__ = generate_docstring(
        FTTransformerConfig,
        """FTTransformer Classifier. This class extends the SklearnBaseClassifier class
        and uses the FTTransformer model with the default FTTransformer configuration.""",
        examples="""
        >>> from deeptab.models import FTTransformerClassifier
        >>> from deeptab.configs import FTTransformerConfig
        >>> model = FTTransformerClassifier(model_config=FTTransformerConfig(d_model=64, n_layers=8))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class FTTransformerLSS(SklearnBaseLSS):
    _model_cls = FTTransformer
    _config_cls = FTTransformerConfig

    __doc__ = generate_docstring(
        FTTransformerConfig,
        """FTTransformer for distributional regression.
        This class extends the SklearnBaseLSS class and uses the
        FTTransformer model with the default FTTransformer configuration.""",
        examples="""
        >>> from deeptab.models import FTTransformerLSS
        >>> from deeptab.configs import FTTransformerConfig
        >>> model = FTTransformerLSS(model_config=FTTransformerConfig(d_model=64, n_layers=8))
        >>> model.fit(X_train, y_train, family="normal")
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
