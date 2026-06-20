from deeptab.architectures.tabtransformer import TabTransformer
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.models.tabtransformer_config import TabTransformerConfig
from ._docstring import generate_docstring


class TabTransformerRegressor(SklearnBaseRegressor):
    _model_cls = TabTransformer
    _config_cls = TabTransformerConfig

    __doc__ = generate_docstring(
        TabTransformerConfig,
        model_description="""
        TabTransformer regressor. This class extends the SklearnBaseRegressor class and uses the TabTransformer model
        with the default TabTransformer configuration.
        """,
        examples="""
        >>> from deeptab.models import TabTransformerRegressor
        >>> model = TabTransformerRegressor()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class TabTransformerClassifier(SklearnBaseClassifier):
    _model_cls = TabTransformer
    _config_cls = TabTransformerConfig

    __doc__ = generate_docstring(
        TabTransformerConfig,
        model_description="""
        TabTransformer classifier. This class extends the SklearnBaseClassifier class and uses the TabTransformer model
        with the default TabTransformer configuration.
        """,
        examples="""
        >>> from deeptab.models import TabTransformerClassifier
        >>> model = TabTransformerClassifier()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class TabTransformerLSS(SklearnBaseLSS):
    _model_cls = TabTransformer
    _config_cls = TabTransformerConfig

    __doc__ = generate_docstring(
        TabTransformerConfig,
        model_description="""
        TabTransformer for distributional regression. This class extends the SklearnBaseLSS class and uses the TabTransformer model
        with the default TabTransformer configuration.
        """,
        examples="""
        >>> from deeptab.models import TabTransformerLSS
        >>> model = TabTransformerLSS()
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
