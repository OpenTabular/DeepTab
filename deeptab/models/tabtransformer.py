from ..base_models.tabtransformer import TabTransformer
from ..configs.tabtransformer_config import DefaultTabTransformerConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class TabTransformerRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultTabTransformerConfig,
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

    def __init__(self, **kwargs):
        super().__init__(model=TabTransformer, config=DefaultTabTransformerConfig, **kwargs)


class TabTransformerClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultTabTransformerConfig,
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

    def __init__(self, **kwargs):
        super().__init__(model=TabTransformer, config=DefaultTabTransformerConfig, **kwargs)


class TabTransformerLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultTabTransformerConfig,
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

    def __init__(self, **kwargs):
        super().__init__(model=TabTransformer, config=DefaultTabTransformerConfig, **kwargs)
