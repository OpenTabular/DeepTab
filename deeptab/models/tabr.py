from ..base_models.tabr import TabR
from ..configs.tabr_config import DefaultTabRConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class TabRRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultTabRConfig,
        model_description="""
        TabR regressor. This class extends the SklearnBaseRegressor class and uses the TabR model
        with the default TabR configuration.
        """,
        examples="""
        >>> from deeptab.models import TabRRegressor
        >>> model = TabRRegressor()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=TabR, config=DefaultTabRConfig, **kwargs)


class TabRClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultTabRConfig,
        model_description="""
        TabR classifier. This class extends the SklearnBaseClassifier class and uses the TabR model
        with the default TabR configuration.
        """,
        examples="""
        >>> from deeptab.models import TabRClassifier
        >>> model = TabRClassifier()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=TabR, config=DefaultTabRConfig, **kwargs)


class TabRLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultTabRConfig,
        model_description="""
        TabR regressor. This class extends the SklearnBaseLSS class and uses the TabR model
        with the default TabR configuration.
        """,
        examples="""
        >>> from deeptab.models import TabRLSS
        >>> model = TabRLSS(d_model=64, family='normal')
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=TabR, config=DefaultTabRConfig, **kwargs)
