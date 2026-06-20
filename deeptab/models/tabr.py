from deeptab.architectures.tabr import TabR
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.models.tabr_config import TabRConfig
from ._docstring import generate_docstring


class TabRRegressor(SklearnBaseRegressor):
    _model_cls = TabR
    _config_cls = TabRConfig

    __doc__ = generate_docstring(
        TabRConfig,
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


class TabRClassifier(SklearnBaseClassifier):
    _model_cls = TabR
    _config_cls = TabRConfig

    __doc__ = generate_docstring(
        TabRConfig,
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


class TabRLSS(SklearnBaseLSS):
    _model_cls = TabR
    _config_cls = TabRConfig

    __doc__ = generate_docstring(
        TabRConfig,
        model_description="""
        TabR regressor. This class extends the SklearnBaseLSS class and uses the TabR model
        with the default TabR configuration.
        """,
        examples="""
        >>> from deeptab.models import TabRLSS
        >>> model = TabRLSS()
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
