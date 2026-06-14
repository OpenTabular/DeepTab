from deeptab.architectures.mambatab import MambaTab
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.models.mambatab_config import MambaTabConfig
from ._docstring import generate_docstring


class MambaTabRegressor(SklearnBaseRegressor):
    _model_cls = MambaTab
    _config_cls = MambaTabConfig

    __doc__ = generate_docstring(
        MambaTabConfig,
        model_description="""
        MambaTab regressor. This class extends the SklearnBaseRegressor class and uses the MambaTab model
        with the default MambaTab configuration.
        """,
        examples="""
        >>> from deeptab.models import MambaTabRegressor
        >>> from deeptab.configs import MambaTabConfig
        >>> model = MambaTabRegressor(model_config=MambaTabConfig(d_model=64, n_layers=2))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class MambaTabClassifier(SklearnBaseClassifier):
    _model_cls = MambaTab
    _config_cls = MambaTabConfig

    __doc__ = generate_docstring(
        MambaTabConfig,
        model_description="""
        MambaTab classifier. This class extends the SklearnBaseClassifier class and uses the MambaTab model
        with the default MambaTab configuration.
        """,
        examples="""
        >>> from deeptab.models import MambaTabClassifier
        >>> from deeptab.configs import MambaTabConfig
        >>> model = MambaTabClassifier(model_config=MambaTabConfig(d_model=64, n_layers=2))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class MambaTabLSS(SklearnBaseLSS):
    _model_cls = MambaTab
    _config_cls = MambaTabConfig

    __doc__ = generate_docstring(
        MambaTabConfig,
        model_description="""
        MambaTab LSS for distributional regression. This class extends the SklearnBaseLSS class and uses the MambaTab model
        with the default MambaTab configuration.
        """,
        examples="""
        >>> from deeptab.models import MambaTabLSS
        >>> from deeptab.configs import MambaTabConfig
        >>> model = MambaTabLSS(model_config=MambaTabConfig(d_model=64, n_layers=2))
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
