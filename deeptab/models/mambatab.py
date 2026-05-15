from ..base_models.mambatab import MambaTab
from ..configs.mambatab_config import DefaultMambaTabConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class MambaTabRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultMambaTabConfig,
        model_description="""
        MambaTab regressor. This class extends the SklearnBaseRegressor class and uses the MambaTab model
        with the default MambaTab configuration.
        """,
        examples="""
        >>> from deeptab.models import MambaTabRegressor
        >>> model = MambaTabRegressor(d_model=64, n_layers=2)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=MambaTab, config=DefaultMambaTabConfig, **kwargs)


class MambaTabClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultMambaTabConfig,
        model_description="""
        MambaTab classifier. This class extends the SklearnBaseClassifier class and uses the MambaTab model
        with the default MambaTab configuration.
        """,
        examples="""
        >>> from deeptab.models import MambaTabClassifier
        >>> model = MambaTabClassifier(d_model=64, n_layers=2)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=MambaTab, config=DefaultMambaTabConfig, **kwargs)


class MambaTabLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultMambaTabConfig,
        model_description="""
        MambaTab LSS for distributional regression. This class extends the SklearnBaseLSS class and uses the MambaTab model
        with the default MambaTab configuration.
        """,
        examples="""
        >>> from deeptab.models import MambaTabLSS
        >>> model = MambaTabLSS(d_model=64, n_layers=2)
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=MambaTab, config=DefaultMambaTabConfig, **kwargs)
