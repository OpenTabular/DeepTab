from ...base_models.modern_nca import ModernNCA
from ...configs.modernnca_config import DefaultModernNCAConfig
from ...utils.docstring_generator import generate_docstring
from ..utils.sklearn_base_classifier import SklearnBaseClassifier
from ..utils.sklearn_base_lss import SklearnBaseLSS
from ..utils.sklearn_base_regressor import SklearnBaseRegressor


class ModernNCARegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultModernNCAConfig,
        model_description="""
        Multi-Layer Perceptron regressor. This class extends the SklearnBaseRegressor class and uses the ModernNCA model
        with the default ModernNCA configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import ModernNCARegressor
        >>> model = ModernNCARegressor(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=ModernNCA, config=DefaultModernNCAConfig, **kwargs)


class ModernNCAClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultModernNCAConfig,
        model_description="""
        Multi-Layer Perceptron classifier This class extends the SklearnBaseClassifier class and uses the ModernNCA model
        with the default ModernNCA configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import ModernNCAClassifier
        >>> model = ModernNCAClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=ModernNCA, config=DefaultModernNCAConfig, **kwargs)


class ModernNCALSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultModernNCAConfig,
        model_description="""
        Multi-Layer Perceptron for distributional regression. This class extends the SklearnBaseLSS class and uses the ModernNCA model
        with the default ModernNCA configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import ModernNCALSS
        >>> model = ModernNCALSS(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=ModernNCA, config=DefaultModernNCAConfig, **kwargs)
