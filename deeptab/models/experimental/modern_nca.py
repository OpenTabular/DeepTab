from deeptab.architectures.experimental.modern_nca import ModernNCA
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ...configs.experimental.modernnca_config import ModernNCAConfig
from .._docstring import generate_docstring


class ModernNCARegressor(SklearnBaseRegressor):
    _model_cls = ModernNCA
    _config_cls = ModernNCAConfig

    __doc__ = generate_docstring(
        ModernNCAConfig,
        model_description="""
        Multi-Layer Perceptron regressor. This class extends the SklearnBaseRegressor class and uses the ModernNCA model
        with the default ModernNCA configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import ModernNCARegressor
        >>> from deeptab.configs import ModernNCAConfig
        >>> model = ModernNCARegressor(model_config=ModernNCAConfig(dim=128, n_blocks=4))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class ModernNCAClassifier(SklearnBaseClassifier):
    _model_cls = ModernNCA
    _config_cls = ModernNCAConfig

    __doc__ = generate_docstring(
        ModernNCAConfig,
        model_description="""
        Multi-Layer Perceptron classifier This class extends the SklearnBaseClassifier class and uses the ModernNCA model
        with the default ModernNCA configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import ModernNCAClassifier
        >>> from deeptab.configs import ModernNCAConfig
        >>> model = ModernNCAClassifier(model_config=ModernNCAConfig(dim=128, n_blocks=4))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class ModernNCALSS(SklearnBaseLSS):
    _model_cls = ModernNCA
    _config_cls = ModernNCAConfig

    __doc__ = generate_docstring(
        ModernNCAConfig,
        model_description="""
        Multi-Layer Perceptron for distributional regression. This class extends the SklearnBaseLSS class and uses the ModernNCA model
        with the default ModernNCA configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import ModernNCALSS
        >>> from deeptab.configs import ModernNCAConfig
        >>> model = ModernNCALSS(model_config=ModernNCAConfig(dim=128, n_blocks=4))
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
