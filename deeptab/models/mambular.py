from deeptab.architectures.mambular import Mambular
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.models.mambular_config import MambularConfig
from ._docstring import generate_docstring


class MambularRegressor(SklearnBaseRegressor):
    _model_cls = Mambular
    _config_cls = MambularConfig

    __doc__ = generate_docstring(
        MambularConfig,
        model_description="""
        Mambular regressor. This class extends the SklearnBaseRegressor class and uses the Mambular model
        with the default Mambular configuration.
        """,
        examples="""
        >>> from deeptab.models import MambularRegressor
        >>> from deeptab.configs import MambularConfig
        >>> model = MambularRegressor(model_config=MambularConfig(d_model=64, n_layers=8))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class MambularClassifier(SklearnBaseClassifier):
    _model_cls = Mambular
    _config_cls = MambularConfig

    __doc__ = generate_docstring(
        MambularConfig,
        model_description="""
        Mambular classifier. This class extends the SklearnBaseClassifier class and uses the Mambular model
        with the default Mambular configuration.
        """,
        examples="""
        >>> from deeptab.models import MambularClassifier
        >>> from deeptab.configs import MambularConfig
        >>> model = MambularClassifier(model_config=MambularConfig(d_model=64, n_layers=8))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class MambularLSS(SklearnBaseLSS):
    _model_cls = Mambular
    _config_cls = MambularConfig

    __doc__ = generate_docstring(
        MambularConfig,
        model_description="""
        Mambular LSS for distributional regression. This class extends the SklearnBaseLSS class and uses the Mambular model
        with the default Mambular configuration.
        """,
        examples="""
        >>> from deeptab.models import MambularLSS
        >>> from deeptab.configs import MambularConfig
        >>> model = MambularLSS(model_config=MambularConfig(d_model=64, n_layers=8))
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
