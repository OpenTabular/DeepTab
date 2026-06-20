from deeptab.architectures.experimental.tangos import Tangos
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ...configs.experimental.tangos_config import TangosConfig
from .._docstring import generate_docstring


class TangosRegressor(SklearnBaseRegressor):
    _model_cls = Tangos
    _config_cls = TangosConfig

    __doc__ = generate_docstring(
        TangosConfig,
        model_description="""
        Tangos regressor. This class extends the SklearnBaseRegressor class and uses the Tangos model
        with the default Tangos configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import TangosRegressor
        >>> from deeptab.configs import TangosConfig
        >>> model = TangosRegressor(model_config=TangosConfig(layer_sizes=[128, 64]))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class TangosClassifier(SklearnBaseClassifier):
    _model_cls = Tangos
    _config_cls = TangosConfig

    __doc__ = generate_docstring(
        TangosConfig,
        model_description="""
        Tangos classifier This class extends the SklearnBaseClassifier class and uses the Tangos model
        with the default Tangos configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import TangosClassifier
        >>> from deeptab.configs import TangosConfig
        >>> model = TangosClassifier(model_config=TangosConfig(layer_sizes=[128, 64]))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class TangosLSS(SklearnBaseLSS):
    _model_cls = Tangos
    _config_cls = TangosConfig

    __doc__ = generate_docstring(
        TangosConfig,
        model_description="""
        Tangos for distributional regression. This class extends the SklearnBaseLSS class and uses the Tangos model
        with the default Tangos configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import TangosLSS
        >>> from deeptab.configs import TangosConfig
        >>> model = TangosLSS(model_config=TangosConfig(layer_sizes=[128, 64]))
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
