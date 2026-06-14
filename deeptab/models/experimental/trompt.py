from deeptab.architectures.experimental.trompt import Trompt
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ...configs.experimental.trompt_config import TromptConfig
from .._docstring import generate_docstring


class TromptRegressor(SklearnBaseRegressor):
    _model_cls = Trompt
    _config_cls = TromptConfig

    __doc__ = generate_docstring(
        TromptConfig,
        model_description="""
        Trompt regressor. This class extends the SklearnBaseRegressor
        class and uses the Trompt model with the default Trompt
        configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import TromptRegressor
        >>> from deeptab.configs import TromptConfig
        >>> model = TromptRegressor(model_config=TromptConfig(d_model=64))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class TromptClassifier(SklearnBaseClassifier):
    _model_cls = Trompt
    _config_cls = TromptConfig

    __doc__ = generate_docstring(
        TromptConfig,
        """Trompt Classifier. This class extends the SklearnBaseClassifier class
        and uses the Trompt model with the default Trompt configuration.""",
        examples="""
        >>> from deeptab.models.experimental import TromptClassifier
        >>> from deeptab.configs import TromptConfig
        >>> model = TromptClassifier(model_config=TromptConfig(d_model=64))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class TromptLSS(SklearnBaseLSS):
    _model_cls = Trompt
    _config_cls = TromptConfig

    __doc__ = generate_docstring(
        TromptConfig,
        """Trompt for distributional regression.
        This class extends the SklearnBaseLSS class and uses the
        Trompt model with the default Trompt configuration.""",
        examples="""
        >>> from deeptab.models.experimental import TromptLSS
        >>> from deeptab.configs import TromptConfig
        >>> model = TromptLSS(model_config=TromptConfig(d_model=64))
        >>> model.fit(X_train, y_train, family="normal")
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
