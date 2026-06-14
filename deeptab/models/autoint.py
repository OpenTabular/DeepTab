from deeptab.architectures.autoint import AutoInt
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.models.autoint_config import AutoIntConfig
from ._docstring import generate_docstring


class AutoIntRegressor(SklearnBaseRegressor):
    _model_cls = AutoInt
    _config_cls = AutoIntConfig

    __doc__ = generate_docstring(
        AutoIntConfig,
        model_description="""
        AutoInt regressor. This class extends the SklearnBaseRegressor
        class and uses the AutoInt model with the default AutoInt
        configuration.
        """,
        examples="""
        >>> from deeptab.models import AutoIntRegressor
        >>> from deeptab.configs import AutoIntConfig
        >>> model = AutoIntRegressor(model_config=AutoIntConfig(d_model=64, n_layers=8))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class AutoIntClassifier(SklearnBaseClassifier):
    _model_cls = AutoInt
    _config_cls = AutoIntConfig

    __doc__ = generate_docstring(
        AutoIntConfig,
        """AutoInt Classifier. This class extends the SklearnBaseClassifier class
        and uses the AutoInt model with the default AutoInt configuration.""",
        examples="""
        >>> from deeptab.models import AutoIntClassifier
        >>> from deeptab.configs import AutoIntConfig
        >>> model = AutoIntClassifier(model_config=AutoIntConfig(d_model=64, n_layers=8))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class AutoIntLSS(SklearnBaseLSS):
    _model_cls = AutoInt
    _config_cls = AutoIntConfig

    __doc__ = generate_docstring(
        AutoIntConfig,
        """AutoInt for distributional regression.
        This class extends the SklearnBaseLSS class and uses the
        AutoInt model with the default AutoInt configuration.""",
        examples="""
        >>> from deeptab.models import AutoIntLSS
        >>> from deeptab.configs import AutoIntConfig
        >>> model = AutoIntLSS(model_config=AutoIntConfig(d_model=64, n_layers=8))
        >>> model.fit(X_train, y_train, family="normal")
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
