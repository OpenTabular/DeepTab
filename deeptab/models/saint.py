from deeptab.architectures.saint import SAINT
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.models.saint_config import SAINTConfig
from ._docstring import generate_docstring


class SAINTRegressor(SklearnBaseRegressor):
    _model_cls = SAINT
    _config_cls = SAINTConfig

    __doc__ = generate_docstring(
        SAINTConfig,
        model_description="""
        SAINT regressor. This class extends the SklearnBaseRegressor
        class and uses the SAINT model with the default SAINT
        configuration.
        """,
        examples="""
        >>> from deeptab.models import SAINTRegressor
        >>> from deeptab.configs import SAINTConfig
        >>> model = SAINTRegressor(model_config=SAINTConfig(d_model=64, n_layers=8))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class SAINTClassifier(SklearnBaseClassifier):
    _model_cls = SAINT
    _config_cls = SAINTConfig

    __doc__ = generate_docstring(
        SAINTConfig,
        """SAINT Classifier. This class extends the SklearnBaseClassifier class
        and uses the SAINT model with the default SAINT configuration.""",
        examples="""
        >>> from deeptab.models import SAINTClassifier
        >>> from deeptab.configs import SAINTConfig
        >>> model = SAINTClassifier(model_config=SAINTConfig(d_model=64, n_layers=8))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class SAINTLSS(SklearnBaseLSS):
    _model_cls = SAINT
    _config_cls = SAINTConfig

    __doc__ = generate_docstring(
        SAINTConfig,
        """SAINT for distributional regression.
        This class extends the SklearnBaseLSS class and uses the
        SAINT model with the default SAINT configuration.""",
        examples="""
        >>> from deeptab.models import SAINTLSS
        >>> from deeptab.configs import SAINTConfig
        >>> model = SAINTLSS(model_config=SAINTConfig(d_model=64, n_layers=8))
        >>> model.fit(X_train, y_train, family="normal")
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
