from deeptab.architectures.tabm import TabM
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.models.tabm_config import TabMConfig
from ._docstring import generate_docstring


class TabMRegressor(SklearnBaseRegressor):
    _model_cls = TabM
    _config_cls = TabMConfig

    __doc__ = generate_docstring(
        TabMConfig,
        model_description="""
        TabM regressor. This class extends the SklearnBaseRegressor class and uses the TabM model
        with the default TabM configuration.
        """,
        examples="""
        >>> from deeptab.models import TabMRegressor
        >>> from deeptab.configs import TabMConfig
        >>> model = TabMRegressor(model_config=TabMConfig(ensemble_size=32, model_type='full'))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class TabMClassifier(SklearnBaseClassifier):
    _model_cls = TabM
    _config_cls = TabMConfig

    __doc__ = generate_docstring(
        TabMConfig,
        model_description="""
        TabM classifier. This class extends the SklearnBaseClassifier class and uses the TabM model
        with the default TabM configuration.
        """,
        examples="""
        >>> from deeptab.models import TabMClassifier
        >>> from deeptab.configs import TabMConfig
        >>> model = TabMClassifier(model_config=TabMConfig(ensemble_size=32, model_type='full'))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class TabMLSS(SklearnBaseLSS):
    _model_cls = TabM
    _config_cls = TabMConfig

    __doc__ = generate_docstring(
        TabMConfig,
        model_description="""
        TabM for distributional regressoion. This class extends the SklearnBaseLSS class and uses the TabM model
        with the default TabM configuration.
        """,
        examples="""
        >>> from deeptab.models import TabMLSS
        >>> from deeptab.configs import TabMConfig
        >>> model = TabMLSS(model_config=TabMConfig(ensemble_size=32, model_type='full'))
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
