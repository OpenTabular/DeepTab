from deeptab.architectures.ndtf import NDTF
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.models.ndtf_config import NDTFConfig
from ._docstring import generate_docstring


class NDTFRegressor(SklearnBaseRegressor):
    _model_cls = NDTF
    _config_cls = NDTFConfig

    __doc__ = generate_docstring(
        NDTFConfig,
        model_description="""
        Neural Decision Forest regressor. This class extends the SklearnBaseRegressor class and uses the NDTF model
        with the default NDTF configuration.
        """,
        examples="""
        >>> from deeptab.models import NDTFRegressor
        >>> from deeptab.configs import NDTFConfig
        >>> model = NDTFRegressor(model_config=NDTFConfig(n_ensembles=12, max_depth=8))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class NDTFClassifier(SklearnBaseClassifier):
    _model_cls = NDTF
    _config_cls = NDTFConfig

    __doc__ = generate_docstring(
        NDTFConfig,
        model_description="""
        Neural Decision Forest classifier. This class extends the SklearnBaseClassifier class and uses the NDTF model
        with the default NDTF configuration.
        """,
        examples="""
        >>> from deeptab.models import NDTFClassifier
        >>> from deeptab.configs import NDTFConfig
        >>> model = NDTFClassifier(model_config=NDTFConfig(n_ensembles=12, max_depth=8))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class NDTFLSS(SklearnBaseLSS):
    _model_cls = NDTF
    _config_cls = NDTFConfig

    __doc__ = generate_docstring(
        NDTFConfig,
        model_description="""
        Neural Decision Forest for distributional regression. This class extends the SklearnBaseLSS class and uses the NDTF model
        with the default NDTF configuration.
        """,
        examples="""
        >>> from deeptab.models import NDTFLSS
        >>> from deeptab.configs import NDTFConfig
        >>> model = NDTFLSS(model_config=NDTFConfig(n_ensembles=12, max_depth=8))
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
