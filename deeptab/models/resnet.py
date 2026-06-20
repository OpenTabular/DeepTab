from deeptab.architectures.resnet import ResNet
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.models.resnet_config import ResNetConfig
from ._docstring import generate_docstring


class ResNetRegressor(SklearnBaseRegressor):
    _model_cls = ResNet
    _config_cls = ResNetConfig

    __doc__ = generate_docstring(
        ResNetConfig,
        model_description="""
        ResNet regressor. This class extends the SklearnBaseRegressor class and uses the ResNet model
        with the default ResNet configuration.
        """,
        examples="""
        >>> from deeptab.models import ResNetRegressor
        >>> model = ResNetRegressor()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class ResNetClassifier(SklearnBaseClassifier):
    _model_cls = ResNet
    _config_cls = ResNetConfig

    __doc__ = generate_docstring(
        ResNetConfig,
        model_description="""
        ResNet classifier This class extends the SklearnBaseClassifier class and uses the ResNet model
        with the default ResNet configuration.
        """,
        examples="""
        >>> from deeptab.models import ResNetClassifier
        >>> model = ResNetClassifier()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class ResNetLSS(SklearnBaseLSS):
    _model_cls = ResNet
    _config_cls = ResNetConfig

    __doc__ = generate_docstring(
        ResNetConfig,
        model_description="""
        ResNet for distributional regressor. This class extends the SklearnBaseLSS class and uses the ResNet model
        with the default ResNet configuration.
        """,
        examples="""
        >>> from deeptab.models import ResNetLSS
        >>> model = ResNetLSS()
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
