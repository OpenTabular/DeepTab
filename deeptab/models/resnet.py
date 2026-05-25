from deeptab.architectures.resnet import ResNet
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.core import PreprocessingConfig, TrainerConfig
from ..configs.models.resnet_config import ResNetConfig
from ._docstring import generate_docstring


class ResNetRegressor(SklearnBaseRegressor):
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

    def __init__(
        self,
        model_config: ResNetConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=ResNet,
            config=ResNetConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class ResNetClassifier(SklearnBaseClassifier):
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

    def __init__(
        self,
        model_config: ResNetConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=ResNet,
            config=ResNetConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class ResNetLSS(SklearnBaseLSS):
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

    def __init__(
        self,
        model_config=None,
        preprocessing_config=None,
        trainer_config=None,
        random_state=None,
        **kwargs,
    ):
        super().__init__(
            model=ResNet,
            config=ResNetConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )
