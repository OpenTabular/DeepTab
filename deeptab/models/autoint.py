from deeptab.architectures.autoint import AutoInt
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.core import PreprocessingConfig, TrainerConfig
from ..configs.models.autoint_config import AutoIntConfig
from ._docstring import generate_docstring


class AutoIntRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        AutoIntConfig,
        model_description="""
        AutoInt regressor. This class extends the SklearnBaseRegressor
        class and uses the AutoInt model with the default AutoInt
        configuration.
        """,
        examples="""
        >>> from deeptab.models import AutoIntRegressor
        >>> model = AutoIntRegressor(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: AutoIntConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=AutoInt,
            config=AutoIntConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class AutoIntClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        AutoIntConfig,
        """AutoInt Classifier. This class extends the SklearnBaseClassifier class
        and uses the AutoInt model with the default AutoInt configuration.""",
        examples="""
        >>> from deeptab.models import AutoIntClassifier
        >>> model = AutoIntClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: AutoIntConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=AutoInt,
            config=AutoIntConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class AutoIntLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        AutoIntConfig,
        """AutoInt for distributional regression.
        This class extends the SklearnBaseLSS class and uses the
        AutoInt model with the default AutoInt configuration.""",
        examples="""
        >>> from deeptab.models import AutoIntLSS
        >>> model = AutoIntLSS(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train, family="normal")
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
            model=AutoInt,
            config=AutoIntConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )
