from ..base_models.autoint import AutoInt
from ..configs.autoint_config import AutoIntConfig
from ..configs.preprocessing_config import PreprocessingConfig
from ..configs.trainer_config import TrainerConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


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
