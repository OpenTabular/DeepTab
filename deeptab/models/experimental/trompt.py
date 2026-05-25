from ...base_models.trompt import Trompt
from ...configs.preprocessing_config import PreprocessingConfig
from ...configs.trainer_config import TrainerConfig
from ...configs.trompt_config import TromptConfig
from ...utils.docstring_generator import generate_docstring
from ..utils.sklearn_base_classifier import SklearnBaseClassifier
from ..utils.sklearn_base_lss import SklearnBaseLSS
from ..utils.sklearn_base_regressor import SklearnBaseRegressor


class TromptRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        TromptConfig,
        model_description="""
        Trompt regressor. This class extends the SklearnBaseRegressor
        class and uses the Trompt model with the default Trompt
        configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import TromptRegressor
        >>> model = TromptRegressor(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: TromptConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=Trompt,
            config=TromptConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class TromptClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        TromptConfig,
        """Trompt Classifier. This class extends the SklearnBaseClassifier class
        and uses the Trompt model with the default Trompt configuration.""",
        examples="""
        >>> from deeptab.models.experimental import TromptClassifier
        >>> model = TromptClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: TromptConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=Trompt,
            config=TromptConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class TromptLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        TromptConfig,
        """Trompt for distributional regression.
        This class extends the SklearnBaseLSS class and uses the
        Trompt model with the default Trompt configuration.""",
        examples="""
        >>> from deeptab.models.experimental import TromptLSS
        >>> model = TromptLSS(d_model=64, n_layers=8)
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
            model=Trompt,
            config=TromptConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )
