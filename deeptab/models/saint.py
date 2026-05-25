from ..base_models.saint import SAINT
from ..configs.preprocessing_config import PreprocessingConfig
from ..configs.saint_config import SAINTConfig
from ..configs.trainer_config import TrainerConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class SAINTRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        SAINTConfig,
        model_description="""
        SAINT regressor. This class extends the SklearnBaseRegressor
        class and uses the SAINT model with the default SAINT
        configuration.
        """,
        examples="""
        >>> from deeptab.models import SAINTRegressor
        >>> model = SAINTRegressor(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: SAINTConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=SAINT,
            config=SAINTConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class SAINTClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        SAINTConfig,
        """SAINT Classifier. This class extends the SklearnBaseClassifier class
        and uses the SAINT model with the default SAINT configuration.""",
        examples="""
        >>> from deeptab.models import SAINTClassifier
        >>> model = SAINTClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: SAINTConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=SAINT,
            config=SAINTConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class SAINTLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        SAINTConfig,
        """SAINT for distributional regression.
        This class extends the SklearnBaseLSS class and uses the
        SAINT model with the default SAINT configuration.""",
        examples="""
        >>> from deeptab.models import SAINTLSS
        >>> model = SAINTLSS(d_model=64, n_layers=8)
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
            model=SAINT,
            config=SAINTConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )
