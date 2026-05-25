from ..base_models.mambular import Mambular
from ..configs.mambular_config import MambularConfig
from ..configs.preprocessing_config import PreprocessingConfig
from ..configs.trainer_config import TrainerConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class MambularRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        MambularConfig,
        model_description="""
        Mambular regressor. This class extends the SklearnBaseRegressor class and uses the Mambular model
        with the default Mambular configuration.
        """,
        examples="""
        >>> from deeptab.models import MambularRegressor
        >>> model = MambularRegressor(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: MambularConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=Mambular,
            config=MambularConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class MambularClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        MambularConfig,
        model_description="""
        Mambular classifier. This class extends the SklearnBaseClassifier class and uses the Mambular model
        with the default Mambular configuration.
        """,
        examples="""
        >>> from deeptab.models import MambularClassifier
        >>> model = MambularClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: MambularConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=Mambular,
            config=MambularConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class MambularLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        MambularConfig,
        model_description="""
        Mambular LSS for distributional regression. This class extends the SklearnBaseLSS class and uses the Mambular model
        with the default Mambular configuration.
        """,
        examples="""
        >>> from deeptab.models import MambularLSS
        >>> model = MambularLSS(d_model=64, n_layers=8)
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
            model=Mambular,
            config=MambularConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )
