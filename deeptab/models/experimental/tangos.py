from ...base_models.tangos import Tangos
from ...configs.preprocessing_config import PreprocessingConfig
from ...configs.tangos_config import TangosConfig
from ...configs.trainer_config import TrainerConfig
from ...utils.docstring_generator import generate_docstring
from ..utils.sklearn_base_classifier import SklearnBaseClassifier
from ..utils.sklearn_base_lss import SklearnBaseLSS
from ..utils.sklearn_base_regressor import SklearnBaseRegressor


class TangosRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        TangosConfig,
        model_description="""
        Tangos regressor. This class extends the SklearnBaseRegressor class and uses the Tangos model
        with the default Tangos configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import TangosRegressor
        >>> model = TangosRegressor(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: TangosConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=Tangos,
            config=TangosConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class TangosClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        TangosConfig,
        model_description="""
        Tangos classifier This class extends the SklearnBaseClassifier class and uses the Tangos model
        with the default Tangos configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import TangosClassifier
        >>> model = TangosClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: TangosConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=Tangos,
            config=TangosConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class TangosLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        TangosConfig,
        model_description="""
        Tangos for distributional regression. This class extends the SklearnBaseLSS class and uses the Tangos model
        with the default Tangos configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import TangosLSS
        >>> model = TangosLSS(d_model=64, n_layers=8)
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
            model=Tangos,
            config=TangosConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )
