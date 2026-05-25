from ..base_models.node import NODE
from ..configs.node_config import NODEConfig
from ..configs.preprocessing_config import PreprocessingConfig
from ..configs.trainer_config import TrainerConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class NODERegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        NODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (NODE) Regressor. Slightly different with a MLP as a tabular task specific head. This class extends the SklearnBaseRegressor class and uses the NODE model
        with the default NODE configuration.
        """,
        examples="""
        >>> from deeptab.models import NODERegressor
        >>> model = NODERegressor()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: NODEConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=NODE,
            config=NODEConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class NODEClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        NODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (NODE) Classifier. Slightly different with a MLP as a tabular task specific head.
        This class extends the SklearnBaseClassifier class and uses the NODE model
        with the default NODE configuration.
        """,
        examples="""
        >>> from deeptab.models import NODEClassifier
        >>> model = NODEClassifier()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: NODEConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=NODE,
            config=NODEConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class NODELSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        NODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (NODE) for distributional regression. Slightly different with a MLP as a tabular task specific head.
        This class extends the SklearnBaseLSS class and uses the NODE model
        with the default NODE configuration.
        """,
        examples="""
        >>> from deeptab.models import NODELSS
        >>> model = NODELSS()
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
            model=NODE,
            config=NODEConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )
