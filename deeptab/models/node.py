from deeptab.architectures.node import NODE
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.core import PreprocessingConfig, TrainerConfig
from ..configs.models.node_config import NODEConfig
from ._docstring import generate_docstring


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
    ):
        super().__init__(
            model=NODE,
            config=NODEConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
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
    ):
        super().__init__(
            model=NODE,
            config=NODEConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
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
    ):
        super().__init__(
            model=NODE,
            config=NODEConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
        )
