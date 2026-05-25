from ..base_models.enode import ENODE
from ..configs.enode_config import ENODEConfig
from ..configs.preprocessing_config import PreprocessingConfig
from ..configs.trainer_config import TrainerConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class ENODERegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        ENODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (ENODE) Regressor. Slightly different with a MLP as a tabular task specific head. This class extends the SklearnBaseRegressor class and uses the ENODE model
        with the default ENODE configuration.
        """,
        examples="""
        >>> from deeptab.models import ENODERegressor
        >>> model = ENODERegressor()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: ENODEConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=ENODE,
            config=ENODEConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class ENODEClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        ENODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (ENODE) Classifier. Slightly different with a MLP as a tabular task specific head.
        This class extends the SklearnBaseClassifier class and uses the ENODE model
        with the default ENODE configuration.
        """,
        examples="""
        >>> from deeptab.models import ENODEClassifier
        >>> model = ENODEClassifier()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: ENODEConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=ENODE,
            config=ENODEConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class ENODELSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        ENODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (ENODE) for distributional regression. Slightly different with a MLP as a tabular task specific head.
        This class extends the SklearnBaseLSS class and uses the ENODE model
        with the default ENODE configuration.
        """,
        examples="""
        >>> from deeptab.models import ENODELSS
        >>> model = ENODELSS()
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
            model=ENODE,
            config=ENODEConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )
