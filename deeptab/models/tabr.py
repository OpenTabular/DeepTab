from deeptab.architectures.tabr import TabR
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.core import PreprocessingConfig, TrainerConfig
from ..configs.models.tabr_config import TabRConfig
from ._docstring import generate_docstring


class TabRRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        TabRConfig,
        model_description="""
        TabR regressor. This class extends the SklearnBaseRegressor class and uses the TabR model
        with the default TabR configuration.
        """,
        examples="""
        >>> from deeptab.models import TabRRegressor
        >>> model = TabRRegressor()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: TabRConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            model=TabR,
            config=TabRConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
        )


class TabRClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        TabRConfig,
        model_description="""
        TabR classifier. This class extends the SklearnBaseClassifier class and uses the TabR model
        with the default TabR configuration.
        """,
        examples="""
        >>> from deeptab.models import TabRClassifier
        >>> model = TabRClassifier()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: TabRConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            model=TabR,
            config=TabRConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
        )


class TabRLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        TabRConfig,
        model_description="""
        TabR regressor. This class extends the SklearnBaseLSS class and uses the TabR model
        with the default TabR configuration.
        """,
        examples="""
        >>> from deeptab.models import TabRLSS
        >>> model = TabRLSS(d_model=64, family='normal')
        >>> model.fit(X_train, y_train)
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
            model=TabR,
            config=TabRConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
        )
