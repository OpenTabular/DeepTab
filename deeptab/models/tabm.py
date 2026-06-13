from deeptab.architectures.tabm import TabM
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.core import PreprocessingConfig, TrainerConfig
from ..configs.models.tabm_config import TabMConfig
from ._docstring import generate_docstring


class TabMRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        TabMConfig,
        model_description="""
        TabM regressor. This class extends the SklearnBaseRegressor class and uses the TabM model
        with the default TabM configuration.
        """,
        examples="""
        >>> from deeptab.models import TabMRegressor
        >>> model = TabMRegressor(ensemble_size=32, model_type='full')
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: TabMConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        observability_config=None,
    ):
        super().__init__(
            model=TabM,
            config=TabMConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            observability_config=observability_config,
        )


class TabMClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        TabMConfig,
        model_description="""
        TabM classifier. This class extends the SklearnBaseClassifier class and uses the TabM model
        with the default TabM configuration.
        """,
        examples="""
        >>> from deeptab.models import TabMClassifier
        >>> model = TabMClassifier(ensemble_size=32, model_type='full')
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: TabMConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        observability_config=None,
    ):
        super().__init__(
            model=TabM,
            config=TabMConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            observability_config=observability_config,
        )


class TabMLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        TabMConfig,
        model_description="""
        TabM for distributional regressoion. This class extends the SklearnBaseLSS class and uses the TabM model
        with the default TabM configuration.
        """,
        examples="""
        >>> from deeptab.models import TabMLSS
        >>> model = TabMLSS(ensemble_size=32, model_type='full')
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
        observability_config=None,
    ):
        super().__init__(
            model=TabM,
            config=TabMConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            observability_config=observability_config,
        )
