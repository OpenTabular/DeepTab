from ..base_models.mambatab import MambaTab
from ..configs.mambatab_config import MambaTabConfig
from ..configs.preprocessing_config import PreprocessingConfig
from ..configs.trainer_config import TrainerConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class MambaTabRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        MambaTabConfig,
        model_description="""
        MambaTab regressor. This class extends the SklearnBaseRegressor class and uses the MambaTab model
        with the default MambaTab configuration.
        """,
        examples="""
        >>> from deeptab.models import MambaTabRegressor
        >>> model = MambaTabRegressor(d_model=64, n_layers=2)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: MambaTabConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=MambaTab,
            config=MambaTabConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class MambaTabClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        MambaTabConfig,
        model_description="""
        MambaTab classifier. This class extends the SklearnBaseClassifier class and uses the MambaTab model
        with the default MambaTab configuration.
        """,
        examples="""
        >>> from deeptab.models import MambaTabClassifier
        >>> model = MambaTabClassifier(d_model=64, n_layers=2)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: MambaTabConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=MambaTab,
            config=MambaTabConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class MambaTabLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        MambaTabConfig,
        model_description="""
        MambaTab LSS for distributional regression. This class extends the SklearnBaseLSS class and uses the MambaTab model
        with the default MambaTab configuration.
        """,
        examples="""
        >>> from deeptab.models import MambaTabLSS
        >>> model = MambaTabLSS(d_model=64, n_layers=2)
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
            model=MambaTab,
            config=MambaTabConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )
