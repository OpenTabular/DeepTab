from deeptab.architectures.mambatab import MambaTab
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.core import PreprocessingConfig, TrainerConfig
from ..configs.models.mambatab_config import MambaTabConfig
from ._docstring import generate_docstring


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
