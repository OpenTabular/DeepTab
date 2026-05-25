from ..base_models.ndtf import NDTF
from ..configs.ndtf_config import NDTFConfig
from ..configs.preprocessing_config import PreprocessingConfig
from ..configs.trainer_config import TrainerConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class NDTFRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        NDTFConfig,
        model_description="""
        Neural Decision Forest regressor. This class extends the SklearnBaseRegressor class and uses the NDTF model
        with the default NDTF configuration.
        """,
        examples="""
        >>> from deeptab.models import NDTFRegressor
        >>> model = NDTFRegressor(n_ensembles=12, max_depth=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: NDTFConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=NDTF,
            config=NDTFConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class NDTFClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        NDTFConfig,
        model_description="""
        Neural Decision Forest classifier. This class extends the SklearnBaseClassifier class and uses the NDTF model
        with the default NDTF configuration.
        """,
        examples="""
        >>> from deeptab.models import NDTFClassifier
        >>> model = NDTFClassifier(n_ensembles=12, max_depth=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: NDTFConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=NDTF,
            config=NDTFConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class NDTFLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        NDTFConfig,
        model_description="""
        Neural Decision Forest for distributional regression. This class extends the SklearnBaseLSS class and uses the NDTF model
        with the default NDTF configuration.
        """,
        examples="""
        >>> from deeptab.models import NDTFLSS
        >>> model = NDTFLSS(n_ensembles=12, max_depth=8)
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
            model=NDTF,
            config=NDTFConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )
