from ..base_models.tabtransformer import TabTransformer
from ..configs.preprocessing_config import PreprocessingConfig
from ..configs.tabtransformer_config import TabTransformerConfig
from ..configs.trainer_config import TrainerConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class TabTransformerRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        TabTransformerConfig,
        model_description="""
        TabTransformer regressor. This class extends the SklearnBaseRegressor class and uses the TabTransformer model
        with the default TabTransformer configuration.
        """,
        examples="""
        >>> from deeptab.models import TabTransformerRegressor
        >>> model = TabTransformerRegressor()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: TabTransformerConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=TabTransformer,
            config=TabTransformerConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class TabTransformerClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        TabTransformerConfig,
        model_description="""
        TabTransformer classifier. This class extends the SklearnBaseClassifier class and uses the TabTransformer model
        with the default TabTransformer configuration.
        """,
        examples="""
        >>> from deeptab.models import TabTransformerClassifier
        >>> model = TabTransformerClassifier()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: TabTransformerConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=TabTransformer,
            config=TabTransformerConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class TabTransformerLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        TabTransformerConfig,
        model_description="""
        TabTransformer for distributional regression. This class extends the SklearnBaseLSS class and uses the TabTransformer model
        with the default TabTransformer configuration.
        """,
        examples="""
        >>> from deeptab.models import TabTransformerLSS
        >>> model = TabTransformerLSS()
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
            model=TabTransformer,
            config=TabTransformerConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )
