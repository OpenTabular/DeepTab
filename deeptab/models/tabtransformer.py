from deeptab.architectures.tabtransformer import TabTransformer
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.core import PreprocessingConfig, TrainerConfig
from ..configs.models.tabtransformer_config import TabTransformerConfig
from ._docstring import generate_docstring


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
        observability_config=None,
    ):
        super().__init__(
            model=TabTransformer,
            config=TabTransformerConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            observability_config=observability_config,
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
        observability_config=None,
    ):
        super().__init__(
            model=TabTransformer,
            config=TabTransformerConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            observability_config=observability_config,
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
        observability_config=None,
    ):
        super().__init__(
            model=TabTransformer,
            config=TabTransformerConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            observability_config=observability_config,
        )
