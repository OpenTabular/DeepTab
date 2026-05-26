from deeptab.architectures.ft_transformer import FTTransformer
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.core import PreprocessingConfig, TrainerConfig
from ..configs.models.fttransformer_config import FTTransformerConfig
from ._docstring import generate_docstring


class FTTransformerRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        FTTransformerConfig,
        model_description="""
        FTTransformer regressor. This class extends the SklearnBaseRegressor
        class and uses the FTTransformer model with the default FTTransformer
        configuration.
        """,
        examples="""
        >>> from deeptab.models import FTTransformerRegressor
        >>> model = FTTransformerRegressor(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: FTTransformerConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            model=FTTransformer,
            config=FTTransformerConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
        )


class FTTransformerClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        FTTransformerConfig,
        """FTTransformer Classifier. This class extends the SklearnBaseClassifier class
        and uses the FTTransformer model with the default FTTransformer configuration.""",
        examples="""
        >>> from deeptab.models import FTTransformerClassifier
        >>> model = FTTransformerClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: FTTransformerConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            model=FTTransformer,
            config=FTTransformerConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
        )


class FTTransformerLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        FTTransformerConfig,
        """FTTransformer for distributional regression.
        This class extends the SklearnBaseLSS class and uses the
        FTTransformer model with the default FTTransformer configuration.""",
        examples="""
        >>> from deeptab.models import FTTransformerLSS
        >>> model = FTTransformerLSS(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train, family="normal")
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
            model=FTTransformer,
            config=FTTransformerConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
        )
