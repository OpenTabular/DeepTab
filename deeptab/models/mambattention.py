from deeptab.architectures.mambattention import MambAttention
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.core import PreprocessingConfig, TrainerConfig
from ..configs.models.mambattention_config import MambAttentionConfig
from ._docstring import generate_docstring


class MambAttentionRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        MambAttentionConfig,
        model_description="""
        MambAttention regressor. This class extends the SklearnBaseRegressor class and uses the MambAttention model
        with the default MambAttention configuration.
        """,
        examples="""
        >>> from deeptab.models import MambAttentionRegressor
        >>> model = MambAttentionRegressor(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: MambAttentionConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            model=MambAttention,
            config=MambAttentionConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
        )


class MambAttentionClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        MambAttentionConfig,
        model_description="""
        MambAttention classifier. This class extends the SklearnBaseClassifier class and uses the MambAttention model
        with the default MambAttention configuration.
        """,
        examples="""
        >>> from MambAttention.models import MambAttentionClassifier
        >>> model = MambAttentionClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: MambAttentionConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            model=MambAttention,
            config=MambAttentionConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
        )


class MambAttentionLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        MambAttentionConfig,
        model_description="""
        MambAttention LSS for distributional regression. This class extends the SklearnBaseLSS class and uses the MambAttention model
        with the default MambAttention configuration.
        """,
        examples="""
        >>> from MambAttention.models import MambAttentionLSS
        >>> model = MambAttentionLSS(d_model=64, n_layers=8)
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
            model=MambAttention,
            config=MambAttentionConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
        )
