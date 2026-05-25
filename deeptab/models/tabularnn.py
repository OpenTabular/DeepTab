from deeptab.architectures.tabularnn import TabulaRNN
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.core import PreprocessingConfig, TrainerConfig
from ..configs.models.tabularnn_config import TabulaRNNConfig
from ._docstring import generate_docstring


class TabulaRNNRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        TabulaRNNConfig,
        model_description="""
        TabulaRNN regressor. This class extends the SklearnBaseRegressor
        class and uses the TabulaRNN model with the default TabulaRNN
        configuration.
        """,
        examples="""
        >>> from deeptab.models import TabulaRNNRegressor
        >>> model = TabulaRNNRegressor(d_model=64)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: TabulaRNNConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=TabulaRNN,
            config=TabulaRNNConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class TabulaRNNClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        TabulaRNNConfig,
        model_description="""
        TabulaRNN classifier. This class extends the SklearnBaseClassifier
        class and uses the TabulaRNN model with the default TabulaRNN
        configuration.
        """,
        examples="""
        >>> from deeptab.models import TabulaRNNClassifier
        >>> model = TabulaRNNClassifier(d_model=64)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: TabulaRNNConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=TabulaRNN,
            config=TabulaRNNConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class TabulaRNNLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        TabulaRNNConfig,
        model_description="""
        TabulaRNN for distributional regression. This class extends the SklearnBaseLSS
        class and uses the TabulaRNN model with the default TabulaRNN configuration.
        Supports RNN, LSTM, GRU, mLSTM, and sLSTM architectures.
        """,
        examples="""
        >>> from deeptab.models import TabulaRNNLSS
        >>> model = TabulaRNNLSS(model_type='LSTM', d_model=128, n_layers=4)
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
            model=TabulaRNN,
            config=TabulaRNNConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )
