from ..base_models.tabularnn import TabulaRNN
from ..configs.tabularnn_config import DefaultTabulaRNNConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class TabulaRNNRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultTabulaRNNConfig,
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

    def __init__(self, **kwargs):
        super().__init__(model=TabulaRNN, config=DefaultTabulaRNNConfig, **kwargs)


class TabulaRNNClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultTabulaRNNConfig,
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

    def __init__(self, **kwargs):
        super().__init__(model=TabulaRNN, config=DefaultTabulaRNNConfig, **kwargs)


class TabulaRNNLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultTabulaRNNConfig,
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

    def __init__(self, **kwargs):
        super().__init__(model=TabulaRNN, config=DefaultTabulaRNNConfig, **kwargs)
