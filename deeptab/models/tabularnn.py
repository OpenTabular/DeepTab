from deeptab.architectures.tabularnn import TabulaRNN
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.models.tabularnn_config import TabulaRNNConfig
from ._docstring import generate_docstring


class TabulaRNNRegressor(SklearnBaseRegressor):
    _model_cls = TabulaRNN
    _config_cls = TabulaRNNConfig

    __doc__ = generate_docstring(
        TabulaRNNConfig,
        model_description="""
        TabulaRNN regressor. This class extends the SklearnBaseRegressor
        class and uses the TabulaRNN model with the default TabulaRNN
        configuration.
        """,
        examples="""
        >>> from deeptab.models import TabulaRNNRegressor
        >>> from deeptab.configs import TabulaRNNConfig
        >>> model = TabulaRNNRegressor(model_config=TabulaRNNConfig(d_model=64))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class TabulaRNNClassifier(SklearnBaseClassifier):
    _model_cls = TabulaRNN
    _config_cls = TabulaRNNConfig

    __doc__ = generate_docstring(
        TabulaRNNConfig,
        model_description="""
        TabulaRNN classifier. This class extends the SklearnBaseClassifier
        class and uses the TabulaRNN model with the default TabulaRNN
        configuration.
        """,
        examples="""
        >>> from deeptab.models import TabulaRNNClassifier
        >>> from deeptab.configs import TabulaRNNConfig
        >>> model = TabulaRNNClassifier(model_config=TabulaRNNConfig(d_model=64))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class TabulaRNNLSS(SklearnBaseLSS):
    _model_cls = TabulaRNN
    _config_cls = TabulaRNNConfig

    __doc__ = generate_docstring(
        TabulaRNNConfig,
        model_description="""
        TabulaRNN for distributional regression. This class extends the SklearnBaseLSS
        class and uses the TabulaRNN model with the default TabulaRNN configuration.
        Supports RNN, LSTM, GRU, mLSTM, and sLSTM architectures.
        """,
        examples="""
        >>> from deeptab.models import TabulaRNNLSS
        >>> from deeptab.configs import TabulaRNNConfig
        >>> model = TabulaRNNLSS(model_config=TabulaRNNConfig(model_type='LSTM', d_model=128, n_layers=4))
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
