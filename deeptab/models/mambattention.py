from deeptab.architectures.mambattention import MambAttention
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.models.mambattention_config import MambAttentionConfig
from ._docstring import generate_docstring


class MambAttentionRegressor(SklearnBaseRegressor):
    _model_cls = MambAttention
    _config_cls = MambAttentionConfig

    __doc__ = generate_docstring(
        MambAttentionConfig,
        model_description="""
        MambAttention regressor. This class extends the SklearnBaseRegressor class and uses the MambAttention model
        with the default MambAttention configuration.
        """,
        examples="""
        >>> from deeptab.models import MambAttentionRegressor
        >>> from deeptab.configs import MambAttentionConfig
        >>> model = MambAttentionRegressor(model_config=MambAttentionConfig(d_model=64, n_layers=8))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class MambAttentionClassifier(SklearnBaseClassifier):
    _model_cls = MambAttention
    _config_cls = MambAttentionConfig

    __doc__ = generate_docstring(
        MambAttentionConfig,
        model_description="""
        MambAttention classifier. This class extends the SklearnBaseClassifier class and uses the MambAttention model
        with the default MambAttention configuration.
        """,
        examples="""
        >>> from deeptab.models import MambAttentionClassifier
        >>> from deeptab.configs import MambAttentionConfig
        >>> model = MambAttentionClassifier(model_config=MambAttentionConfig(d_model=64, n_layers=8))
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class MambAttentionLSS(SklearnBaseLSS):
    _model_cls = MambAttention
    _config_cls = MambAttentionConfig

    __doc__ = generate_docstring(
        MambAttentionConfig,
        model_description="""
        MambAttention LSS for distributional regression. This class extends the SklearnBaseLSS class and uses the MambAttention model
        with the default MambAttention configuration.
        """,
        examples="""
        >>> from deeptab.models import MambAttentionLSS
        >>> from deeptab.configs import MambAttentionConfig
        >>> model = MambAttentionLSS(model_config=MambAttentionConfig(d_model=64, n_layers=8))
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
