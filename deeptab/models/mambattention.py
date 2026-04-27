from ..base_models.mambattn import MambAttention
from ..configs.mambattention_config import DefaultMambAttentionConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class MambAttentionRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultMambAttentionConfig,
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

    def __init__(self, **kwargs):
        super().__init__(model=MambAttention, config=DefaultMambAttentionConfig, **kwargs)


class MambAttentionClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultMambAttentionConfig,
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

    def __init__(self, **kwargs):
        super().__init__(model=MambAttention, config=DefaultMambAttentionConfig, **kwargs)


class MambAttentionLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultMambAttentionConfig,
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

    def __init__(self, **kwargs):
        super().__init__(model=MambAttention, config=DefaultMambAttentionConfig, **kwargs)
