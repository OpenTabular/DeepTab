from ...base_models.tangos import Tangos
from ...configs.tangos_config import DefaultTangosConfig
from ...utils.docstring_generator import generate_docstring
from ..utils.sklearn_base_classifier import SklearnBaseClassifier
from ..utils.sklearn_base_lss import SklearnBaseLSS
from ..utils.sklearn_base_regressor import SklearnBaseRegressor


class TangosRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultTangosConfig,
        model_description="""
        Tangos regressor. This class extends the SklearnBaseRegressor class and uses the Tangos model
        with the default Tangos configuration.
        """,
        examples="""
        >>> from deeptab.models import TangosRegressor
        >>> model = TangosRegressor(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=Tangos, config=DefaultTangosConfig, **kwargs)


class TangosClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultTangosConfig,
        model_description="""
        Tangos classifier This class extends the SklearnBaseClassifier class and uses the Tangos model
        with the default Tangos configuration.
        """,
        examples="""
        >>> from deeptab.models import TangosClassifier
        >>> model = TangosClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=Tangos, config=DefaultTangosConfig, **kwargs)


class TangosLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultTangosConfig,
        model_description="""
        Tangos for distributional regression. This class extends the SklearnBaseLSS class and uses the Tangos model
        with the default Tangos configuration.
        """,
        examples="""
        >>> from deeptab.models import TangosLSS
        >>> model = TangosLSS(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=Tangos, config=DefaultTangosConfig, **kwargs)
