from deeptab.architectures.node import NODE
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.models.node_config import NODEConfig
from ._docstring import generate_docstring


class NODERegressor(SklearnBaseRegressor):
    _model_cls = NODE
    _config_cls = NODEConfig

    __doc__ = generate_docstring(
        NODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (NODE) Regressor. Slightly different with a MLP as a tabular task specific head. This class extends the SklearnBaseRegressor class and uses the NODE model
        with the default NODE configuration.
        """,
        examples="""
        >>> from deeptab.models import NODERegressor
        >>> model = NODERegressor()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class NODEClassifier(SklearnBaseClassifier):
    _model_cls = NODE
    _config_cls = NODEConfig

    __doc__ = generate_docstring(
        NODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (NODE) Classifier. Slightly different with a MLP as a tabular task specific head.
        This class extends the SklearnBaseClassifier class and uses the NODE model
        with the default NODE configuration.
        """,
        examples="""
        >>> from deeptab.models import NODEClassifier
        >>> model = NODEClassifier()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class NODELSS(SklearnBaseLSS):
    _model_cls = NODE
    _config_cls = NODEConfig

    __doc__ = generate_docstring(
        NODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (NODE) for distributional regression. Slightly different with a MLP as a tabular task specific head.
        This class extends the SklearnBaseLSS class and uses the NODE model
        with the default NODE configuration.
        """,
        examples="""
        >>> from deeptab.models import NODELSS
        >>> model = NODELSS()
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
