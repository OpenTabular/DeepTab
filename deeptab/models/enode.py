from deeptab.architectures.enode import ENODE
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.models.enode_config import ENODEConfig
from ._docstring import generate_docstring


class ENODERegressor(SklearnBaseRegressor):
    _model_cls = ENODE
    _config_cls = ENODEConfig

    __doc__ = generate_docstring(
        ENODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (ENODE) Regressor. Slightly different with a MLP as a tabular task specific head. This class extends the SklearnBaseRegressor class and uses the ENODE model
        with the default ENODE configuration.
        """,
        examples="""
        >>> from deeptab.models import ENODERegressor
        >>> model = ENODERegressor()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class ENODEClassifier(SklearnBaseClassifier):
    _model_cls = ENODE
    _config_cls = ENODEConfig

    __doc__ = generate_docstring(
        ENODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (ENODE) Classifier. Slightly different with a MLP as a tabular task specific head.
        This class extends the SklearnBaseClassifier class and uses the ENODE model
        with the default ENODE configuration.
        """,
        examples="""
        >>> from deeptab.models import ENODEClassifier
        >>> model = ENODEClassifier()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )


class ENODELSS(SklearnBaseLSS):
    _model_cls = ENODE
    _config_cls = ENODEConfig

    __doc__ = generate_docstring(
        ENODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (ENODE) for distributional regression. Slightly different with a MLP as a tabular task specific head.
        This class extends the SklearnBaseLSS class and uses the ENODE model
        with the default ENODE configuration.
        """,
        examples="""
        >>> from deeptab.models import ENODELSS
        >>> model = ENODELSS()
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )
