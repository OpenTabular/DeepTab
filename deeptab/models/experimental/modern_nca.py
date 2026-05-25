from deeptab.architectures.experimental.modern_nca import ModernNCA
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ...configs.modernnca_config import ModernNCAConfig
from ...configs.preprocessing_config import PreprocessingConfig
from ...configs.trainer_config import TrainerConfig
from .._docstring import generate_docstring


class ModernNCARegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        ModernNCAConfig,
        model_description="""
        Multi-Layer Perceptron regressor. This class extends the SklearnBaseRegressor class and uses the ModernNCA model
        with the default ModernNCA configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import ModernNCARegressor
        >>> model = ModernNCARegressor(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: ModernNCAConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=ModernNCA,
            config=ModernNCAConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class ModernNCAClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        ModernNCAConfig,
        model_description="""
        Multi-Layer Perceptron classifier This class extends the SklearnBaseClassifier class and uses the ModernNCA model
        with the default ModernNCA configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import ModernNCAClassifier
        >>> model = ModernNCAClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: ModernNCAConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=ModernNCA,
            config=ModernNCAConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )


class ModernNCALSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        ModernNCAConfig,
        model_description="""
        Multi-Layer Perceptron for distributional regression. This class extends the SklearnBaseLSS class and uses the ModernNCA model
        with the default ModernNCA configuration.
        """,
        examples="""
        >>> from deeptab.models.experimental import ModernNCALSS
        >>> model = ModernNCALSS(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(
        self,
        model_config: ModernNCAConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=ModernNCA,
            config=ModernNCAConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
            **kwargs,
        )
