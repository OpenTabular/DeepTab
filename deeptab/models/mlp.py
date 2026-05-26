from deeptab.architectures.mlp import MLP
from deeptab.models.classifier_base import SklearnBaseClassifier
from deeptab.models.lss_base import SklearnBaseLSS
from deeptab.models.regressor_base import SklearnBaseRegressor

from ..configs.core import PreprocessingConfig, TrainerConfig
from ..configs.models.mlp_config import MLPConfig
from ._docstring import generate_docstring


class MLPRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        MLPConfig,
        model_description="""
        Multi-Layer Perceptron regressor. This class extends the SklearnBaseRegressor class and uses the MLP model
        with the default MLP configuration.
        """,
        examples="""
        >>> from deeptab.models import MLPRegressor
        >>> from deeptab.configs import MLPConfig, TrainerConfig
        >>> model = MLPRegressor(
        ...     model_config=MLPConfig(layer_sizes=[128, 64]),
        ...     trainer_config=TrainerConfig(max_epochs=100, lr=1e-3),
        ... )
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        """,
    )

    def __init__(
        self,
        model_config: MLPConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            model=MLP,
            config=MLPConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
        )


class MLPClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        MLPConfig,
        model_description="""
        Multi-Layer Perceptron classifier This class extends the SklearnBaseClassifier class and uses the MLP model
        with the default MLP configuration.
        """,
        examples="""
        >>> from deeptab.models import MLPClassifier
        >>> from deeptab.configs import MLPConfig, TrainerConfig
        >>> model = MLPClassifier(
        ...     model_config=MLPConfig(layer_sizes=[128, 64]),
        ...     trainer_config=TrainerConfig(max_epochs=100, lr=1e-3),
        ... )
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        """,
    )

    def __init__(
        self,
        model_config: MLPConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            model=MLP,
            config=MLPConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
        )


class MLPLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        MLPConfig,
        model_description="""
        Multi-Layer Perceptron for distributional regression. This class extends the SklearnBaseLSS class and uses the MLP model
        with the default MLP configuration.
        """,
        examples="""
        >>> from deeptab.models import MLPLSS
        >>> model = MLPLSS(d_model=64, n_layers=8)
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
            model=MLP,
            config=MLPConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
        )
