import textwrap


def generate_docstring(config, model_description, examples):
    """Build a model class docstring from its description, constructor
    parameters, and usage examples.

    DeepTab estimators accept a small, fixed set of config objects rather than
    flat hyperparameters, so the documented ``Parameters`` mirror the real
    constructor signature. The architecture hyperparameters live on the model
    config and are documented on that config class, which avoids listing config
    fields as if they were constructor arguments.
    """
    config_name = config.__name__

    description = textwrap.indent(textwrap.dedent(model_description).strip(), "    ")
    examples_block = textwrap.indent(textwrap.dedent(examples).strip(), "    ")

    parameters = textwrap.indent(
        textwrap.dedent(
            f"""\
            model_config : {config_name}, optional
                Architecture hyperparameters for the model. If ``None``, a
                default :class:`~deeptab.configs.{config_name}` is used. See
                that class for the full list of available fields.
            preprocessing_config : PreprocessingConfig, optional
                Feature preprocessing settings such as scaling, encoding, and
                numerical embeddings. If ``None``, defaults from
                :class:`~deeptab.configs.PreprocessingConfig` are used.
            trainer_config : TrainerConfig, optional
                Training-loop settings such as epochs, batch size, learning
                rate, and early stopping. If ``None``, defaults from
                :class:`~deeptab.configs.TrainerConfig` are used.
            observability_config : ObservabilityConfig, optional
                Optional logging, experiment tracking, and run-directory
                settings (``deeptab.core.observability.ObservabilityConfig``).
                If ``None``, observability is disabled and the estimator emits
                nothing.
            random_state : int, optional
                Seed for reproducible weight initialisation and data shuffling."""
        ),
        "    ",
    )

    return f"""
{description}

    Parameters
    ----------
{parameters}

    Examples
    --------
{examples_block}
    """
