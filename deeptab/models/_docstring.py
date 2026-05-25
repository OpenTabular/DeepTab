import inspect
import textwrap

from pretab.preprocessor import Preprocessor


def generate_docstring(config, model_description, examples):
    """Generates the complete docstring for any model class by combining config and Preprocessor docstrings.

    The `Parameters` tag is stripped from the Preprocessor docstring to avoid duplication.
    """
    # inspect.cleandoc is the correct tool for Python docstrings: it strips
    # leading blank lines, then removes the common indentation from lines 2+
    # (the class-body indent).  textwrap.dedent cannot do this because Python
    # stores line 1 without any leading whitespace, making the common indent 0.
    config_doc = inspect.cleandoc(config.__doc__ or "No documentation.")
    preprocessor_doc = inspect.cleandoc(Preprocessor.__doc__ or "No documentation.")

    # After cleandoc the section header is at column 0: "Parameters\n----------\n"
    preprocessor_doc_cleaned = preprocessor_doc.split("Parameters\n----------\n", 1)[-1].strip()
    preprocessor_doc_cleaned = preprocessor_doc_cleaned.split("Attributes")[0].strip()

    # Combine config doc + preprocessor params, then re-indent uniformly at 4 spaces.
    config_doc_indented = textwrap.indent(config_doc + "\n\n" + preprocessor_doc_cleaned, "    ")

    description_indented = textwrap.indent(textwrap.dedent(model_description).strip(), "    ")
    examples_indented = textwrap.indent(textwrap.dedent(examples).strip(), "    ")

    return f"""
{description_indented}

    Notes
    -----
    The parameters for this class include the attributes from the config
    dataclass as well as preprocessing arguments handled by the base class.

{config_doc_indented}

    Examples
    --------
{examples_indented}
    """
