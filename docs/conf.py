# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(1, os.path.dirname(os.path.abspath("../")) + os.sep + "deeptab")

project = "deeptab"
project_copyright = "2024, BASF SE"
author = "Anton Frederik Thielmann, Manish Kumar, Christoph Weisser, Benjamin Saefken, Soheila Samiee"

try:
    version = _version("deeptab")
except PackageNotFoundError:
    version = "0+unknown"
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "nbsphinx",
    "numpydoc",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "myst_parser",
    "mdinclude",  # custom module
    "sphinx_rtd_theme",
    # "pydata_sphinx_theme",
    "sphinx_autodoc_typehints",
]
autodoc_mock_imports = [
    "lightning",
    "torch",
    "torchmetrics",
    "pytorch_lightning",
    "numpy",
    "pandas",
    "sklearn",
    "properscoring",
    "tqdm",
    "einops",
    "accelerate",
    "scikit-optimize",
    "scipy",
    "skopt",
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".md"]
# source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "_templates", "homepage.md"]

# The reST default role (single back ticks `dict`) cross links to any code
# object (including Python, but others as well).
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# 'sphinx_rtd_theme'  # 'furo', 'press', 'pydata_sphinx_theme'
html_theme = "sphinx_book_theme"
# html_static_path = ['_static']
# html_css_files = ['custom.css']
# html_js_files = ['custom.js']

html_theme_options = {
    "globaltoc_collapse": False,
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "images/logo/mamba_tabular.jpg"

# Override the Sphinx default title that appends `documentation`
html_title = f"{project} {project_copyright}"
# Format of the last updated section in the footer
html_last_updated_fmt = "%Y-%m-%d"

# -- Options for autodoc ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "exclude-members": "set_output",
}

# generate autosummary even if no references
autosummary_generate = True

# -- Options for numpydoc -----------------------------------------------------

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69

numpydoc_show_class_members = False
