# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from sphinxawesome_theme.postprocess import Icons

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(1, os.path.dirname(os.path.abspath("../")) + os.sep + "deeptab")

project = "deeptab"
project_copyright = "2024-2026, OpenTabular"

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
    "sphinx_togglebutton",
    "nbsphinx",
    "numpydoc",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "myst_parser",
    "sphinxcontrib.mermaid",
    # "pydata_sphinx_theme",
    "sphinx_autodoc_typehints",
    "sphinx_design",
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

# The root toctree document.
root_doc = "index"

# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# Suppress RemovedInSphinx10Warning from nbsphinx (upstream issue, not ours)
filterwarnings = [
    "ignore::sphinx.deprecation.RemovedInSphinx10Warning",
]


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
pygments_style = "github-light"
pygments_style_dark = "github-dark"

# -- Options for nbsphinx -----------------------------------------------------

# Don't execute notebooks during build
nbsphinx_execute = "never"

# -- Options for HTML output -------------------------------------------------

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# sphinxawesome_theme
html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
templates_path = ["_templates"]
html_extra_path = ["llms.txt"]

html_theme_options = {
    "show_breadcrumbs": True,
    "show_prev_next": True,
    "show_scrolltop": True,
    "awesome_headerlinks": True,
    "awesome_external_links": True,
    "main_nav_links": {
        "GitHub": "https://github.com/OpenTabular/DeepTab",
        "PyPI": "https://pypi.org/project/deeptab/",
    },
}

# Use the theme's own permalink icon
html_permalinks_icon = Icons.permalinks_icon

# Keep full navigation sidebar on all pages including API reference
# Remove this to use theme's default sidebars everywhere
# html_sidebars = {
#     "api/**": ["sidebar_main_nav_links.html"],
# }

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "images/logo/deeptab-v1.png"

# Override the Sphinx default title that appends `documentation`
html_title = "DeepTab"
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

# -- Options for MyST parser --------------------------------------------------

myst_enable_extensions = [
    "colon_fence",  # Enable ```{note}, ```{tip}, etc.
    "deflist",  # Definition lists
    "dollarmath",  # LaTeX math with $...$
    "fieldlist",  # Field lists
    "html_admonition",  # HTML admonitions
    "html_image",  # HTML images
    "replacements",  # Text replacements
    "smartquotes",  # Smart quotes
    "strikethrough",  # ~~strikethrough~~
    "substitution",  # Variable substitution
    "tasklist",  # Task lists [ ]
]

# Use sphinx-design for admonitions (better styling with icons)
myst_fence_as_directive = [
    "note",
    "warning",
    "tip",
    "important",
    "caution",
    "attention",
    "danger",
    "error",
    "hint",
    "seealso",
]
