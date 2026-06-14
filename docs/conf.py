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
    "sphinxext.opengraph",
]
autodoc_mock_imports = [
    "properscoring",
    "tqdm",
    "einops",
    "accelerate",
    "scikit-optimize",
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

# Suppress unresolvable cross-references in third-party docstrings.
# sphinx-autodoc-typehints 3.x still attempts to format signatures for
# dataclass __init__ methods even with typehints_use_signature=False, and
# crashes on nn.Module defaults like activation=nn.ReLU().
suppress_warnings = [
    "autodoc",  # nn.ReLU() default value signature crash
    "intersphinx.fetch_inventory",  # SSL/network failures when building offline
]


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "_templates", "homepage.md", "tutorials/notebooks/*.ipynb"]

# The reST default role (single back ticks `dict`) cross links to any code
# object (including Python, but others as well).
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# Move type hints into the parameter description body rather than the
# function signature. This avoids "list assignment index out of range"
# errors from sphinx-autodoc-typehints when a default value is an
# nn.Module instance (e.g. activation=nn.ReLU()).
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
# Do NOT rewrite signatures — that is the step that crashes on nn.Module defaults.
typehints_use_signature = False
typehints_use_signature_return = False

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
    "awesome_external_links": False,
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
html_logo = "images/logo/deeptab-favicon.png"
html_favicon = "images/logo/deeptab-favicon.png"

# Override the Sphinx default title that appends `documentation`
html_title = "DeepTab"
# Format of the last updated section in the footer
html_last_updated_fmt = "%Y-%m-%d"
# Hide [source] links in API docs
html_show_sourcelink = False

# -- Options for autodoc ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    # Exclude sklearn's inherited metadata-routing boilerplate. Their docstrings
    # contain :ref:`metadata_routing`, which only resolves in sklearn's own
    # Sphinx build and otherwise emits "undefined label" warnings here. They are
    # not part of DeepTab's public API surface.
    "exclude-members": (
        "set_output,"
        "get_metadata_routing,"
        "set_fit_request,"
        "set_predict_request,"
        "set_predict_proba_request,"
        "set_predict_log_proba_request,"
        "set_score_request,"
        "set_partial_fit_request,"
        "set_transform_request,"
        "set_inverse_transform_request"
    ),
}

# -- Options for sphinxext-opengraph ------------------------------------------

ogp_site_url = "https://deeptab.readthedocs.io/"
ogp_image = "https://deeptab.readthedocs.io/en/latest/_images/deeptab-v1.png"
ogp_description_length = 200
ogp_type = "website"

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
