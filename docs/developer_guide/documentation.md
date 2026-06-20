# Documentation

DeepTab documentation is built with [Sphinx 9](https://www.sphinx-doc.org/) using the [sphinxawesome-theme](https://sphinxawesome.xyz/). Pages are written in [MyST Markdown](https://myst-parser.readthedocs.io/) or reStructuredText. API reference pages are generated automatically from docstrings via [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) and [numpydoc](https://numpydoc.readthedocs.io/).

## Building locally

```bash
just docs
```

This runs:

```bash
poetry run sphinx-build -b html docs/ docs/_build/html -W --keep-going
```

The `-W` flag treats every Sphinx warning as a build error; `--keep-going` collects all warnings before stopping so you can fix them in one pass. Open `docs/_build/html/index.html` in a browser to preview the result.

## Directory layout

```
docs/
├── conf.py                  # Sphinx configuration
├── index.rst                # Root toctree (navigation structure)
├── _static/
│   └── custom.css           # Theme overrides and syntax highlight palette
├── homepage.md              # Landing page content
├── getting_started/         # Initial onboarding
├── core_concepts/           # Deep-dive concept guides
├── tutorials/               # Hands-on tutorials with notebooks
├── model_zoo/               # Model documentation and comparisons
├── api/                     # Auto-generated API reference
└── developer_guide/         # This section
```

## Adding a new page

1. Create a `.md` file in the appropriate sub-directory (e.g. `docs/developer_guide/my_topic.md`).
2. Add the path (without extension) to the matching `.. toctree::` block in `docs/index.rst`:

```rst
.. toctree::
   :caption: Developer Guide
   :maxdepth: 1
   :hidden:

   developer_guide/my_topic
```

3. Run `just docs` and fix any warnings before opening a PR.

## Docstring style

All public classes and functions use **NumPy-style docstrings**. The full specification is at [numpydoc.readthedocs.io](https://numpydoc.readthedocs.io/en/latest/format.html). A minimal example:

```python
def fit(self, X, y, val_size=0.2):
    """Fit the model to training data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y : array-like of shape (n_samples,)
        Target values.
    val_size : float, default=0.2
        Fraction of training data used for validation.

    Returns
    -------
    self : object
        Fitted estimator.

    Raises
    ------
    ValueError
        If ``X`` and ``y`` have incompatible shapes.
    """
```

## Preventing duplicate-description warnings

Sphinx raises a warning when `autodoc` documents the same symbol more than once. If a class is re-exported from a package `__init__`, add `:noindex:` to the second occurrence's directive:

```rst
.. autoclass:: deeptab.models.MLPClassifier
   :noindex:
```

## Code blocks in pages

Use fenced code blocks with a language tag for syntax highlighting:

````markdown
```python
model = MLPClassifier()
model.fit(X_train, y_train)
```
````

For shell commands use ` ```bash ` or ` ```sh `.

## Mermaid diagrams

The `sphinxcontrib-mermaid` extension is enabled. Use a `mermaid` code fence:

````markdown
```{mermaid}
flowchart LR
    A[Start] --> B[End]
```
````

## ReadTheDocs

Documentation is hosted on ReadTheDocs and built automatically on every push to `main` and on every new release tag. The build configuration lives in `readthedocs.yaml` at the repository root. RTD uses Python 3.12 on Ubuntu 24.04.
