# Contribution Guidelines

Thanks for contributing to DeepTab. This page covers environment setup, the local workflow, and what a pull request needs to pass review.

## Code of Conduct

All contributors are expected to follow the project [Code of Conduct](https://github.com/OpenTabular/DeepTab/blob/main/CODE_OF_CONDUCT.md), which sets the standard for respectful and inclusive participation.

## Setting up the development environment

The project uses [Poetry](https://python-poetry.org/docs/) for dependency management and the [just](https://just.systems/man/en/) command runner for common tasks (`justfile` defines testing, building, and formatting).

1. Clone the repository:

```bash
git clone https://github.com/OpenTabular/DeepTab
cd DeepTab
```

2. Install Python 3.12 or 3.13 for development that includes documentation builds, Poetry 2.x (`pip install "poetry>=2,<3"`), and `just` (see the [just install guide](https://just.systems/man/en/packages.html), e.g. `brew install just`). Python 3.11 remains supported for core development and testing, but not for the docs toolchain.

3. Install dependencies and register the pre-commit hooks:

```bash
just install
```

Without `just`, run the same steps directly:

```bash
poetry install
poetry run pre-commit install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push
```

To work on the docs, select the Python 3.12 or 3.13 interpreter with
`poetry env use 3.12` or `poetry env use 3.13`, then explicitly install the
docs group with `poetry install --with docs`.

Use `just install-extras` and `just test-extras` for the logging/tracking-positive
test lane, or `just install-core` and `just test-core` for the core-only lane.
These synchronize only `main,dev` and remove other packages, including docs
dependencies; use separate environments for comparisons. Before building docs,
use Python 3.12 or 3.13 and run `poetry install --with docs` to restore the
docs toolchain. `just test` retains the 90% line-coverage gate.
For a selected file or marker, use `just test-run tests/data/ -v` or
`just test-run -m integration`; these targeted runs do not enforce coverage.
See [Testing](testing.md) for profile checks, categories, and setup details.

## How to contribute

1. Branch off `main` with a short, descriptive name.
2. Make your changes, keeping each pull request to a single logical focus.
3. Add or update tests, and run the full check suite locally before pushing:

```bash
just test     # full suite with coverage
just check    # shared file checks, formatters, lint, and type checking
just docs     # build HTML docs (warnings treated as errors)
```

4. Commit using Conventional Commits via `just commit`. If `just check` reformats files, review and commit those separately with `style: apply formatting`.
5. Open a pull request to `main`, reference any related issues, and address review feedback until approved and merged.

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to enforce code quality automatically. `just install` registers all three hook types so each fires at the right time:

| Stage        | Hook                                                                                  |
| ------------ | ------------------------------------------------------------------------------------- |
| `commit-msg` | Validates the message against Conventional Commits.                                   |
| `pre-commit` | File hygiene, Ruff format/lint for Python, and Prettier for Markdown, YAML, and JSON. |
| `pre-push`   | The shared checks above plus project-wide Pyright type checking.                      |

```{important}
Run `just check` before opening a PR. It runs push-stage hooks against all files,
including the shared file checks and Pyright. Commit-message validation runs
separately when committing; the test suite and docs build are separate gates too.
```

Individual recipes are available when you want to run one step:

| Command       | Action                                 |
| ------------- | -------------------------------------- |
| `just lint`   | Lint and auto-fix with ruff.           |
| `just format` | Run the ruff formatter.                |
| `just types`  | Run the pyright type checker.          |
| `just check`  | Run push-stage hooks across all files. |

If a hook reformats files, review its changes and rerun `just check` until it
passes. `just format` formats Python; the Prettier hook handles Markdown, YAML,
and JSON. Keep formatting-only changes separate from behavioral fixes.

## Documentation

For a full description of the Sphinx setup, how to add pages, write docstrings, and how the ReadTheDocs deployment works, see the dedicated **[Documentation](documentation.md)** page.

Quick reference:

```bash
just docs          # build HTML locally
open docs/_build/html/index.html
```

## Release workflow

For the end-to-end release procedure (version bump, tags, PyPI publishing) see:

- **[Release process](release.md)**: step-by-step instructions.
- **[Versioning](versioning.md)**: SemVer rules, commit types, `cz bump`.
- **[CI/CD](ci_cd.md)**: what each GitHub Actions workflow does.

## Submitting Contributions

Before requesting review, make sure your pull request:

- Has a clear description and references any related issues.
- Keeps a single, logical focus.
- Includes passing tests and updated docs for the changes made.

## Issue Tracker

Report bugs, request features, or ask for help on the [Issue Tracker](https://github.com/OpenTabular/DeepTab/issues). Search existing issues before opening a new one.

## License

By contributing, you agree that your contributions are licensed under the project LICENSE.
