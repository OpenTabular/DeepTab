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

2. Install the prerequisites: `pip install poetry` and `just` (see the [just install guide](https://just.systems/man/en/packages.html), e.g. `brew install just`).

3. Install dependencies and register the pre-commit hooks:

```bash
just install
```

Without `just`, run the same steps directly:

```bash
poetry install
poetry run pre-commit install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push
```

To work on the docs, also install the docs group with `poetry install --with docs`.

## How to contribute

1. Branch off `main` with a short, descriptive name.
2. Make your changes, keeping each pull request to a single logical focus.
3. Add or update tests, and run the full check suite locally before pushing:

```bash
just test     # full suite with coverage
just check    # lint, format, type-check, all pre-commit hooks (what CI runs)
just docs     # build HTML docs (warnings treated as errors)
```

4. Commit using Conventional Commits via `just commit`. If `just check` reformats files, commit those separately with `style: apply ruff formatting`.
5. Open a pull request to `main`, reference any related issues, and address review feedback until approved and merged.

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to enforce code quality automatically. `just install` registers all three hook types so each fires at the right time:

| Stage        | Hook                                                                    |
| ------------ | ----------------------------------------------------------------------- |
| `commit-msg` | Validates the message against Conventional Commits.                     |
| `pre-commit` | `ruff` format and lint, plus file hygiene (whitespace, EOF, conflicts). |
| `pre-push`   | `pyright` type checking (slower, so deferred to push). Also runs in CI. |

```{important}
Run `just check` before opening a PR. It executes the commit and push stage hooks against every file, giving you the same signal CI will see.
```

Individual recipes are available when you want to run one step:

| Command       | Action                                          |
| ------------- | ----------------------------------------------- |
| `just lint`   | Lint and auto-fix with ruff.                    |
| `just format` | Run the ruff formatter.                         |
| `just types`  | Run the pyright type checker.                   |
| `just check`  | Run all hooks across all files (commit + push). |

If pre-commit reports files that _would be reformatted_, run `just format` and commit the result separately with `style: apply ruff formatting`.

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
