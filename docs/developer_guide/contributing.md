# Contribution Guidelines

Thank you for considering contributing to our Python package! We appreciate your time and effort in helping us improve our project. Please take a moment to review the following guidelines to ensure a smooth and efficient contribution process.

## Code of Conduct

We kindly request all contributors to adhere to our Code of Conduct when participating in this project. It outlines our expectations for respectful and inclusive behavior within the community.

## Setting Up Development Environment

Before you start contributing to the project, you need to set up your development environment. This will allow you to make changes to the codebase, run tests, and build the documentation locally. The project uses `poetry` for dependency management and packaging. Along with that, `ruff` is used for source code formatting and linting.

To set up the development environment for this Python package, follow these steps:

1. Clone the repository to your local machine using the command:

```
git clone https://github.com/OpenTabular/DeepTab

cd DeepTab
```

2. Install tools required for setting up development environment:

- Install `poetry` for dependency management and packaging. You can install it using the following command or refer to the [official documentation](https://python-poetry.org/docs/) for more information.

```
pip install poetry
```

- Install `just` command runner. You can install it using the following command or refer to the [official documentation](https://just.systems/man/en/) for more information.

`justfile` in the source directory is used to define and run common tasks like testing, building, and formatting the codebase.

3. In case you are able to successfully install `poetry` and `just`, you can run the following command to install the dependencies and set up the development environment:

```
# it will install the dependencies as defined in the pyproject.toml file
# it will also install the pre-commit hooks

just install
```

In case you are not able to install `just`, you can follow the below steps to set up the development environment:

```
cd DeepTab

poetry install

poetry run pre-commit install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push
```

If you need to update the documentation, please install the documentation dependencies:

```bash
# Recommended: install via the docs dependency group
poetry install --with docs

# Alternative: install directly
pip install -r docs/requirements_docs.txt
```

**Note:** You can also set up a virtual environment to isolate your development environment.

## How to Contribute

1. Create a new branch from `main` for your contributions. Please use descriptive and concise branch names.
2. Make your desired changes or additions to the codebase.
3. Ensure that your code adheres to [PEP8](https://peps.python.org/pep-0008/) coding style guidelines.
4. Write appropriate tests for your changes and verify they pass:

```bash
just test
```

5. Update the documentation and examples, if necessary.
6. Build the HTML documentation and verify it works as expected:

```bash
just docs
```

Verify the output under `docs/_build/html/`. `index.html` is the entry point. 7. Run the full local check suite before pushing (lint, format, type-check, and all pre-commit hooks):

```bash
just check
```

If `ruff-format` modifies any files, commit those changes before pushing:

```bash
git add -u
git commit -m "style: apply ruff formatting"
```

8. Commit your changes following the Conventional Commits specification (see below):

```bash
just commit
```

9. Submit a pull request from your branch to `main` in the original repository.
10. Wait for the maintainers to review your pull request. Address any feedback or comments if required.
11. Once approved, your changes will be merged into the main codebase.

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to enforce code quality automatically. The hooks run on two stages:

- **commit** — `ruff` format and lint checks, plus general file hygiene hooks
- **push** — `pyright` type checking (slow, so deferred to push)

`just install` registers all three hook types (`commit-msg`, `pre-commit`, `pre-push`) so everything fires at the right time automatically.

> **Important:** Run `just check` before opening a PR. It executes all hooks against every file in the repo (both commit and push stages), giving you the same signal CI will see.

```bash
# Run commit-stage hooks on all files (ruff format, ruff lint, file hygiene)
just lint

# Run ruff formatter
just format

# Run pyright type checker
just typecheck

# Run ALL hooks across ALL files (commit + push stages) — equivalent to what CI checks
just check
```

If pre-commit reports files that _would be reformatted_, run `just format`, stage the changes, and commit before pushing. Formatting-only changes should be committed separately with `style: apply ruff formatting`.

### Type checking (pyright)

Type checking with `pyright` runs automatically on `git push` via the pre-push hook (registered by `just install`). It also runs in CI as the `typecheck` job in `.github/workflows/ci.yml`.

To run it manually at any time:

```bash
just typecheck
```

Fix any reported errors before opening a PR.

## Documentation

For a full description of the Sphinx setup, how to add pages, write docstrings, and how the ReadTheDocs deployment works, see the dedicated **[Documentation](documentation.md)** page.

Quick reference:

```bash
just docs          # build HTML locally
open docs/_build/html/index.html
```

## Release workflow

For the end-to-end release procedure (version bump, tags, PyPI publishing) see:

- **[Release process](release.md)** — step-by-step instructions
- **[Versioning](versioning.md)** — SemVer rules, commit types, `cz bump`
- **[CI/CD](ci_cd.md)** — what each GitHub Actions workflow does

## Submitting Contributions

When submitting your contributions, please ensure the following:

- Include a clear and concise description of the changes made in your pull request.
- Reference any relevant issues or feature requests in the pull request description.
- Make sure your code follows the project's coding style and conventions.
- Include appropriate tests that cover your changes, ensuring they pass successfully.
- Update the documentation if necessary to reflect the changes made.
- Ensure that your pull request has a single, logical focus.

## Issue Tracker

If you encounter any bugs, have feature requests, or need assistance, please visit our [Issue Tracker](https://github.com/OpenTabular/DeepTab/issues). Make sure to search for existing issues before creating a new one.

## License

By contributing to this project, you agree that your contributions will be licensed under the LICENSE of the project.
Please note that the above guidelines are subject to change, and the project maintainers hold the right to reject or request modifications to any contributions. Thank you for your understanding and support in making this project better!
