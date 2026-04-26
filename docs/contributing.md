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

poetry run pre-commit install
```

If you need to update the documentation, please install the dependencies requried for documentation:

```
pip install -r docs/requirements_docs.txt
```

**Note:** You can also set up a virtual environment to isolate your development environment.

## How to Contribute

1. Create a new branch from `main` for your contributions. Please use descriptive and concise branch names.
2. Make your desired changes or additions to the codebase.
3. Ensure that your code adheres to [PEP8](https://peps.python.org/pep-0008/) coding style guidelines.
4. Write appropriate tests for your changes, ensuring that they pass.
   - `make test`
5. Update the documentation and examples, if necessary.
6. Build the html documentation and verify if it works as expected. We have used Sphinx for documentation, you could build the documents as follows:
   - `cd docs`
   - `make clean`
   - `make html`
7. Verify the html documents created under `docs/_build/html` directory. `index.html` file is the main file which contains link to all other files and doctree.

8. Commit your changes following the Conventional Commits specification (see below).
9. Submit a pull request from your branch to `main` in the original repository.
10. Wait for the maintainers to review your pull request. Address any feedback or comments if required.
11. Once approved, your changes will be merged into the main codebase.

## Docs Workflow

Documentation is built with [Sphinx](https://www.sphinx-doc.org/) and hosted on [Read the Docs](https://readthedocs.org/).

The docs CI is defined in `.github/workflows/docs.yml` and is separate from the main CI workflow.

### How docs are published

| Trigger                                                               | CI (`docs.yml`)                                       | Read the Docs                                                 |
| --------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------- |
| PR touching `docs/**`, `deeptab/**`, `README.md`, or `pyproject.toml` | Sphinx build check — PR is blocked if docs are broken | No publish                                                    |
| Merge to `main`                                                       | Sphinx build check                                    | Publishes **latest** (dev) version                            |
| Stable tag pushed (`vX.Y.Z`)                                          | Sphinx build check from that exact tagged commit      | Publishes **versioned** snapshot and updates **stable** alias |
| RC tag pushed (`vX.Y.Zrc1`)                                           | Sphinx build check from that exact tagged commit      | Publishes versioned pre-release snapshot                      |

> **Note:** The docs CI `push` trigger has **no `paths:` filter** — tag pushes always run the full docs build regardless of which files changed in the tagged commit. The `paths:` filter only applies to PRs to keep checks fast.

> **Note:** Versioned and stable docs require **"Build tags"** to be enabled in the Read the Docs project settings under _Admin → Advanced settings_. RTD automatically sets the `stable` alias to the highest non-pre-release tag.

### Tag → versioned docs flow

```
git tag -a v1.7.0 -m "Release v1.7.0"
git push origin v1.7.0
         ↓
docs.yml triggers on that exact tagged commit
         ↓
Sphinx build succeeds (or blocks if broken)
         ↓
RTD webhook fires → builds docs from v1.7.0 source
         ↓
RTD publishes /en/v1.7.0/ (versioned)
         ↓
RTD updates /en/stable/ → points to v1.7.0
```

RC tags (`vX.Y.Zrc1`) follow the same flow but RTD does **not** update the `stable` alias for pre-release tags.

### Building docs locally

```bash
# Install system dependency (macOS)
brew install pandoc
# or on Ubuntu
sudo apt-get install pandoc

# Install doc dependencies
pip install -r docs/requirements_docs.txt

# Build HTML docs (warnings treated as errors)
sphinx-build -b html docs/ docs/_build/html -W --keep-going

# Open in browser
open docs/_build/html/index.html
```

### Version resolution

The docs version is read at build time from the installed package metadata via `importlib.metadata.version("deeptab")`, which reflects the version in `pyproject.toml`. No separate version file is maintained.

## Release Workflow

This project uses conventional commits and intentional, maintainer-controlled releases.

### Release Process Overview

```
1. Make Changes → 2. Conventional Commit → 3. Merge to Main → 4. CI runs
                                                                     ↓
                                          (no automatic release — main is not a release trigger)

5. Maintainer opens Release PR → version bump + CHANGELOG update → merge to main
6. Maintainer creates Git tag → PyPI publish triggered automatically
```

**Step-by-Step:**

1. **Development Phase**
   - Create feature branch from `main`
   - Make your changes
   - Commit using conventional commits (e.g., `feat:`, `fix:`)

2. **Merge to Main** (CI only — no release)
   - Create PR to `main`
   - After review, merge to `main`
   - GitHub Actions runs tests
   - **No version bump, no tag, no PyPI publish happens automatically**

3. **Maintainer Release PR** (periodic, intentional)
   - Maintainer creates a `release/vX.Y.Z` branch
   - Runs `cz bump` to update `pyproject.toml` and `CHANGELOG.md`
   - Opens PR to `main`, merges after review

4. **Maintainer Creates Git Tag**
   - After the release PR is merged:
     ```bash
     git checkout main && git pull
     git tag -a vX.Y.Z -m "Release vX.Y.Z"
     git push origin vX.Y.Z
     ```
   - This tag push triggers `publish-pypi.yml` → builds and publishes to PyPI + creates GitHub Release
   - For RC tags (`vX.Y.Zrc1`), push triggers `publish-testpypi.yml` → publishes to TestPyPI instead

### What Triggers a Release?

| Event                         | Result                                |
| ----------------------------- | ------------------------------------- |
| Push to `main`                | CI tests only                         |
| Maintainer pushes `v*` tag    | PyPI publish + GitHub Release         |
| Maintainer pushes `v*rc*` tag | PyPI pre-release + GitHub pre-release |

### Commit Types and Their Effect on Version

Commit messages determine the version bump chosen by the maintainer when running `cz bump`:

| Commit Type                                              | Version Bump      |
| -------------------------------------------------------- | ----------------- |
| `feat:`                                                  | Minor (1.x.0)     |
| `fix:`, `perf:`                                          | Patch (1.6.x)     |
| `feat!:` / `BREAKING CHANGE:`                            | Major (x.0.0)     |
| `docs:`, `style:`, `refactor:`, `test:`, `chore:`, `ci:` | No release needed |

### Example Scenarios

**Scenario 1: Documentation Update (No Release)**

```bash
git commit -m "docs: update API reference"
# Merge to main → CI only, no release
```

**Scenario 2: Bug Fix (Patch Release)**

```bash
git commit -m "fix: resolve memory leak in dataloader"
# Merged to main → later, maintainer runs cz bump → creates v1.6.2 tag → PyPI release
```

**Scenario 3: New Feature (Minor Release)**

```bash
git commit -m "feat(models): add TabNet architecture"
# Merged to main → later, maintainer runs cz bump → creates v1.7.0 tag → PyPI release
```

**Scenario 4: RC for risky feature**

```bash
# Maintainer tags manually:
git tag -a v1.7.0rc1 -m "Release candidate v1.7.0rc1"
git push origin v1.7.0rc1
# → PyPI pre-release, GitHub pre-release
```

### Important Notes

- **Merging to `main` never triggers a PyPI release**
- **Only a manually pushed `v*` tag triggers publishing**
- **Never manually edit the version number** in `pyproject.toml` — use `cz bump` on a release branch
- **PyPI publishing** uses OIDC Trusted Publishing — no API tokens are stored in GitHub secrets

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
