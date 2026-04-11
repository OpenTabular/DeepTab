## Contribution Guidelines

Thank you for considering contributing to our Python package! We appreciate your time and effort in helping us improve our project. Please take a moment to review the following guidelines to ensure a smooth and efficient contribution process.

### Code of Conduct

We kindly request all contributors to adhere to our Code of Conduct when participating in this project. It outlines our expectations for respectful and inclusive behavior within the community.

### Setting Up Development Environment

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

### How to Contribute

1. Create a new branch from the `develop` branch for your contributions. Please use descriptive and concise branch names.
2. Make your desired changes or additions to the codebase.
3. Ensure that your code adheres to [PEP8](https://peps.python.org/pep-0008/) coding style guidelines.
4. Write appropriate tests for your changes, ensuring that they pass.
   - `make test`
5. Update the documentation and examples, if necessary.
6. Build the html documentation and verify if it works as expected. We have used Sphinx for documentation, you could build the documents as follows:
   - `cd src/docs`
   - `make clean`
   - `make html`
7. Verify the html documents created under `docs/_build/html` directory. `index.html` file is the main file which contains link to all other files and doctree.

8. Commit your changes following the Conventional Commits specification (see below).
9. Submit a pull request from your branch to the development branch of the original repository.
10. Wait for the maintainers to review your pull request. Address any feedback or comments if required.
11. Once approved, your changes will be merged into the main codebase.

### Release Workflow

This project uses automated semantic versioning and releases. Here's how releases work:

#### Automated Release Process

```
1. Make Changes → 2. Conventional Commit → 3. Merge to Master → 4. Automated Release
```

**Step-by-Step:**

1. **Development Phase**

   - Create feature branch from `develop`
   - Make your changes
   - Commit using conventional commits (e.g., `feat:`, `fix:`)

2. **Merge to Develop**

   - Create PR to `develop` branch
   - After review, merge to `develop`
   - ReadTheDocs dev documentation updates automatically

3. **Merge to Master** (Triggers Release)

   - Merge `develop` to `master`
   - GitHub Actions semantic-release workflow runs automatically

4. **Automated Release (on Master)**
   - ✅ Analyzes conventional commits since last release
   - ✅ Determines version bump (major/minor/patch)
   - ✅ Updates version in `pyproject.toml` and `__version__.py`
   - ✅ Generates/updates `CHANGELOG.md`
   - ✅ Creates git tag (e.g., `v1.7.0`)
   - ✅ Builds package (`poetry build`)
   - ✅ Publishes to PyPI
   - ✅ Creates GitHub Release with notes

#### What Triggers a Release?

| Commit Type                                              | Version Bump  | PyPI Release |
| -------------------------------------------------------- | ------------- | ------------ |
| `feat:`                                                  | Minor (1.x.0) | ✅ Yes       |
| `fix:`                                                   | Patch (1.6.x) | ✅ Yes       |
| `perf:`                                                  | Patch (1.6.x) | ✅ Yes       |
| `feat!:` or `BREAKING CHANGE:`                           | Major (x.0.0) | ✅ Yes       |
| `docs:`, `style:`, `refactor:`, `test:`, `chore:`, `ci:` | None          | ❌ No        |

#### Example Scenarios

**Scenario 1: Documentation Update (No Release)**

```bash
git commit -m "docs: update API reference"
# Merge to master → No version bump, no PyPI release
```

**Scenario 2: Bug Fix (Patch Release)**

```bash
git commit -m "fix: resolve memory leak in dataloader"
# Merge to master → Version 1.6.1 → 1.6.2 → PyPI release
```

**Scenario 3: New Feature (Minor Release)**

```bash
git commit -m "feat(models): add TabNet architecture"
# Merge to master → Version 1.6.1 → 1.7.0 → PyPI release
```

**Scenario 4: Breaking Change (Major Release)**

```bash
git commit -m "feat!: remove Python 3.9 support

BREAKING CHANGE: Python 3.10 is now the minimum required version"
# Merge to master → Version 1.6.1 → 2.0.0 → PyPI release
```

#### Important Notes

- **Only master branch** triggers releases
- **Semantic-release is fully automated** - no manual version bumping needed
- **Never manually edit version numbers** in `pyproject.toml` or `deeptab/__version__.py` - they are automatically updated by semantic-release
- **PyPI token** is configured in GitHub repository secrets
- **Review commits carefully** before merging to master (they determine the version!)

#### Working with Updated Versions

**Q: What happens to `develop` branch after a release?**

After semantic-release completes on `master`, the version files are automatically updated. The `develop` branch syncs automatically:

**Automatic Sync Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│              Release & Sync Process                         │
└─────────────────────────────────────────────────────────────┘

1. Merge develop → master
   │
   ▼
2. Semantic Release runs on master
   │
   ├─→ Version: 1.6.1 → 1.7.0
   ├─→ Update pyproject.toml
   ├─→ Update __version__.py
   ├─→ Update CHANGELOG.md
   └─→ Create tag v1.7.0
   │
   ▼
3. Auto-Sync Workflow triggers
   │
   ├─→ [No Conflicts] ✅
   │   │
   │   ├─→ Merge master → develop automatically
   │   └─→ Develop updated within 60 seconds
   │
   └─→ [Conflicts Detected] ⚠️
       │
       ├─→ Create PR: "chore: sync develop with master"
       ├─→ Notify maintainers
       └─→ Manual merge required (rare)
```

**For Contributors:**

Before starting new work, always pull the latest `develop`:

```bash
# Pull latest develop (already synced automatically)
git checkout develop
git pull origin develop

# Create your feature branch
git checkout -b feature/my-new-feature
```

**Note:** 95% of the time, `develop` syncs automatically. If you see a PR titled "sync develop with master", it means manual conflict resolution is needed (maintainers handle this).

Don't worry if the version seems "old" in your branch - semantic-release calculates the correct new version based on commits when merging to master.

### Submitting Contributions

When submitting your contributions, please ensure the following:

- Include a clear and concise description of the changes made in your pull request.
- Reference any relevant issues or feature requests in the pull request description.
- Make sure your code follows the project's coding style and conventions.
- Include appropriate tests that cover your changes, ensuring they pass successfully.
- Update the documentation if necessary to reflect the changes made.
- Ensure that your pull request has a single, logical focus.

### Issue Tracker

If you encounter any bugs, have feature requests, or need assistance, please visit our [Issue Tracker](https://github.com/OpenTabular/DeepTab/issues). Make sure to search for existing issues before creating a new one.

### License

By contributing to this project, you agree that your contributions will be licensed under the LICENSE of the project.
Please note that the above guidelines are subject to change, and the project maintainers hold the right to reject or request modifications to any contributions. Thank you for your understanding and support in making this project better!
