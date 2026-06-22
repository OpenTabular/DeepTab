# CI/CD

DeepTab uses GitHub Actions for continuous integration and delivery. All workflows live in `.github/workflows/`.

## Workflow overview

| Workflow file          | Trigger                      | Purpose                                  |
| ---------------------- | ---------------------------- | ---------------------------------------- |
| `ci.yml`               | Push / PR to `main`          | Lint, type-check, build, test, and cover |
| `docs.yml`             | Push / PR (docs paths), tags | Build Sphinx docs; deploy to ReadTheDocs |
| `build-check.yml`      | Manual (`workflow_dispatch`) | Dry-run build validation before tagging  |
| `publish-testpypi.yml` | Push of `vX.Y.ZrcN` tag      | Publish release candidate to TestPyPI    |
| `publish-pypi.yml`     | Push of `vX.Y.Z` stable tag  | Publish stable release to PyPI           |

---

## ci.yml (continuous integration)

Runs on every push to `main` and every pull request targeting `main`. In-progress runs for the same branch are cancelled via `concurrency`.

| Job         | Runner / Python | What it does                               | Depends on |
| ----------- | --------------- | ------------------------------------------ | ---------- |
| `lint`      | ubuntu / 3.10   | `ruff check .` and `ruff format --check .` | -          |
| `typecheck` | ubuntu / 3.10   | `pyright`                                  | -          |
| `build`     | ubuntu / 3.10   | `poetry build` and `twine check dist/*`    | -          |
| `tests`     | full matrix     | `pytest tests/ -v`                         | -          |
| `smoke`     | ubuntu / 3.12   | `pytest tests/ -m smoke --tb=short`        | `lint`     |
| `coverage`  | ubuntu / 3.12   | branch coverage uploaded to Codecov        | `tests`    |

The `lint`, `typecheck`, `build`, and `tests` jobs run in parallel with `fail-fast: false`, so one failing matrix cell does not cancel the rest. The `tests` matrix covers every supported Python and OS combination listed in the [Support Matrix](support_matrix.md).

```{note}
The two Python versions are intentional. Compatibility-sensitive jobs (`lint`, `typecheck`, `build`) pin to the **lowest** supported version (3.10) so they catch code that relies on newer syntax or stdlib than we promise to support. The fast single-version gates (`smoke`, `coverage`) use a recent version (3.12), since they check behavior rather than compatibility. Full compatibility is verified by the `tests` matrix.
```

---

## docs.yml (documentation build)

Runs on:

- Every push to `main` (always, regardless of changed paths, so tag pushes rebuild docs).
- Pull requests that touch `docs/**`, `README.md`, `pyproject.toml`, or `deeptab/**`.
- Every version tag (`v*`).

The job installs `pandoc`, installs the `docs` dependency group via Poetry, then runs:

```bash
poetry run sphinx-build -b html docs/ docs/_build/html -W --keep-going
```

On `main` pushes, the built HTML is deployed to ReadTheDocs automatically via the RTD webhook configured in `readthedocs.yaml`.

---

## build-check.yml (manual dry-run)

A `workflow_dispatch`-only workflow. Builds the package with Poetry and validates it with `twine check` without publishing anywhere. Use it to validate a release candidate before tagging:

1. Navigate to **Actions → Build & Check (dry run)** in the GitHub UI.
2. Click **Run workflow** and select the release branch.
3. Confirm `twine check dist/*` passes before cutting the tag.

---

## Publishing (publish-testpypi.yml / publish-pypi.yml)

Both publish jobs use [OIDC trusted publishing](https://docs.pypi.org/trusted-publishers/), so no `PYPI_TOKEN` secret is required. Each builds with `poetry build`, publishes, then creates a GitHub release.

| Workflow               | Tag pattern        | Target                                             | Release type |
| ---------------------- | ------------------ | -------------------------------------------------- | ------------ |
| `publish-testpypi.yml` | `v*.*.*rc*`        | [TestPyPI](https://test.pypi.org/project/deeptab/) | Pre-release  |
| `publish-pypi.yml`     | `v*.*.*` (no `rc`) | [PyPI](https://pypi.org/project/deeptab/)          | Release      |

The `pypi-publish` GitHub Environment must allow the matching tag pattern in its protection rules. See the [Release process](release.md) page for when and how to push these tags.

---

## Adding or modifying a workflow

1. Edit the relevant YAML file in `.github/workflows/`.
2. Keep job names consistent, since they appear in PR status checks and on the Actions tab.
3. Pin third-party actions to a tagged version or full commit SHA (e.g. `actions/checkout@v4`) and keep them current via `just update` (which runs `pre-commit autoupdate`).
