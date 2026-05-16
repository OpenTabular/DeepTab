# CI/CD

DeepTab uses GitHub Actions for continuous integration and delivery. All workflows live in `.github/workflows/`.

## Workflow overview

| Workflow file          | Trigger                      | Purpose                                  |
| ---------------------- | ---------------------------- | ---------------------------------------- |
| `ci.yml`               | Push / PR → `main`           | Lint, type-check, build, and test        |
| `docs.yml`             | Push / PR (docs paths), tags | Build Sphinx docs; deploy to ReadTheDocs |
| `build-check.yml`      | Manual (`workflow_dispatch`) | Dry-run build validation before tagging  |
| `publish-testpypi.yml` | Push of `vX.Y.ZrcN` tag      | Publish release candidate to TestPyPI    |
| `publish-pypi.yml`     | Push of `vX.Y.Z` stable tag  | Publish stable release to PyPI           |

---

## ci.yml — Continuous integration

Runs on every push to `main` and every pull request targeting `main`. Cancels in-progress runs for the same branch via `concurrency`.

### Jobs

**`lint`** — runs on `ubuntu-latest` / Python 3.10:

```bash
ruff check .          # style and correctness
ruff format --check . # formatting (no changes applied)
```

**`typecheck`** — runs on `ubuntu-latest` / Python 3.10:

```bash
pyright
```

**`build`** — runs on `ubuntu-latest` / Python 3.10:

```bash
poetry build
twine check dist/*
```

**`tests`** — runs across a full matrix:

| Dimension | Values                                            |
| --------- | ------------------------------------------------- |
| OS        | `ubuntu-latest`, `macos-latest`, `windows-latest` |
| Python    | `3.10`, `3.11`, `3.12`, `3.13`                    |

```bash
pytest tests/ -v
```

All jobs are independent; `fail-fast: false` ensures a failure in one matrix cell does not cancel the others.

---

## docs.yml — Documentation build

Runs on:

- Every push to `main` (always, regardless of changed paths — needed so tag pushes rebuild docs).
- Pull requests that touch `docs/**`, `README.md`, `pyproject.toml`, or `deeptab/**`.
- Every version tag (`v*`).

The job installs `pandoc`, installs the `docs` dependency group via Poetry, then runs:

```bash
poetry run sphinx-build -b html docs/ docs/_build/html -W --keep-going
```

On `main` pushes, the built HTML is deployed to ReadTheDocs automatically via the RTD webhook configured in `readthedocs.yaml`.

---

## build-check.yml — Manual dry-run

A `workflow_dispatch`-only workflow. Builds the package with Poetry and validates it with `twine check` without publishing anywhere. Use it to validate a release candidate before tagging:

1. Navigate to **Actions → Build & Check (dry run)** in the GitHub UI.
2. Click **Run workflow** and select the release branch.
3. Confirm `twine check dist/*` passes before cutting the tag.

---

## publish-testpypi.yml — Release candidate publishing

Triggered by any tag matching `v*.*.*rc*`. Uses [OIDC trusted publishing](https://docs.pypi.org/trusted-publishers/) — no `PYPI_TOKEN` secret is required.

Steps:

1. Build the package with `poetry build`.
2. Publish to [TestPyPI](https://test.pypi.org/project/deeptab/).
3. Create a GitHub pre-release via `gh release create`.

The `pypi-publish` GitHub Environment is required; it must have the `v*rc*` tag pattern in its protection rules.

---

## publish-pypi.yml — Stable release publishing

Triggered by any tag matching `v*.*.*` that does **not** contain `rc` (stable only). Also uses OIDC trusted publishing.

Steps:

1. Build the package with `poetry build`.
2. Publish to [PyPI](https://pypi.org/project/deeptab/).
3. Create a GitHub release with auto-generated release notes.

See the [Release process](release.md) page for the full end-to-end procedure including when and how to push these tags.

---

## Adding or modifying a workflow

1. Edit the relevant YAML file in `.github/workflows/`.
2. Use [act](https://github.com/nektos/act) to test locally before pushing:

```bash
act push --job tests
```

3. Keep job names consistent — they are displayed in PR status checks and on the Actions tab.
4. Pin third-party actions to a full commit SHA or a tagged version (e.g. `actions/checkout@v4`) and keep them up to date via `just update` (which runs `pre-commit autoupdate`).
