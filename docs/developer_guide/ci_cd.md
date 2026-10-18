# CI/CD

DeepTab uses GitHub Actions for continuous integration and delivery. All workflows live in `.github/workflows/`.

## Workflow overview

| Workflow file          | Trigger                      | Purpose                                  |
| ---------------------- | ---------------------------- | ---------------------------------------- |
| `ci.yml`               | Push / PR to `main`          | Lint, type-check, build, test, and cover |
| `docs.yml`             | Push / PR (docs paths), tags | Validate the Sphinx documentation build  |
| `build-check.yml`      | Manual (`workflow_dispatch`) | Dry-run build validation before tagging  |
| `publish-testpypi.yml` | Push of `vX.Y.ZrcN` tag      | Publish release candidate to TestPyPI    |
| `publish-pypi.yml`     | Push of `vX.Y.Z` stable tag  | Publish stable release to PyPI           |

---

## ci.yml (continuous integration)

Runs on every push to `main` and every pull request targeting `main`. In-progress runs for the same branch are cancelled via `concurrency`.

The CI, docs, and build-check workflows also expose `workflow_call` so release workflows can require their checks on the exact tagged commit. Concurrency groups distinguish a release invocation from ordinary CI and docs runs, so those runs do not cancel one another.

| Job                    | Runner / Python            | What it does                                                             | Depends on                                                    |
| ---------------------- | -------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------- |
| `lint`                 | ubuntu / 3.11              | `ruff check .` and `ruff format --check .`                               | -                                                             |
| `typecheck`            | ubuntu / 3.11              | `pyright`                                                                | -                                                             |
| `build`                | ubuntu / 3.11              | `poetry build` and `twine check dist/*`                                  | -                                                             |
| `tests`                | full matrix                | Full core-only suite with verified dependency profile                    | -                                                             |
| `tests-extras`         | ubuntu / 3.12              | Full suite with verified logs/tracking extras                            | -                                                             |
| `tests-tabr`           | ubuntu and windows / 3.12  | Full suite with native retrieval and observability extras                | -                                                             |
| `dependency-endpoints` | three OSes / 3.11 and 3.13 | Independent minimum/newest core, extras, and supported native CPU suites | `build`                                                       |
| `smoke`                | ubuntu / 3.12              | `pytest tests/ -m smoke --tb=short`                                      | `lint`                                                        |
| `coverage`             | ubuntu / 3.12              | Native plus observability suite; line coverage >=90%; upload to Codecov  | `tests`, `tests-extras`, `tests-tabr`, `dependency-endpoints` |

The `lint`, `typecheck`, `build`, and `tests` jobs run in parallel. Matrix jobs use `fail-fast: false`, so one failing matrix cell does not cancel the rest. The `tests` matrix has nine lanes: Python 3.11/3.12/3.13 on three operating systems, as listed in the [Support Matrix](support_matrix.md).

The matrix uses `poetry sync --only main,dev` and
`pytest tests/ --dependency-profile=core --strict-markers`. The extras job uses
the same groups with `--extras "logs tracking"` and `--dependency-profile=extras`.
Both check the installed graph with `pip check`, verify actual optional-package
availability, and collect all categories without marker filtering. Their cache
keys distinguish core from extras; synchronization removes stale packages from
restored environments. The two structured-logging fit tests skip in core-only
and must execute with extras. Native jobs install `--extras "logs tracking tabr"`
and verify `--dependency-profile=tabr`; a missing native dependency fails that
profile. Coverage also installs and verifies the native profile. Core and
observability-only profiles require FAISS/delu to be absent, while exact-index
retrieval tests continue to run without native dependencies.

Local equivalents are `just install-core` / `just test-core` and
`just install-extras` / `just test-extras`, followed by `just test` for coverage.
On Linux or Windows, use `just install-tabr` / `just test-tabr` for native cases.
See [Testing](testing.md) for environment-switching cautions and selectors.
Pushes to integration branches do not trigger automatic CI. Open a pull request
targeting `main` or dispatch CI manually, and retain the run URL and tested SHA.
Reusable release calls remain available; configured lanes alone are not passing
evidence for a release revision. The locked extras lane covers Ubuntu/Python 3.12;
independent extras endpoints add Python 3.11/3.13 on each configured platform.

The build job retains `candidate-dist`. Ordinary CI's 16 endpoint lanes download
that artifact. Release callers wait for the package gate and pass its `dist`
artifact instead, so endpoints validate the wheel that publication will consume.
All lanes independently resolve wheel-metadata requirements without `poetry.lock`
and verify exact wheel identity before training/reload checks and the full CPU
source suite. Endpoint failures block coverage and fail reusable release CI.
The four native endpoint lanes cover Linux and Windows, excluding macOS.
Each lane uploads its reports even after failure, named by OS, architecture,
Python, endpoint, profile and revision. These CPU jobs do not certify CUDA/MPS.
See the {ref}`endpoint procedure <dependency-endpoints>` and support limitations.

```{note}
Compatibility-sensitive jobs (`lint`, `typecheck`, `build`) pin to the **lowest** supported version (3.11) so they catch code that relies on newer syntax or stdlib than we promise to support. The installed-wheel package gate and post-publication RC smoke also use 3.11. The single-version gates (`smoke`, `tests-extras`, `coverage`) retain 3.12, and documentation retains its newer interpreter and tool versions. The `tests` matrix verifies the core profile across the declared combinations; extras compatibility remains scoped to its own jobs.
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

This workflow validates the build; it does not upload its HTML to Read the Docs.
Read the Docs builds and publishes separately through the repository's RTD
integration, using `readthedocs.yaml`. That configuration installs the package
and `docs/requirements_docs.txt`, not Poetry's docs group. Keep the requirements
file compatible with the docs group and its patched dependency versions; a
passing GitHub docs job does not validate RTD's separate installation or prove
that publication succeeded. Versioned tag builds also depend on the RTD
project's build settings.

---

## build-check.yml (manual dry-run)

A manually triggered or reusable workflow. It builds the wheel and source distribution with Poetry, validates both with `twine check`, and installs the wheel with its dependencies into a fresh virtual environment. The installed-wheel check runs outside the checkout with Python's isolated mode. It trains an MLP classifier and regressor with default and explicit preprocessing widths, saves each estimator, reloads it on CPU, and checks predictions, original class labels, and class probabilities. Importing the checkout instead of the installed package fails the check.

For a version tag, the package version must match the tag before building. The workflow uploads the validated distributions as the `dist` artifact only after all package checks succeed. Use it without publishing to validate a release candidate before tagging:

1. Navigate to **Actions → Build & Check (dry run)** in the GitHub UI.
2. Click **Run workflow** and select the release branch.
3. Confirm package validation and the installed-wheel training and reload checks pass before cutting the tag.

---

## Publishing (publish-testpypi.yml / publish-pypi.yml)

Both publish jobs use [OIDC trusted publishing](https://docs.pypi.org/trusted-publishers/), so no `PYPI_TOKEN` secret is required. Before either publisher runs, three required jobs validate the tagged commit:

- `ci` waits for `package` and calls full CI with that validated distribution artifact, including locked and endpoint matrices, lint, type checking, smoke tests and coverage.
- `docs` calls the Sphinx workflow, which builds with warnings treated as errors.
- `package` calls build-check, which validates the tag version, wheel, source distribution, and installed-wheel training and reload behavior.

```{important}
If any required job fails, is cancelled, or is skipped, publication does not run. The CI coverage job enforces at least 90% line coverage across `deeptab` using `--cov-fail-under=90`. A result below that threshold blocks publication even when all tests pass or the Codecov upload succeeds.
```

The publisher downloads the validated `dist` artifact from the same workflow run and publishes those files without rebuilding them. Only the publish job receives write permissions and an OIDC token; validation jobs have read-only repository permissions. After publication, the workflow creates the matching GitHub release. The RC workflow retains its post-publication TestPyPI installation check.

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
