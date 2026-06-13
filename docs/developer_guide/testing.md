# Testing

[![codecov](https://codecov.io/gh/OpenTabular/DeepTab/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenTabular/DeepTab)

DeepTab uses [pytest](https://docs.pytest.org/) with [pytest-cov](https://pytest-cov.readthedocs.io/) for test coverage. The test suite runs against all supported Python versions and operating systems on every push and pull request.

## Running the test suite

```bash
just test
```

This expands to:

```bash
poetry run pytest --cov=deeptab tests/
```

To run a single file or a specific test:

```bash
poetry run pytest tests/test_models.py -v
poetry run pytest tests/test_models.py::test_classifier_fit_predict_shape -v
```

To print live log output and stop on the first failure:

```bash
poetry run pytest tests/ -x -s
```

## Writing new tests

- Place tests in `tests/` using the `test_*.py` naming convention.
- Prefer parametrize over copy-paste for variations of the same test:

```python
import pytest

@pytest.mark.parametrize("n_layers", [1, 2, 4])
def test_depth(n_layers):
    ...
```

- Use small synthetic datasets (`n=64`, `d=8`) to keep tests fast. Avoid downloading external data in tests.
- Models rely on PyTorch Lightning internally. To suppress verbose trainer output in tests, set:

```python
import logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
```

## Coverage

A coverage report is printed to the terminal after every `just test` run. To generate an interactive HTML report:

```bash
poetry run pytest --cov=deeptab --cov-report=html tests/
open htmlcov/index.html
```

## CI test matrix

The `ci.yml` workflow runs the full suite on every push to `main` and on every pull request, across:

| Dimension  | Values                                            |
| ---------- | ------------------------------------------------- |
| **Python** | 3.10, 3.11, 3.12, 3.13                            |
| **OS**     | `ubuntu-latest`, `macos-latest`, `windows-latest` |

All 12 combinations run in parallel with `fail-fast: false`, so a failure in one combination does not cancel the others.

## Pre-push checks

The pre-commit configuration includes a push-stage hook that runs `pyright` type checking before `git push`. This is installed automatically by `just install`. To run it manually:

```bash
just check
```

The full test suite is not part of the push hook; it runs in CI on every push and pull request. Run `just test` locally before pushing if your change touches model or training code.
