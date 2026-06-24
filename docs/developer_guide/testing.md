# Testing

[![codecov](https://codecov.io/gh/OpenTabular/DeepTab/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenTabular/DeepTab)

DeepTab uses [pytest](https://docs.pytest.org/) with [pytest-cov](https://pytest-cov.readthedocs.io/) for test coverage. The test suite runs against all supported Python versions and operating systems on every push and pull request.

## Running the test suite

| Goal                          | Command                                                |
| ----------------------------- | ------------------------------------------------------ |
| Full suite with coverage      | `just test`                                            |
| A single file                 | `poetry run pytest tests/test_models.py -v`            |
| A single test                 | `poetry run pytest tests/test_models.py::test_name -v` |
| Live logs, stop on first fail | `poetry run pytest tests/ -x -s`                       |

`just test` expands to `poetry run pytest --cov=deeptab tests/`.

## Writing new tests

| Convention    | Guideline                                                                                       |
| ------------- | ----------------------------------------------------------------------------------------------- |
| Location      | Place tests in `tests/` using the `test_*.py` naming convention.                                |
| Variations    | Use `@pytest.mark.parametrize` instead of copy-pasting near-identical tests.                    |
| Data          | Use small synthetic datasets (`n=64`, `d=8`); never download external data.                     |
| Trainer noise | Silence Lightning output with `logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)`. |

```python
import pytest

@pytest.mark.parametrize("n_layers", [1, 2, 4])
def test_depth(n_layers):
    ...
```

## Coverage

A coverage report is printed after every `just test` run. For an interactive HTML report:

```bash
poetry run pytest --cov=deeptab --cov-report=html tests/
open htmlcov/index.html
```

The full suite also runs in CI across every supported Python and OS combination. See [CI/CD](ci_cd.md) for the matrix.

## Pre-push checks

The pre-commit configuration includes a push-stage hook that runs `pyright` type checking before `git push`. This is installed automatically by `just install`. To run it manually:

```bash
just check
```

The full test suite is not part of the push hook; it runs in CI on every push and pull request. Run `just test` locally before pushing if your change touches model or training code.
