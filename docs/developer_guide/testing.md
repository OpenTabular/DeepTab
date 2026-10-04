# Testing

[![codecov](https://codecov.io/gh/OpenTabular/DeepTab/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenTabular/DeepTab)

DeepTab uses [pytest](https://docs.pytest.org/) with [pytest-cov](https://pytest-cov.readthedocs.io/) for test coverage. The test suite runs against all supported Python versions and operating systems on every push and pull request.

## Running the test suite

| Goal                                     | Command                                                                                                                          |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Full suite with coverage                 | `just test`                                                                                                                      |
| One component                            | `poetry run pytest tests/data/ -v`                                                                                               |
| A single file                            | `poetry run pytest tests/metrics/test_regression_metrics.py -v`                                                                  |
| A single test                            | `poetry run pytest tests/metrics/test_regression_metrics.py::TestRegressionMetrics::test_perfect_predictions_give_zero_error -v` |
| Cross-module workflows                   | `poetry run pytest tests/ -m integration -v`                                                                                     |
| Package-owned tests                      | `poetry run pytest tests/ -m "not integration" -v`                                                                               |
| Smoke selection                          | `poetry run pytest tests/ -m smoke --tb=short`                                                                                   |
| Inspect collection without running tests | `poetry run pytest tests/ --collect-only -q`                                                                                     |
| Live logs, stop on first fail            | `poetry run pytest tests/ -x -s`                                                                                                 |

`just test` expands to `poetry run pytest --cov=deeptab tests/`.

## Organization

Tests are grouped by their owning package: `architectures`, `configs`, `core`,
`data`, `distributions`, `hpo`, `metrics`, `models`, `nn`, and `training`.
Cross-module workflows live in `tests/integration/`. Estimator mixin tests live
in `tests/models/mixins/`, and neural block tests in `tests/nn/blocks/`.

Name files and classes after observable behavior, not issue numbers or past
refactors. A test file can cover several related source files when they share
one contract. Split it when ownership or setup becomes genuinely different.

The root `tests/conftest.py` automatically marks tests under `integration/`
with `integration`. This is a directory-based selection, not a claim that every
other test is an isolated unit test. Some package-owned checks train estimators
to verify inference, inspection, serialization helpers, or sklearn contracts.
Likewise, integration files can retain related constructor checks beside their
workflow tests to preserve class-level setup.

Keep test filenames unique across directories. The suite uses pytest's default
import mode and does not make each test directory a Python package; duplicate
basenames can cause import mismatches.

## Writing new tests

| Convention    | Guideline                                                                                             |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| Location      | Use `test_*.py` in the owning package directory, or `tests/integration/` for a cross-module workflow. |
| Variations    | Use `@pytest.mark.parametrize` instead of copy-pasting near-identical tests.                          |
| Data          | Use small synthetic datasets (`n=64`, `d=8`); never download external data.                           |
| Trainer noise | Silence Lightning output with `logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)`.       |
| Assertions    | Verify values, invariants, error messages, and state transitions, not just successful execution.      |
| Training      | Use small configurations and explicit seeds; prefer CPU for new portable workflow checks.             |
| Artifacts     | Use `tmp_path` for new checkpoints, logs, and saved models.                                           |
| Fixtures      | Keep setup local unless several compatible consumers genuinely need a shared fixture.                 |

```python
import pytest

@pytest.mark.parametrize("n_layers", [1, 2, 4])
def test_depth(n_layers):
    ...
```

Fixtures provide reusable setup through test parameters. Function scope is the
default; broader scopes share objects and need care when tests mutate fitted
estimators. Existing class-scoped and module-scoped fixtures are preserved.

Use `monkeypatch` or mocks at the dependency lookup point to control failures or
expensive collaborators. Integration tests should leave the interaction they
claim to verify real, even when an unrelated dependency is replaced. For
example, HPO workflow tests can control proposed candidates while performing
real preprocessing, fitting, winner refitting, and prediction.

Use `pytest.approx`, NumPy testing helpers, or `torch.testing.assert_close` for
floating-point comparisons with meaningful tolerances. Use exact equality for
labels, shapes, metadata, and contracts that require bit-exact results.

Skip tests only when prerequisites are unavailable. Record known behavioral
failures with explicit `xfail` reasons; `strict=True` makes an unexpected pass
fail so the exception can be removed. Run `pytest -ra` to inspect both categories.

## Coverage

A coverage report is printed after every `just test` run. For an interactive HTML report:

```bash
poetry run pytest --cov=deeptab --cov-report=html tests/
open htmlcov/index.html
```

To include branch coverage:

```bash
poetry run pytest tests/ --cov=deeptab --cov-branch --cov-report=term-missing
```

Coverage measures execution, not assertion quality, numerical correctness, or
compatibility on untested devices. Review uncovered decisions and pair
regression tests with assertions that would fail if the defect returned.

The full suite also runs in CI across every supported Python and OS combination. See [CI/CD](ci_cd.md) for the matrix.

## Pre-push checks

The pre-commit configuration includes a push-stage hook that runs `pyright` type checking before `git push`. This is installed automatically by `just install`. To run it manually:

```bash
just check
```

The full test suite is not part of the push hook; it runs in CI on every push and pull request. Run `just test` locally before pushing if your change touches model or training code.
