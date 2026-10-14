# Testing

[![codecov](https://codecov.io/gh/OpenTabular/DeepTab/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenTabular/DeepTab)

DeepTab uses [pytest](https://docs.pytest.org/) with [pytest-cov](https://pytest-cov.readthedocs.io/) for test coverage. CI defines a Python/OS test matrix and runs automatically for pushes to `main` and pull requests targeting `main`. It also supports manual and reusable workflow invocations. A configured matrix is not evidence that a particular revision passed it.

## Running the test suite

| Goal                                        | Command                                                                                                                      |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Full suite with coverage                    | `just test`                                                                                                                  |
| Verify core-only installation and suite     | `just test-core`                                                                                                             |
| Verify logs/tracking installation and suite | `just test-extras`                                                                                                           |
| One component                               | `just test-run tests/data/ -v`                                                                                               |
| A single file                               | `just test-run tests/metrics/test_regression_metrics.py -v`                                                                  |
| A single test                               | `just test-run tests/metrics/test_regression_metrics.py::TestRegressionMetrics::test_perfect_predictions_give_zero_error -v` |
| Cross-module workflows                      | `just test-run -m integration -v`                                                                                            |
| Package-owned tests                         | `just test-run -m "not integration" -v`                                                                                      |
| Environment contracts                       | `just test-run -m environment --dependency-profile=extras -v`                                                                |
| Smoke selection                             | `just test-run -m smoke --tb=short`                                                                                          |
| Inspect collection without running tests    | `just test-run --collect-only -q`                                                                                            |
| Live logs, stop on first fail               | `just test-run -x -s`                                                                                                        |

`just test` expands to `poetry run pytest --cov=deeptab --cov-fail-under=90 tests/`.
`just test-run` forwards pytest arguments, preserving quoted marker expressions,
and rejects unknown markers. With no arguments it uses the configured `tests/`
test path. `test-run`, `test-core`, and `test-extras` do not collect or enforce
coverage unless coverage flags are supplied; use `just test` for the coverage
gate. Just is the command interface; Poetry runs each command in the project
environment. CI uses the corresponding Poetry commands directly.

## Dependency Profiles

The profile test commands verify an existing installation; they do not install
packages. Prepare the desired environment first:

```bash
just install-core
just deps-check
just test-core
```

```bash
just install-extras
just deps-check
just test-extras
just test
```

`install-core` uses `poetry sync --only main,dev`. `install-extras` uses the same
groups with `--extras "logs tracking"`. Both register the usual Git hooks.
The core profile requires `structlog`, `mlflow`, and `tensorboard` to be absent;
the extras profile requires their real import guards to succeed. An incorrect
installation fails rather than silently skipping the dependency checks.
Both profiles run the full source suite. Optional-positive tests skip only
when their dependency is absent; missing-extra error tests still execute.
The two structured-logging fit tests execute in the extras lane.

```{warning}
These setup commands synchronize the active project environment. They remove
packages outside `main,dev`, including documentation dependencies; `install-core`
also removes logging/tracking extras. Use separate environments when retaining
both profiles or comparing them. `just install` restores the ordinary developer
setup, including the configured default groups, before a docs build.
```

(dependency-security)=

## Dependency Security

Run `just audit` in the environment being evaluated. It calls `pip-audit`
without advisory ignores and returns nonzero when findings remain. The audit
tool must be provisioned separately; it is not a core runtime dependency.
Keep audit-tool installation from silently replacing the target runtime pins.
Retain a machine-readable unfiltered result as well as the command's exit code:

```bash
just audit
poetry run pip-audit --format json --output "$TMPDIR/deeptab-audit.json"
just deps-check
```

Audit core runtime, requested logging/tracking extras, development tools and
documentation dependencies separately. Traverse installed `Requires-Dist`
metadata with environment markers and requested extras, then audit each exact
resolved closure. Lightning requests `fsspec[http]`, so aiohttp and multidict
are core-reachable, not merely development dependencies. A package may belong
to several scopes; do not sum overlapping scope totals.

Keep the original advisory records, including duplicates, and derive a separate
ID/alias-grouped triage inventory. Record package versions, paths from declared
roots, missing metadata, query times, source/lock hashes and actual exit codes.
An incomplete graph or unavailable audit service is not a clean audit.
`pip check`, import/fit/reload checks and advisory queries answer different
questions; a successful dependency resolver alone does not establish compatibility.

The release audit has no unconditional accepted-risk list. An exception needs
explicit maintainer approval, matching advisory IDs/aliases and affected package
versions, exposure, rationale, mitigations, an owner and a review date. A new
version or advisory outside that record must fail closed. Do not implement a
blanket ignore to obtain a green result. If compatible remediation is unavailable,
keep release sign-off open pending a delay, reviewed support/dependency change
or bounded approved risk decision.

```{warning}
Only load model files and checkpoints from trusted sources. This reduces
exposure but does not repair vulnerabilities in PyTorch's loader.
`weights_only=True` is not proof that arbitrary input is safe, and passing
behavioral tests is not security certification. Test loader updates with benign
generated artifacts; do not construct or execute malicious checkpoints.
```

Test out-of-range fixes in separate environments before changing constraints.
Exercise native import orders, actual operations and optional integrations,
not only `pip check`. An environment layered over the old site-packages can
borrow removed APIs and produce a false compatibility pass; inspect module
origins and repeat relevant failures in a clean environment. See the
{ref}`support policy <dependency-policy>`.

## Reproducible baselines

Record the local and remote commit, worktree changes, interpreter executable,
Python version, OS/architecture, device, thread count, installed package versions,
and selected extras before comparing results. Retain the command, test node IDs,
skip/xfail reasons, and machine-readable results alongside each run:

```bash
git rev-parse HEAD
git status --short
git ls-remote origin "refs/heads/$(git branch --show-current)"
just deps-list
just test-run -q -ra --junitxml=test-results.xml
```

For ordinary development, install the test tooling and the extras exercised by
the tests:

```bash
just install-extras
just deps-check
just test-extras
```

For a core-versus-extras comparison, provision two fresh isolated environments
with the same interpreter, mandatory dependency pins, and test tooling. Install
the same built wheel in both, adding `logs` and `tracking` only to the extras
environment. Retain the resolved package inventories and run `pip check` in
each. A working environment that already contains extras is not a core-only
baseline. Source tests and an installed-wheel check outside the checkout prove
different things; identify which import path each run exercises.

Select CPU explicitly for portable training reproductions, use fixed seeds and
small configurations, and isolate generated artifacts. If existing tests
auto-select an accelerator, control that selection and disclose it in the
result. A CPU-only run does not validate CUDA or MPS. Likewise, an unavailable
FAISS runtime, optional package, or Array API prerequisite is pending evidence,
not a passing test. Do not combine counts from unlike environments or treat
preemptive `pytest.xfail()` calls as executed checks. See [CI/CD](ci_cd.md) and
the [Support Matrix](support_matrix.md) for workflow and platform scope.

## Organization

Tests are grouped by their owning package: `architectures`, `configs`, `core`,
`data`, `distributions`, `hpo`, `metrics`, `models`, `nn`, and `training`.
Cross-module workflows live in `tests/integration/`. Estimator mixin tests live
in `tests/models/mixins/`, and neural block tests in `tests/nn/blocks/`.

Additional shared behavior is organized by category. Add a directory with its
first substantive test, not an empty placeholder. The categories below now
contain installation checks, shared model/persistence contracts, learning checks,
lifecycle transitions, metric oracles and input/configuration boundaries.

| Category Directory | Marker          | Responsibility                                                     |
| ------------------ | --------------- | ------------------------------------------------------------------ |
| `contracts/`       | `contract`      | Shared estimator and fresh-process persistence behavior            |
| `learnability/`    | `learnability`  | Seeded learning, baseline improvement, parameter updates           |
| `lifecycle/`       | `lifecycle`     | Refit, continuation, pretraining, and loaded-estimator transitions |
| `metrics/oracles/` | `metric_oracle` | Training metrics checked against independent answers               |
| `boundaries/`      | `boundary`      | Input and configuration boundaries                                 |
| `environments/`    | `environment`   | Pre-provisioned dependencies, installation, and device contracts   |

For example, use `just test-run -m lifecycle` for lifecycle transitions.
Environment tests never install or download packages during pytest. Their
dependency checks require `--dependency-profile=core` or `extras`; ordinary
unprofiled runs skip only those three installation assertions. CI supplies
the profile, so they must execute there.

Name files and classes after observable behavior, not issue numbers or past
refactors. A test file can cover several related source files when they share
one contract. Split it when ownership or setup becomes genuinely different.

The root `tests/conftest.py` automatically applies the category marker to files
under each category directory, including nested files. Tests under `integration/`
receive `integration`. Real cross-module workflows elsewhere must also declare
`@pytest.mark.integration`; a category marker does not imply it automatically.
This is a selection policy, not a claim that every
other test is an isolated unit test. Some package-owned checks train estimators
to verify inference, inspection, serialization helpers, or sklearn contracts.
Likewise, integration files can retain related constructor checks beside their
workflow tests to preserve class-level setup.

Keep test filenames unique across directories. The suite uses pytest's default
import mode and does not make each test directory a Python package; duplicate
basenames can cause import mismatches.

(behavioral-evidence)=

## Behavioral Evidence

Shape and finite-value checks verify API behavior, not learning. Likewise,
`R2 <= 1` accepts a score of `-100`, and a probability row summing to one does
not show that the labels were learned. Score API checks compare against sklearn's
R2 oracle. Learning checks instead require improvement over a training-only
dummy baseline and actual parameter updates.

The shared tiny-model factory in [tests/conftest.py](https://github.com/OpenTabular/DeepTab/blob/main/tests/conftest.py)
uses supported configuration fields, explicit CPU training, seed `17`, zero
dropout and temporary checkpoints. Stable learning and persistence matrices
select all architectures marked stable in the registry, across classification,
regression and normal-family LSS. A newly declared stable model automatically
enters those matrices; an unknown or incompatible configuration must fail,
not silently disappear from collection.

The learning fixture has 96 rows, a three-level noisy numerical signal, a
nuisance feature and a categorical version of the signal. The split is an
explicit 63 training/33 validation rows. This deliberately easy problem probes
whether the complete training path can learn, not generalization superiority
on real-world datasets. Baselines use training targets only.

| Task           | Fixed Acceptance Criterion                                                                                        |
| -------------- | ----------------------------------------------------------------------------------------------------------------- |
| Classification | Accuracy at least 0.85 and at least 0.4 above the most-frequent dummy                                             |
| Regression     | R2 at least 0.75 and at least 0.6 above the mean dummy                                                            |
| Normal LSS     | NLL at least 0.2 below both the constant training-fitted Gaussian and initial full-validation NLL; positive scale |

The budget is 40 epochs, batch size 16 and learning rate 0.01 without a scheduler.
Parameter snapshots at training start must change, including parameters outside
the output head. Negative controls reject dummy-level accuracy/R2, `R2=-100`
and unchanged LSS loss. Record scores, baselines, initial loss and update counts
in JUnit properties. Do not reduce margins to disguise a failure; investigate
the architecture, optimizer, preprocessing or output semantics first.

Stable contracts also check fit return values, labels/class mappings, probability
ranges and sums, task-specific output axes, finite values, normal scale,
one-row/partial batches and generated-artifact reloads in fresh interpreters.
The child process receives query data and the saved artifact, not the original
model or datamodule. Float32 reload/row comparisons use `rtol=1e-5, atol=1e-6`.
No binary artifact fixture or network dataset is required.

```{note}
SAINT's row attention uses the current batch as context. Its contract checks
permutations within unchanged batch groups and reloads with identical query
batching; it does not claim that changing batch membership or predicting a
single row reproduces a full-batch result. Other stable contracts check
reversed rows and one-row consistency explicitly.
```

Representative boundary tests cover one feature, categorical-only data, unseen
categories, missing values, Inf rejection and invalid targets. TabTransformer
requires categorical input and explicitly rejects a numerical-only feature.
Invalid targets preserve fitted state during preflight; a late preprocessing
failure invalidates the estimator and subsequent prediction must raise a
not-fitted error. These are different safe outcomes, not interchangeable passes.
Embedding tests combine raw/embedded MLP and transformer pooling/normalization
with standardization/PLE and reloads. LSS external-embedding fit support is not
implied by those point-regressor tests.

```bash
just test-run -m learnability -q
just test-run tests/contracts/test_stable_behavior.py -q
just test-run -m "boundary or lifecycle" -q
just test-run tests/learnability/test_stable_learning.py \
    tests/contracts/test_stable_behavior.py -k TabR -q
```

The last command requires a supported native FAISS platform. Darwin arm64 skips
are specific platform limits, not native passing evidence. Run the same cases
on Linux or Windows and retain the exact runtime and source identity. CPU/meta
movement tests do not validate accelerators.

Keep these bounded known-answer, lifecycle, boundary and learning regressions
in ordinary PR coverage. Use category or representative architecture selectors
for quick local feedback; a subset does not replace the full suite, extras/profile
checks and unchanged coverage gate before release. Current CI still runs the
full source suite. Larger multi-seed experiments, exhaustive family/configuration
combinations and cross-dataset statistical claims belong in separately budgeted
nightly or post-release studies. No nightly workflow is implied by these selectors.

## Writing new tests

| Convention    | Guideline                                                                                                                                                  |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Location      | Keep localized regressions in the owning package. Put shared failure classes in the matching category, or uncategorized workflows in `tests/integration/`. |
| Variations    | Use `@pytest.mark.parametrize` instead of copy-pasting near-identical tests.                                                                               |
| Data          | Use small synthetic datasets (`n=64`, `d=8`); never download external data.                                                                                |
| Trainer noise | Silence Lightning output with `logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)`.                                                            |
| Assertions    | Verify values, invariants, error messages, and state transitions, not just successful execution.                                                           |
| Training      | Use small configurations and explicit seeds; prefer CPU for new portable workflow checks.                                                                  |
| Artifacts     | Use `tmp_path` for new checkpoints, logs, and saved models.                                                                                                |
| Fixtures      | Keep setup local unless several compatible consumers genuinely need a shared fixture.                                                                      |

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

Skip tests only when prerequisites are unavailable; use `pytest.importorskip`
for optional-positive checks and keep missing-extra error tests active. Record known behavioral
failures with explicit `xfail` reasons; `strict=True` makes an unexpected pass
fail so the exception can be removed. Run `pytest -ra` to inspect both categories.

## Expected-Failure Policy

An expected-failure marker must let the test body run. Do not short-circuit a
known check with `pytest.xfail()` before calling it, or use `run=False`: neither
can reveal an unexpected pass. Use `pytest.mark.xfail(reason=..., strict=True)`
and narrow the annotation to the failing estimator or parameterization when
necessary. A newly passing check must fail CI until its stale annotation is
removed; changing the historical number of xfails is not an acceptance criterion.

The sklearn suite checks `MLPClassifier` and `MLPRegressor` on CPU, with automatic
CUDA/MPS selection disabled within each estimator check. Its two marker-policy
regressions verify that both registries execute failing bodies and require
strict unexpected passes. Shared unsupported-weight checks live in
`tests/contracts/` and are selected by the `contract` marker.

```bash
just test-run tests/models/test_sklearn_contract.py tests/contracts/ -q -ra
just test-run -m contract -q
```

To inspect the actual failures behind annotations:

```bash
just test-run tests/models/test_sklearn_contract.py --runxfail -q --tb=short
```

`--runxfail` is a diagnostic and normally exits nonzero while gaps remain. Do not
add it to ordinary CI or interpret its failure as a new regression without
comparing the case identities and tracebacks. Record the check name, estimator,
partial-check keyword arguments, device, reason, and actual outcome. Sklearn
versions can collect different checks, so recompute the inventory in each tested
environment. Accelerator results need separate real-device runs; a CPU failure
must not be described as an MPS-only limitation.

Input-safety defects are not intentional unsupported features. Keep their
validation/state fixes visible, and do not add xfails to hide a release-blocking
regression. Unsupported sparse input and fitted standard pickle, classifying
sampling weights as weighted loss, and missing Array API prerequisites are
different contracts; see the [sklearn guide](../core_concepts/sklearn_api.md)
and [model operations](../core_concepts/model_operations.md).

## Coverage

`just test` and the CI coverage job require at least **90% line coverage** across `deeptab`. Both pass `--cov-fail-under=90`, so a lower result fails the command even when every test passes. CI publishes its coverage report to Codecov, but the local threshold determines success independently of that upload. A failed CI coverage job also blocks release publication.

For an interactive HTML report with the same threshold:

```bash
just test-run --cov=deeptab --cov-fail-under=90 --cov-report=html
open htmlcov/index.html
```

Branch coverage is an optional diagnostic, not the release gate's metric:

```bash
just test-run --cov=deeptab --cov-branch --cov-report=term-missing
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

The full test suite is not part of the push hook. Automatic CI currently covers pushes to `main` and pull requests targeting `main`; other branches need an explicit workflow invocation. Run `just test` locally before pushing if your change touches model or training code.
