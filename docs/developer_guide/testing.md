# Testing

[![codecov](https://codecov.io/gh/OpenTabular/DeepTab/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenTabular/DeepTab)

DeepTab uses [pytest](https://docs.pytest.org/) and [pytest-cov](https://pytest-cov.readthedocs.io/). Start with the commands below; the later sections cover test design and release validation.

## Running the test suite

| Goal                              | Command                                                       |
| --------------------------------- | ------------------------------------------------------------- |
| Full suite with 90% line coverage | `just test`                                                   |
| One package                       | `just test-run tests/data/ -v`                                |
| One file                          | `just test-run tests/metrics/test_regression_metrics.py -v`   |
| Match test names                  | `just test-run tests/metrics/ -k perfect_predictions -v`      |
| Cross-module workflows            | `just test-run -m integration -v`                             |
| Package-owned tests               | `just test-run -m "not integration" -v`                       |
| Environment contracts             | `just test-run -m environment --dependency-profile=extras -v` |
| Smoke tests                       | `just test-run -m smoke --tb=short`                           |
| Inspect collection                | `just test-run --collect-only -q`                             |
| Live logs, stop on first failure  | `just test-run -x -s`                                         |

Just runs commands through Poetry in the project environment. `test-run` forwards
pytest arguments, preserves quoted marker expressions, and rejects unknown
markers. With no arguments, it selects `tests/`. For one test, append
`::TestClass::test_name` to its filename.

```{important}
Only `just test` enforces coverage by default. `test-run` and the profile test
commands below need explicit coverage flags to enforce the same gate.
```

## Dependency Profiles

Profile tests verify an existing installation and run the full source suite;
they do not install packages. Choose a setup, run `just deps-check`, then run
its test command.

| Profile     | Setup                 | Test               | Installation contract                                                        |
| ----------- | --------------------- | ------------------ | ---------------------------------------------------------------------------- |
| Core        | `just install-core`   | `just test-core`   | No structlog, MLflow, TensorBoard, FAISS, or delu.                           |
| Extras      | `just install-extras` | `just test-extras` | Logging/tracking imports succeed; FAISS and delu are absent.                 |
| Native TabR | `just install-tabr`   | `just test-tabr`   | Retrieval and observability dependencies are present; Linux or Windows only. |

```{warning}
Setup synchronizes the active environment to `main,dev`, removes other groups
(including docs), and registers Git hooks. Extras adds `logs tracking`; native
TabR adds `logs tracking tabr`. Use separate environments to retain profiles.
To restore docs, select Python 3.12 or 3.13 and run `poetry install --with docs`.
```

An incorrect profile fails its installation checks. Unprofiled runs skip those
assertions; CI must supply `--dependency-profile`. Environment tests never
install or download packages. Optional-positive tests skip missing dependencies,
but missing-extra error tests stay active. The extras lane runs both logging
fit tests and writes real TensorBoard hyperparameter events to check NumPy compatibility.

```{note}
Native TabR is blocked on macOS arm64 before importing conflicting libraries.
Exact-index TabR tests still run without native packages; they do not count as
native FAISS validation.
```

(dependency-endpoints)=

## Dependency Endpoints

Endpoints test a candidate wheel in fresh environments without modifying Poetry's
active environment. Install `uv` (CI uses 0.12.5), provide the requested Python
versions, and run each profile separately:

```bash
uv python install 3.11 3.13
poetry build --format wheel
just test-endpoint dist 3.11 minimum core "$TMPDIR/deeptab-min-core"
just test-endpoint dist 3.11 minimum extras "$TMPDIR/deeptab-min-extras"
just test-endpoint dist 3.13 newest core "$TMPDIR/deeptab-new-core"
just test-endpoint dist 3.13 newest extras "$TMPDIR/deeptab-new-extras"
```

| Aspect            | Requirement                                                                                                                                                                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Inputs            | An empty output path per run; exactly one wheel in the input directory, or its filename. Use an interpreter path to select a particular installation. On Windows, replace `$TMPDIR` with a writable directory.                                                            |
| Resolution        | Read wheel `Requires-Dist`, evaluate markers/extras, and resolve all selected runtime roots without development-lock pins. `minimum` uses joint `lowest-direct` resolution, not the oldest transitive versions; `newest` selects highest compatible stable versions.      |
| Prereleases       | Direct runtime dependencies must be stable. Required transitive prereleases, such as MLflow's OpenTelemetry semantic conventions, are permitted and recorded.                                                                                                             |
| Runtime integrity | Retain download URLs, SHA256 hashes, runtime lock, and pip report. Constrain test tooling to runtime versions and compare before/after inventories.                                                                                                                       |
| Validation        | Run `pip check`, native imports, and an isolated installed-wheel check outside the checkout. Verify import/distribution identity and wheel hash before four classifier/regressor fit/save/reload workflows, then the full profile suite with JUnit and 90% line coverage. |
| Evidence          | Retain resolver/interpreter versions and environment path, OS/architecture, CPU scope, revision/patch, source and wheel hashes, logs, inventories, and reports. Failures produce a failed manifest and nonzero exit.                                                      |
| CI artifact       | Ordinary CI uses `candidate-dist`. Release CI and the publisher consume the package gate's same `dist` artifact, with no rebuild after validation.                                                                                                                        |

```{important}
`--provision-only` is diagnostic, not full validation. Release endpoint approval
also needs the hosted artifact/run URL for the same commit. Endpoints use CPU
(including Linux's PyTorch CPU index), not CUDA or MPS validation. See the
{ref}`support limitations <dependency-policy>` for separate device requirements.
```

(dependency-security)=

## Dependency Security

Run the audit in the target environment and retain its unfiltered JSON result
and exit code. Provision `pip-audit` separately without replacing runtime pins;
it is not a core dependency. `just audit` uses no advisory ignores and fails
when findings remain.

```bash
just audit
poetry run pip-audit --format json --output "$TMPDIR/deeptab-audit.json"
just deps-check
```

| Area                | Requirement                                                                                                                                                                                                                                                             |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scopes              | Audit core, requested extras, development tools, and docs separately. Follow installed `Requires-Dist` markers and extras through all dependencies; overlapping scope totals must not be summed. Lightning's `fsspec[http]` makes aiohttp and multidict core-reachable. |
| Evidence            | Preserve raw advisories, including duplicates, plus a separate ID/alias-grouped inventory. Record versions, dependency paths, missing metadata, query times, source/lock hashes, and exit codes.                                                                        |
| Clean result        | An incomplete dependency graph or unavailable audit service is not a pass. Resolution, `pip check`, behavioral tests, and advisory queries answer different questions.                                                                                                  |
| Exceptions          | Require explicit maintainer approval, advisory IDs/aliases, affected versions, exposure, rationale, mitigations, owner, and review date. Unlisted versions/advisories fail closed; no blanket ignores.                                                                  |
| Compatibility fixes | Test constraint changes in clean environments with native import orders, real operations, and optional integrations. Check module origins: layered environments can borrow removed APIs and give false passes.                                                          |

Unresolved findings keep release sign-off open pending remediation, a release
delay, a reviewed support/dependency change, or a specifically approved risk.
See the {ref}`support policy <dependency-policy>`.

```{warning}
Only load model files and checkpoints from trusted sources. This reduces
exposure but does not repair vulnerabilities in PyTorch's loader.
`weights_only=True` is not proof that arbitrary input is safe, and passing
behavioral tests is not security certification. Test loader updates with benign
generated artifacts; do not construct or execute malicious checkpoints.
```

## Reproducible baselines

| Record                 | Include                                                                                                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Source                 | Local and remote commits, worktree changes, and whether imports come from the checkout or an installed wheel.                                                                  |
| Environment            | Interpreter path/version, OS/architecture, device, thread count, exact package versions, and extras.                                                                           |
| Run                    | Command, test node IDs, seeds, skip/xfail reasons, and machine-readable results. Use small configurations and temporary artifacts.                                             |
| Core/extras comparison | Two fresh environments with the same interpreter, mandatory pins, test tooling, and wheel. Add only `logs tracking` to extras; retain inventories and run `pip check` in each. |

For example, capture revision and package information alongside a JUnit report:

```bash
git rev-parse HEAD
git status --short
git ls-remote origin "refs/heads/$(git branch --show-current)"
just deps-list
just test-run -q -ra --junitxml=test-results.xml
```

```{note}
Select CPU explicitly for portable reproductions and disclose device overrides.
A CPU run does not validate CUDA/MPS; an unavailable prerequisite is not a pass.
Do not combine counts from different environments or count preemptive xfails as
executed checks. A working environment with extras is not a core-only baseline.
```

## Organization

Package tests follow ownership: `architectures`, `configs`, `core`,
`data`, `distributions`, `hpo`, `metrics`, `models`, `nn`, and `training`.
Estimator mixins live in `tests/models/mixins/`; neural blocks in `tests/nn/blocks/`.

| Directory under `tests/` | Marker          | Responsibility                                                      |
| ------------------------ | --------------- | ------------------------------------------------------------------- |
| `integration/`           | `integration`   | Cross-module workflows.                                             |
| `contracts/`             | `contract`      | Shared estimator and fresh-process persistence behavior.            |
| `learnability/`          | `learnability`  | Seeded learning, baseline improvement, parameter updates.           |
| `lifecycle/`             | `lifecycle`     | Refit, continuation, pretraining, and loaded-estimator transitions. |
| `metrics/oracles/`       | `metric_oracle` | Metrics checked against independent answers.                        |
| `boundaries/`            | `boundary`      | Input and configuration boundaries.                                 |
| `environments/`          | `environment`   | Pre-provisioned installation and device contracts.                  |

The root `conftest.py` applies these markers recursively. Cross-module workflows
elsewhere must declare `@pytest.mark.integration`; category markers do not imply it.
Package-owned tests can train models, so `-m "not integration"` is not necessarily fast.

| Shared fixture         | Purpose                                                                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `easy_problem`         | Function-scoped, seeded 96-row numerical/categorical problem with a fixed 63/33 split and classification/regression targets. Does not fit models.                  |
| `stable_model_factory` | Function-scoped, small registry-selected estimators; tests choose when to fit. Native TabR skips unsupported Apple Silicon rather than substituting another model. |

Name tests after observable behavior, not issue numbers or past refactors. Keep
related contracts together; split on different ownership/setup, and create new
directories only with substantive tests. Reuse shared fixtures for shared contracts.
Keep test filenames unique across directories. The suite uses pytest's default
import mode without test packages, so duplicate basenames can cause import mismatches.

(behavioral-evidence)=

## Behavioral Evidence

Valid shapes and probabilities do not prove learning; `R2 <= 1` even accepts
`-100`. Score API checks use sklearn's R2 oracle. Learning checks require
improvement over training-only dummy baselines and actual parameter updates.

Stable learning and persistence matrices select every stable registry model
across classification, regression, and normal-family LSS. New stable models
automatically enter; incompatible configurations must fail, not disappear.

| Shared setup | Value                                                                                                                        |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| Data         | 96 rows: noisy three-level numerical signal, nuisance feature, and categorical signal; 63 training/33 validation rows.       |
| Training     | CPU, seed `17`, zero dropout, 40 epochs, batch size 16, learning rate 0.01, no scheduler, temporary checkpoints.             |
| Baselines    | Fit on training targets only. The deliberately easy problem tests the training path, not superiority on real-world datasets. |
| Results      | Record scores, baselines, initial loss, and parameter-update counts in JUnit properties.                                     |

The fixed thresholds below use negative log-likelihood (NLL) for normal LSS:

| Task           | Fixed Acceptance Criterion                                                                                        |
| -------------- | ----------------------------------------------------------------------------------------------------------------- |
| Classification | Accuracy at least 0.85 and at least 0.4 above the most-frequent dummy                                             |
| Regression     | R2 at least 0.75 and at least 0.6 above the mean dummy                                                            |
| Normal LSS     | NLL at least 0.2 below both the constant training-fitted Gaussian and initial full-validation NLL; positive scale |

| Check              | Required behavior                                                                                                                                                                                           |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Parameter updates  | Snapshots at training start must change, including parameters outside the output head. Negative controls reject dummy-level scores, `R2=-100`, and unchanged LSS loss.                                      |
| Estimator contract | Correct fit return value, labels/class mappings, probability ranges/sums, output axes, finite values, positive normal scale, and one-row/partial batches.                                                   |
| Persistence        | Fresh interpreters receive query data and a generated saved artifact, not the original model/datamodule. Float32 reload/row comparisons use `rtol=1e-5, atol=1e-6`; no binary fixtures or network datasets. |
| Boundaries         | One feature, categorical-only data, unseen categories, missing values, Inf rejection, and invalid targets. TabTransformer rejects numerical-only input.                                                     |
| Failed fits        | Invalid-target preflight preserves fitted state; late preprocessing failure invalidates the estimator and prediction raises a not-fitted error.                                                             |
| Embeddings         | Raw/embedded MLP and transformer pooling/normalization with standardization/PLE and reloads. Point-regressor tests do not establish LSS external-embedding fit support.                                     |

Investigate failed learning thresholds rather than lowering margins to hide a failure.

```{note}
SAINT's row attention uses the current batch as context. Its contract checks
permutations within unchanged batch groups and reloads with identical query
batching; it does not claim that changing batch membership or predicting a
single row reproduces a full-batch result. Other stable contracts check
reversed rows and one-row consistency explicitly.
```

| Selection             | Command                                                               |
| --------------------- | --------------------------------------------------------------------- |
| Learning              | `just test-run -m learnability -q`                                    |
| Stable contracts      | `just test-run tests/contracts/test_stable_behavior.py -q`            |
| Boundaries/lifecycle  | `just test-run -m "boundary or lifecycle" -q`                         |
| Native TabR learning  | `just test-run tests/learnability/test_stable_learning.py -k TabR -q` |
| Native TabR contracts | `just test-run tests/contracts/test_stable_behavior.py -k TabR -q`    |

Native TabR selections require Linux or Windows; retain source/runtime identity.
CPU/meta movement checks are not accelerator validation. Keep these bounded
regressions in ordinary PR coverage; subsets do not replace full-suite, profile,
or coverage gates. Budget large multi-seed/configuration/dataset studies separately;
no nightly workflow is implied. See [CI/CD](ci_cd.md) and the
[Support Matrix](support_matrix.md).

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

Fixtures provide setup through test parameters; function scope is the default.
Broader scopes share state, so take care with mutable fitted estimators and
preserve existing class/module setup. Mock at the dependency lookup point,
keeping the interaction under test real. HPO tests, for example, may control
candidates while retaining real preprocessing, fitting, refitting, and prediction.

Use `pytest.approx`, NumPy helpers, or `torch.testing.assert_close` with meaningful
tolerances for floats; exact equality for labels, shapes, metadata, and bit-exact contracts.

## Expected-Failure Policy

| Situation                   | Policy                                                                                                                                                                                                         |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Missing prerequisite        | Skip only when unavailable; use `pytest.importorskip` for optional-positive checks and keep missing-extra errors active.                                                                                       |
| Known behavioral failure    | Use `pytest.mark.xfail(reason=..., strict=True)`, limited to the failing estimator/parameters. Let the body execute; an unexpected pass must fail CI until the annotation is removed.                          |
| Inspect skips/xfails        | Run with `-ra`. Record check name, estimator, partial-check arguments, device, reason, and outcome; recompute inventories for each sklearn version.                                                            |
| Inspect underlying failures | `just test-run tests/models/test_sklearn_contract.py --runxfail -q --tb=short`. Diagnostic only; normally nonzero while gaps remain. Compare case identities/tracebacks before calling a failure a regression. |

```{warning}
Do not use preemptive `pytest.xfail()` or `run=False`: they cannot detect an
unexpected pass. Do not add `--runxfail` to ordinary CI or hide input-safety
defects behind xfails. Historical xfail counts are not acceptance criteria.
```

The sklearn suite checks `MLPClassifier`/`MLPRegressor` on CPU with automatic
CUDA/MPS selection disabled. Two policy regressions verify body execution and
strict unexpected passes in both registries. Run these and the shared
unsupported-weight contracts together:

```bash
just test-run tests/models/test_sklearn_contract.py tests/contracts/ -q -ra
```

Accelerator conclusions require real-device runs; a CPU failure is not an MPS-only limit.

Unsupported sparse input, fitted standard pickle, sampling weights versus weighted
loss, and missing Array API prerequisites are distinct contracts. See the
[sklearn guide](../core_concepts/sklearn_api.md) and
[model operations](../core_concepts/model_operations.md).

## Coverage

| Check             | Command or rule                                                                                                                                                            |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Required gate     | `just test` and CI require **90% line coverage** across `deeptab` with `--cov-fail-under=90`. Lower coverage fails even if all tests pass, and blocks release publication. |
| HTML report       | `just test-run --cov=deeptab --cov-fail-under=90 --cov-report=html`, then open `htmlcov/index.html`.                                                                       |
| Branch diagnostic | `just test-run --cov=deeptab --cov-branch --cov-report=term-missing`; optional, not the release metric.                                                                    |
| Codecov           | CI uploads reports; threshold enforcement does not depend on upload success.                                                                                               |

Coverage measures execution, not assertion quality, numerical correctness, or
untested-device compatibility. Assertions must fail if the defect returns.

## Pre-push checks

Run `just check` for file checks and Pyright; `just install` registers the hooks.
The push hook does not run tests. Run `just test` for model/training changes.
CI runs the full Python/OS matrix on pushes to `main` and PRs targeting `main`,
with manual/reusable invocations also available. Other branches need an explicit
invocation. A configured matrix alone is not evidence that a revision passed it.
See [CI/CD](ci_cd.md) for details.
