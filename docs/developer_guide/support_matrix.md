# Support Matrix

This page lists the officially supported versions of Python and core dependencies. These combinations are included in the configured CI matrix. Automatic CI covers pushes to `main` and pull requests targeting `main`; other revisions need a manual or reusable workflow invocation. Confirm results for the exact revision before claiming release validation. Combinations not listed here may work but are not guaranteed.

---

## Python and Operating Systems

| Category | Supported                                                      | Notes                                                     |
| -------- | -------------------------------------------------------------- | --------------------------------------------------------- |
| Python   | 3.11, 3.12, 3.13                                               | 3.14+ excluded pending dependency and release validation. |
| OS       | Linux, macOS (Apple Silicon), Windows (all `*-latest` runners) | CI's macOS runners are Apple Silicon (arm64).             |

```{warning}
**Intel Mac (macOS x86_64) is not supported.** Upstream PyTorch no longer publishes
x86_64 macOS wheels for `torch >= 2.3` or Python 3.13. **Apple Silicon PyTorch
wheels are available for Python 3.13**, including `torch==2.9.1`. Intel Macs are
outside the CI matrix. The older Intel PyTorch 2.2.2 build does not satisfy the
current dependency range and is not a supported workaround.
```

```{warning}
**TabR is not validated on Apple Silicon.** The PyPI `faiss-cpu` build can
segfault on macOS arm64, so CI skips TabR classifier, regressor, and LSS tests
on that platform. Use Linux or Windows for TabR. An alternative FAISS build is
not covered by this CI matrix.
```

---

(dependency-policy)=

## Core Dependencies and Policy

The authoritative version constraints live in [`pyproject.toml`](https://github.com/OpenTabular/deeptab/blob/main/pyproject.toml) under `[tool.poetry.dependencies]`, which is updated on every release. Treat that file as the source of truth; the list below names the core packages and the policy that governs their bounds.

- **PyTorch, Lightning, NumPy, pandas, scikit-learn, torchmetrics, scipy, PreTab** are the pinned core dependencies. NumPy 1.x is **not** supported.
- **delu, faiss-cpu** are mandatory but model-specific dependencies (used only by TabR), pinned to a tested range rather than a broader policy window; they are candidates for optional extras in a future release.

DeepTab targets a tested support window, informed by [SPEC 0](https://scientific-python.org/specs/spec-0000/) used by the broader scientific Python ecosystem:

- **Python**: maintain a three-minor window, currently 3.11-3.13. A new upstream release is not admitted automatically: compatible dependencies and release checks must pass first. Review older versions near upstream end of life, no earlier than 42 months after release, and announce removals in the release notes.
- **NumPy / pandas / scikit-learn**: support the two most recent minor releases of each dependency at the time of a new DeepTab minor release.
- **PyTorch**: support the current stable release and the one prior. The upper bound in `pyproject.toml` is updated when a new PyTorch stable is released and CI passes.

This is a compatibility target, not a claim that every current stable release
is already admitted or free of known advisories. The current Torch `<2.10.0`
ceiling remains a compatibility exception: an isolated Apple Silicon probe
with Torch `2.13.0` and FAISS `1.15.1` aborted with a duplicate OpenMP-runtime
error in both import orders. That probe does not validate a patched stack on
other platforms, and an unsafe OpenMP override is not an approved workaround.

The setuptools `<80` ceiling also remains under review. TensorBoard `2.20.0`
imports `pkg_resources`; a clean environment with setuptools `83.0.0` passes
`pip check` but fails those server imports because that module is removed.
Raising the bound without updating and verifying the optional integration
would trade a reported security finding for a compatibility regression.

A newer TensorBoard `2.21.0` passed those clean server imports with setuptools
`83.0.0`, so a coordinated minimum-version update is under review. The patched
Torch candidate also passed a small non-retrieval CPU fit without importing
FAISS. Those component probes do not validate a combined stack or authorize
changed constraints; full core/extras and supported native-platform checks
are still required before adopting a revised installation policy.

These reproduced limits are not accepted-risk approvals or security sign-off.
The {ref}`dependency audit policy <dependency-security>` requires an
unfiltered audit and a verified remediation or explicit reviewed decision.
Trusted-checkpoint guidance does not make an affected loader secure. Preserve
the original findings and keep release sign-off open while unapproved findings
remain.

When a version is dropped, it is announced in the release notes and the `pyproject.toml` lower bound is bumped in the same minor release.

## Dependency Endpoint Validation

The nine locked core lanes cover Python 3.11-3.13 on three operating systems.
An additional 12 lanes resolve minimum-direct dependencies on Python 3.11 and
newest stable compatible dependencies on Python 3.13, separately for core and
`logs`/`tracking` installations on each platform. Resolution uses the built wheel's
metadata, not the development lock. The minimum set is jointly resolved; it is
not a promise that every direct lower bound can be installed simultaneously.

Both endpoints require native imports, `pip check`, the full CPU source suite,
90% line coverage, and installed-wheel fit/save/reload checks outside the checkout.
The wheel version, import origin and SHA256 must match the requested artifact.
See the {ref}`reproduction and evidence requirements <dependency-endpoints>`.

Local probes reproduced pandas 2.0.3/NumPy 2.0.0 binary incompatibility and Torch
2.2.2/NumPy 2.0.0 conversion failure. The pandas and Torch floors are now 2.2.2
and 2.4.0 respectively. Torch 2.4.0 conversion passed on the local Apple Silicon
Python 3.10 runtime; that component result is not a full endpoint pass.

Historical Python 3.10 captures retain the SciPy 1.15.0-1.15.3 PROPACK loader
failures. Raising Python alone did not fix the issue: SciPy 1.15.0 also failed
on local Apple Silicon Python 3.11. The SciPy minimum is now 1.16.0; clean minimum
core/extras imports and NumPy/Torch conversion passed on Python 3.11 with that
bound. Those import probes do not replace full source, installed-wheel or hosted
endpoint checks. CPU results do not validate CUDA or MPS, and the Intel Mac and
TabR arm64 limitations above remain in effect.

The TensorBoard extra requires `>=2.18,<3`. A minimum-extras full suite exposed
TensorBoard 2.14's use of NumPy's removed `np.string_` alias during hyperparameter
logging. Version 2.17 reproduced the failure; 2.18 passed real event writing
with NumPy 2 and minimum Torch. This compatibility repair does not change the
separate setuptools security/compatibility review.

Python 3.10 is no longer supported in DeepTab 2.1. Follow the
[Migration Guide](../getting_started/migration) before upgrading an older
environment. Documentation tooling retains its Python 3.12-3.13 requirement.
