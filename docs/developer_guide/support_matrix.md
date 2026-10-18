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
wheels are available for Python 3.13** in the supported Torch range. Intel Macs are
outside the CI matrix. The older Intel PyTorch 2.2.2 build does not satisfy the
current dependency range and is not a supported workaround.
```

```{warning}
**Native TabR is unsupported on Apple Silicon.** Patched Torch and the PyPI
`faiss-cpu` build load conflicting OpenMP runtimes. Construction fails before
native imports, and CI runs native cases only on Linux and Windows with the
`tabr` extra. Exact-index tests still run without FAISS. Alternative FAISS
builds are not covered by this CI matrix.
```

---

(dependency-policy)=

## Core Dependencies and Policy

The authoritative version constraints live in [`pyproject.toml`](https://github.com/OpenTabular/deeptab/blob/main/pyproject.toml) under `[tool.poetry.dependencies]`, which is updated on every release. Treat that file as the source of truth; the list below names the core packages and the policy that governs their bounds.

- **PyTorch, Lightning, NumPy, pandas, scikit-learn, torchmetrics, scipy, PreTab** are core dependencies with declared version ranges; `poetry.lock` records exact development versions. NumPy 1.x is **not** supported.
- **delu, faiss-cpu** are model-specific dependencies installed only by the `tabr` extra. The default install and the `all` observability bundle exclude them.

DeepTab targets a tested support window, informed by [SPEC 0](https://scientific-python.org/specs/spec-0000/) used by the broader scientific Python ecosystem:

- **Python**: maintain a three-minor window, currently 3.11-3.13. A new upstream release is not admitted automatically: compatible dependencies and release checks must pass first. Review older versions near upstream end of life, no earlier than 42 months after release, and announce removals in the release notes.
- **NumPy / pandas / scikit-learn**: support the two most recent minor releases of each dependency at the time of a new DeepTab minor release.
- **PyTorch**: support the current stable release and the one prior. The upper bound in `pyproject.toml` is updated when a new PyTorch stable is released and CI passes.

The declared windows are Torch `>=2.13,<2.15`, setuptools `>=83,<84`,
Lightning `>=2.6.6,<3`, and Accelerate `>=1.15,<2`. They exclude older versions
with reported vulnerabilities. TensorBoard extras require `>=2.21,<3` to work
without the removed `pkg_resources` module and to write events with NumPy 2.

This is a compatibility target, not a claim that every admitted dependency
combination is free of advisories. Apple Silicon probes with both Torch 2.13
and 2.14.1 plus FAISS 1.15.1 aborted in both import orders. Optional retrieval
keeps that native conflict out of default installs; an unsafe OpenMP override
is not an approved workaround. Configured native Linux/Windows jobs and fresh
core/extras endpoints still require passing evidence for the exact revision.

Dependency bounds alone are not security sign-off.
The {ref}`dependency audit policy <dependency-security>` requires an
unfiltered audit and a verified remediation or explicit reviewed decision.
Trusted-checkpoint guidance does not make an affected loader secure. Preserve
the original findings and keep release sign-off open while unapproved findings
remain.

When a supported version is dropped, announce it in the release notes and update
the `pyproject.toml` lower bound in the same release. Choose the release increment
according to the [versioning policy](versioning.md), rather than assuming that a
support removal is backward-compatible.

## Dependency Endpoint Validation

The nine locked core lanes cover Python 3.11-3.13 on three operating systems.
An additional 16 lanes resolve minimum-direct dependencies on Python 3.11 and
newest stable compatible dependencies on Python 3.13, separately for core and
`logs`/`tracking` installations on each platform, plus native `tabr` profiles
on Linux and Windows. Two locked native lanes use Python 3.12 on those platforms.
Resolution uses the built wheel's
metadata, not the development lock. The minimum set is jointly resolved; it is
not a promise that every direct lower bound can be installed simultaneously.

Both endpoints require native imports, `pip check`, the full CPU source suite,
90% line coverage, and installed-wheel fit/save/reload checks outside the checkout.
The wheel version, import origin and SHA256 must match the requested artifact.
See the {ref}`reproduction and evidence requirements <dependency-endpoints>`.

Local probes reproduced pandas 2.0.3/NumPy 2.0.0 binary incompatibility and Torch
2.2.2/NumPy 2.0.0 conversion failure. The pandas floor remains 2.2.2; Torch
now requires 2.13 or newer for the security fixes above. Historical Torch 2.4
conversion checks do not validate the revised stack.

Historical Python 3.10 captures retain the SciPy 1.15.0-1.15.3 PROPACK loader
failures. Raising Python alone did not fix the issue: SciPy 1.15.0 also failed
on local Apple Silicon Python 3.11. The SciPy minimum is now 1.16.0; clean minimum
core/extras imports and NumPy/Torch conversion passed on Python 3.11 with that
bound. Those import probes do not replace full source, installed-wheel or hosted
endpoint checks. CPU results do not validate CUDA or MPS, and the Intel Mac and
TabR arm64 limitations above remain in effect.

The TensorBoard extra now requires `>=2.21,<3`. A historical minimum-extras full suite exposed
TensorBoard 2.14's use of NumPy's removed `np.string_` alias during hyperparameter
logging. Version 2.17 reproduced the failure; 2.18 passed real event writing
with NumPy 2 and the former minimum Torch. Those old results do not replace
validation of the patched setuptools/TensorBoard combination.

Before these dependency changes, on 2026-10-15, the candidate passed all four local CPU endpoint suites
on Apple Silicon: Python 3.11.16 minimum-direct and Python 3.13.15 newest-compatible,
each with core and logs/tracking profiles. Each lane passed four installed-wheel
fit/save/reload workflows and exceeded 91.6% line coverage, with matching
source and wheel hashes. The candidate still carries package version `2.0.0`;
these results do not certify a final-version 2.1 artifact, hosted platforms,
native TabR, accelerators or security sign-off.

Python 3.10 is no longer supported in DeepTab 2.1. Follow the
[Migration Guide](../getting_started/migration) before upgrading an older
environment. Documentation tooling retains its Python 3.12-3.13 requirement.
