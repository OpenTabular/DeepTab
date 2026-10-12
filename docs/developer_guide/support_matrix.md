# Support Matrix

This page lists the officially supported versions of Python and core dependencies. These combinations are included in the configured CI matrix. Automatic CI covers pushes to `main` and pull requests targeting `main`; other revisions need a manual or reusable workflow invocation. Confirm results for the exact revision before claiming release validation. Combinations not listed here may work but are not guaranteed.

---

## Python and Operating Systems

| Category | Supported                                                      | Notes                                                           |
| -------- | -------------------------------------------------------------- | --------------------------------------------------------------- |
| Python   | 3.10, 3.11, 3.12, 3.13                                         | 3.14+ pending `scipy` wheels; added once dependencies catch up. |
| OS       | Linux, macOS (Apple Silicon), Windows (all `*-latest` runners) | CI's macOS runners are Apple Silicon (arm64).                   |

```{warning}
**Intel Mac (macOS x86_64) is not supported.** Upstream PyTorch no longer publishes
x86_64 macOS wheels for `torch >= 2.3` or Python 3.13. **Apple Silicon PyTorch
wheels are available for Python 3.13**, including `torch==2.9.1`. Intel Macs are
outside the CI matrix; see [Installation](../getting_started/installation) for the
Python 3.12 / PyTorch 2.2.2 workaround.
```

```{warning}
**TabR is not validated on Apple Silicon.** The PyPI `faiss-cpu` build can
segfault on macOS arm64, so CI skips TabR classifier, regressor, and LSS tests
on that platform. Use Linux or Windows for TabR. An alternative FAISS build is
not covered by this CI matrix.
```

---

## Core Dependencies and Policy

The authoritative version constraints live in [`pyproject.toml`](https://github.com/OpenTabular/deeptab/blob/main/pyproject.toml) under `[tool.poetry.dependencies]`, which is updated on every release. Treat that file as the source of truth; the list below names the core packages and the policy that governs their bounds.

- **PyTorch, Lightning, NumPy, pandas, scikit-learn, torchmetrics, scipy, PreTab** are the pinned core dependencies. NumPy 1.x is **not** supported.
- **delu, faiss-cpu** are mandatory but model-specific dependencies (used only by TabR), pinned to a tested range rather than a broader policy window; they are candidates for optional extras in a future release.

DeepTab follows a rolling support window, similar to [SPEC 0](https://scientific-python.org/specs/spec-0000/) used by the broader scientific Python ecosystem:

- **Python**: support the three most recent minor releases that have reached General Availability. Drop a version no earlier than 42 months after its release.
- **NumPy / pandas / scikit-learn**: support the two most recent minor releases of each dependency at the time of a new DeepTab minor release.
- **PyTorch**: support the current stable release and the one prior. The upper bound in `pyproject.toml` is updated when a new PyTorch stable is released and CI passes.

When a version is dropped, it is announced in the release notes and the `pyproject.toml` lower bound is bumped in the same minor release.
