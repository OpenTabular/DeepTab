# Support Matrix

This page lists the officially supported versions of Python and core dependencies. "Supported" means the combination is tested in CI on every commit and pull request. Combinations not listed here may work but are not guaranteed.

---

## Python and Operating Systems

| Category | Supported                                      | Notes                                                           |
| -------- | ---------------------------------------------- | --------------------------------------------------------------- |
| Python   | 3.10, 3.11, 3.12, 3.13                         | 3.14+ pending `scipy` wheels; added once dependencies catch up. |
| OS       | Linux, macOS, Windows (all `*-latest` runners) |                                                                 |

---

## Core Dependencies and Policy

The authoritative version constraints live in [`pyproject.toml`](https://github.com/OpenTabular/deeptab/blob/main/pyproject.toml) under `[tool.poetry.dependencies]`, which is updated on every release. Treat that file as the source of truth; the list below names the core packages and the policy that governs their bounds.

- **PyTorch, Lightning, NumPy, pandas, scikit-learn, torchmetrics, scipy** are the pinned core dependencies. NumPy 1.x is **not** supported.

DeepTab follows a rolling support window, similar to [SPEC 0](https://scientific-python.org/specs/spec-0000/) used by the broader scientific Python ecosystem:

- **Python**: support the three most recent minor releases that have reached General Availability. Drop a version no earlier than 42 months after its release.
- **NumPy / pandas / scikit-learn**: support the two most recent minor releases of each dependency at the time of a new DeepTab minor release.
- **PyTorch**: support the current stable release and the one prior. The upper bound in `pyproject.toml` is updated when a new PyTorch stable is released and CI passes.

When a version is dropped, it is announced in the release notes and the `pyproject.toml` lower bound is bumped in the same minor release.
