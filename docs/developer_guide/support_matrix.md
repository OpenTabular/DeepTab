# Support Matrix

This page lists the officially supported versions of Python and core dependencies. "Supported" means the combination is tested in CI on every commit and pull request. Combinations not listed here may work but are not guaranteed.

---

## Python

| Version | Status         |
| ------- | -------------- |
| 3.10    | Supported      |
| 3.11    | Supported      |
| 3.12    | Supported      |
| 3.13    | Supported      |
| 3.14    | Supported      |
| < 3.10  | Not supported  |
| 3.15+   | Not yet tested |

---

## Operating Systems

| OS                       | Status    |
| ------------------------ | --------- |
| Linux (ubuntu-latest)    | Supported |
| macOS (macos-latest)     | Supported |
| Windows (windows-latest) | Supported |

---

## Core Dependencies

The table below shows the range of versions supported by the package metadata (`pyproject.toml`). The lower bound is the minimum version that has been tested; the upper bound is what Poetry's caret/range constraint allows.

| Package                                              | Minimum | Upper bound | Notes                                                      |
| ---------------------------------------------------- | ------- | ----------- | ---------------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                      | 2.2.2   | < 2.8.0     | Pinned range; update when a new PyTorch stable is released |
| [Lightning](https://lightning.ai/)                   | 2.3.3   | < 3.0       |                                                            |
| [NumPy](https://numpy.org/)                          | 2.0.0   | < 3.0       | NumPy 1.x is **not** supported                             |
| [pandas](https://pandas.pydata.org/)                 | 2.0.3   | < 3.0       |                                                            |
| [scikit-learn](https://scikit-learn.org/)            | 1.3.2   | < 2.0       |                                                            |
| [torchmetrics](https://torchmetrics.readthedocs.io/) | 1.5.2   | < 2.0       |                                                            |
| [scipy](https://scipy.org/)                          | 1.15.0  | < 2.0       |                                                            |

---

## Policy

DeepTab follows a rolling support window, similar to [SPEC 0](https://scientific-python.org/specs/spec-0000/) used by the broader scientific Python ecosystem:

- **Python**: support the three most recent minor releases that have reached General Availability. Drop a version no earlier than 42 months after its release.
- **NumPy / pandas / scikit-learn**: support the two most recent minor releases of each dependency at the time of a new DeepTab minor release.
- **PyTorch**: support the current stable release and the one prior. The upper bound in `pyproject.toml` is updated when a new PyTorch stable is released and CI passes.

When a version is dropped, it is announced in the release notes and the `pyproject.toml` lower bound is bumped in the same minor release.

---

## Updating the Matrix

When CI is extended to cover a new Python or dependency version:

1. Update `pyproject.toml` (the `[tool.poetry.dependencies]` version constraint).
2. Update `.github/workflows/ci.yml` (`matrix.python-version` or equivalent).
3. Update the table on this page to reflect the new status.

All three changes should land in the same PR so the docs and the CI are always in sync.
