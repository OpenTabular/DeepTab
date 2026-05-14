# CHANGELOG

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/) and uses
[Conventional Commits](https://www.conventionalcommits.org/).

Going forward, this file is updated automatically by `cz bump` on each release.

---

## v1.7.0 (2026-05-14)

### Documentation

- Added **Developer Guide** section with dedicated pages for contributing, the release process, model promotion policy (experimental → stable), and a SPEC 0–aligned support matrix
- Added **Getting Started** section: new Overview page with full model reference table, Installation guide (prerequisites, PyPI, source, optional Mamba CUDA kernels), and Key Concepts page covering the sklearn API, task variants, config system, and preprocessing
- Rewrote all three example pages (classification, regression, distributional regression) as narrative tutorials with runnable code and "using your own data" sections
- Added `llms.txt` index for LLM tool discovery
- Switched to `sphinx_design` for cards and grid layouts; added custom CSS with JetBrains Mono font and brand colour palette
- Added `pygments_style = "friendly"` and dark-mode code block colours via CSS
- Updated release workflow documentation; fixed Mermaid diagram node line breaks

### CI

- Dropped Python 3.14 from the test matrix (`scipy` wheels are not yet available); ceiling reverted to `<3.14` in `pyproject.toml`
- Fixed `torch` upper bound from `<=2.7.0` to `<2.8.0` to allow patch releases
- Replaced `tomllib` with `poetry version --short` for Python 3.10 compatibility in CI scripts
- Added Lightning log filtering to reduce test output noise
- Regenerated `poetry.lock` after dependency constraint changes

### Bug Fixes

- Fixed Pyright type errors in test fixtures (DataFrame construction and unused variable bindings)

---

## v1.7.0rc2 (2026-05-09)

### Documentation

- Updated and fixed release workflow documentation
- Applied doc theme and header size style refinements

### CI

- Fixed `tomllib` usage replaced with `poetry version --short` for Python 3.10 compatibility

---

## v1.7.0rc1 (2026-05-08)

### Documentation

- Added **Developer Guide** section with dedicated pages for contributing, the release process, model promotion policy (experimental → stable), and a SPEC 0–aligned support matrix
- Added **Getting Started** section: new Overview page with full model reference table, Installation guide (prerequisites, PyPI, source, optional Mamba CUDA kernels), and Key Concepts page covering the sklearn API, task variants, config system, and preprocessing
- Rewrote all three example pages (classification, regression, distributional regression) as narrative tutorials with runnable code and "using your own data" sections
- Added `llms.txt` index for LLM tool discovery
- Switched to `sphinx_design` for cards and grid layouts; added custom CSS with JetBrains Mono font and brand colour palette
- Added `pygments_style = "friendly"` and dark-mode code block colours via CSS

### CI

- Dropped Python 3.14 from the test matrix (`scipy` wheels are not yet available); ceiling reverted to `<3.14` in `pyproject.toml`
- Fixed `torch` upper bound from `<=2.7.0` to `<2.8.0` to allow patch releases
- Added Lightning log filtering to reduce test output noise
- Regenerated `poetry.lock` after dependency constraint changes

### Bug Fixes

- Fixed Pyright type errors in test fixtures (DataFrame construction and unused variable bindings)

---

## v1.6.1 (2025-04-26)

### Changes

- Renamed package from `mambular` / `deeptabular` to `deeptab`
- Dynamic versioning: version is now sourced from `pyproject.toml` via `importlib.metadata`; removed `__version__.py`
- CI rework: split into lint, typecheck, build, and test jobs; manual tagging with OIDC PyPI publishing; removed semantic-release automation

---

## v1.5.0 (2025-04-14)

### Changes

- Moved preprocessing to the `pretab` package; removed bundled preprocessor
- Added TabR model integration
- Fixed LSS bug affecting distributional output
- Updated docstrings for documentation generation compatibility

---

## v1.4.0 (2025-03-24)

### Features

- Added ModernNCA model
- Added training candidates support during prediction and validation in the lightning module

---

## v1.3.2 (2025-03-19)

### Bug Fixes

- Fixed `num_classes` argument for binary classification
- Fixed version info extraction

---

## v1.3.1 (2025-03-17)

### Features

- Added Tangos model (classifier, regressor, and distributional variants)

---

## v1.3.0 (2025-03-13)

### Features

- Added AutoInt model
- Added Trompt model
- Added ENode (embedding oblivious trees) model
- Fixed preprocessor bug causing `ValueError: not enough values to unpack`

---

## v1.2.0 (2025-02-17)

### Features

- Added `BaseConfig` parent class; restructured all configs to inherit from it
- Added `JohnsonSU` distribution and individual preprocessing per column
- Adapted embedding layer for new preprocessing pipeline
- Added unit tests for PRs
- Fixed column name handling (int → string) in datamodule

---

## v1.1.0 (2025-01-03)

### Features

- Added `BaseConfig` class to public init
- Added `JohnsonSU` distribution support
- Removed dependency on rotary embeddings

---

## v1.0.0 (2024-12-04)

Initial stable release.
