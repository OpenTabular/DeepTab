# CHANGELOG

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/) and uses
[Conventional Commits](https://www.conventionalcommits.org/).

Going forward, this file is updated automatically by `cz bump` on each release.

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
