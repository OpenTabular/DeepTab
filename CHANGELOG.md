# CHANGELOG

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/) and uses
[Conventional Commits](https://www.conventionalcommits.org/).

Commitizen generates initial entries with `cz bump`; release notes are reviewed
and consolidated for each stable version before publication.

## v2.1.0 (2026-10-20)

### Added

- PreTab presets and a representation registry for selecting preprocessing strategies.
- Canonical `PreprocessingConfig` fields with legacy alias resolution and actionable
  errors for unsupported preprocessing settings or missing optional dependencies.
- Block and dense output options in `build_preprocessor`.
- Saved preprocessor specifications, fingerprints, and feature widths for inspecting
  and restoring fitted preprocessing.

### Changed

- **Python support:** Require Python 3.11-3.13 (`>=3.11,<3.14`). Python 3.10
  [reached end of life on 2026-10-01](https://peps.python.org/pep-0619/#lifespan)
  and is no longer supported. Users requiring Python 3.10 must remain on DeepTab
  2.0.x with compatible dependencies; older releases do not guarantee security updates.
- **Preprocessing:** Integrate PreTab 1.0 and default numerical `output_dim` to 7
  for new fits. Explicit widths remain honored. Resolve the preprocessing task from
  the estimator type and map `ObservabilityConfig.verbosity` to preprocessor verbosity.
- **Dependency compatibility:** Require pandas >=2.2.2,<3 and SciPy >=1.16,<2.
  The SciPy minimum avoids a reproduced native loader failure on Apple Silicon.
  Documentation builds require Python 3.12 or 3.13.
- **Optional retrieval:** Move FAISS and delu to the explicit `tabr` extra. The
  default install and `all` observability bundle no longer install native retrieval
  packages. Native TabR remains unsupported on macOS arm64 and fails before
  conflicting native imports.
- **Artifact compatibility:** Historical 2.0 PLE artifacts require their original
  dependency environment; upgrading does not convert them. See the
  [Migration Guide](docs/getting_started/migration.md) before upgrading older environments.
- Use configured logging rather than `print()` for model save/load status messages.

### Fixed

#### Estimator State and Persistence

- Preserve an existing fitted estimator when a refit's input validation fails.
  Failures after rebuilding starts invalidate prediction until a successful fit
  (#424, #426, #449).
- Apply `set_params` overrides to model configuration, execute estimator contract
  checks, and reject unsupported weights (#410, #449).
- Refresh supplied data, validation splits, and loaders during weight continuation
  with `rebuild=False`, while retaining compatible weights and fitted preprocessing.
- Continue from pretrained weights by default in `fit()` (#446).
- Persist and restore the fitted loss function and NDTF's generated tree architecture
  across save/load (#444, #445).
- Default `load()` to CPU instead of automatically selecting hardware (#465).
- Preserve TabR and ModernNCA candidate data across save/load and loaded continuation,
  and prevent query-label leakage during retrieval (#413). Older incomplete or
  incompatible artifacts still require refitting.
- Correct ModernNCA regression and binary-classification reloads, and convert binary
  classification votes to real logits (#268).

#### Training, Metrics, and HPO

- Preserve string class labels in `Accuracy` and `F1Score`, fixing classifier
  `evaluate()` returning NaN accuracy for valid string predictions. Add an explicit
  `pos_label` option for binary F1 with string labels; numeric labels and probability
  inputs retain their existing behavior.
- Honor fit-level arguments, optimizer options, and custom regression losses instead
  of silently dropping them or overriding them with trainer settings (#415, #430, #442).
- Respect explicit checkpoint paths and requested monitor/mode settings, resolve
  `best_epoch`, and timestamp default checkpoint directories (#411, #425).
- Correct seeding, callbacks, fit-state transitions, and device handling (#452).
- Encode classification labels consistently and route metrics to labels,
  probabilities, or raw outputs as required, including ensemble evaluation
  (#409, #443).
- Aggregate training and validation metrics over complete epochs and reset metric
  state between sanity validation and epochs.
- Accept probabilities rather than raw logits in expected calibration error and
  include confidence 1.0 in its bins (#421).
- Correct HPO trial selection, pruning, and winner refitting (#420).

#### Data and Preprocessing

- Correct model encoding, categorical-feature metadata, input-dimension typing, and
  preprocessing type compatibility with PreTab (#422, #451).
- Cast concatenated features to floating point for non-embedding architectures.
- Normalize `None` to `np.nan` in object columns, reject duplicate columns, and impute
  all-NaN columns in `ensure_dataframe`.
- Reject `categorical_method="none"` in `PreprocessingConfig` with a clear error.
- Handle regression target shapes correctly (#441).

#### Architectures and Distributions

- Correct architecture configuration options and neural-block behavior (#423, #450).
- Preserve strict custom architecture constructor signatures when supplying binary
  task metadata.
- Correct MLP bias handling and NDTF backward computation on MPS without changing
  saved parameter layouts.
- Apply the Mamba block in `MambaTab.forward`, use pre-normalization and dropout
  around RowColTransformer attention, and align contrastive anchors with their pairs.
- Preserve parameter axes in ensemble LSS outputs (#447).
- Correct LSS scoring, distribution handling, and persistence (#419).
- Fix multinomial softmax, JohnsonSU naming, and singleton-dimension handling (#453).
- Use PyTorch's probability convention in `NegativeBinomial` and correct the
  log-term sign in `GammaDeviance`.

### Security

- Require Torch >=2.13,<2.15, setuptools >=83,<84, Lightning >=2.6.6,<3, and
  Accelerate >=1.15,<2 to exclude versions with reported vulnerabilities.
- Require TensorBoard >=2.21,<3 for TensorBoard and tracking extras, compatible with
  patched setuptools and NumPy 2 hyperparameter logging.
- Update vulnerable locked dependencies, remove blanket audit ignores, and update
  the separately installed documentation HTML cleaner to 0.4.5.

### Documentation and Maintenance

- Clarify installation profiles, preprocessing widths, historical artifact limits,
  supported environments, and the maintainer release process; correct docstrings
  and examples (#454).
- Run CI on release-branch pushes and add the aggregate `CI Success` check.
  Validate release-tag ancestry before packaging, testing, and publication.
- Validate independently resolved minimum/newest dependency profiles and installed-wheel
  training and save/load workflows in CI.
- Correct CI dependency installation, runner-cache scope, and NumPy typing checks;
  enable Prettier in pre-commit hooks and update regression tests.
- Centralize preprocessor construction and neural-block imports, and remove unused
  MLP/batch helpers and stub modules.

## v2.0.0 (2026-06-24)

### BREAKING CHANGE

- internal package layout, configuration objects, and import
  paths have changed. See the migration guide for details.

### Feat

- DeepTab v2 API with split-config design (#400)
- **config**: warn on misplaced config slots
- **training**: add unregister_optimizer, unregister_scheduler with built-in protection
- **inspection**: expose public read-only task_model property
- **models**: thread observability_config through all estimators
- **core**: add ObservabilityConfig
- **models**: expose ObservabilityConfig on base estimator constructor
- **models**: add observability mixin wiring ObservabilityConfig to base estimators
- **models**: integrate ObservabilityConfig into fit mixin
- **training**: rewrite configure_optimizers, add contrastive pretraining fixes, and cleanup
- introduce IDataModule/ITaskModel protocols and default factories, wire into SklearnBase
- **configs**: add optimizer/scheduler fields to TrainerConfig and InferenceModel support
- **training**: wire optimizer/scheduler registry into LightningModule and extend losses
- **training**: add optimizer/scheduler registry with all torch.optim classes
- **api**: export exception and warning types from deeptab and deeptab.core
- **configs,models**: add **post_init** validation using typed exceptions
- **core**: add exception hierarchy and message factories
- **models**: wire evaluate() in lss_base, regressor_base, and classifier_base to new deeptab.metrics registry
- **metrics**: add deeptab metrics ABC, regression, classification, lss
- add tweedie, inflated poissons, log normal etc. distribution
- light weight inference wrapper
- **serialization**: warn when save/load path lacks .deeptab extension
- **inspection**: add profile() method for pre-training dry-run diagnostics
- **training**: add class-imbalance loss registry and weighted sampling
- **core**: add set_seed/seed_context reproducibility helpers
- **core**: add sklearn_compat module and update serialization/core exports
- add rich model artifact serialization metadata
- model inspection api added
- **data**: add optional TabularBatch return mode
- **data**: add stratified splitting for classification and schema property
- **data**: add FeatureSchema and TabularBatch typed containers
- **configs**: add SplitConfig for train/validation splitting parameters
- **root**: expose configs, data, distributions, metrics, models in top-level **init**
- **models**: add \_docstring helper to centralize generate_docstring for all models
- **models**: expose stable classes in **all** and add **getattr** shim for experimental
- **models**: add split base classes for classifier, regressor, and LSS task variants
- **configs**: add configs/core.py with shared base configuration definitions
- **configs**: add configs/experimental sub module for ModernNCA, Tangos, Trompt
- **configs**: add configs sub module with per-model config modules
- **hpo**: add hpo module with get_search_space mapper
- **metrics**: add metrics module stubs for classification, regression, distributional
- **distributions**: add distributions module with 12 distribution classes
- **data**: add data module with MambularDataModule, MambularDataset, batch, schema, split
- **training**: add training module with lightning module, losses, optimizers, schedulers
- **core**: add core module with BaseModel, registry, embeddings, pooling, serialization
- **architectures**: add experimental sub-package with ModernNCA, Tangos, Trompt
- **architectures**: add architectures module with all stable model definitions
- **nn**: add nn module with blocks, normalization, and initialization
- **config**: split config into trainer, model and preprocessing config
- **sklearn_parent**: implement split-config path in SklearnBase.**init**, get_params, set_params
- **models**: add split config **init** to all Classifier and Regressor wrappers
- **base_models**: replace DefaultXXConfig with XXConfig in all base model constructors
- **configs**: add \*Config for all architectures
- **configs**: add ENODEConfig architecture only config
- **hardware**: add print_hardware_info for CPU/CUDA/MPS detection

### Fix

- **sklearn_compat**: satisfy pandas typing in ensure_dataframe
- **training**: register custom torchmetrics via nn.ModuleDict so state moves to device
- **sklearn_compat**: cast pandas category columns to object in ensure_dataframe
- **modernnca**: support LSS prediction and add experimental model tests
- **models**: adapt child class to use class var, update docstring example
- **transformer**: use batch_first attention to prevent cross-sample leakage
- **hpo**: rebuild model per trial and map activation names to modules
- save default artificats to <run_dir>/artifacts/model.deeptab
- **base**: add **sklearn_is_fitted**, use check_is_fitted
- **sklearn_compat**: raise ValueError for 1D array input in ensure_dataframe
- **exceptions**: inherit EmptyDataError and ColumnCountError from ValueError for sklearn compat
- add seed to DataLoader/sampler generators
- data validation for parameters
- **models**: read optimizer_type and preprocessor live from config in \_build_model
- **test**: add typed error, fix preprocessing config
- **architectures,distributions**: replace ValueError with typed exceptions
- **docs**: remove dead cross-reference links and fix tables
- **training**: apply distribution parameter transform before passing predictions to metrics
- use r2 metric for regresion as default
- use getattr for task_model access in InspectionMixin
- enable side bar navigation for api reference
- **tests**: update flat-kwarg error assertions to match native TypeError message
- **tests**: update config lookup to search configs.models and configs.experimental
- training parameter added
- modernca config and model update
- **lss**: use getattr fallback for lr/weight_decay in SklearnBaseLSS.fit()

### Refactor

- **models**: drop legacy flat-kwargs constructor
- **core**: centralize optional-dependency
- replace SplitConfig with TrainerConfig.stratify and refresh docs
- **models**: adopt declarative class variable estimator pattern
- **hpo**: rename mapper.py to search_space.py and fix lss_base error
- **core**: update inspection and serialization for \_ attribute rename
- **models**: prefix non-constructor attributes with \_ for sklearn compliance
- extract \_FitMixin, \_PredictMixin, \_SerializationMixin, \_HyperparameterMixin, \_ObservabilityMixin from SklearnBase
- **configs**: remove legacy BaseConfig class
- **distributions**: separate dist classes, add registry
- consolidate save/load into core.serialization helpers
- **models**: update base classifier/regressor/lss model internals
- **data**: update datamodule and dataset internals
- **models**: update imports to use TabularDataModule
- **data**: rename to TabularDataset/TabularDataModule and move task-specific label logic to DataModule
- **models**: replace \*\*kwargs with explicit signatures in stable model constructors
- **hpo**: add missing exports to hpo/**init**.py
- **models**: update training and hpo imports to go through package boundaries
- **architectures**: update core imports to go through package boundary
- **architectures**: add lazy **getattr** boundary with TYPE_CHECKING guards
- **nn**: expose public API via nn/**init**.py boundary
- **training**: expose public API via training/**init**.py boundary
- **core**: expose public API via core/**init**.py boundary
- **architectures**: update config imports to use configs/models/ and configs/experimental/
- **models**: update config imports to use configs/models/, configs/experimental/, and configs/core
- **configs**: update **init** to import from core, models/, and experimental/
- **configs**: remove deprecated flat config files superseded by models/ and experimental/
- **models**: update import paths in experimental ModernNCA, Tangos, Trompt modules
- **models**: update import paths in ndtf, node, resnet, saint, tabm, tabr, tabtransformer, tabularnn
- **modules**: remove legacy arch_utils, base_models, data_utils, utils

## v1.8.0 (2026-05-24)

### Feat

- stable/experimental model split, save/load, and bug fixes
- **models**: add save/load, fix get_params and LSS family pickling
- stable and experimental model separation

### Fix

- duplicate entry in docs for model classes
- test case fixed for windows
- suppress family hparam warning
- **tabr**: cast candidate_y to float in regression/LSS label encoder path
- **tabr**: use regression label encoder for LSS by forwarding lss flag
- **types**: guard task_model None, use getattr for get_params, annotate test kwargs
- **lint**: drop unused unpacked variables and dead code
- **tabtrans**: add dedicated LayerNorm for numerical features instead of reusing encoder norm
- **ndtf**: correct ensemble aggregation for multi-class and LSS outputs
- added faiss-cpu
- add delu dependency for TabR
- DefaultTabRConfig export
- import error after experimental namespace changes
- resolve stale site-packages

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
