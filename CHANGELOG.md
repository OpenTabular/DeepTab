# CHANGELOG

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/) and uses
[Conventional Commits](https://www.conventionalcommits.org/).

Going forward, this file is updated automatically by `cz bump` on each release.

## v2.1.0rc1 (2026-10-18)

### Feat

- **preprocessing**: add PreTab optional-dependency errors to ImportErrors in deeptab
- **preprocessing**: add PreTab presets and representation registry
- **serialization**: persist preprocessor spec, fingerprint, and feature widths
- **preprocessing**: map ObservabilityConfig.verbosity to Preprocessor verbose
- **preprocessing**: resolve task from estimator type, override config
- **config**: add canonical PreprocessingConfig fields with legacy alias resolution

### Fix

- **deps**: update docs HTML cleaner to 0.4.5
- package update
- mlp refactor for mps device
- linting, formatting
- resolve dependency installation, type checks and test failures in CI
- **hooks**: enable Prettier and clean up formatting
- **tabr**: install FAISS and delu only when requested
- **ci**: move runner cache configuration to step scope
- correct categorical feature metadata typing
- correct ModernNCA regression and binary model reloads
- **compat**: drop Python 3.10 support and validate dependency endpoints
- **deps**: update vulnerable dependencies and remove audit ignores
- preserve retrieval inference state across save and load
- compute metrics over complete epochs
- adapt classification metric inputs during training
- refresh data and loaders during weight continuation
- preserve fitted state when refit validation fails
- **models**: execute contract checks and reject unsupported weights
- clarify preprocessing widths and historical artifact compatibility
- preserve parameter axes in ensemble LSS outputs (#447)
- correct model encoding (#451)
- correct architecture options and neural block behavior (#423)
- harden model state, data handling, and artifact loading (#424)
- correct model state, validation, scoring, and training behavior (#426)
- preserve estimator state and sklearn compatibility (#449)
- **lss**: correct scoring, distribution handling, and persistence (#419)
- correct HPO trial selection, pruning and winner refitting (#420)
- prevent query label leakage in retrieval models (#413)
- broken config options across architectures (#450)
- correct seeding, fit state, callbacks, and device handling (#452)
- apply set_params overrides to model config for sklearn compatibility (#410)
- cast concatenated features to float for for non-embedding architectures
- normalize None to np.nan for object columns
- reject categorical_method="none" in PreprocessingConfig
- convert ModernNCA binary classification vote to a real logit (#268)
- reject duplicate columns and impute all-NaN columns in ensure_dataframe
- persist NDTF's generated tree architecture across save/load (#444)
- default load() to cpu instead of auto-detecting hardware (#465)
- continue from pretrained weights by default in fit() (#446)
- persist and restore the fitted loss function across save/load (#445)
- honor explicit checkpoint_path, resolve best_epoch, and timestamp default checkpoint dirs (#411)
- **distributions**: use torch probs convention in NegativeBinomial
- **metrics**: correct inverted log-term sign in GammaDeviance
- **nn**: apply pre-norm and dropout around RowColTransformer attention
- **training**: align contrastive pretraining anchors with their pairs
- **architectures**: apply the Mamba block in MambaTab.forward
- ModelCheckpoint ignoring requested monitor/mode (#425)
- update docstring, correct docs (#454), fix test cases
- multinomial softmax, johnsonsu naming, and singleton squeeze bugs (#453)
- fit() metric args being swapped and crashing on tensors (#443)
- feature_preprocessing type mismatch with pretab (#422)
- optimizer_kwargs and regression loss_fct being silently dropped (#415, #430)
- fit() arguments being overridden by TrainerConfig (#442)
- regression label shape handling (#441)
- type correction for input_dim
- **metrics**: ECE accepts proba instead of raw logit
- encode labels for classification input (#409)
- **metrics**: count confidence 1.0 in ECE binning (#421)
- **core**: use logging for BaseModel.save_model/load_model status messages instead of print
- **tests**: pyright type-widening in legacy fields
- **preprocessing**: add blocks/dense output in build_preprocessor
- **config**: default output_dim for preprocessor to 7

### Refactor

- **nn**: consolidate scattered imports and drop E402 noqa in nn/blocks
- **core**: remove dead MLP_Block and make_random_batches
- **core**: delete unused docstring-only stub modules
- **preprocessing**: centralize Preprocessor construction in build_preprocessor

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
