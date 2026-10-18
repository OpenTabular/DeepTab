# Model Promotion Policy

This document defines the minimum bar a model must meet before it can be promoted from experimental to stable. The goal is to make promotion a deterministic, reviewable decision rather than a subjective one.

## Stability Tiers

DeepTab ships models at two tiers:

| Tier             | Meaning                                                                                                                           |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Experimental** | API may change without a deprecation cycle. No stability guarantees. Flagged in docs and docstrings.                              |
| **Stable**       | Public API is frozen under semantic versioning. Breaking changes require a deprecation notice ≥ one minor release before removal. |

A model enters the codebase as experimental. A maintainer promotes it to stable by opening a dedicated promotion PR once every criterion below is met.

## Promotion Requirements

Every criterion below must be met before promotion. Each is objective and reviewable.

| #   | Requirement        | What it means                                                                                                                                                                                                                                                             |
| --- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Public API         | Estimators use the shared split-config constructor (`model_config`, `preprocessing_config`, `trainer_config`, ...), with sklearn-compatible cloning and nested parameter access. Architecture settings belong in a `BaseModelConfig` subclass, not flat estimator kwargs. |
| 2   | Documentation      | A page under `docs/model_zoo/` with architecture summary, **When to use**, **Limitations**, and configuration guidance, plus an API reference entry. All public methods render docstrings without `just docs` warnings.                                                   |
| 3   | End-to-end example | A runnable tutorial or example under `docs/tutorials/` covering data loading, construction, fitting, and prediction, passing against current `main`.                                                                                                                      |
| 4   | Save / Load        | If save/load is part of the task's stable contract, the model round-trips via the standard mechanism with a test asserting identical predictions.                                                                                                                         |
| 5   | Tests              | A behavioral test covering fit on synthetic data, predict shape/dtype, and config serialization round-trip. All pass in CI (`just test`).                                                                                                                                 |
| 6   | No critical bugs   | No open `bug` issues describing a failure in fit, predict, or save/load. Non-bug limitations are documented in the Limitations section.                                                                                                                                   |
| 7   | Registry           | Config class exported from `deeptab/configs/__init__.py`; model exported from the experimental package; `MODEL_REGISTRY` entry present with correct `status` and `import_path`.                                                                                           |

## Promotion PR

Open a PR titled `feat(<model-name>): promote to stable`. The PR must:

1. Move the model file from `deeptab/models/experimental/` to `deeptab/models/` using `git mv`.
2. Move the architecture from `deeptab/architectures/experimental/` to `deeptab/architectures/` and its config from `deeptab/configs/experimental/` to `deeptab/configs/models/`. Update imports and the estimator's `_model_cls` / `_config_cls` bindings.
3. Remove the model from `deeptab/models/experimental/__init__.py` and its `__all__`.
4. Update exports in `deeptab/models/__init__.py`, `deeptab/architectures/__init__.py`, and `deeptab/configs/__init__.py` for the new paths.
5. Update `MODEL_REGISTRY` in `deeptab/core/registry.py`: change `status` to `"stable"` and `import_path` to `"deeptab.models"`.
6. Move its guide from `docs/model_zoo/experimental/` to `docs/model_zoo/stable/`, update navigation and links, and remove experimental notices.
7. Update the API reference imports and remove its experimental badge. Run the registry-selected stable learning and persistence tests for the newly stable model; see [Testing](testing.md).
8. Add the model to the changelog under `### Promoted to Stable`.

Approval requires at least one maintainer review beyond the author. Use the promotion checklist in the PR template to track each requirement.

## Demotion

Demotion back to experimental is only warranted if a critical correctness bug is found that cannot be fixed without a breaking API change. In that case:

1. Open an issue labelled `regression` and `breaking`.
2. Add an `ExperimentalWarning` in the next patch release pointing to the issue.
3. Fix the problem and re-promote via the standard requirements above.

Document demotion prominently in the release notes. A change in stability label
does not authorize incompatible public API changes; those must still follow
the [versioning policy](versioning.md).
