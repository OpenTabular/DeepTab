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

### 1. Public API

The model's public constructor signature must be consistent with other stable estimators in `deeptab/models/`. Parameter names must follow existing conventions (e.g. `n_layers`, `d_model`, `dropout`). `__init__` must accept a config object from `deeptab/configs/` with all config fields reflected as constructor kwargs.

### 2. Documentation

A model page must exist under `docs/api/models/` and include:

- A one-paragraph description of the architecture.
- A **When to use** section: what problem or data type this model is suited for.
- A **Limitations** section: known failure modes, dataset-size requirements, or computational constraints.
- A full parameter table generated from the config docstring.

All public methods must have docstrings that render without warnings under `just docs`.

### 3. End-to-end Example

At least one runnable example must exist in `examples/` or `docs/examples/` that demonstrates loading data, constructing the model, fitting, and predicting. The example must run to completion without error against the current main branch.

### 4. Save / Load Support

If save/load is part of the stable core contract for the task type, the model must be saveable and reloadable via the standard DeepTab mechanism, with a round-trip test confirming identical predictions before and after reload.

### 5. Tests

A behavioral test must exist (in a dedicated file or in `tests/test_base.py`) covering:

- Fit on a small synthetic dataset.
- Predict returning an array of the expected shape and dtype.
- Config serialization round-trip.

All tests must pass in CI (`just test`).

### 6. No Open Critical Bugs

No open GitHub issues labelled `bug` for the model may describe a failure in a core workflow (fit, predict, save/load). Known limitations that are not bugs must be documented in the model's Limitations section.

### 7. Registry

A config class must exist in `deeptab/configs/` and be exported from `deeptab/configs/__init__.py`. The model must be exported from `deeptab/models/experimental/__init__.py` while experimental, or from `deeptab/models/__init__.py` once stable. The `MODEL_REGISTRY` in `deeptab/core/registry.py` must contain a `ModelInfo` entry with the correct `status` and `import_path`.

## Promotion PR

Open a PR titled `feat(<model-name>): promote to stable`. The PR must:

1. Move the model file from `deeptab/models/experimental/` to `deeptab/models/` using `git mv`.
2. Update relative imports in the moved file (reduce one `..` level).
3. Remove the model from `deeptab/models/experimental/__init__.py` and its `__all__`.
4. Add the model to `deeptab/models/__init__.py` imports and `__all__`.
5. Update `MODEL_REGISTRY` in `deeptab/core/registry.py`: change `status` to `"stable"` and `import_path` to `"deeptab.models"`.
6. Remove any `.. experimental::` admonition from the model's doc page.
7. Remove the experimental badge from the API reference entry.
8. Add the model to the changelog under `### Promoted to Stable`.

Approval requires at least one maintainer review beyond the author. Use the promotion checklist in the PR template to track each requirement.

## Demotion

Demotion back to experimental is only warranted if a critical correctness bug is found that cannot be fixed without a breaking API change. In that case:

1. Open an issue labelled `regression` and `breaking`.
2. Add an `ExperimentalWarning` in the next patch release pointing to the issue.
3. Fix the problem and re-promote via the standard requirements above.

Demotion is not itself a breaking change, but it must be documented prominently in the release notes.
