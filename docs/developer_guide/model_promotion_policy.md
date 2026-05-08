# Model Promotion Policy: Experimental → Stable

This document defines the minimum bar that every model in DeepTab must meet before it can be promoted from experimental to stable. The goal is to make promotion a deterministic, reviewable decision rather than a subjective one.

---

## Background

DeepTab ships models at two tiers:

| Tier | Meaning |
|---|---|
| **Experimental** | API may change without a deprecation cycle. No stability guarantees. Clearly flagged in docs and docstrings. |
| **Stable** | Public API is frozen under semantic versioning. Breaking changes require a deprecation notice ≥ one minor release before removal. |

A model enters the codebase as experimental. A maintainer promotes it to stable by opening a dedicated promotion PR once every criterion below is met.

---

## Promotion Checklist

The promoting maintainer must verify each item and mark it in the PR description before requesting review.

### 1. Public API

- [ ] The model's public constructor signature (parameters and their defaults) is consistent with other stable estimators in `deeptab/models/`.
- [ ] Parameter names follow existing naming conventions (e.g. `n_layers`, `d_model`, `dropout`).
- [ ] No positional-only parameters that would break if reordered in a future patch.
- [ ] `__init__` accepts a config object from `deeptab/configs/` and all config fields are reflected as constructor kwargs.

### 2. Documentation

- [ ] A model page exists under `docs/api/models/` or is reachable from the API reference.
- [ ] The model page includes:
  - A one-paragraph description of the architecture.
  - A **"When to use"** section — what problem or data type this model is suited for.
  - A **"Limitations"** section — known failure modes, dataset-size requirements, or computational constraints.
  - Full parameter table generated from the config docstring.
- [ ] All public methods have docstrings that pass `make doctest`.

### 3. End-to-end Example

- [ ] At least one runnable example exists in `examples/` or `docs/examples/` that demonstrates: loading data → constructing the model → fitting → predicting.
- [ ] The example runs to completion without error against the current main branch.

### 4. Save / Load Support

- [ ] The model can be saved and reloaded via the standard DeepTab save/load mechanism (if save/load is part of the stable core contract for the task type).
- [ ] A round-trip test confirms predictions are identical before and after reload.

### 5. Tests

- [ ] A behavioral test file exists (or an entry in `tests/test_base.py`) that covers:
  - Fit on a small synthetic dataset (classification and/or regression as applicable).
  - Predict returns an array of the expected shape and dtype.
  - Config serialization round-trip.
- [ ] All tests pass in CI (`just test`).

### 6. No Open Critical Bugs

- [ ] No open GitHub issues tagged `bug` + `component:<model-name>` describe a failure in a core workflow (fit, predict, save/load).
- [ ] If known limitations exist that are not bugs (e.g. performance on very small datasets), they are documented in the "Limitations" section of the model page.

### 7. Config Registration

- [ ] A config class exists in `deeptab/configs/` and is exported from `deeptab/configs/__init__.py`.
- [ ] The model is exported from `deeptab/models/__init__.py`.
- [ ] The model is listed in `deeptab/utils/config_mapper.py` (or equivalent registry) so it is discoverable programmatically.

---

## Promotion PR Requirements

Open a PR titled `feat(<model-name>): promote to stable` and include the checklist above in the PR description before requesting review. The PR must:

1. Remove any `.. experimental::` admonition from the model's doc page.
2. Remove any `ExperimentalWarning` raised in `__init__` or `fit`.
3. Update the model's entry in the API reference to remove the experimental badge.
4. Add the model to the changelog under `### Promoted to Stable`.

Approval requires **at least one maintainer review** beyond the author.

---

## Demoting a Stable Model Back to Experimental

Demotion is rare and only warranted if a critical correctness bug is discovered that cannot be fixed without breaking the API. Process:

1. Open an issue describing the correctness failure and tag it `regression` + `breaking`.
2. Add an `ExperimentalWarning` in the next patch release with a message pointing to the issue.
3. Fix the underlying problem; re-promote via the standard checklist.

Demotion itself is **not** a breaking change because experimental has no stability guarantee; however it must be prominently documented in the release notes.
