# Conventional Commits Quick Reference

## Commit Format

```
<type>(<scope>): <subject>
```

## Types

| Type       | Description             | Version Bump  |
| ---------- | ----------------------- | ------------- |
| `feat`     | New feature             | Minor (0.x.0) |
| `fix`      | Bug fix                 | Patch (0.0.x) |
| `docs`     | Documentation only      | None          |
| `style`    | Code style/formatting   | None          |
| `refactor` | Code refactoring        | None          |
| `perf`     | Performance improvement | Patch         |
| `test`     | Adding/updating tests   | None          |
| `build`    | Build system changes    | None          |
| `ci`       | CI/CD changes           | None          |
| `chore`    | Other changes           | None          |

## Examples

```bash
# Feature (minor bump: 1.6.1 → 1.7.0)
git commit -m "feat(models): add TabNet architecture"

# Bug fix (patch bump: 1.6.1 → 1.6.2)
git commit -m "fix(datamodule): resolve memory leak in batch loading"

# Performance (patch bump)
git commit -m "perf(transformer): optimize attention computation"

# Documentation (no bump)
git commit -m "docs: update API reference for MambaTab"

# Breaking change (major bump: 1.6.1 → 2.0.0)
git commit -m "feat!: remove Python 3.9 support

BREAKING CHANGE: Python 3.10+ is now required"
```

## Scopes (Optional)

Common scopes in this project:

- `models`: Model implementations
- `configs`: Configuration classes
- `data`: Data utilities and dataloaders
- `arch`: Architecture utilities
- `utils`: General utilities
- `ci`: CI/CD related
- `deps`: Dependencies

## Quick Commands

```bash
# Interactive commit (recommended)
just commit

# Version bump
just bump

# View changelog
cat CHANGELOG.md

# Dry-run semantic release
just release-dry
```

## Breaking Changes

Use `!` after type and explain in footer:

```
feat!: change API signature

BREAKING CHANGE: The `fit()` method now requires `x_train` and `y_train` as separate arguments instead of a tuple.
```

## Multi-line Commits

```bash
# Using editor
git commit

# In editor:
feat(models): add multi-head attention support

This commit introduces multi-head attention mechanism
to improve model performance on large datasets.

Closes #123
```

## Pre-commit Hook

Commits are validated automatically. If rejected:

1. Check format: `type(scope): description`
2. Use allowed types only
3. Keep header under 72 characters
4. Don't end subject with period
