# Versioning

DeepTab follows [Semantic Versioning 2.0](https://semver.org/) and uses [Conventional Commits](https://www.conventionalcommits.org/) to automate version bumps and changelog generation via [commitizen](https://commitizen-tools.github.io/commitizen/).

## Version format

```
MAJOR.MINOR.PATCH
```

| Segment | When it increments                                                         |
| ------- | -------------------------------------------------------------------------- |
| `MAJOR` | Breaking change (`feat!:` or `BREAKING CHANGE:` footer)                    |
| `MINOR` | New backwards-compatible feature (`feat:`)                                 |
| `PATCH` | Backwards-compatible bug fix (`fix:`) or performance improvement (`perf:`) |

Release candidates use the suffix `rcN`, e.g. `1.8.0rc1`.

The version is defined **in one place only**, `pyproject.toml`, and read at runtime via `importlib.metadata` in `deeptab/_version.py`, so it never needs to be hard-coded in the package.

## Commit types and their effect

| Commit type | Example                                     | Version bump |
| ----------- | ------------------------------------------- | ------------ |
| `feat`      | `feat(models): add TabM architecture`       | Minor        |
| `fix`       | `fix(datamodule): resolve NaN propagation`  | Patch        |
| `perf`      | `perf(transformer): fuse attention kernels` | Patch        |
| `feat!`     | `feat!: remove Python 3.9 support`          | Major        |
| `docs`      | `docs: update API reference`                | None         |
| `test`      | `test: add save/load round-trip test`       | None         |
| `ci`        | `ci: add Python 3.13 to matrix`             | None         |
| `refactor`  | `refactor: simplify config validation`      | None         |
| `style`     | `style: apply ruff formatting`              | None         |
| `chore`     | `chore: update pre-commit revisions`        | None         |

Commit messages that do not match any of these types do not trigger a version bump.

## Making a conventional commit

Use commitizen's interactive prompt rather than writing the message by hand:

```bash
just commit      # opens the cz commit wizard
```

Or write the message directly:

```bash
git commit -m "feat(models): add NODE architecture"
git commit -m "fix(configs): validate n_layers > 0"
```

The `commit-msg` pre-commit hook validates every commit message against the conventional commits format and rejects non-conforming messages.

## Bumping the version

Version bumps are driven by [commitizen](https://commitizen-tools.github.io/commitizen/), wrapped in `just` recipes. Preview first with the `-preview` (dry-run) variant, then apply. Each apply recipe updates `version` in `pyproject.toml`, appends to `CHANGELOG.md`, and creates the bump commit and tag.

| Goal              | Preview                | Apply          |
| ----------------- | ---------------------- | -------------- |
| Stable release    | `just bump-preview`    | `just bump`    |
| Release candidate | `just bump-rc-preview` | `just bump-rc` |

The next version is inferred from the conventional commits since the last tag. To force a level when it is not auto-detected, append the increment, e.g. `just bump --increment MAJOR`.

## Changelog

`CHANGELOG.md` at the repository root is the authoritative changelog, updated automatically by the bump recipes. Changes are grouped under their commit types (`feat`, `fix`, `perf`, ...) with the subject line of every matching commit since the previous release.

## Tags

Release tags follow `vMAJOR.MINOR.PATCH` (or `vMAJOR.MINOR.PATCHrcN` for RCs) and trigger the PyPI publish workflows. See the [Release process](release.md) page for the full end-to-end procedure.
