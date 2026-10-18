# Versioning

DeepTab follows [Semantic Versioning 2.0](https://semver.org/) and uses [Conventional Commits](https://www.conventionalcommits.org/) to automate version bumps and changelog generation via [commitizen](https://commitizen-tools.github.io/commitizen/).

## Version format

```
MAJOR.MINOR.PATCH
```

| Segment | When it increments                                                                                          |
| ------- | ----------------------------------------------------------------------------------------------------------- |
| `MAJOR` | Public API break or removal of an upstream-supported Python version (`feat!:` or `BREAKING CHANGE:` footer) |
| `MINOR` | Backwards-compatible feature or support update permitted by the policy below (`feat:`)                      |
| `PATCH` | Backwards-compatible bug fix (`fix:`) or performance improvement (`perf:`)                                  |

Release candidates use the suffix `rcN`, e.g. `1.8.0rc1`.

The version is defined **in one place only**, `pyproject.toml`, and read at runtime via `importlib.metadata` in `deeptab/_version.py`, so it never needs to be hard-coded in the package.

## Python and dependency support

DeepTab distinguishes public API compatibility from the supported runtime and
dependency window:

| Change                                           | Release level                                                                                  |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Breaking public API change                       | Major.                                                                                         |
| Remove a Python version still supported upstream | Major.                                                                                         |
| Remove an end-of-life Python version             | Minor is allowed when required by the Scientific Python ecosystem or DeepTab dependencies.     |
| Raise a dependency minimum                       | Minor is allowed if the public API remains compatible and retained Python versions still work. |

Announce support changes, their reasons, and migration requirements in the release
notes. Update package metadata and CI together, and validate the supported
combinations. Retiring an EOL Python version must not happen in a patch release.

```{important}
These exceptions do not permit breaking DeepTab's public API in a minor release
or using a dependency update to drop an upstream-supported Python version.
```

[SPEC 0](https://scientific-python.org/specs/spec-0000/) informs dependency
retirement windows; the release-level rules above are DeepTab's policy, not a
SemVer requirement imposed by SPEC 0. Upstream Python support status is listed
in the [Python Developer's Guide](https://devguide.python.org/versions/).

## Commit types and their effect

| Commit type | Example                                      | Version bump |
| ----------- | -------------------------------------------- | ------------ |
| `feat`      | `feat(models): add TabM architecture`        | Minor        |
| `feat`      | `feat(compat): retire EOL Python 3.10`       | Minor        |
| `fix`       | `fix(datamodule): resolve NaN propagation`   | Patch        |
| `perf`      | `perf(transformer): fuse attention kernels`  | Patch        |
| `feat!`     | `feat!: remove a public estimator parameter` | Major        |
| `docs`      | `docs: update API reference`                 | None         |
| `test`      | `test: add save/load round-trip test`        | None         |
| `ci`        | `ci: add Python 3.13 to matrix`              | None         |
| `refactor`  | `refactor: simplify config validation`       | None         |
| `style`     | `style: apply ruff formatting`               | None         |
| `chore`     | `chore: update pre-commit revisions`         | None         |

Commit messages that do not match any of these types do not trigger a version bump.

Use a non-breaking `feat(compat):` message for an eligible EOL support retirement
or dependency-floor change. Commitizen reads commit messages, not Python's EOL
status; review the proposed increment with `just bump-rc-preview` before applying it.

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
