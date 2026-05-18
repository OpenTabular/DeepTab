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

The version is defined **in one place only** — `pyproject.toml` — and read at runtime via `importlib.metadata`:

```python
from importlib.metadata import version
__version__ = version("deeptab")
```

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

On a `release/vX.Y.Z` branch, let commitizen determine the next version automatically:

```bash
cz bump
```

This command:

1. Reads all conventional commits since the last tag.
2. Determines the next version (`MAJOR`, `MINOR`, or `PATCH`).
3. Updates `version` in `pyproject.toml`.
4. Appends the new section to `CHANGELOG.md`.
5. Creates a local commit `bump: version X.Y.Z-1 → X.Y.Z`.

For a release candidate, set the version explicitly instead:

```bash
poetry version 1.8.0rc1
poetry lock
git add pyproject.toml poetry.lock CHANGELOG.md
git commit -m "bump: version 1.7.0 → 1.8.0rc1"
```

## Changelog

`CHANGELOG.md` at the repository root is the authoritative changelog. It is updated automatically by `cz bump` on stable releases. For release candidates, update it manually before tagging.

The changelog format groups changes under the commit types (`feat`, `fix`, `perf`, etc.) and lists the subject line of every matching commit since the previous release.

## Tags

All release tags follow the format `vMAJOR.MINOR.PATCH` (or `vMAJOR.MINOR.PATCHrcN` for RCs). Tags are what trigger the PyPI publish workflows — see the [Release process](release.md) page for the full end-to-end procedure.
