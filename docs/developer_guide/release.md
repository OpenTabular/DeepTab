# Build and Release

The document outlines the steps to build and release `deeptab` package. It is assumed that all feature branches and PRs for the release have been reviewed, approved, and merged into `main` before starting this process.

## Release workflow

```{mermaid}
%%{init: {'theme': 'base', 'flowchart': {'useMaxWidth': false}, 'themeVariables': {
  'primaryColor': '#dbeafe',
  'primaryTextColor': '#1e3a5f',
  'primaryBorderColor': '#3b82f6',
  'lineColor': '#6b7280',
  'secondaryColor': '#ede9fe',
  'tertiaryColor': '#f0fdf4',
  'edgeLabelBackground': '#f9fafb'
}}}%%
flowchart LR
    A[Create release branch]:::git --> B[Hotfixes and doc updates]:::setup
    B --> QA["Quality checks<br/>lint, format, hooks, tests"]:::ci
    QA --> QAP{All pass?}:::decision
    QAP -->|No| B
    QAP -->|Yes| D[Build docs]:::setup
    D --> E[Commit changes and push branch]:::git
    E --> F{Release type?}:::decision
    F -->|RC| RC1[Preview and bump RC]:::setup
    F -->|Stable| ST1[Preview and bump stable]:::setup
    RC1 --> RC2[Push branch and exact RC tag]:::git
    RC2 --> RC3[CI: publish-testpypi.yml]:::ci
    RC3 --> RC4[TestPyPI + GitHub pre-release]:::rc
    RC4 -->|Issues found| B
    RC4 -->|RC approved| ST1
    ST1 --> ST2["Open PR: release/vX.Y.Z → main"]:::pr
    ST2 --> ST3{Review and approve}:::decision
    ST3 --> ST4[Merge PR into main]:::pr
    ST4 --> ST5["Create annotated stable tag<br/>on merged main and push"]:::git
    ST5 --> ST6[CI: publish-pypi.yml]:::ci
    ST6 --> ST7[PyPI + GitHub Release]:::stable

    classDef setup    fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef pr       fill:#ede9fe,stroke:#8b5cf6,color:#3b0764
    classDef decision fill:#fef9c3,stroke:#ca8a04,color:#713f12
    classDef git      fill:#f0fdf4,stroke:#22c55e,color:#14532d
    classDef ci       fill:#fff7ed,stroke:#f97316,color:#7c2d12
    classDef rc       fill:#fdf4ff,stroke:#d946ef,color:#701a75
    classDef stable   fill:#ecfdf5,stroke:#10b981,color:#064e3b
```

## 1. Create the release branch

After all PRs for the release are merged into `main`, create a dedicated release branch.

```{important}
Always branch off from an up-to-date `main`. Never start a release from a stale local copy.
```

```bash
git checkout main && git pull --ff-only
git checkout -b release/vX.Y.Z
git push -u origin release/vX.Y.Z
```

Use Python 3.12 or 3.13 for release checks, since the documentation toolchain
requires it. Install the development and documentation groups with
`poetry install --with dev,docs`. Prepare and validate optional dependency
profiles separately as described in [Testing](testing.md).

## 2. Apply hotfixes and documentation updates

Use the release branch to apply any last-minute bug fixes, dependency security patches, or documentation updates required before the release.

```{note}
Only **bug fixes and documentation changes** belong on the release branch. New features must target the next release cycle via `main`.
```

```{warning}
For a dependency managed in `pyproject.toml`, adjust its constraint if necessary
and update the affected package rather than upgrading the entire environment:

    poetry update "<package>"

Review the resulting `poetry.lock` changes and run the relevant tests.
Read the Docs installs `docs/requirements_docs.txt` separately from Poetry;
update affected pins there too. A patched Poetry lock does not repair a stale
Read the Docs requirement.
```

**Security audit:** run `just audit` and resolve any vulnerability with an available fix before bumping the version:

```bash
just audit
```

A missing upstream fix is not automatic risk acceptance. Follow the
{ref}`dependency security policy <dependency-security>`: keep release sign-off
open until a compatible fix is applied or a maintainer records explicit
approval of the risk for the affected versions and advisories.

## 3. Quality checks

Run all checks in the order shown below. **Each step must pass cleanly before proceeding to the next.**

| Step   | Command       | What it does                                                          |
| ------ | ------------- | --------------------------------------------------------------------- |
| Lint   | `just lint`   | `ruff check --fix`; review and resolve any errors it cannot auto-fix. |
| Format | `just format` | `ruff format` for consistent code style.                              |
| Hooks  | `just check`  | All pre-commit hooks across all files: ruff, prettier, and Pyright.   |
| Tests  | `just test`   | Full test suite with at least 90% line coverage.                      |

```{important}
If `just check` modifies any files, stage and commit them before continuing:

    git add -u && git commit -m "style: apply pre-commit formatting"
```

```{warning}
A test failure at this stage must be fixed on the release branch before proceeding. Do not skip, suppress, or comment out failing tests.
```

## 4. Documentation

Build the HTML docs locally. Sphinx treats all warnings as errors (`-W`), so every warning must be resolved before proceeding.

```bash
just docs
```

Review the rendered output in `docs/_build/html/`. Check for broken links, missing API entries, and any rendering issues on new or changed pages.

```{note}
See the **[Documentation Guide](documentation.md)** for docstring conventions and tips on building docs locally.
```

## 5. Commit all changes

Once all checks and the documentation build pass cleanly, stage and commit any outstanding changes:

```bash
git add -A
git commit -m "chore(release): pre-release fixes and QA for vX.Y.Z"
git push origin release/vX.Y.Z
```

```{note}
Prefer `just commit` over a manual `git commit` to stay consistent with the conventional commit style enforced by commitizen.
```

After pushing the release branch, wait for the full hosted CI run, including
**CI Success**, before applying the version bump. Confirm that the run validated
the pushed commit. See [CI/CD](ci_cd.md) for required-check configuration; local
checks do not replace the hosted platform and dependency matrices.

## 6. Version bump

The version bump is driven entirely by [Commitizen](https://commitizen-tools.github.io/commitizen/). The increment (MAJOR / MINOR / PATCH) is inferred from the conventional-commit messages since the last tag, the version in `pyproject.toml` is updated, and the matching `CHANGELOG.md` section is generated and committed with a tag. The current recipes create lightweight tags unless `--annotated-tag` is supplied.

```{important}
Always run `--dry-run` first and review the proposed version and CHANGELOG entries carefully before applying the bump.
```

The following Commitizen settings in `pyproject.toml` shape this behaviour:

| Setting                      | Value  | Effect                                                                                        |
| ---------------------------- | ------ | --------------------------------------------------------------------------------------------- |
| `prerelease_offset`          | `1`    | Prereleases start at `rc1` (not the default `rc0`)                                            |
| `changelog_merge_prerelease` | `true` | On the stable bump, all `rcN` CHANGELOG sections are rolled up into a single `vX.Y.Z` section |
| `update_changelog_on_bump`   | `true` | `CHANGELOG.md` is regenerated automatically on every bump                                     |

The bump commands are wrapped in `just` recipes so the correct Commitizen flags are applied consistently:

| Recipe                 | Wraps                               | Use for               |
| ---------------------- | ----------------------------------- | --------------------- |
| `just bump-rc-preview` | `cz bump --prerelease rc --dry-run` | Preview an RC bump    |
| `just bump-rc`         | `cz bump --prerelease rc`           | Apply an RC bump      |
| `just bump-preview`    | `cz bump --dry-run`                 | Preview a stable bump |
| `just bump`            | `cz bump`                           | Apply a stable bump   |

Each recipe forwards extra arguments to Commitizen, so flags such as `--increment MAJOR` can be appended directly (e.g. `just bump-rc-preview --increment MAJOR`).

### 6a. Release candidate bump

Use the `bump-rc` recipes to cut an RC. The first prerelease of a cycle is `rc1` (thanks to `prerelease_offset = 1`); subsequent ones increment to `rc2`, `rc3`, and so on.

```{note}
Commitizen can only infer the target version from commits. For a **major** release where the commit history does not contain a `feat!:` / `BREAKING CHANGE:` marker, force the increment explicitly with `--increment MAJOR` (likewise `MINOR` / `PATCH`).
```

**Step 1, preview:**

```bash
# Append --increment MAJOR/MINOR/PATCH if the bump is not inferred correctly
just bump-rc-preview
```

Confirm the proposed tag (e.g. `v2.0.0rc1`) and that the CHANGELOG entries are complete and correctly classified.

**Step 2, apply:**

```bash
just bump-rc
```

This updates `version` in `pyproject.toml`, appends the `vX.Y.ZrcN` section to `CHANGELOG.md`, creates the `bump:` commit, and creates the `vX.Y.ZrcN` tag locally.

**Step 3, review and push the branch and exact RC tag:**

```bash
git show HEAD
git push origin release/vX.Y.Z
git push origin vX.Y.ZrcN
```

```{note}
Pushing the exact RC tag triggers `publish-testpypi.yml`. Do not rely on
`git push --follow-tags` with the default bump recipes: it pushes annotated
tags, not the lightweight tag created by default. Explicitly pushing the
reviewed tag works with either tag type and avoids publishing unrelated tags.
```

### 6b. Stable bump

When all RCs are approved, cut the final stable version with the plain `bump` recipes. Because `changelog_merge_prerelease = true`, the intermediate `rcN` sections are merged into a single complete `vX.Y.Z` CHANGELOG section, so end users see the full release notes in one place.

```bash
just bump-preview   # append --increment MAJOR if needed
just bump
```

This updates `pyproject.toml`, regenerates `CHANGELOG.md` (with prereleases merged), and creates the `bump:` commit plus the `vX.Y.Z` tag.

**Review the bump commit:**

```bash
git show HEAD
```

Check that `pyproject.toml` shows the correct version and that `CHANGELOG.md`
reads cleanly. Manually amend duplicate entries if present, then push the
branch. Push the stable tag only after the release PR is merged, as described
in step 9:

```bash
git push origin release/vX.Y.Z
```

See **[Versioning](versioning.md)** for the full SemVer rules and commit-type reference.

## 7. Tag and publish a release candidate

`cz bump --prerelease rc` (step 6a) already created the `vX.Y.ZrcN` tag locally. RC tags are pushed **directly from the release branch**, with no PR to `main` required.

If you explicitly pushed the RC tag in step 6a, it is already live and you can
skip to verifying CI. Otherwise, review and push that exact tag:

```bash
git push origin vX.Y.ZrcN
```

This triggers `publish-testpypi.yml`, which publishes to **TestPyPI** and creates a GitHub pre-release.

Before CI, documentation, or package checks start, the workflow validates the
exact `vX.Y.ZrcN` format and requires the tagged commit to be reachable from the
matching remote `release/vX.Y.Z` branch. Push the branch before the tag, and keep
it available until publication completes. A different release branch does not
satisfy this check.

If issues are found, fix them on the release branch (return to step 2), bump to the next RC with `cz bump --prerelease rc` (which yields `rcN+1`), and repeat.

## 8. Release PR (stable only)

Once all RCs are approved (or skipping RC for a straightforward release), open a PR from `release/vX.Y.Z` to `main` on GitHub.

- After review and approval, merge the PR
- **Merging to `main` does NOT trigger a PyPI release**

```{important}
**Promoting a model from experimental to stable?** Verify the [Model Promotion Policy](model_promotion_policy.md) checklist is complete before merging a release PR that includes a promotion.
```

## 9. Create and push the stable tag

The stable `cz bump` in step 6b created a local `vX.Y.Z` tag on the release branch, but the authoritative tag must point at the merge commit on `main`. After the release PR is merged into `main`, drop the local tag and re-create it on `main`:

```bash
git checkout main && git pull --ff-only
git tag -d vX.Y.Z                    # remove the local tag created by cz bump
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

```{warning}
Pushing the tag starts the release validation workflow. Publication runs only after CI, documentation, and package checks succeed. Once the package is uploaded, that release cannot be overwritten. Confirm that `main` is in the expected state and the version in `pyproject.toml` is correct before pushing.
```

The stable workflow accepts only the exact `vX.Y.Z` format and requires the tag's
commit to be reachable from remote `main`. It does not require that commit to
remain the tip of `main`, so later commits do not invalidate the release.

## 10. Publish package

The tag push automatically triggers the appropriate GitHub Actions workflow. Both first validate the tag and required remote branch ancestry, then require the full CI test matrix and coverage collection, a successful Sphinx build, and package validation including classifier/regressor training and save/load from an installed wheel. Any failed, cancelled, or skipped prerequisite blocks publication. Both use **OIDC Trusted Publishing**, so no API tokens are required. The publish job downloads the validated distributions from the same run rather than rebuilding them. See **[CI/CD](ci_cd.md)** for full details.

| Tag pushed  | Workflow               | Result                        |
| ----------- | ---------------------- | ----------------------------- |
| `vX.Y.Z`    | `publish-pypi.yml`     | PyPI + GitHub Release         |
| `vX.Y.ZrcN` | `publish-testpypi.yml` | TestPyPI + GitHub pre-release |

## 11. GitHub Release

The GitHub Release is created automatically by the publish workflow. Once it appears, verify that:

- The release notes reflect the correct CHANGELOG section
- Assets (wheel and sdist) are attached
- The release is marked stable (not pre-release) for a stable tag

Add any manual context or migration notes to the release description if needed.
