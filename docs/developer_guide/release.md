# Build and Release

The document outlines the steps to build and release the `deeptab` package. It is assumed that all feature branches and PRs for the release have been reviewed, approved, and merged into `main` before starting this process.

## Release workflow

```{mermaid}
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#dbeafe',
  'primaryTextColor': '#1e3a5f',
  'primaryBorderColor': '#3b82f6',
  'lineColor': '#6b7280',
  'secondaryColor': '#ede9fe',
  'tertiaryColor': '#f0fdf4',
  'edgeLabelBackground': '#f9fafb'
}}}%%
flowchart TD
    A["git checkout main &#38;&#38; git pull<br/>git checkout -b release/vX.Y.Z"]:::git --> B[Hotfixes &#38; doc updates]:::setup
    B --> QA["Quality checks<br/>lint → format → check → test"]:::ci
    QA --> QAP{All pass?}:::decision
    QAP -->|No| B
    QAP -->|Yes| D["Build docs: just docs"]:::setup
    D --> E[Commit changes &#38; push branch]:::git
    E --> F{Release type?}:::decision
    F -->|RC| RC1["just bump-rc-preview<br/>then just bump-rc"]:::setup
    F -->|Stable| ST1["just bump-preview<br/>then just bump"]:::setup
    RC1 --> RC2["git push --follow-tags<br/>(pushes vX.Y.ZrcN)"]:::git
    RC2 --> RC3[CI: publish-testpypi.yml]:::ci
    RC3 --> RC4[TestPyPI + GitHub pre-release]:::rc
    RC4 -->|Issues found| B
    RC4 -->|RC approved| ST1
    ST1 --> ST2["Open PR: release/vX.Y.Z → main"]:::pr
    ST2 --> ST3{Review &#38; approve}:::decision
    ST3 --> ST4[Merge PR into main]:::pr
    ST4 --> ST5["git checkout main &#38;&#38; git pull<br/>git tag -d vX.Y.Z (local)<br/>git tag vX.Y.Z &#38;&#38; git push origin vX.Y.Z"]:::git
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
git checkout main && git pull
git checkout -b release/vX.Y.Z
git push -u origin release/vX.Y.Z
```

## 2. Apply hotfixes and documentation updates

Use the release branch to apply any last-minute bug fixes, dependency security patches, or documentation updates required before the release.

```{note}
Only **bug fixes and documentation changes** belong on the release branch. New features must target the next release cycle via `main`.
```

```{warning}
If you update any dependencies (e.g. to resolve security findings), regenerate the lock file immediately:

    poetry update <package>

Then verify the change does not break any tests.
```

**Security audit:** run `just audit` and resolve any vulnerability with an available fix before bumping the version:

```bash
just audit
```

Vulnerabilities with no upstream fix available should be noted and tracked as known accepted risks.

## 3. Quality checks

Run all checks in the order shown below. **Each step must pass cleanly before proceeding to the next.**

### 3.1 Linting

```bash
just lint
```

Runs `ruff check --fix` and auto-corrects fixable issues. Review and manually resolve any remaining errors.

### 3.2 Formatting

```bash
just format
```

Runs `ruff format` to ensure consistent code style across the codebase.

### 3.3 Pre-commit hooks

```bash
just check
```

Runs all pre-commit hooks across all files: ruff lint/format, prettier (YAML/Markdown/JSON), and Pyright type checking.

```{important}
If `just check` modifies any files, stage and commit them before continuing:

    git add -u && git commit -m "style: apply pre-commit formatting"
```

### 3.4 Unit tests

```bash
just test
```

Runs the full test suite with coverage reporting.

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

## 6. Version bump

The version bump is driven entirely by [Commitizen](https://commitizen-tools.github.io/commitizen/). The increment (MAJOR / MINOR / PATCH) is inferred from the conventional-commit messages since the last tag, the version in `pyproject.toml` is updated, and the matching `CHANGELOG.md` section is generated and committed with an annotated tag.

```{important}
Always run `--dry-run` first and review the proposed version and CHANGELOG entries carefully before applying the bump.
```

The following Commitizen settings in `pyproject.toml` shape this behaviour:

| Setting | Value | Effect |
| --- | --- | --- |
| `prerelease_offset` | `1` | Prereleases start at `rc1` (not the default `rc0`) |
| `changelog_merge_prerelease` | `true` | On the stable bump, all `rcN` CHANGELOG sections are rolled up into a single `vX.Y.Z` section |
| `update_changelog_on_bump` | `true` | `CHANGELOG.md` is regenerated automatically on every bump |

The bump commands are wrapped in `just` recipes so the correct Commitizen flags are applied consistently:

| Recipe | Wraps | Use for |
| --- | --- | --- |
| `just bump-rc-preview` | `cz bump --prerelease rc --dry-run` | Preview an RC bump |
| `just bump-rc` | `cz bump --prerelease rc` | Apply an RC bump |
| `just bump-preview` | `cz bump --dry-run` | Preview a stable bump |
| `just bump` | `cz bump` | Apply a stable bump |

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

**Step 3, review and push** (commit and tag together):

```bash
git show HEAD
git push --follow-tags origin release/vX.Y.Z
```

```{note}
`--follow-tags` pushes the annotated `vX.Y.ZrcN` tag along with the branch in one step, which triggers `publish-testpypi.yml`. See **[Tag and publish a release candidate](#7-tag-and-publish-a-release-candidate)** below if you prefer to push the tag separately.
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

Check that `pyproject.toml` shows the correct version and that `CHANGELOG.md` reads cleanly. Manually amend duplicate entries if present, then push the branch (the stable tag is pushed later, after the release PR is merged — see step 9):

```bash
git push origin release/vX.Y.Z
```

See **[Versioning](versioning.md)** for the full SemVer rules and commit-type reference.

## 7. Tag and publish a release candidate

`cz bump --prerelease rc` (step 6a) already created the annotated `vX.Y.ZrcN` tag locally. RC tags are pushed **directly from the release branch**, with no PR to `main` required.

If you pushed with `git push --follow-tags` in step 6a, the tag is already live and you can skip to verifying CI. Otherwise, push the tag explicitly:

```bash
git push origin vX.Y.ZrcN
```

This triggers `publish-testpypi.yml`, which publishes to **TestPyPI** and creates a GitHub pre-release.

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
git checkout main && git pull
git tag -d vX.Y.Z                    # remove the local tag created by cz bump
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

```{warning}
Pushing the tag triggers PyPI publication immediately and cannot be undone. Confirm that `main` is in the expected state and the version in `pyproject.toml` is correct before pushing.
```

## 10. Publish package

The tag push automatically triggers the appropriate GitHub Actions workflow. See **[CI/CD](ci_cd.md)** for full details. In summary:

- Stable tag (`vX.Y.Z`) → `publish-pypi.yml` → PyPI + GitHub Release
- RC tag (`vX.Y.ZrcN`) → `publish-testpypi.yml` → TestPyPI + GitHub pre-release

Both workflows use **OIDC Trusted Publishing**, so no API tokens are required.

## 11. GitHub Release

The GitHub Release is created automatically by the publish workflow. Once it appears, verify that:

- The release notes reflect the correct CHANGELOG section
- Assets (wheel and sdist) are attached
- The release is marked stable (not pre-release) for a stable tag

Add any manual context or migration notes to the release description if needed.
