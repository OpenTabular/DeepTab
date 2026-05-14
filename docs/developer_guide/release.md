# Build and release

The document outlines the steps to build and release the `deeptab` package. At this point, it is assumed that the development and testing of the package have been completed successfully.

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
    A[Create release/vX.Y.Z branch]:::setup --> B{Release type?}:::decision
    B -->|RC| RC1[Bump version e.g. 1.7.0rc1]:::setup
    B -->|Stable| ST1[Bump version e.g. 1.7.0]:::setup
    RC1 --> RC2[Update CHANGELOG.md]:::setup
    RC2 --> RC3[Commit & push branch]:::setup
    RC3 --> RC4["git tag vX.Y.ZrcN<br/>git push origin vX.Y.ZrcN"]:::git
    RC4 --> RC5[CI: publish-testpypi.yml]:::ci
    RC5 --> RC6[TestPyPI + GitHub pre-release]:::rc
    RC6 -->|Issues found| RC1
    RC6 -->|RC approved| ST1
    ST1 --> ST2[Update CHANGELOG.md]:::setup
    ST2 --> ST3[Commit & push branch]:::setup
    ST3 --> ST4[Open PR: release/vX.Y.Z → main]:::pr
    ST4 --> ST5{Review & approve}:::decision
    ST5 --> ST6[Merge PR into main]:::pr
    ST6 --> ST7[git checkout main && git pull]:::git
    ST7 --> ST8["git tag vX.Y.Z<br/>git push origin vX.Y.Z"]:::git
    ST8 --> ST9[CI: publish-pypi.yml]:::ci
    ST9 --> ST10[PyPI + GitHub Release]:::stable

    classDef setup   fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef pr      fill:#ede9fe,stroke:#8b5cf6,color:#3b0764
    classDef decision fill:#fef9c3,stroke:#ca8a04,color:#713f12
    classDef git     fill:#f0fdf4,stroke:#22c55e,color:#14532d
    classDef ci      fill:#fff7ed,stroke:#f97316,color:#7c2d12
    classDef rc      fill:#fdf4ff,stroke:#d946ef,color:#701a75
    classDef stable  fill:#ecfdf5,stroke:#10b981,color:#064e3b
```

## 1. Test documentation

It is expected from the contributor to update the documentation as an when required along side the change in source code. Please use the below process to test the documentation:

```sh
cd docs/

make doctest
```

Fix any docstring related issue, then proceed with next steps.

## 2. Version update

The package version is maintained in `pyproject.toml` only. The version is read at runtime via `importlib.metadata`.

On a `release/vX.Y.Z` branch:

**For a stable release** — let `cz bump` derive the next version automatically from conventional commits:

```bash
cz bump
```

This will:

- Determine the next version from conventional commits since the last tag
- Update the version in `pyproject.toml`
- Update `CHANGELOG.md`
- Create a local commit `bump: version X.Y.Z-1 → X.Y.Z`

**For a release candidate** — set the version explicitly (e.g. `1.7.0rc1`):

```bash
poetry version 1.7.0rc1
poetry lock
```

Then update `CHANGELOG.md` manually and commit:

```bash
git add pyproject.toml poetry.lock CHANGELOG.md
git commit -m "bump: version 1.6.1 → 1.7.0rc1"
```

The version number follows the format `major.minor.patch` for stable, or `major.minor.patchrcN` for release candidates.

## 3. Tag and publish a release candidate

RC tags are pushed **directly from the release branch** — no PR to `main` required.

```bash
git tag -a vX.Y.ZrcN -m "Release candidate vX.Y.ZrcN"
git push origin vX.Y.ZrcN
```

This triggers `publish-testpypi.yml`, which publishes to **TestPyPI** and creates a GitHub pre-release.

If issues are found, fix them on the release branch, bump to the next RC (`rcN+1`), and repeat.

## 4. Release PR (stable only)

Once all RCs are approved, open a PR from `release/vX.Y.Z` to `main`:

- After review and approval, merge the PR
- **Merging to `main` does NOT trigger a PyPI release**

> **Promoting a model from experimental to stable?** Verify the [Model Promotion Policy](model_promotion_policy.md) checklist is complete before merging a release PR that includes a promotion.

## 5. Create and push the stable tag

After the release PR is merged:

```bash
git checkout main && git pull
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

## 6. Publish package to PyPI

The tag push automatically triggers the appropriate workflow in GitHub Actions:

- Stable tag (`vX.Y.Z`) → `publish-pypi.yml` → publishes to **PyPI** + creates GitHub Release
- RC tag (`vX.Y.ZrcN`) → `publish-testpypi.yml` → publishes to **TestPyPI** + creates GitHub pre-release

Both workflows:

- Build the package (`poetry build`)
- Validate with `twine check`
- Publish via **OIDC Trusted Publishing** (no API tokens required)
- Create a GitHub Release with auto-generated notes

> **Note:** A `pypi-publish` GitHub Environment (for stable) and `testpypi-publish` environment (for RCs) must be configured with tag-based deployment protection rules.

## 7. GitHub Release

The GitHub Release is created automatically by the publish workflow. Verify the release notes are correct and add any manual context if needed.
