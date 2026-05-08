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
    A[Create release/vX.Y.Z branch]:::setup --> B[Bump version]:::setup
    B --> C[Update CHANGELOG.md]:::setup
    C --> D[Commit & push branch]:::setup
    D --> E[Open PR: release/vX.Y.Z → main]:::pr
    E --> F{Review & approve}:::decision
    F --> G[Merge PR into main]:::pr
    G --> H[git checkout main && git pull]:::git
    H --> I{Release type?}:::decision
    I -->|RC| J["git tag vX.Y.ZrcN\ngit push origin vX.Y.ZrcN"]:::git
    I -->|Stable| K["git tag vX.Y.Z\ngit push origin vX.Y.Z"]:::git
    J --> L[CI: publish-testpypi.yml]:::ci
    K --> M[CI: publish-pypi.yml]:::ci
    L --> N[TestPyPI + GitHub pre-release]:::rc
    M --> O[PyPI + GitHub Release]:::stable

    classDef setup  fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef pr     fill:#ede9fe,stroke:#8b5cf6,color:#3b0764
    classDef decision fill:#fef9c3,stroke:#ca8a04,color:#713f12
    classDef git    fill:#f0fdf4,stroke:#22c55e,color:#14532d
    classDef ci     fill:#fff7ed,stroke:#f97316,color:#7c2d12
    classDef rc     fill:#fdf4ff,stroke:#d946ef,color:#701a75
    classDef stable fill:#ecfdf5,stroke:#10b981,color:#064e3b
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

## 3. Release PR

- Open a PR from `release/vX.Y.Z` to `main`
- After review and approval, merge the PR
- **Merging to `main` does NOT trigger a PyPI release**

> **Promoting a model from experimental to stable?** Verify the [Model Promotion Policy](model_promotion_policy.md) checklist is complete before merging a release PR that includes a promotion.

## 4. Create and push the Git tag

After the release PR is merged, the maintainer creates the release tag:

```bash
git checkout main && git pull
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

For a release candidate:

```bash
git tag -a vX.Y.Zrc1 -m "Release candidate vX.Y.Zrc1"
git push origin vX.Y.Zrc1
```

## 5. Publish package to PyPI

The tag push automatically triggers the appropriate workflow in GitHub Actions:

- Stable tag (`vX.Y.Z`) → `publish-pypi.yml` → publishes to **PyPI** + creates GitHub Release
- RC tag (`vX.Y.Zrc1`) → `publish-testpypi.yml` → publishes to **TestPyPI** + creates GitHub pre-release

Both workflows:

- Build the package (`poetry build`)
- Validate with `twine check`
- Publish via **OIDC Trusted Publishing** (no API tokens required)
- Create a GitHub Release with auto-generated notes

> **Note:** A `pypi-publish` GitHub Environment (for stable) and `testpypi-publish` environment (for RCs) must be configured with tag-based deployment protection rules.

## 6. GitHub Release

The GitHub Release is created automatically by the publish workflow. Verify the release notes are correct and add any manual context if needed.
