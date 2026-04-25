# Build and release

The document outlines the steps to build and release the `deeptab` package. At this point, it is assumed that the development and testing of the package have been completed successfully.

## 1. Test documentation

It is expected from the contributor to update the documentation as an when required along side the change in source code. Please use the below process to test the documentation:

```sh
cd deeptab/docs/

make doctest
```

Fix any docstring related issue, then proceed with next steps.

## 2. Version update

The package version is maintained in `pyproject.toml` only. The version is read at runtime via `importlib.metadata`.

On a `release/vX.Y.Z` branch, run:

```bash
cz bump
```

This will:

- Determine the next version from conventional commits since the last tag
- Update the version in `pyproject.toml`
- Update `CHANGELOG.md`
- Create a local commit `bump: version X.Y.Z-1 → X.Y.Z`

The version number follows the format `major.minor.patch`. For example, `1.0.1`.

## 3. Release PR

- Open a PR from `release/vX.Y.Z` to `main`
- After review and approval, merge the PR
- **Merging to `main` does NOT trigger a PyPI release**

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
