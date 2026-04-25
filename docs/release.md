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
The package version is mantained in `deeptab/__version__.py` and `pyproject.toml` file. Increment the version according to the changes such as patch, minor, major or all.

- The version number should be in the format `major.minor.patch`. For example, `1.0.1`.

**Note:** Don't forget to update the version in the `pyproject.toml` file as well.


## 3. Release

- Create a pull request from your `feature` branch to the `develop` branch.
- Once the pull request is approved and merged to develop. The maintainer will test the package and documentation. If everything is fine, the maintainer will proceed further to merge the changed to `main` and `release` branch.
- Ideally content of `main` and `release` branch should be same. The `release` branch is used to publish the package to PyPi while `main` branch is used to publish the documentation to readthedocs and can be accesseed at [deeptab.readthedocs.io](https://deeptab.readthedocs.io/en/latest/).


## 4. Publish package to PyPi

The package is published to PyPi using GitHub Actions. The workflow is triggered when a new tag is pushed to the repository. The workflow will build the package, upload it to PyPi.

## 5. GitHub Release

Create a new release on GitHub with the version number and release notes. The release notes should include the changes made in the release.
