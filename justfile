# list recipes
default:
	@just --list --unsorted

# install dependencies and set up pre-commit hooks
install:
	poetry install
	poetry run pre-commit install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push

# update dependencies and pre-commit hook revisions
update:
	poetry update
	poetry run pre-commit autoupdate

# regenerate poetry.lock without upgrading dependencies
lock:
    poetry lock --no-update

# remove Python file artifacts
clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

# run ruff linting and fix all fixable errors
lint:
    poetry run ruff check --fix .

# run ruff formatter
format:
    poetry run ruff format .

# run pyright type checking
types:
    poetry run pyright

# run tests with coverage
test:
    poetry run pytest --cov=deeptab tests/

# build HTML docs locally (warnings treated as errors)
docs:
    poetry run sphinx-build -b html docs/ docs/_build/html -W --keep-going

# run all pre-commit hooks on all files including push-stage hooks (ruff, pyright, prettier)
check:
    poetry run pre-commit run --hook-stage push --all-files

# create a conventional commit using commitizen
commit:
    poetry run cz commit

# audit installed dependencies for known vulnerabilities
audit:
    poetry run pip-audit

# preview the next stable version bump (pass extra cz args, e.g. `just bump-preview --increment MAJOR`)
bump-preview *args:
    poetry run cz bump --dry-run {{ args }}

# apply the next stable version bump (updates version, CHANGELOG, commit, and tag)
bump *args:
    poetry run cz bump {{ args }}

# preview the next release-candidate bump (pass extra cz args, e.g. `just bump-rc-preview --increment MAJOR`)
bump-rc-preview *args:
    poetry run cz bump --prerelease rc --dry-run {{ args }}

# apply the next release-candidate bump (rcN; updates version, CHANGELOG, commit, and tag)
bump-rc *args:
    poetry run cz bump --prerelease rc {{ args }}
