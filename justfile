# list recipes
default:
	@just --list --unsorted

# install dependencies and set up pre-commit hooks
install:
	poetry install
	poetry run pre-commit install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push

# synchronize core test dependencies, removing optional extras and other groups
install-core:
    poetry sync --only main,dev
    poetry run pre-commit install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push

# synchronize test dependencies with structured logging and experiment tracking
install-extras:
    poetry sync --only main,dev --extras "logs tracking"
    poetry run pre-commit install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push

# check the installed dependency graph
deps-check:
    poetry run python -m pip check

# show exact installed package versions
deps-list:
    poetry run python -m pip freeze --all

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

# run tests with at least 90% line coverage
test:
    poetry run pytest --cov=deeptab --cov-fail-under=90 tests/

# run selected tests with ordinary pytest arguments, without a coverage gate
[positional-arguments]
test-run *args:
    poetry run pytest --strict-markers "$@"

# verify the existing core-only installation and run the full source suite
test-core:
    poetry run pytest tests/ --dependency-profile=core --strict-markers -ra

# verify the existing logs/tracking installation and run the full source suite
test-extras:
    poetry run pytest tests/ --dependency-profile=extras --strict-markers -ra

# build HTML docs locally (warnings treated as errors)
docs:
    rm -rf docs/_build
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
