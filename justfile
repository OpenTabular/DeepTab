# list recipes
default:
	@just --list --unsorted

# install dependencies and set up all pre-commit hooks
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

# run docformatter and ruff formatter
format:
    poetry run docformatter --in-place --recursive --wrap-summaries 120 --wrap-descriptions 120 .
    poetry run ruff format .

# run pyright type checking
typecheck:
    poetry run pyright

# run tests with coverage
test:
    poetry run pytest --cov=deeptab tests/

# build HTML docs locally (warnings treated as errors)
docs:
    sphinx-build -b html docs/ docs/_build/html -W --keep-going

# run all pre-commit hooks on all files (commit + push stage)
check:
    poetry run pre-commit run --all-files
    poetry run pre-commit run --all-files --hook-stage push

# create a conventional commit using commitizen
commit:
    poetry run cz commit
