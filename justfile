# list recipes
defaut:
	@just --list --unsorted

# install dependencies
install:
	poetry install
	poetry run pre-commit install --hook-type commit-msg --hook-type pre-commit

# update dependencies
update:
	poetry update
	poetry run pre-commit autoupdate

# update the poetry.lock file if the pyproject.toml file has been updated
lock:
    poetry lock

# remove Python file artifacts
clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +


# run ruff linting and fix all fixable linting errors
lint:
    poetry run ruff check --fix .

# run ruff formatter to format all files
format:
    poetry run docformatter --in-place --recursive --wrap-summaries 120 --wrap-descriptions 120 .
    poetry run ruff format .

# create a conventional commit using commitizen
commit:
    poetry run cz commit
