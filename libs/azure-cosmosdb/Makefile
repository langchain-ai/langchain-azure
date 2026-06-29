.PHONY: all format lint test tests integration_tests help

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

# uv run helpers. `--frozen` keeps the lockfile authoritative (no implicit
# updates), so CI's "did the run create new files" check stays green.
UV_RUN_TEST = uv run --frozen --all-extras --group test
UV_RUN_INTEGRATION = uv run --frozen --all-extras --group test_integration --group test
UV_RUN_LINT = uv run --frozen --all-extras --group lint --group typing --group test --group test_integration

test:
	$(UV_RUN_TEST) pytest $(TEST_FILE)

tests:
	$(UV_RUN_TEST) pytest $(TEST_FILE)

integration_tests:
	$(UV_RUN_INTEGRATION) pytest tests/integration_tests/

test_watch:
	$(UV_RUN_TEST) ptw --now . -- -vv $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/azure-cosmosdb --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=src/langchain_azure_cosmosdb
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && $(UV_RUN_LINT) mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	uv run --frozen --group codespell codespell --toml pyproject.toml

spell_fix:
	uv run --frozen --group codespell codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
