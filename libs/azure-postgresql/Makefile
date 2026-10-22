.PHONY: all format lint lint_package lint_tests \
	test tests integration_test integration_tests \
	create-env type package vuln help

all: help

TEST_FILE ?= --pyargs tests langchain_azure_postgresql

UV_RUN_TEST = uv run --frozen --group test
UV_RUN_LINT = uv run --frozen --group lint

test tests integration_test integration_tests:
	$(UV_RUN_TEST) pytest $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

PYTHON_FILES=src tests
lint_package: PYTHON_FILES=src/langchain_azure_postgresql
lint_tests: PYTHON_FILES=tests

# Preserve report-only linting without hiding tool failures.
lint lint_package lint_tests:
	$(UV_RUN_LINT) ruff check --exit-zero $(PYTHON_FILES)

format:
	$(UV_RUN_LINT) ruff format $(PYTHON_FILES)
	$(UV_RUN_LINT) ruff check --select I --fix $(PYTHON_FILES)

create-env:
	@echo "Creating virtual environment..."
	uv sync --frozen
	@echo "Virtual environment created."

type:
	uv run --frozen --group test --group type mypy src tests

package:
	uv run --frozen --group package python -m build --installer uv --outdir dist
	uv run --frozen --group package twine check "dist/*"

vuln:
	uv run --frozen --group package uv-secure


######################
# HELP
######################

help:
	@echo "----"
	@echo "create-env                   - create the development environment"
	@echo "format                       - format Python code and sort imports"
	@echo "lint                         - run lint checks on sources and tests"
	@echo "lint_package                 - run lint checks on package sources"
	@echo "lint_tests                   - run lint checks on tests"
	@echo "test                         - run tests and package doctests"
	@echo "tests                        - run tests and package doctests"
	@echo "test TEST_FILE=<test_file>   - run tests in the selected file"
	@echo "integration_test             - run tests and package doctests"
	@echo "integration_tests            - run tests and package doctests"
	@echo "type                         - run type checks"
	@echo "package                      - build and check distributions"
	@echo "vuln                         - check dependencies for vulnerabilities"
