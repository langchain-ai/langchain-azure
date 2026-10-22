TEST_FILE ?= --pyargs tests langchain_azure_postgresql

create-env:
	@echo "Creating virtual environment..."
	uv sync --frozen
	@echo "Virtual environment created."

# Preserve report-only linting without hiding tool failures.
lint lint_package lint_tests:
	@echo "Running linters..."
	uv run --frozen --group lint ruff check --exit-zero src tests
	@echo "Linters completed."

test tests integration_test integration_tests:
	@echo "Running tests..."
	uv run --frozen --group test pytest $(TEST_FILE)
	@echo "Tests completed."

type:
	uv run --frozen --group test --group type mypy src tests

package:
	uv run --frozen --group package python -m build --installer uv --outdir dist
	uv run --frozen --group package twine check "dist/*"

vuln:
	uv run --frozen --group package uv-secure

.PHONY: create-env lint lint_package lint_tests test tests integration_test integration_tests type package vuln
