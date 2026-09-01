create-env:
	@echo "Creating virtual environment..."
	uv sync --frozen
	@echo "Virtual environment created."

lint lint_package lint_tests: create-env
	@echo "Running linters..."
	uv run --frozen tox run -e lint || true
	@echo "Linters completed."

test integration_tests: create-env
	@echo "Running tests..."
	uv run --frozen tox run -m test
	@echo "Tests completed."

integration_test:
	uv run --frozen --all-extras --group test_integration --group test pytest -m compile tests/integration_tests

.PHONY: create-env lint lint_package lint_tests test integration_test integration_tests
