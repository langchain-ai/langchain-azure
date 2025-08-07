create-env:
	@echo "Creating virtual environment..."
	poetry sync
	@echo "Virtual environment created."

lint lint_package lint_tests: create-env
	@echo "Running linters..."
	poetry run tox run -e lint || true
	@echo "Linters completed."

test: create-env
	@echo "Running tests..."
	poetry run tox run -m test
	@echo "Tests completed."

.PHONY: create-env lint test
