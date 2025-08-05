create-env:
	@echo "Creating virtual environment..."
	uv sync --all-extras
	@echo "Virtual environment created."

lint: create-env
	@echo "Running linters..."
	uv run tox run -e lint || true
	@echo "Linters completed."

test: create-env
	@echo "Running tests..."
	uv run tox run -m test
	@echo "Tests completed."

.PHONY: create-env lint test
