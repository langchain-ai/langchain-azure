.PHONY: all format format_diff lint lint_diff lint_package lint_tests \
	test tests test_watch coverage integration_test integration_tests \
	check_imports spell_check spell_fix help

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

# uv run helpers. `--frozen` keeps the lockfile authoritative (no implicit
# updates), so CI's "did the run create new files" check stays green.
UV_RUN_TEST = uv run --frozen --all-extras --group test
UV_RUN_LINT = uv run --frozen --all-extras --group lint --group typing --group test --group test_integration
UV_RUN = uv run --frozen --all-extras

test tests:
	$(UV_RUN_TEST) pytest $(TEST_FILE)

# Not folded into `test`: modules gated behind optional extras skip their
# tests on Pythons that cannot install them (deepagents needs >= 3.11), and
# coverage would fall short of the 100% floor there. The shared CI workflows
# run `make test`, not this target, so the floor is a local/reviewer gate
# rather than a CI-enforced one.
coverage:
	$(UV_RUN_TEST) pytest tests/unit_tests/ \
		--cov=langchain_azure_compute --cov-report=term-missing

# Release CI runs this target unconditionally and has no live Azure resources,
# so tests under tests/integration_tests must skip themselves when their
# required environment variables are absent rather than fail.
integration_test integration_tests:
	$(UV_RUN_TEST) --group test_integration pytest tests/integration_tests/

test_watch:
	$(UV_RUN_TEST) ptw --now . -- -vv $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/azure-compute --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_azure_compute
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || { mkdir -p $(MYPY_CACHE) && $(UV_RUN_LINT) mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE); }

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	uv run --frozen --group codespell codespell --toml pyproject.toml

spell_fix:
	uv run --frozen --group codespell codespell --toml pyproject.toml -w

check_imports: $(shell find langchain_azure_compute -name '*.py')
	$(UV_RUN) python ./scripts/check_imports.py $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports                - check imports'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'spell_check                  - run codespell'
	@echo 'integration_test             - run integration tests (needs live Azure)'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'coverage                     - run unit tests, enforcing 100% coverage'
	@echo 'integration_tests            - run integration tests (needs live Azure)'
