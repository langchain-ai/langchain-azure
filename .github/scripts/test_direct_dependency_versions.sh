#!/usr/bin/env bash
set -euo pipefail

resolution="${1:-}"
case "$resolution" in
  lowest-direct | highest) ;;
  *)
    echo "Usage: $0 {lowest-direct|highest}" >&2
    exit 2
    ;;
esac

: "${PKG_NAME:?PKG_NAME must be set}"
: "${PYTHON_VERSION:?PYTHON_VERSION must be set}"
: "${RUNNER_TEMP:?RUNNER_TEMP must be set}"
: "${UV_PROJECT_ENVIRONMENT:?UV_PROJECT_ENVIRONMENT must be set}"

export UV_NO_SYNC=1

requirements_file="$RUNNER_TEMP/$resolution-direct-dependencies.txt"

uv pip compile pyproject.toml \
  --no-sources \
  --python "$UV_PROJECT_ENVIRONMENT" \
  --python-version "$PYTHON_VERSION" \
  --resolution "$resolution" \
  --output-file "$requirements_file"

echo "Testing direct dependencies resolved with: $resolution"
cat "$requirements_file"

uv pip install \
  --python "$UV_PROJECT_ENVIRONMENT" \
  --no-sources \
  --group test \
  --group test_integration \
  --requirements "$requirements_file"

echo "Test environment ($resolution):"
uv run --no-sync python - <<'PY'
import sys
from importlib.metadata import distributions

print(f"Python: {sys.version}")
for package in sorted(f"{dist.name}=={dist.version}" for dist in distributions()):
    print(package)
PY

if [[ "$PKG_NAME" == "langchain-azure-postgresql" ]]; then
  echo "Running unit and integration tests"
  uv run --no-sync pytest --pyargs tests langchain_azure_postgresql
else
  echo "Running unit tests"
  make tests

  echo "Running integration tests"
  make integration_tests
fi