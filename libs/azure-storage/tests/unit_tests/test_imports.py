import re
from pathlib import Path

import pytest

import langchain_azure_storage


def test_import_package() -> None:
    try:
        import langchain_azure_storage  # noqa: F401
    except ImportError:
        pytest.fail("langchain_azure_storage package is expected to be importable.")


def test_package_version_matches_pyproject_version() -> None:
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()
    # Note: Using a regex here instead of toml parser to avoid pulling in tomli as
    # a test dependency. If we ever only support Python 3.11+ (which includes
    # tomlib) or pull in tomli as a dependency, we could switch to using a toml
    # parser and make this logic more robust.
    version_match = re.search(r'^version\s*=\s*"([\w.]+)"', content, re.MULTILINE)
    if version_match is None:
        pytest.fail("Could not find version in pyproject.toml")
    pyproject_version = version_match.group(1)

    assert pyproject_version == langchain_azure_storage.__version__
