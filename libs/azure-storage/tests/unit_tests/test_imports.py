import tomllib
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
    with pyproject_path.open("rb") as pyproject_file:
        pyproject_version = tomllib.load(pyproject_file)["project"]["version"]

    assert pyproject_version == langchain_azure_storage.__version__
