from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--docs-samples-dir",
        help="Repository-relative docs or samples directory; defaults to all examples.",
    )


def _example_directories(config: pytest.Config) -> list[Path]:
    selected = config.getoption("--docs-samples-dir")
    if selected is None:
        return [
            REPO_ROOT / "docs",
            REPO_ROOT / "samples",
            *sorted((REPO_ROOT / "libs").glob("*/docs")),
        ]

    directory = (REPO_ROOT / selected).resolve()
    if not directory.is_relative_to(REPO_ROOT) or not directory.is_dir():
        raise pytest.UsageError(
            "--docs-samples-dir must name an existing directory in this repository."
        )
    parts = directory.relative_to(REPO_ROOT).parts
    if parts and (
        parts[0] in {"docs", "samples"}
        or (len(parts) >= 3 and parts[0] == "libs" and parts[2] == "docs")
    ):
        return [directory]
    raise pytest.UsageError(
        "--docs-samples-dir must name a directory under docs/, samples/, "
        "or libs/<sdk>/docs/."
    )


@pytest.fixture(scope="session")
def docs_samples_dirs(pytestconfig: pytest.Config) -> list[Path]:
    return _example_directories(pytestconfig)
