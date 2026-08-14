# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Stage the current langchain-azure-ai source for azd remote build."""

from __future__ import annotations

import shutil
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = SERVER_ROOT.parents[5]
STAGED_PACKAGE_ROOT = SERVER_ROOT / "vendor" / "langchain-azure-ai"


def main() -> None:
    """Copy only the files needed to build the current package source."""

    shutil.rmtree(STAGED_PACKAGE_ROOT, ignore_errors=True)
    STAGED_PACKAGE_ROOT.mkdir(parents=True)
    for filename in ("pyproject.toml", "README.md"):
        shutil.copy2(PACKAGE_ROOT / filename, STAGED_PACKAGE_ROOT / filename)
    shutil.copytree(
        PACKAGE_ROOT / "langchain_azure_ai",
        STAGED_PACKAGE_ROOT / "langchain_azure_ai",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


if __name__ == "__main__":
    main()