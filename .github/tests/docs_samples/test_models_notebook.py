"""Smoke tests for the models notebook examples."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_file_search_tool_binding(docs_samples_dirs: list[Path]) -> None:
    notebook_path = (
        Path(__file__).resolve().parents[3]
        / "libs"
        / "azure-ai"
        / "docs"
        / "azure_ai_models_demo.ipynb"
    )
    if not any(
        notebook_path.is_relative_to(directory) for directory in docs_samples_dirs
    ):
        pytest.skip("This smoke test covers the azure-ai models notebook.")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "79055a62")
    model = Mock()
    namespace = {"model": model, "vector_store": SimpleNamespace(id="vs_test")}

    cell_filename = f"<{notebook_path.name}:{cell['id']}>"
    exec(compile("".join(cell["source"]), cell_filename, "exec"), namespace)

    model.bind_tools.assert_called_once_with(
        [{"type": "file_search", "vector_store_ids": ["vs_test"]}]
    )
    assert namespace["model_with_tools"] is model.bind_tools.return_value
