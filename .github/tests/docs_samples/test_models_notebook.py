"""Smoke tests for the models notebook examples."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

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
    file_search_tool = Mock()
    builtin_tools = ModuleType("langchain_azure_ai.tools.builtin")
    builtin_tools.__dict__["FileSearchTool"] = file_search_tool
    namespace = {"model": model, "vector_store": SimpleNamespace(id="vs_test")}

    cell_filename = f"<{notebook_path.name}:{cell['id']}>"
    with patch.dict(sys.modules, {builtin_tools.__name__: builtin_tools}):
        exec(compile("".join(cell["source"]), cell_filename, "exec"), namespace)

    file_search_tool.assert_called_once_with(vector_store_ids=["vs_test"])
    model.bind_tools.assert_called_once_with([file_search_tool.return_value])
    assert namespace["model_with_tools"] is model.bind_tools.return_value
