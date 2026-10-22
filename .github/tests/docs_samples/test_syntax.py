import ast
from pathlib import Path

import nbformat
import pytest
from IPython.core.inputtransformer2 import TransformerManager


def test_example_syntax(example_file: Path) -> None:
    if example_file.suffix == ".py":
        compile(example_file.read_bytes(), str(example_file), "exec", dont_inherit=True)
        return

    notebook = nbformat.read(example_file, as_version=4)
    nbformat.validate(notebook)
    language = notebook.metadata.get("language_info", {}).get("name", "python")
    if language != "python":
        pytest.skip(f"Syntax checking is only supported for Python, not {language}.")

    transformer = TransformerManager()
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(
                transformer.transform_cell(cell.source),
                f"{example_file}:cell {index + 1}",
                "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                dont_inherit=True,
            )
