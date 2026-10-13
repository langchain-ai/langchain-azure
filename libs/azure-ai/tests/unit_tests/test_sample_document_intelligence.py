"""Regression test for the Document Intelligence sample's agent imports."""

import ast
from pathlib import Path
from typing import Any

from langchain_azure_ai.agents import AgentServiceAgentState

SAMPLES_ROOT = Path(__file__).resolve().parents[4] / "samples"


def test_document_intelligence_sample_agent_imports() -> None:
    sample_path = SAMPLES_ROOT / "react-agent-docintelligence/src/react_agent/graph.py"
    sample_tree = ast.parse(sample_path.read_text(encoding="utf-8"))
    imports: list[ast.stmt] = [
        node
        for node in sample_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("langchain_azure_ai.agents")
    ]
    namespace: dict[str, Any] = {}

    exec(
        compile(ast.Module(body=imports, type_ignores=[]), str(sample_path), "exec"),
        namespace,
    )

    assert namespace["AgentServiceAgentState"] is AgentServiceAgentState
