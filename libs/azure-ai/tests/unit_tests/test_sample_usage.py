"""Regression tests for sample imports and documented Memory usage."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from langchain_azure_ai.agents import AgentServiceAgentState

SAMPLES_ROOT = Path(__file__).resolve().parents[4] / "samples"


def test_document_intelligence_sample_agent_imports() -> None:
    sample_path = SAMPLES_ROOT / "react-agent-docintelligence/src/react_agent/graph.py"
    sample_tree = ast.parse(sample_path.read_text(encoding="utf-8"))
    imports = [
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


def test_memory_readme_minimal_usage() -> None:
    readme_path = SAMPLES_ROOT / "foundry-memory/README.md"
    minimal_usage = readme_path.read_text(encoding="utf-8").split(
        "## Minimal Usage", 1
    )[1]
    example = minimal_usage.split("```python\n", 1)[1].split("```", 1)[0]
    client = Mock()
    client.beta.memory_stores.search_memories.return_value = SimpleNamespace(
        search_id="search-1", memories=[]
    )
    credential = Mock()
    namespace: dict[str, Any] = {}

    with (
        patch("azure.ai.projects.AIProjectClient", return_value=client) as factory,
        patch("azure.identity.DefaultAzureCredential", return_value=credential),
    ):
        exec(compile(example, str(readme_path), "exec"), namespace)

    factory.assert_called_once_with(
        endpoint="https://<resource>.services.ai.azure.com/api/projects/<project>",
        credential=credential,
        user_agent="langchain-azure-ai",
    )
    assert namespace["history"].scope == "user:123"
    assert namespace["retriever"].history_ref is namespace["history"]
    assert namespace["docs"] == []
    search_arguments = client.beta.memory_stores.search_memories.call_args.kwargs
    assert search_arguments["name"] == "my_store"
    assert search_arguments["scope"] == "user:123"
