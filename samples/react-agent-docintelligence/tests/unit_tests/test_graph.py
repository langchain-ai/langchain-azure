"""Smoke test for the Document Intelligence sample's graph entry point."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


def test_graph_import_and_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    agents = [
        SimpleNamespace(
            id=f"{name}:1",
            name=name,
            version="1",
        )
        for name in ("document-parser", "document-analyst")
    ]
    client = Mock()
    client.agents.create_version.side_effect = agents
    client.agents.get_version.side_effect = agents
    monkeypatch.setenv(
        "AZURE_AI_PROJECT_ENDPOINT",
        "https://example.services.ai.azure.com/api/projects/test",
    )
    monkeypatch.delenv("AZURE_AI_INFERENCE_CREDENTIAL", raising=False)
    monkeypatch.setenv("OTEL_AUTO_CONFIGURE_AZURE_MONITOR", "false")

    with (
        patch(
            "langchain_azure_ai.agents._v2.prebuilt.factory.AIProjectClient",
            return_value=client,
        ),
        patch(
            "langchain_azure_ai._resources.get_service_endpoint_from_project",
            return_value=("https://document-intelligence.example.test", "test-key"),
        ),
        patch(
            "langchain_azure_ai.tools.services.document_intelligence."
            "DocumentIntelligenceClient"
        ) as document_client,
    ):
        from react_agent import graph

        assert graph.name == "document-intake-agent"
        assert set(graph.get_graph().nodes) == {
            "__start__",
            "parser",
            "tools",
            "prepare_analysis",
            "analyst",
            "__end__",
        }
        assert [
            call.kwargs["agent_name"]
            for call in client.agents.create_version.call_args_list
        ] == ["document-parser", "document-analyst"]
        document_client.assert_called_once()
