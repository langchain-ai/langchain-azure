"""Verify hosted samples send complete history without model-side storage."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from azure.core.credentials import TokenCredential
from azure.identity import get_bearer_token_provider
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

SAMPLES = (
    Path(__file__).resolve().parents[3] / "samples/hosting/langgraph-hosted-agents"
)


# Async client cleanup uses the event loop's local wakeup socket.
@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
@pytest.mark.parametrize(
    "path",
    sorted(SAMPLES.glob("*/*/main.py")),
    ids=lambda path: path.parent.relative_to(SAMPLES).as_posix(),
)
def test_model_request_keeps_history_without_storage(
    path: Path, docs_samples_dirs: list[Path]
) -> None:
    if not any(path.is_relative_to(directory) for directory in docs_samples_dirs):
        pytest.skip("This test covers LangGraph hosted-agent samples.")
    constructor = next(
        node
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"ChatOpenAI", "AzureAIOpenAIApiChatModel"}
    )
    base_url = "https://example.invalid/openai/v1"
    namespace = {
        "ChatOpenAI": ChatOpenAI,
        "AzureAIOpenAIApiChatModel": AzureAIOpenAIApiChatModel,
        "os": SimpleNamespace(
            environ={"FOUNDRY_PROJECT_ENDPOINT": "https://example.invalid/project"}
        ),
        "deployment": "test-model",
        "base_url": base_url,
        "openai_client": SimpleNamespace(base_url=base_url),
        "token_provider": "test-key",
        "credential": Mock(spec=TokenCredential),
        "get_bearer_token_provider": get_bearer_token_provider,
        "_AZURE_AI_SCOPE": "https://ai.azure.com/.default",
    }
    # Execute the sample's actual constructor without starting its host or tools.
    model = eval(compile(ast.Expression(constructor), str(path), "eval"), namespace)
    try:
        payload = model._get_request_payload(
            [
                HumanMessage(content="What is 137 * 139?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "call_calculate", "name": "calculate", "args": {}}
                    ],
                ),
                ToolMessage(content="19043", tool_call_id="call_calculate"),
            ]
        )
        assert payload.get("store") is False
        assert "previous_response_id" not in payload
        assert payload["input"][0]["role"] == "user"
        assert [item["call_id"] for item in payload["input"][1:]] == [
            "call_calculate",
            "call_calculate",
        ]
        assert payload["input"][-1]["output"] == "19043"
    finally:
        model.root_client.close()
        asyncio.run(model.root_async_client.close())
