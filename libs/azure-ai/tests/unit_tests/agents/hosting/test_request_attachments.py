# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Request attachments survive hosting conversion and conversation history."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest

pytest.importorskip("azure.ai.agentserver.responses")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from pydantic import SecretStr  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from langchain_azure_ai.agents.hosting import ResponsesHostServer  # noqa: E402
from langchain_azure_ai.agents.hosting._converters import (  # noqa: E402
    items_to_messages,
)

ATTACHMENTS = [
    {
        "type": "input_image",
        "image_url": "https://example.com/chart.png",
        "detail": "high",
    },
    {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="},
    {"type": "input_image", "file_id": "file-image", "detail": "auto"},
    {"type": "input_file", "file_url": "https://example.com/report.pdf"},
    {"type": "input_file", "file_id": "file-report", "filename": "report.pdf"},
    {
        "type": "input_file",
        "file_data": "data:application/pdf;base64,aGVsbG8=",
        "filename": "report.pdf",
    },
]


@pytest.mark.parametrize("attachment", ATTACHMENTS)
@pytest.mark.parametrize("role", ["user", "system", "developer", "tool"])
def test_attachments_reach_responses_model_payload(
    attachment: dict[str, Any], role: str
) -> None:
    content = [
        {"type": "input_text", "text": "before"},
        attachment,
        {"type": "input_text", "text": "after"},
    ]
    original = deepcopy(content)
    item = (
        {"type": "function_call_output", "call_id": "call-1", "output": content}
        if role == "tool"
        else {"type": "message", "role": role, "content": content}
    )
    messages = items_to_messages([item])
    model = ChatOpenAI(model="test", api_key=SecretStr("test"), use_responses_api=True)
    payload = model._get_request_payload(messages)["input"][0]
    assert payload["output" if role == "tool" else "content"] == original
    assert content == original
    assert isinstance(messages[0].content, list)
    assert isinstance(messages[0].content[1], dict)
    messages[0].content[1]["file_id"] = "changed"
    assert content == original


@pytest.mark.parametrize("attachment", ATTACHMENTS)
@pytest.mark.parametrize("with_text", [False, True])
def test_assistant_attachments_preserve_role_and_content_in_graph_input(
    attachment: dict[str, Any], with_text: bool
) -> None:
    content = [{"type": "input_text", "text": "prior context"}] if with_text else []
    content.append(attachment)
    messages = items_to_messages(
        [{"type": "message", "role": "assistant", "content": content}]
    )
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    expected = [{"type": "text", "text": "prior context"}] if with_text else []
    assert messages[0].content == [*expected, attachment]
    assert isinstance(messages[0].content, list)
    assert isinstance(messages[0].content[-1], dict)
    messages[0].content[-1]["file_id"] = "changed"
    assert content[-1] == attachment


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_http_attachments_survive_second_turn(
    streaming: bool, checkpointed: bool
) -> None:
    captured: list[list[Any]] = []

    def inspect_input(state: MessagesState) -> dict[str, Any]:
        captured.append(deepcopy(state["messages"]))
        return {"messages": [AIMessage(content="received")]}

    graph = StateGraph(MessagesState)
    graph.add_node("inspect", inspect_input)
    graph.add_edge(START, "inspect")
    graph.add_edge("inspect", END)
    server = ResponsesHostServer(
        graph.compile(checkpointer=InMemorySaver() if checkpointed else None)
    )
    content = [{"type": "input_text", "text": "Inspect attachments"}, *ATTACHMENTS]

    def completed(response: Any) -> dict[str, Any]:
        assert response.status_code == 200, response.text
        if not streaming:
            return response.json()
        events = [
            json.loads(line.removeprefix("data:").strip())
            for line in response.text.splitlines()
            if line.startswith("data:")
            and line.removeprefix("data:").strip() != "[DONE]"
        ]
        return next(e["response"] for e in events if e["type"] == "response.completed")

    with TestClient(server.app) as client:
        first = completed(
            client.post(
                "/responses",
                json={
                    "input": [{"type": "message", "role": "user", "content": content}],
                    "stream": streaming,
                },
            )
        )
        second = completed(
            client.post(
                "/responses",
                json={
                    "input": "continue",
                    "previous_response_id": first["id"],
                    "stream": streaming,
                },
            )
        )
    assert first["status"] == second["status"] == "completed"
    expected = [{"type": "text", "text": "Inspect attachments"}, *ATTACHMENTS]
    assert captured[0][0].content == expected
    assert captured[1][0].content == expected
    assert sum(message.content == expected for message in captured[1]) == 1
