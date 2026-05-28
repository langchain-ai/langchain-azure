# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end tests for ``ResponsesHostServer`` via Starlette TestClient."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.ai.agentserver.responses.models import (
    ItemMessage,
    MessageContentInputTextContent,
)
from starlette.testclient import TestClient

from langchain_azure_ai.agents.hosting import ResponsesHostServer

from .conftest import (
    make_checkpointed_echo_graph,
    make_custom_state_graph,
    make_echo_graph,
    make_streaming_graph,
)


def _client(server: ResponsesHostServer) -> TestClient:
    return TestClient(server.app)


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    current_type = ""
    for line in body.splitlines():
        if line.startswith("event:"):
            current_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data = line.split(":", 1)[1].strip()
            if not data:
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                payload = {"raw": data}
            events.append((current_type, payload))
    return events


def _message_item(text: str) -> ItemMessage:
    return ItemMessage(
        role="user",
        content=[MessageContentInputTextContent({"type": "input_text", "text": text})],
    )


def _request(**kwargs: object) -> MagicMock:
    request = MagicMock()
    request.instructions = kwargs.get("instructions")
    request.previous_response_id = kwargs.get("previous_response_id")
    return request


def _context(
    *,
    response_id: str = "resp-current",
    conversation_id: str | None = "conv-test",
    current_text: str = "hello",
    history: list[object] | None = None,
    provider: object | None = None,
) -> MagicMock:
    context = MagicMock()
    context.response_id = response_id
    context.conversation_id = conversation_id
    context.isolation = None
    context.get_input_items = AsyncMock(return_value=[_message_item(current_text)])
    context.get_history = AsyncMock(return_value=history or [])
    context._provider = provider
    return context


def test_non_streaming_request_returns_completed_response() -> None:
    server = ResponsesHostServer(make_echo_graph())
    with _client(server) as client:
        resp = client.post("/responses", json={"input": "hello", "model": "test"})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["status"] == "completed"
    output = payload["output"]
    assert any(item.get("type") == "message" for item in output)
    text = "".join(
        part.get("text", "")
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", [])
    )
    assert "Echo: hello" in text


def test_streaming_request_emits_sse_lifecycle_events() -> None:
    server = ResponsesHostServer(make_streaming_graph())
    with _client(server) as client:
        resp = client.post(
            "/responses",
            json={"input": "ignored", "stream": True, "model": "test"},
        )
    assert resp.status_code == 200, resp.text
    events = _parse_sse(resp.text)
    types = [t for t, _ in events]
    assert "response.created" in types
    assert "response.in_progress" in types
    assert "response.output_text.delta" in types
    assert "response.completed" in types
    deltas = [p["delta"] for t, p in events if t == "response.output_text.delta"]
    assert "".join(deltas) == "Hello, world!"


def test_readiness_endpoint_is_available() -> None:
    server = ResponsesHostServer(make_echo_graph())
    with _client(server) as client:
        resp = client.get("/readiness")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


def test_constructor_rejects_non_messages_state_schema() -> None:
    with pytest.raises(ValueError, match="messages"):
        ResponsesHostServer(make_custom_state_graph())


async def test_checkpointed_graph_uses_current_input_only() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context()

    result = await server.build_input(_request(), context)

    context.get_history.assert_not_called()
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "hello"


async def test_checkpointed_previous_response_id_does_not_duplicate_history() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(
        conversation_id=None,
        history=[_message_item("from responses transcript")],
        current_text="current turn",
    )

    result = await server.build_input(
        _request(previous_response_id="resp-previous"),
        context,
    )

    context.get_history.assert_not_called()
    assert [message.content for message in result["messages"]] == ["current turn"]


async def test_non_checkpointed_graph_uses_responses_history() -> None:
    server = ResponsesHostServer(make_echo_graph())
    context = _context(history=[_message_item("from history")])

    result = await server.build_input(_request(), context)

    context.get_history.assert_awaited_once()
    assert [message.content for message in result["messages"]] == [
        "from history",
        "hello",
    ]


async def test_non_checkpointed_previous_response_id_includes_history_once() -> None:
    server = ResponsesHostServer(make_echo_graph())
    context = _context(
        conversation_id=None,
        history=[_message_item("turn one"), _message_item("turn two")],
        current_text="turn three",
    )

    result = await server.build_input(
        _request(previous_response_id="resp-2"),
        context,
    )

    context.get_history.assert_awaited_once()
    assert [message.content for message in result["messages"]] == [
        "turn one",
        "turn two",
        "turn three",
    ]


async def test_checkpointed_previous_response_id_restores_graph_history_once() -> None:
    class _Provider:
        async def get_response(
            self,
            response_id: str,
            *,
            isolation: object = None,
        ) -> dict[str, str | None]:
            del isolation
            responses: dict[str, dict[str, str | None]] = {
                "resp-2": {"previous_response_id": "resp-1"},
                "resp-1": {"previous_response_id": None},
            }
            return responses[response_id]

    server = ResponsesHostServer(make_checkpointed_echo_graph())
    first_context = _context(
        response_id="resp-1",
        conversation_id=None,
        current_text="turn one",
    )
    first_config = await server.build_runnable_config(_request(), first_context)
    first_input = await server.build_input(_request(), first_context)

    first_state = await server.graph.ainvoke(first_input, config=first_config)

    second_context = _context(
        response_id="resp-3",
        conversation_id=None,
        current_text="turn two",
        history=[_message_item("turn one from responses transcript")],
        provider=_Provider(),
    )
    second_request = _request(previous_response_id="resp-2")
    second_config = await server.build_runnable_config(second_request, second_context)
    second_input = await server.build_input(second_request, second_context)

    second_state = await server.graph.ainvoke(second_input, config=second_config)

    second_context.get_history.assert_not_called()
    assert first_config["configurable"]["thread_id"] == "resp-resp-1"
    assert second_config["configurable"]["thread_id"] == "resp-resp-1"
    assert [message.content for message in first_state["messages"]] == [
        "turn one",
        "Echo: turn one",
    ]
    assert [message.content for message in second_state["messages"]] == [
        "turn one",
        "Echo: turn one",
        "turn two",
        "Echo: turn two",
    ]


async def test_conversation_id_is_thread_id() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    config = await server.build_runnable_config(_request(), _context())

    assert config["configurable"]["thread_id"] == "conv-test"


async def test_previous_response_id_chain_resolves_root_thread_id() -> None:
    class _Provider:
        async def get_response(
            self,
            response_id: str,
            *,
            isolation: object = None,
        ) -> dict[str, str | None]:
            del isolation
            responses: dict[str, dict[str, str | None]] = {
                "resp-2": {"previous_response_id": "resp-1"},
                "resp-1": {"previous_response_id": None},
            }
            return responses[response_id]

    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(
        response_id="resp-3",
        conversation_id=None,
        provider=_Provider(),
    )

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-2"),
        context,
    )

    assert config["configurable"]["thread_id"] == "resp-resp-1"
    assert (
        server.build_runnable_config_sync(
            _request(previous_response_id="resp-3"),
            _context(response_id="resp-4", conversation_id=None),
        )["configurable"]["thread_id"]
        == "resp-resp-3"
    )


async def test_conversation_management_debug_log_has_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = ResponsesHostServer(make_echo_graph())
    context = _context(history=[_message_item("from history")])

    caplog.set_level(
        logging.DEBUG,
        logger="langchain_azure_ai.agents.hosting._responses_host",
    )
    await server.build_input(_request(), context)

    assert "mode=responses_history" in caplog.text
    assert "history_items=1" in caplog.text
    assert "history_messages=1" in caplog.text
    assert "current_items=1" in caplog.text
    assert "current_messages=1" in caplog.text
