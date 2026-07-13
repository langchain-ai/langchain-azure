# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end tests for ``ResponsesHostServer`` via Starlette TestClient."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("starlette")

from azure.ai.agentserver.responses.models import (  # noqa: E402
    ItemMessage,
    MessageContentInputTextContent,
)
from starlette.testclient import TestClient  # noqa: E402

from langchain_azure_ai.agents.hosting import (  # noqa: E402
    ResponsesHostServer,
    ResponsesServerOptions,
)

from .conftest import (  # noqa: E402
    make_checkpointed_echo_graph,
    make_checkpointed_two_node_graph,
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
    context.conversation_chain_id = conversation_id or f"resp-{response_id}"
    context.is_recovery = False
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


def test_constructor_accepts_reexported_response_options() -> None:
    options = ResponsesServerOptions(default_model="test")
    server = ResponsesHostServer(make_echo_graph(), options=options)

    with _client(server) as client:
        resp = client.post("/responses", json={"input": "hello"})

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"


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


async def test_root_response_creates_direct_path_mapping(tmp_path: Path) -> None:
    mapping_dir = tmp_path / "responses_mapping"
    server = ResponsesHostServer(
        make_checkpointed_echo_graph(),
        responses_mapping_dir=mapping_dir,
    )
    context = _context(response_id="resp-1", conversation_id=None)

    config = await server.build_runnable_config(_request(), context)

    assert config["configurable"]["thread_id"] == "resp-1"
    assert "checkpoint_id" not in config["configurable"]
    assert (mapping_dir / "resp-1").read_text(encoding="utf-8") == "resp-1"


async def test_conversation_uses_root_response_id_and_latest_checkpoint(
    tmp_path: Path,
) -> None:
    mapping_dir = tmp_path / "responses_mapping"
    conversation_dir = tmp_path / "conversations_mapping"
    checkpoint_dir = tmp_path / "checkpoints_mapping"
    server = ResponsesHostServer(
        make_checkpointed_echo_graph(),
        responses_mapping_dir=mapping_dir,
    )
    first_context = _context(response_id="resp-1", conversation_id="conv-test")
    first_config = await server.build_runnable_config(_request(), first_context)

    checkpoint_dir.mkdir()
    (checkpoint_dir / "resp-2").write_text("checkpoint-current", encoding="utf-8")
    second_context = _context(response_id="resp-2", conversation_id="conv-test")
    second_context.is_recovery = True
    second_config = await server.build_runnable_config(_request(), second_context)

    assert first_config["configurable"]["thread_id"] == "resp-1"
    assert second_config["configurable"]["thread_id"] == "resp-1"
    assert "checkpoint_id" not in first_config["configurable"]
    assert "checkpoint_id" not in second_config["configurable"]
    assert second_config["configurable"]["responses_context"] is second_context
    assert (conversation_dir / "conv-test").read_text(encoding="utf-8") == "resp-1"
    assert (mapping_dir / "resp-1").read_text(encoding="utf-8") == "resp-1"
    assert (mapping_dir / "resp-2").read_text(encoding="utf-8") == "resp-1"


async def test_previous_response_uses_root_and_exact_parent_checkpoint(
    tmp_path: Path,
) -> None:
    mapping_dir = tmp_path / "responses_mapping"
    checkpoint_dir = tmp_path / "checkpoints_mapping"
    server = ResponsesHostServer(
        make_checkpointed_echo_graph(),
        responses_mapping_dir=mapping_dir,
    )
    mapping_dir.mkdir()
    checkpoint_dir.mkdir()
    (mapping_dir / "resp-2").write_text("resp-1", encoding="utf-8")
    (checkpoint_dir / "resp-2").write_text("checkpoint-2", encoding="utf-8")
    context = _context(response_id="resp-3", conversation_id=None)

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-2"),
        context,
    )

    assert config["configurable"]["thread_id"] == "resp-1"
    assert config["configurable"]["checkpoint_id"] == "checkpoint-2"
    assert config["configurable"]["responses_context"] is context
    assert (mapping_dir / "resp-3").read_text(encoding="utf-8") == "resp-1"

    sync_context = _context(response_id="resp-4", conversation_id=None)
    sync_config = server.build_runnable_config_sync(
        _request(previous_response_id="resp-2"),
        sync_context,
    )
    assert sync_config["configurable"]["thread_id"] == "resp-1"
    assert sync_config["configurable"]["checkpoint_id"] == "checkpoint-2"
    assert sync_config["configurable"]["responses_context"] is sync_context


async def test_recovery_uses_latest_checkpoint_on_root_thread(tmp_path: Path) -> None:
    mapping_dir = tmp_path / "responses_mapping"
    checkpoint_dir = tmp_path / "checkpoints_mapping"
    server = ResponsesHostServer(
        make_checkpointed_echo_graph(),
        responses_mapping_dir=mapping_dir,
    )
    mapping_dir.mkdir()
    checkpoint_dir.mkdir()
    (mapping_dir / "resp-parent").write_text("resp-root", encoding="utf-8")
    (checkpoint_dir / "resp-parent").write_text(
        "checkpoint-parent", encoding="utf-8"
    )
    (mapping_dir / "resp-current").write_text("resp-root", encoding="utf-8")
    (checkpoint_dir / "resp-current").write_text(
        "checkpoint-current", encoding="utf-8"
    )
    context = _context(response_id="resp-current", conversation_id=None)
    context.is_recovery = True

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-parent"),
        context,
    )

    assert config["configurable"]["thread_id"] == "resp-root"
    assert "checkpoint_id" not in config["configurable"]


async def test_later_turn_recovery_does_not_repeat_completed_node(
    tmp_path: Path,
) -> None:
    server = ResponsesHostServer(
        make_checkpointed_two_node_graph(),
        responses_mapping_dir=tmp_path / "responses_mapping",
    )

    parent_request = _request()
    parent_context = _context(
        response_id="resp-root",
        conversation_id=None,
        current_text="turn one",
    )
    parent_config = await server.build_runnable_config(
        parent_request,
        parent_context,
    )
    parent_input = await server.build_input(parent_request, parent_context)
    await server.graph.ainvoke(parent_input, parent_config, durability="sync")
    await server._persist_response_checkpoint(  # noqa: SLF001
        parent_context,
        parent_config,
    )

    child_request = _request(previous_response_id="resp-root")
    child_context = _context(
        response_id="resp-child",
        conversation_id=None,
        current_text="turn two",
    )
    child_config = await server.build_runnable_config(child_request, child_context)
    child_input = await server.build_input(child_request, child_context)
    child_stream = server.graph.astream(
        child_input,
        child_config,
        stream_mode="updates",
        durability="sync",
    )
    first_update = await anext(child_stream)
    assert list(first_update) == ["plan"]
    await child_stream.aclose()

    recovery_context = _context(
        response_id="resp-child",
        conversation_id=None,
        current_text="turn two",
    )
    recovery_context.is_recovery = True
    recovery_config = await server.build_runnable_config(
        child_request,
        recovery_context,
    )
    recovered_nodes = [
        node
        async for update in server.graph.astream(
            None,
            recovery_config,
            stream_mode="updates",
            durability="sync",
        )
        for node in update
    ]

    assert recovered_nodes == ["research"]


async def test_previous_response_checkpoint_supports_sibling_fork(
    tmp_path: Path,
) -> None:
    mapping_dir = tmp_path / "responses_mapping"
    checkpoint_dir = tmp_path / "checkpoints_mapping"
    server = ResponsesHostServer(
        make_checkpointed_echo_graph(),
        responses_mapping_dir=mapping_dir,
    )

    async def invoke(
        response_id: str,
        text: str,
        previous_response_id: str | None = None,
    ) -> dict[str, object]:
        request = _request(previous_response_id=previous_response_id)
        context = _context(
            response_id=response_id,
            conversation_id=None,
            current_text=text,
        )
        config = await server.build_runnable_config(request, context)
        graph_input = await server.build_input(request, context)
        state = await server.graph.ainvoke(
            graph_input,
            config=config,
            durability="sync",
        )
        post_run_config = await server._persist_response_checkpoint(  # noqa: SLF001
            context,
            config,
        )
        assert post_run_config["configurable"]["checkpoint_id"] == (
            checkpoint_dir / response_id
        ).read_text(encoding="utf-8")
        return state

    root_state = await invoke("resp-root", "root")
    child_state = await invoke("resp-child", "child", "resp-root")
    fork_state = await invoke("resp-fork", "fork", "resp-root")

    assert [message.content for message in root_state["messages"]] == [
        "root",
        "Echo: root",
    ]
    assert [message.content for message in child_state["messages"]] == [
        "root",
        "Echo: root",
        "child",
        "Echo: child",
    ]
    assert [message.content for message in fork_state["messages"]] == [
        "root",
        "Echo: root",
        "fork",
        "Echo: fork",
    ]
    assert (mapping_dir / "resp-child").read_text(encoding="utf-8") == "resp-root"
    assert (mapping_dir / "resp-fork").read_text(encoding="utf-8") == "resp-root"


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
