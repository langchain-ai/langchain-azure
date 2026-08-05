# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end tests for ``InvocationsHostServer`` via Starlette TestClient."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("starlette")

from azure.ai.agentserver.core.streaming import streams  # noqa: E402
from azure.ai.agentserver.core.tasks import TaskContext, TaskMetadata  # noqa: E402
from azure.ai.agentserver.invocations import InvocationAgentServerHost  # noqa: E402
from azure.ai.agentserver.responses import (  # noqa: E402
    InMemoryResponseProvider,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
)
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from langchain_azure_ai.agents.hosting import (  # noqa: E402
    InvocationsHostServer,
    ResponsesHostServer,
)

from .conftest import (  # noqa: E402
    make_checkpointed_echo_graph,
    make_checkpointed_steering_graph,
    make_custom_state_graph,
    make_echo_graph,
    make_recovery_probe_graph,
    make_shutdown_checkpoint_graph,
    make_streaming_graph,
)


def _client(server: InvocationsHostServer) -> TestClient:
    return TestClient(server.app)


def test_non_streaming_invocation_returns_response_text() -> None:
    server = InvocationsHostServer(make_echo_graph())
    with _client(server) as client:
        resp = client.post("/invocations", json={"message": "hi"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"response": "Echo: hi"}


def test_non_streaming_invocation_accepts_runnable_without_builder() -> None:
    def invoke(payload: dict[str, object]) -> dict[str, list[AIMessage]]:
        del payload
        return {"messages": [AIMessage(content="Runnable response")]}

    server = InvocationsHostServer(RunnableLambda(invoke))
    with _client(server) as client:
        resp = client.post("/invocations", json={"message": "hi"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"response": "Runnable response"}


def test_non_streaming_invocation_uses_output_parser() -> None:
    def invoke(payload: dict[str, object]) -> dict[str, str]:
        del payload
        return {"answer": "custom response"}

    def parse_output(output: dict[str, str]) -> str:
        return output["answer"]

    server = InvocationsHostServer(
        RunnableLambda(invoke),
        output_parser=parse_output,
    )
    with _client(server) as client:
        resp = client.post("/invocations", json={"message": "hi"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"response": "custom response"}


def test_streaming_invocation_emits_sse_tokens_and_done() -> None:
    server = InvocationsHostServer(make_streaming_graph())
    with _client(server) as client:
        resp = client.post("/invocations", json={"message": "ignored", "stream": True})
    assert resp.status_code == 200, resp.text
    body = resp.text
    tokens: list[str] = []
    saw_done = False
    for line in body.splitlines():
        if line.startswith("data:"):
            data = line.split(":", 1)[1].strip()
            if not data or data == "{}":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "token" in payload:
                tokens.append(payload["token"])
        elif line.startswith("event:") and line.split(":", 1)[1].strip() == "done":
            saw_done = True
    assert "".join(tokens) == "Hello, world!"
    assert saw_done


def test_session_id_is_propagated_to_response_headers() -> None:
    server = InvocationsHostServer(make_echo_graph())
    with _client(server) as client:
        resp = client.post("/invocations", json={"message": "hi"})
    assert resp.status_code == 200
    assert resp.headers.get("x-agent-session-id")
    assert resp.headers.get("x-agent-invocation-id")


def test_missing_message_returns_400() -> None:
    server = InvocationsHostServer(make_echo_graph())
    with _client(server) as client:
        resp = client.post("/invocations", json={})
    assert resp.status_code == 400
    assert "message" in resp.json()["error"].lower()


def test_constructor_rejects_non_messages_state_schema() -> None:
    with pytest.raises(ValueError, match="messages"):
        InvocationsHostServer(make_custom_state_graph())


def test_constructor_rejects_resilient_background_without_checkpointer() -> None:
    options = ResponsesServerOptions(resilient_background=True)

    with pytest.raises(
        ValueError,
        match="requires a LangGraph checkpointer when resilient_background=True",
    ):
        InvocationsHostServer(make_echo_graph(), options=options)


def test_constructor_accepts_resilient_background_with_checkpointer() -> None:
    options = ResponsesServerOptions(resilient_background=True)

    InvocationsHostServer(make_checkpointed_echo_graph(), options=options)


def test_resilient_background_invocation_can_be_retrieved() -> None:
    options = ResponsesServerOptions(resilient_background=True)
    server = InvocationsHostServer(
        make_checkpointed_echo_graph(),
        options=options,
    )

    with _client(server) as client:
        accepted = client.post(
            "/invocations",
            json={"message": "hi", "background": True},
        )
        assert accepted.status_code == 202, accepted.text
        invocation = accepted.json()
        assert invocation["id"] == accepted.headers["x-agent-invocation-id"]
        assert invocation["status"] in {"queued", "in_progress"}

        result = client.get(f"/invocations/{invocation['id']}")
        for _ in range(20):
            if result.json().get("status") == "completed":
                break
            result = client.get(f"/invocations/{invocation['id']}")

    assert result.status_code == 200, result.text
    assert result.json()["status"] == "completed"
    assert result.json()["response"] == "Echo: hi"


def test_background_streaming_is_rejected() -> None:
    options = ResponsesServerOptions(resilient_background=True)
    server = InvocationsHostServer(
        make_checkpointed_echo_graph(),
        options=options,
    )

    with _client(server) as client:
        response = client.post(
            "/invocations",
            json={"message": "hi", "stream": True, "background": True},
        )

    assert response.status_code == 400
    assert "background" in response.json()["error"].lower()


def test_get_unknown_invocation_returns_404() -> None:
    server = InvocationsHostServer(make_echo_graph())

    with _client(server) as client:
        response = client.get("/invocations/missing")

    assert response.status_code == 404


def test_steerable_conversation_supersedes_active_turn() -> None:
    first_turn_started = threading.Event()
    options = ResponsesServerOptions(steerable_conversations=True)
    server = InvocationsHostServer(
        make_checkpointed_steering_graph(first_turn_started),
        options=options,
    )

    with _client(server) as client, ThreadPoolExecutor(max_workers=1) as executor:
        first_future = executor.submit(
            client.post,
            "/invocations?agent_session_id=steer-session",
            json={"message": "first"},
        )
        assert first_turn_started.wait(timeout=5)

        second = client.post(
            "/invocations?agent_session_id=steer-session",
            json={"message": "second"},
        )
        first = first_future.result(timeout=5)

    assert first.status_code == 409, first.text
    assert "steered" in first.json()["error"].lower()
    assert second.status_code == 200, second.text
    assert second.json() == {"response": "Echo: second"}


def test_queued_steering_turn_can_be_cancelled() -> None:
    first_turn_started = threading.Event()
    cancellation_started = threading.Event()
    release_cancellation = threading.Event()
    server = InvocationsHostServer(
        make_checkpointed_steering_graph(
            first_turn_started,
            cancellation_started,
            release_cancellation,
        ),
        options=ResponsesServerOptions(steerable_conversations=True),
    )
    second_invocation_id = f"queued-{uuid.uuid4()}"

    with _client(server) as client, ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            client.post,
            "/invocations?agent_session_id=queued-session",
            json={"message": "first"},
        )
        assert first_turn_started.wait(timeout=5)

        second_future = executor.submit(
            client.post,
            "/invocations?agent_session_id=queued-session",
            json={"message": "second"},
            headers={"x-agent-invocation-id": second_invocation_id},
        )
        assert cancellation_started.wait(timeout=5)

        cancelled = client.post(f"/invocations/{second_invocation_id}/cancel")
        second = second_future.result(timeout=5)
        release_cancellation.set()
        first = first_future.result(timeout=5)

    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"
    assert second.status_code == 409, second.text
    assert "cancelled" in second.json()["error"].lower()
    assert first.status_code == 409, first.text


def test_background_invocation_can_be_cancelled() -> None:
    first_turn_started = threading.Event()
    options = ResponsesServerOptions(resilient_background=True)
    server = InvocationsHostServer(
        make_checkpointed_steering_graph(first_turn_started),
        options=options,
    )

    with _client(server) as client:
        accepted = client.post(
            "/invocations",
            json={"message": "first", "background": True},
        )
        assert accepted.status_code == 202, accepted.text
        assert first_turn_started.wait(timeout=5)

        invocation_id = accepted.json()["id"]
        cancelled = client.post(f"/invocations/{invocation_id}/cancel")
        assert cancelled.status_code == 200, cancelled.text

        result = client.get(f"/invocations/{invocation_id}")
        for _ in range(20):
            if result.json().get("status") == "cancelled":
                break
            result = client.get(f"/invocations/{invocation_id}")

    assert result.status_code == 200, result.text
    assert result.json()["status"] == "cancelled"


def test_steerable_streaming_invocation_emits_tokens_and_done() -> None:
    options = ResponsesServerOptions(steerable_conversations=True)
    server = InvocationsHostServer(make_streaming_graph(), options=options)

    with _client(server) as client:
        response = client.post(
            "/invocations?agent_session_id=stream-session",
            json={"message": "ignored", "stream": True},
        )

    assert response.status_code == 200, response.text
    tokens: list[str] = []
    saw_done = False
    for line in response.text.splitlines():
        if line.startswith("data:"):
            data = line.split(":", 1)[1].strip()
            if data and data != "{}":
                payload = json.loads(data)
                if "token" in payload:
                    tokens.append(payload["token"])
        elif line == "event: done":
            saw_done = True

    assert "".join(tokens) == "Hello, world!"
    assert saw_done


def test_steerable_conversation_does_not_require_checkpointer() -> None:
    options = ResponsesServerOptions(steerable_conversations=True)
    server = InvocationsHostServer(make_echo_graph(), options=options)

    with _client(server) as client:
        response = client.post(
            "/invocations?agent_session_id=no-checkpointer",
            json={"message": "hi"},
        )

    assert response.status_code == 200, response.text
    assert response.json() == {"response": "Echo: hi"}


def test_steerable_conversation_accepts_runnable_without_builder() -> None:
    def invoke(payload: dict[str, object]) -> dict[str, list[AIMessage]]:
        del payload
        return {"messages": [AIMessage(content="Runnable response")]}

    server = InvocationsHostServer(
        RunnableLambda(invoke),
        options=ResponsesServerOptions(steerable_conversations=True),
    )

    with _client(server) as client:
        response = client.post(
            "/invocations?agent_session_id=runnable-session",
            json={"message": "hi"},
        )

    assert response.status_code == 200, response.text
    assert response.json() == {"response": "Runnable response"}


def test_resilient_foreground_invocation_honors_chain_precondition() -> None:
    server = InvocationsHostServer(
        make_checkpointed_echo_graph(),
        options=ResponsesServerOptions(resilient_background=True),
    )

    with _client(server) as client:
        first = client.post(
            "/invocations?agent_session_id=resilient-session",
            json={"message": "first"},
        )
        assert first.status_code == 200, first.text

        second = client.post(
            "/invocations?agent_session_id=resilient-session",
            json={
                "message": "second",
                "previous_invocation_id": "wrong-id",
            },
        )

    assert second.status_code == 409, second.text
    assert "previous_invocation_id" in second.json()["error"]


@pytest.mark.asyncio
async def test_recovered_background_invocation_resumes_checkpoint() -> None:
    captured: dict[str, object] = {}
    options = ResponsesServerOptions(resilient_background=True)
    server = InvocationsHostServer(
        make_recovery_probe_graph(captured),
        options=options,
    )
    invocation_id = f"recovery-{uuid.uuid4()}"
    session_id = "recovery-session"
    metadata = TaskMetadata(
        {
            "invocation_input_id": invocation_id,
            "langgraph_thread_id": session_id,
            "langgraph_checkpoint_id": "checkpoint-1",
        }
    )
    context = TaskContext(
        task_id=session_id,
        session_id=session_id,
        input={
            "invocation_id": invocation_id,
            "session_id": session_id,
            "message": "hi",
            "stream": False,
        },
        input_id=invocation_id,
        metadata=metadata,
        entry_mode="recovered",
    )

    result = await server._execute_task_invocation(context)

    assert captured["input"] is None
    configurable = captured["config"]["configurable"]  # type: ignore[index]
    assert configurable["thread_id"] == session_id
    assert configurable["checkpoint_id"] == "checkpoint-1"
    assert result["status"] == "completed"
    assert result["response"] == "Recovered"


@pytest.mark.asyncio
async def test_recovered_completed_invocation_tolerates_closed_stream() -> None:
    captured: dict[str, object] = {}
    server = InvocationsHostServer(
        make_recovery_probe_graph(captured),
        options=ResponsesServerOptions(resilient_background=True),
    )
    invocation_id = f"completed-recovery-{uuid.uuid4()}"
    session_id = "completed-recovery-session"
    event_stream = await streams.get_or_create(invocation_id)
    await server._emit_invocation_status(
        event_stream,
        invocation_id=invocation_id,
        session_id=session_id,
        status="completed",
        response="Recovered",
        close=True,
    )
    context = TaskContext(
        task_id=session_id,
        session_id=session_id,
        input={
            "invocation_id": invocation_id,
            "session_id": session_id,
            "message": "hi",
            "stream": False,
        },
        input_id=invocation_id,
        metadata=TaskMetadata(
            {
                "invocation_input_id": invocation_id,
                "invocation_response": "Recovered",
            }
        ),
        entry_mode="recovered",
    )

    try:
        result = await server._execute_task_invocation(context)
    finally:
        await streams.delete(invocation_id)

    assert result["status"] == "completed"
    assert result["response"] == "Recovered"
    assert captured == {}


@pytest.mark.asyncio
async def test_recovered_cancelled_invocation_uses_terminal_stream() -> None:
    captured: dict[str, object] = {}
    server = InvocationsHostServer(
        make_recovery_probe_graph(captured),
        options=ResponsesServerOptions(resilient_background=True),
    )
    invocation_id = f"cancelled-recovery-{uuid.uuid4()}"
    session_id = "cancelled-recovery-session"
    event_stream = await streams.get_or_create(invocation_id)
    await server._emit_invocation_status(
        event_stream,
        invocation_id=invocation_id,
        session_id=session_id,
        status="cancelled",
        error={"code": "cancelled", "message": "Invocation was cancelled."},
        close=True,
    )
    context = TaskContext(
        task_id=session_id,
        session_id=session_id,
        input={
            "invocation_id": invocation_id,
            "session_id": session_id,
            "message": "hi",
            "stream": False,
        },
        input_id=invocation_id,
        metadata=TaskMetadata({"invocation_input_id": invocation_id}),
        entry_mode="recovered",
    )

    try:
        result = await server._execute_task_invocation(context)
    finally:
        await streams.delete(invocation_id)

    assert result["status"] == "cancelled"
    assert result["error"]["code"] == "cancelled"
    assert captured == {}


@pytest.mark.asyncio
async def test_shutdown_persists_ready_checkpoint_before_recovery_exit() -> None:
    server = InvocationsHostServer(
        make_shutdown_checkpoint_graph(),
        options=ResponsesServerOptions(resilient_background=True),
    )
    invocation_id = f"shutdown-checkpoint-{uuid.uuid4()}"
    session_id = "shutdown-checkpoint-session"
    metadata = TaskMetadata()
    context = TaskContext(
        task_id=session_id,
        session_id=session_id,
        input={
            "invocation_id": invocation_id,
            "session_id": session_id,
            "message": "hi",
            "stream": False,
        },
        input_id=invocation_id,
        metadata=metadata,
    )

    try:
        await server._execute_task_invocation(context)
    finally:
        await streams.delete(invocation_id)

    assert metadata["langgraph_thread_id"] == "shutdown-thread"
    assert metadata["langgraph_checkpoint_id"] == "checkpoint-ready"


@pytest.mark.asyncio
async def test_invocation_cancel_request_stops_running_turn() -> None:
    turn_started = threading.Event()
    server = InvocationsHostServer(
        make_checkpointed_steering_graph(turn_started),
        options=ResponsesServerOptions(steerable_conversations=True),
    )
    invocation_id = f"promoted-cancel-{uuid.uuid4()}"
    session_id = "promoted-cancel-session"
    context = TaskContext(
        task_id=session_id,
        session_id=session_id,
        input={
            "invocation_id": invocation_id,
            "session_id": session_id,
            "message": "first",
            "stream": False,
        },
        input_id=invocation_id,
        metadata=TaskMetadata(),
    )

    execution = asyncio.create_task(server._execute_task_invocation(context))
    assert await asyncio.to_thread(turn_started.wait, 5)
    server._cancel_requests[invocation_id].set()

    try:
        result = await asyncio.wait_for(execution, timeout=5)
    finally:
        await streams.delete(invocation_id)

    assert result["status"] == "cancelled"
    assert result["error"]["code"] == "cancelled"
    assert invocation_id not in server._cancel_requests


def test_invocations_and_responses_streams_can_share_host() -> None:
    class CombinedHost(InvocationAgentServerHost, ResponsesAgentServerHost):
        pass

    options = ResponsesServerOptions(steerable_conversations=True)
    app = CombinedHost(options=options, store=InMemoryResponseProvider())
    ResponsesHostServer(make_streaming_graph(), app=app, options=options)
    InvocationsHostServer(make_streaming_graph(), app=app, options=options)

    with TestClient(app) as client:
        invocation = client.post(
            "/invocations?agent_session_id=combined-session",
            json={"message": "ignored", "stream": True},
        )
        response = client.post(
            "/responses",
            json={"input": "ignored", "stream": True, "model": "test"},
        )

    assert invocation.status_code == 200, invocation.text
    assert "event: done" in invocation.text
    assert response.status_code == 200, response.text
    assert "event: response.completed" in response.text


def test_previous_invocation_id_mismatch_returns_conflict() -> None:
    options = ResponsesServerOptions(steerable_conversations=True)
    server = InvocationsHostServer(
        make_checkpointed_echo_graph(),
        options=options,
    )

    with _client(server) as client:
        first = client.post(
            "/invocations?agent_session_id=linear-session",
            json={"message": "first"},
        )
        assert first.status_code == 200, first.text

        second = client.post(
            "/invocations?agent_session_id=linear-session",
            json={
                "message": "second",
                "previous_invocation_id": "wrong-id",
            },
        )

    assert second.status_code == 409, second.text
    assert "previous_invocation_id" in second.json()["error"]
