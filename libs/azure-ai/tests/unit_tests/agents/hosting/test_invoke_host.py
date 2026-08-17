# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end tests for ``InvocationsHostServer`` via Starlette TestClient."""

from __future__ import annotations

import asyncio
import errno
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("starlette")

from azure.ai.agentserver.core.streaming import EventStream, streams
from azure.ai.agentserver.core.tasks import TaskContext, TaskMetadata
from azure.ai.agentserver.invocations import InvocationAgentServerHost
from azure.ai.agentserver.responses import (
    InMemoryResponseProvider,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command, Interrupt
from starlette.testclient import TestClient

from langchain_azure_ai.agents.hosting import (
    HostingFeature,
    InvocationsHostServer,
    ResponsesHostServer,
)
from langchain_azure_ai.agents.hosting._converters import (  # noqa: E402
    HITL_FUNCTION_NAME,
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
from .hitl.conftest import REAL_INTERRUPT_ASYNC_XFAIL  # noqa: E402
from .hitl.graphs import (  # noqa: E402
    build_parallel_empty_update_interrupt_graph,
    build_parallel_interrupt_graph,
    build_simple_interrupt_graph,
)


def _client(server: InvocationsHostServer) -> TestClient:
    return TestClient(server.app)


def test_constructor_registers_invocations_feature() -> None:
    with patch(
        "langchain_azure_ai.agents.hosting._invoke_host._add_process_hosting_features"
    ) as add_process_features:
        server = InvocationsHostServer(make_echo_graph())

    assert server._hosting_features == HostingFeature.INVOCATIONS
    add_process_features.assert_called_once_with(HostingFeature.INVOCATIONS)


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


@REAL_INTERRUPT_ASYNC_XFAIL
def test_invocation_emits_and_resumes_structured_hitl_items() -> None:
    server = InvocationsHostServer(build_simple_interrupt_graph())
    session_id = "invocations-hitl"

    with _client(server) as client:
        first = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={"message": "Ask for my name."},
        )
        assert first.status_code == 200, first.text
        pending = [
            item
            for item in first.json()["output"]
            if item.get("type") == "function_call"
            and item.get("name") == HITL_FUNCTION_NAME
        ]
        assert len(pending) == 1

        second = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={
                "message": [
                    {
                        "type": "function_call_output",
                        "call_id": pending[0]["call_id"],
                        "output": json.dumps({"resume": "Alice"}),
                    }
                ]
            },
        )

    assert second.status_code == 200, second.text
    assert second.json() == {"response": "ok:Alice"}


@REAL_INTERRUPT_ASYNC_XFAIL
def test_invocation_accepts_mcp_approval_response() -> None:
    server = InvocationsHostServer(build_simple_interrupt_graph())
    session_id = "invocations-mcp-approval"

    with _client(server) as client:
        first = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={"message": "Ask for my name."},
        )
        approval = next(
            item
            for item in first.json()["output"]
            if item.get("type") == "mcp_approval_request"
        )
        assert approval["id"].startswith("mcpr_")

        second = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={
                "message": [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": approval["id"],
                        "approve": True,
                    }
                ]
            },
        )

    assert second.status_code == 200, second.text
    assert second.json() == {"response": "ok:name?"}


@pytest.mark.parametrize(
    "options",
    [None, ResponsesServerOptions(steerable_conversations=True)],
)
@REAL_INTERRUPT_ASYNC_XFAIL
def test_partial_parallel_resume_emits_only_active_interrupts(
    options: ResponsesServerOptions | None,
) -> None:
    server = InvocationsHostServer(build_parallel_interrupt_graph(), options=options)
    session_id = f"parallel-active-{options is not None}"

    with _client(server) as client:
        first = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={"message": "Ask both."},
        )
        assert first.status_code == 200, first.text
        pending = {
            json.loads(item["arguments"])["value"]: item["call_id"]
            for item in first.json()["output"]
            if item.get("type") == "function_call"
        }
        assert set(pending) == {"question_a", "question_b"}

        second = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={
                "message": [
                    {
                        "type": "function_call_output",
                        "call_id": pending["question_a"],
                        "output": json.dumps({"resume": "A"}),
                    }
                ]
            },
        )

    assert second.status_code == 200, second.text
    remaining = [
        json.loads(item["arguments"])["value"]
        for item in second.json()["output"]
        if item.get("type") == "function_call"
    ]
    assert remaining == ["question_b"]


@pytest.mark.parametrize(
    "options",
    [None, ResponsesServerOptions(steerable_conversations=True)],
)
@REAL_INTERRUPT_ASYNC_XFAIL
def test_parallel_rejection_blocks_other_resume(
    options: ResponsesServerOptions | None,
) -> None:
    server = InvocationsHostServer(build_parallel_interrupt_graph(), options=options)
    session_id = f"parallel-rejection-{options is not None}"

    with _client(server) as client:
        first = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={"message": "Ask both."},
        )
        assert first.status_code == 200, first.text
        function_calls = {
            json.loads(item["arguments"])["value"]: item["call_id"]
            for item in first.json()["output"]
            if item.get("type") == "function_call"
        }
        approvals = {
            json.loads(item["arguments"])["value"]: item["id"]
            for item in first.json()["output"]
            if item.get("type") == "mcp_approval_request"
        }

        second = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={
                "message": [
                    {
                        "type": "function_call_output",
                        "call_id": function_calls["question_a"],
                        "output": json.dumps({"resume": "A"}),
                    },
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": approvals["question_b"],
                        "approve": False,
                        "reason": "Not authorized",
                    },
                ]
            },
        )

    assert second.status_code == 409, second.text
    assert "Not authorized" in second.json()["error"]


@REAL_INTERRUPT_ASYNC_XFAIL
def test_streaming_partial_resume_omits_answered_empty_update_branch() -> None:
    server = InvocationsHostServer(build_parallel_empty_update_interrupt_graph())
    session_id = "streaming-parallel-empty-update"

    with _client(server) as client:
        first = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={"message": "Ask both."},
        )
        pending = {
            json.loads(item["arguments"])["value"]: item["call_id"]
            for item in first.json()["output"]
            if item.get("type") == "function_call"
        }
        second = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={
                "message": [
                    {
                        "type": "function_call_output",
                        "call_id": pending["question_a"],
                        "output": json.dumps({"resume": "A"}),
                    }
                ],
                "stream": True,
            },
        )

    assert second.status_code == 200, second.text
    output_items = [
        json.loads(line.removeprefix("data: "))
        for line in second.text.splitlines()
        if line.startswith("data: {") and line != "data: {}"
    ]
    remaining = [
        json.loads(item["arguments"])["value"]
        for item in output_items
        if item.get("type") == "function_call"
    ]
    assert remaining == ["question_b"]


@pytest.mark.parametrize(
    ("item", "field"),
    [
        (
            {
                "type": "mcp_approval_response",
                "approval_request_id": "mcpr_1",
                "approve": "false",
            },
            "approve",
        ),
        (
            {"type": "function_call_output", "output": "Alice"},
            "call_id",
        ),
        (
            {
                "type": "function_call_output",
                "call_id": "interrupt-1",
                "output": 42,
            },
            "output",
        ),
    ],
)
def test_malformed_structured_hitl_items_return_400(
    item: dict[str, object], field: str
) -> None:
    server = InvocationsHostServer(build_simple_interrupt_graph())

    with _client(server) as client:
        response = client.post("/invocations", json={"message": [item]})

    assert response.status_code == 400, response.text
    assert field in response.json()["error"]


@REAL_INTERRUPT_ASYNC_XFAIL
def test_pending_string_honors_command_build_input_override() -> None:
    class LegacyApprovalHost(InvocationsHostServer):
        def build_input(self, message: str) -> object:
            if message == "deny":
                return Command(resume=False)
            return super().build_input(message)

    server = LegacyApprovalHost(build_simple_interrupt_graph())
    session_id = "legacy-build-input-command"

    with _client(server) as client:
        first = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={"message": "Ask for my name."},
        )
        assert first.status_code == 200, first.text
        second = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={"message": "deny"},
        )

    assert second.status_code == 200, second.text
    assert second.json() == {"response": "ok:False"}


@REAL_INTERRUPT_ASYNC_XFAIL
def test_task_backed_foreground_invocation_preserves_hitl_output() -> None:
    server = InvocationsHostServer(
        build_simple_interrupt_graph(),
        options=ResponsesServerOptions(steerable_conversations=True),
    )

    with _client(server) as client:
        response = client.post(
            "/invocations?agent_session_id=foreground-hitl",
            json={"message": "Ask for my name."},
        )

    assert response.status_code == 200, response.text
    assert any(
        item.get("type") == "function_call" and item.get("name") == HITL_FUNCTION_NAME
        for item in response.json()["output"]
    )


@REAL_INTERRUPT_ASYNC_XFAIL
def test_task_backed_mcp_rejection_does_not_resume_interrupt() -> None:
    server = InvocationsHostServer(
        build_simple_interrupt_graph(),
        options=ResponsesServerOptions(steerable_conversations=True),
    )
    session_id = "foreground-hitl-rejection"

    with _client(server) as client:
        first = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={"message": "Ask for my name."},
        )
        approval = next(
            item
            for item in first.json()["output"]
            if item.get("type") == "mcp_approval_request"
        )
        rejected = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={
                "message": [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": approval["id"],
                        "approve": False,
                        "reason": "Not authorized",
                    }
                ]
            },
        )

    assert rejected.status_code == 409, rejected.text
    assert "rejected" in rejected.json()["error"]


def test_task_backed_invalid_hitl_input_returns_400() -> None:
    server = InvocationsHostServer(
        make_echo_graph(),
        options=ResponsesServerOptions(steerable_conversations=True),
    )

    with _client(server) as client:
        response = client.post(
            "/invocations?agent_session_id=no-pending-hitl",
            json={
                "message": [
                    {
                        "type": "function_call_output",
                        "call_id": "not-pending",
                        "output": "Alice",
                    }
                ]
            },
        )

    assert response.status_code == 400, response.text
    assert "pending" in response.json()["error"]


@REAL_INTERRUPT_ASYNC_XFAIL
def test_streaming_invocation_emits_structured_hitl_items() -> None:
    server = InvocationsHostServer(build_simple_interrupt_graph())

    with _client(server) as client:
        response = client.post(
            "/invocations?agent_session_id=streaming-hitl",
            json={"message": "Ask for my name.", "stream": True},
        )

    assert response.status_code == 200, response.text
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {") and line != "data: {}"
    ]
    pending = [
        item
        for item in events
        if item.get("type") == "function_call"
        and item.get("name") == HITL_FUNCTION_NAME
    ]
    assert len(pending) == 1
    assert "event: done" in response.text


@REAL_INTERRUPT_ASYNC_XFAIL
def test_streaming_invocation_reemits_unmatched_pending_hitl_items() -> None:
    server = InvocationsHostServer(build_simple_interrupt_graph())
    session_id = "streaming-unmatched-hitl"

    with _client(server) as client:
        first = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={"message": "Ask for my name."},
        )
        assert first.status_code == 200, first.text
        response = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={
                "message": [
                    {
                        "type": "function_call_output",
                        "call_id": "not-the-pending-interrupt",
                        "output": "Alice",
                    }
                ],
                "stream": True,
            },
        )

    assert response.status_code == 200, response.text
    assert "event: output_item" in response.text
    assert HITL_FUNCTION_NAME in response.text
    assert "event: done" in response.text


@REAL_INTERRUPT_ASYNC_XFAIL
def test_background_invocation_emits_and_resumes_structured_hitl_items() -> None:
    server = InvocationsHostServer(
        build_simple_interrupt_graph(),
        options=ResponsesServerOptions(resilient_background=True),
    )
    session_id = "background-hitl"

    with _client(server) as client:
        first = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={"message": "Ask for my name.", "background": True},
        )
        assert first.status_code == 202, first.text
        first_result = client.get(f"/invocations/{first.json()['id']}")
        for _ in range(20):
            if first_result.json().get("status") == "completed":
                break
            first_result = client.get(f"/invocations/{first.json()['id']}")
        pending = [
            item
            for item in first_result.json()["output"]
            if item.get("type") == "function_call"
            and item.get("name") == HITL_FUNCTION_NAME
        ]
        assert len(pending) == 1

        second = client.post(
            f"/invocations?agent_session_id={session_id}",
            json={
                "message": [
                    {
                        "type": "function_call_output",
                        "call_id": pending[0]["call_id"],
                        "output": json.dumps({"resume": "Alice"}),
                    }
                ],
                "background": True,
                "previous_invocation_id": first.json()["id"],
            },
        )
        assert second.status_code == 202, second.text
        second_result = client.get(f"/invocations/{second.json()['id']}")
        for _ in range(20):
            if second_result.json().get("status") == "completed":
                break
            second_result = client.get(f"/invocations/{second.json()['id']}")

    assert second_result.status_code == 200, second_result.text
    assert second_result.json()["response"] == "ok:Alice"
    assert "output" not in second_result.json()


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
async def test_recovered_invocation_reclaims_stale_replay_stream_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    captured: dict[str, object] = {}
    server = InvocationsHostServer(
        make_recovery_probe_graph(captured),
        options=ResponsesServerOptions(resilient_background=True),
    )
    invocation_id = f"stale-lock-{uuid.uuid4()}"
    session_id = "stale-lock-recovery-session"
    stream_path = tmp_path / "streams" / f"{invocation_id}.jsonl"
    stream_path.touch()
    lock_path = stream_path.with_suffix(".jsonl.lock")
    lock_path.touch()
    original_get_or_create = streams.get_or_create
    attempts = 0

    async def fail_once(candidate_id: str) -> EventStream:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            cause = FileExistsError(
                errno.EEXIST,
                "File exists",
                str(lock_path),
            )
            raise RuntimeError("replay stream lock contention") from cause
        return await original_get_or_create(candidate_id)

    monkeypatch.setattr(streams, "get_or_create", fail_once)
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
                "langgraph_thread_id": session_id,
                "langgraph_checkpoint_id": "checkpoint-1",
            }
        ),
        entry_mode="recovered",
    )

    try:
        result = await server._execute_task_invocation(context)
    finally:
        await streams.delete(invocation_id)

    assert attempts > 1
    assert not lock_path.exists()
    assert captured["input"] is None
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_fresh_invocation_does_not_reclaim_replay_stream_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    server = InvocationsHostServer(
        make_recovery_probe_graph({}),
        options=ResponsesServerOptions(resilient_background=True),
    )
    invocation_id = f"live-lock-{uuid.uuid4()}"
    session_id = "live-lock-session"
    stream_path = tmp_path / "streams" / f"{invocation_id}.jsonl"
    stream_path.touch()
    lock_path = stream_path.with_suffix(".jsonl.lock")
    lock_path.touch()

    async def locked(_candidate_id: str) -> EventStream:
        cause = FileExistsError(
            errno.EEXIST,
            "File exists",
            str(lock_path),
        )
        raise RuntimeError("replay stream lock contention") from cause

    monkeypatch.setattr(streams, "get_or_create", locked)
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
        metadata=TaskMetadata(),
    )

    with pytest.raises(RuntimeError, match="lock contention"):
        await server._execute_task_invocation(context)

    assert lock_path.exists()


@pytest.mark.asyncio
async def test_recovered_invocation_reads_pending_hitl_from_latest_checkpoint() -> None:
    captured: dict[str, object] = {}
    pending = Interrupt(value="Approve recovered action?", id="recovered-interrupt")
    server = InvocationsHostServer(
        make_recovery_probe_graph(captured, pending),
        options=ResponsesServerOptions(resilient_background=True),
    )
    invocation_id = f"recovered-hitl-{uuid.uuid4()}"
    session_id = "recovered-hitl-session"
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
                "langgraph_thread_id": session_id,
                "langgraph_checkpoint_id": "checkpoint-before-recovery",
            }
        ),
        entry_mode="recovered",
    )

    result = await server._execute_task_invocation(context)

    assert any(
        item.get("type") == "function_call"
        and item.get("call_id") == "recovered-interrupt"
        for item in result["output"]
    )


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
