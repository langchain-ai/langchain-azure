"""HTTP regression tests for steering conversations with pending HITL decisions."""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import pytest

pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("azure.ai.agentserver.invocations")
pytest.importorskip("starlette")

from azure.ai.agentserver.core.tasks import (  # noqa: E402
    resilient_tasks_enabled,
    set_resilient_tasks_enabled,
)
from azure.ai.agentserver.responses import ResponsesServerOptions  # noqa: E402
from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import HumanInTheLoopMiddleware  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel,
)
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatResult  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402
from langgraph.types import Command, Interrupt, interrupt  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from langchain_azure_ai.agents.hosting import (  # noqa: E402
    InvocationsHostServer,
    ResponsesHostServer,
)
from langchain_azure_ai.agents.hosting._converters import (  # noqa: E402
    HITL_FUNCTION_NAME,
)

from .conftest import (  # noqa: E402
    approval_requests,
    assistant_text,
    resume_item,
    sentinels,
    sse_payloads,
)

Protocol = Literal["responses", "invocations"]


class _ApprovalProtocol:
    async def build_rejection_command(
        self,
        rejections: Mapping[str, str | None],
        pending: Sequence[Interrupt],
        config: RunnableConfig,
    ) -> Command:
        return Command(
            resume={
                interrupt_id: {"type": "reject", "reason": reason}
                for interrupt_id, reason in rejections.items()
            }
        )

    async def build_steering_command(
        self,
        graph_input: dict[str, Any],
        pending: Sequence[Interrupt],
        config: RunnableConfig,
    ) -> Command:
        return Command(
            resume={item.id: {"type": "superseded"} for item in pending},
            update=graph_input,
        )


class _CustomResponsesHost(_ApprovalProtocol, ResponsesHostServer):
    pass


class _CustomInvocationsHost(_ApprovalProtocol, InvocationsHostServer):
    pass


@pytest.fixture(autouse=True)
def isolated_hosting_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[None]:
    previous_enablement = resilient_tasks_enabled()
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    for exporter in (
        "OTEL_TRACES_EXPORTER",
        "OTEL_METRICS_EXPORTER",
        "OTEL_LOGS_EXPORTER",
    ):
        monkeypatch.setenv(exporter, "none")
    monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    set_resilient_tasks_enabled(False)
    try:
        yield
    finally:
        set_resilient_tasks_enabled(previous_enablement)


def _build_approval_graph(
    *,
    checkpointer: InMemorySaver | None = None,
    before_approval: Callable[[RunnableConfig], Awaitable[None]] | None = None,
) -> CompiledStateGraph:
    async def respond(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        user_text = next(
            message.content
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )
        if user_text == "first":
            if before_approval is not None:
                await before_approval(config)
            decision = interrupt({"question": "Approve first?"})
            if isinstance(decision, dict) and decision.get("type") == "reject":
                return {
                    "messages": [AIMessage(content=f"Rejected: {decision['reason']}")]
                }
            return {"messages": [AIMessage(content=f"Decision: {decision}")]}
        return {"messages": [AIMessage(content=f"Echo: {user_text}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def _post_turn(
    client: TestClient,
    protocol: Protocol,
    conversation_id: str,
    message: str | list[dict[str, Any]],
    *,
    background: bool = False,
    stream: bool = False,
) -> Any:
    if protocol == "responses":
        return client.post(
            "/responses",
            json={
                "input": message,
                "conversation": {"id": conversation_id},
                "background": background,
                "stream": stream,
            },
        )
    return client.post(
        f"/invocations?agent_session_id={conversation_id}",
        json={"message": message, "background": background, "stream": stream},
        headers={"x-agent-invocation-id": f"test-{uuid.uuid4().hex}"},
    )


def _settled_payload(
    client: TestClient,
    protocol: Protocol,
    response: Any,
    mode: str,
) -> dict[str, Any]:
    if mode == "stream":
        assert response.status_code == 200, response.text
        if protocol == "responses":
            terminal = [
                payload["response"]
                for payload in sse_payloads(response.text)
                if payload.get("type") in {"response.completed", "response.failed"}
            ]
            assert len(terminal) == 1, response.text
            return terminal[0]
        assert "event: done" in response.text or "event: error" in response.text
        invocation_id = response.request.headers["x-agent-invocation-id"]
        fetched = client.get(f"/invocations/{invocation_id}")
        assert fetched.status_code == 200, fetched.text
        payload = fetched.json()
        if payload["status"] == "failed":
            assert "event: error" in response.text, response.text
            assert "event: done" not in response.text, response.text
            emitted = sse_payloads(response.text)
            assert emitted[:-1] == payload.get("output", []), response.text
            assert emitted[-1] == payload["error"], response.text
        return payload
    payload = response.json()
    if mode == "background":
        assert response.status_code in {200, 202}, response.text
        for _attempt in range(100):
            fetched = client.get(f"/{protocol}/{payload['id']}")
            assert fetched.status_code == 200, fetched.text
            payload = fetched.json()
            if payload["status"] in {"completed", "failed", "cancelled"}:
                break
        assert payload["status"] in {"completed", "failed", "cancelled"}, payload
    return payload


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
@pytest.mark.parametrize("steerable", [False, True])
def test_reject_allows_next_message_in_same_conversation(
    protocol: Protocol,
    steerable: bool,
) -> None:
    """Rejecting an action must not trap later messages behind its approval."""
    host_type = (
        _CustomResponsesHost if protocol == "responses" else _CustomInvocationsHost
    )
    host = host_type(
        _build_approval_graph(),
        options=ResponsesServerOptions(
            steerable_conversations=steerable,
            resilient_background=steerable,
        ),
    )
    conversation_id = f"reject-followup-{uuid.uuid4().hex}"

    with TestClient(host.app) as client:
        first = _post_turn(client, protocol, conversation_id, "first")
        assert first.status_code == 200, first.text
        approvals = approval_requests(first.json())
        assert len(approvals) == 1, first.json()

        rejected = _post_turn(
            client,
            protocol,
            conversation_id,
            [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approvals[0]["id"],
                    "approve": False,
                    "reason": "Not authorized",
                }
            ],
        )
        assert rejected.status_code == 200, rejected.text
        rejection_text = (
            assistant_text(rejected.json())
            if protocol == "responses"
            else rejected.json()["response"]
        )
        assert rejection_text == "Rejected: Not authorized", rejected.json()

        followup = _post_turn(client, protocol, conversation_id, "second")

    assert followup.status_code == 200, followup.text
    payload = followup.json()
    assert not any(
        item.get("name") == HITL_FUNCTION_NAME for item in payload.get("output", [])
    ), "The next message re-emitted the rejected approval instead of continuing."
    text = assistant_text(payload) if protocol == "responses" else payload["response"]
    assert text == "Echo: second", payload


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
@pytest.mark.parametrize("with_hook", [False, True])
@pytest.mark.parametrize("mode", ["task", "stream", "background"])
def test_steering_handles_new_message_after_old_interrupt_is_persisted(
    protocol: Protocol,
    with_hook: bool,
    mode: str,
) -> None:
    """A late checkpointed approval must not consume the queued steering turn."""
    interrupt_persisted = threading.Event()
    cancellation_signal: asyncio.Event | None = None

    async def capture_cancellation_signal(config: RunnableConfig) -> None:
        nonlocal cancellation_signal
        configurable = config["configurable"]
        cancellation_signal = (
            configurable["response_cancellation_signal"]
            if protocol == "responses"
            else configurable["invocation_context"].cancel
        )

    class PausingSaver(InMemorySaver):
        async def aput_writes(
            self,
            config: RunnableConfig,
            writes: Sequence[tuple[str, Any]],
            task_id: str,
            task_path: str = "",
        ) -> None:
            await super().aput_writes(config, writes, task_id, task_path)
            if any(channel == "__interrupt__" for channel, _value in writes):
                assert cancellation_signal is not None
                interrupt_persisted.set()
                await asyncio.wait_for(cancellation_signal.wait(), timeout=5)

    host_type = (
        ResponsesHostServer if protocol == "responses" else InvocationsHostServer
    )
    if with_hook:
        host_type = (
            _CustomResponsesHost if protocol == "responses" else _CustomInvocationsHost
        )
    host = host_type(
        _build_approval_graph(
            checkpointer=PausingSaver(),
            before_approval=capture_cancellation_signal,
        ),
        options=ResponsesServerOptions(
            steerable_conversations=True,
            resilient_background=True,
        ),
    )
    conversation_id = f"late-approval-{uuid.uuid4().hex}"

    with TestClient(host.app) as client:
        first = _post_turn(client, protocol, conversation_id, "first", background=True)
        assert first.status_code in {200, 202}, first.text
        assert interrupt_persisted.wait(timeout=5), "The first interrupt was not saved."
        second = _post_turn(
            client,
            protocol,
            conversation_id,
            "second",
            stream=mode == "stream",
            background=mode == "background",
        )
        payload = _settled_payload(client, protocol, second, mode)
        if not with_hook:
            if protocol == "responses" or mode != "task":
                assert payload["status"] == "failed", payload
                assert payload["error"]["code"] == "pending_hitl_conflict", payload
            else:
                assert second.status_code == 409, second.text
                assert payload["code"] == "pending_hitl_conflict", payload
            pending = sentinels(payload)
            assert len(pending) == 1, payload
            resumed = _post_turn(
                client,
                protocol,
                conversation_id,
                [resume_item(pending[0]["call_id"], "answered")],
            )
            assert resumed.status_code == 200, resumed.text
            resumed_text = (
                assistant_text(resumed.json())
                if protocol == "responses"
                else resumed.json()["response"]
            )
            assert resumed_text == "Decision: answered", resumed.json()
            retried = _post_turn(client, protocol, conversation_id, "second")
            assert retried.status_code == 200, retried.text
            payload = retried.json()

    assert payload.get("status", "completed") == "completed", payload
    assert not any(
        item.get("name") == HITL_FUNCTION_NAME for item in payload.get("output", [])
    ), "The steering turn re-emitted the superseded approval instead of its answer."
    text = assistant_text(payload) if protocol == "responses" else payload["response"]
    assert text == "Echo: second", payload


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
@pytest.mark.parametrize("steerable", [False, True])
def test_unknown_rejection_preserves_pending_interrupt(
    protocol: Protocol,
    steerable: bool,
) -> None:
    host_type = (
        ResponsesHostServer if protocol == "responses" else InvocationsHostServer
    )
    host = host_type(
        _build_approval_graph(),
        options=ResponsesServerOptions(
            steerable_conversations=steerable,
            resilient_background=steerable,
        ),
    )
    conversation_id = f"unknown-rejection-{uuid.uuid4().hex}"
    with TestClient(host.app) as client:
        first = _post_turn(client, protocol, conversation_id, "first")
        assert first.status_code == 200, first.text
        approval = approval_requests(first.json())[0]
        rejected = _post_turn(
            client,
            protocol,
            conversation_id,
            [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval["id"],
                    "approve": False,
                    "reason": "Denied",
                }
            ],
        )
        payload = rejected.json()
        if protocol == "responses":
            assert rejected.status_code == 200, rejected.text
            assert payload["status"] == "failed", payload
            assert payload["error"]["code"] == "unsupported_hitl_rejection", payload
        else:
            assert rejected.status_code == 409, rejected.text
            assert payload["code"] == "unsupported_hitl_rejection", payload
        assert sentinels(payload)[0]["call_id"] == sentinels(first.json())[0]["call_id"]
        resumed = _post_turn(
            client,
            protocol,
            conversation_id,
            [
                resume_item(sentinels(payload)[0]["call_id"], "answered"),
            ],
        )
        assert resumed.status_code == 200, resumed.text
        resumed_text = (
            assistant_text(resumed.json())
            if protocol == "responses"
            else resumed.json()["response"]
        )
        assert resumed_text == "Decision: answered", resumed.json()


@pytest.mark.parametrize("reject", [False, True])
def test_steering_does_not_drop_text_submitted_with_hitl_decision(reject: bool) -> None:
    host = _CustomResponsesHost(
        _build_approval_graph(),
        options=ResponsesServerOptions(
            steerable_conversations=True,
            resilient_background=True,
        ),
    )
    conversation_id = f"mixed-hitl-{uuid.uuid4().hex}"
    with TestClient(host.app) as client:
        first = _post_turn(client, "responses", conversation_id, "first")
        assert first.status_code == 200, first.text
        pending_id = sentinels(first.json())[0]["call_id"]
        decision = resume_item(pending_id, "answered")
        expected = "Decision: answered"
        if reject:
            decision = {
                "type": "mcp_approval_response",
                "approval_request_id": approval_requests(first.json())[0]["id"],
                "approve": False,
                "reason": "Not authorized",
            }
            expected = "Rejected: Not authorized"
        combined = _post_turn(
            client,
            "responses",
            conversation_id,
            [decision, {"role": "user", "content": "second"}],
        )
        payload = combined.json()
        assert payload["status"] == "failed", payload
        assert payload["error"]["code"] == "pending_hitl_conflict", payload
        assert sentinels(payload)[0]["call_id"] == pending_id

        resumed = _post_turn(client, "responses", conversation_id, [decision])
        assert resumed.json()["status"] == "completed", resumed.json()
        assert assistant_text(resumed.json()) == expected
        followup = _post_turn(client, "responses", conversation_id, "second")
        assert assistant_text(followup.json()) == "Echo: second", followup.json()


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
@pytest.mark.parametrize("mode", ["direct", "task", "stream", "background"])
def test_standard_middleware_rejection_skips_tools_and_continues(
    protocol: Protocol,
    mode: str,
) -> None:
    executed: list[str] = []
    model_inputs: list[list[BaseMessage]] = []

    @tool
    def perform_action(value: str) -> str:
        """Record an action that requires approval."""
        executed.append(value)
        return value

    class ToolCallingModel(FakeMessagesListChatModel):
        def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
            return self

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            model_inputs.append([message.model_copy(deep=True) for message in messages])
            return super()._generate(messages, stop, run_manager, **kwargs)

    model = ToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "perform_action",
                        "args": {"value": "first"},
                        "id": "call-first",
                    },
                    {
                        "name": "perform_action",
                        "args": {"value": "other"},
                        "id": "call-other",
                    },
                ],
            ),
            AIMessage(content="Both actions rejected."),
            AIMessage(content="Echo: second"),
        ]
    )
    graph = create_agent(
        model,
        tools=[perform_action],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"perform_action": True})],
        checkpointer=InMemorySaver(),
    )
    host_type = (
        ResponsesHostServer if protocol == "responses" else InvocationsHostServer
    )
    host = host_type(
        graph,
        options=ResponsesServerOptions(
            steerable_conversations=mode != "direct",
            resilient_background=mode != "direct",
        ),
    )
    conversation_id = f"middleware-rejection-{uuid.uuid4().hex}"

    with TestClient(host.app) as client:
        first = _post_turn(client, protocol, conversation_id, "first")
        assert first.status_code == 200, first.text
        approvals = approval_requests(first.json())
        assert len(approvals) == 1, first.json()
        rejected = _post_turn(
            client,
            protocol,
            conversation_id,
            [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approvals[0]["id"],
                    "approve": False,
                    "reason": "Not authorized",
                }
            ],
            background=mode == "background",
            stream=mode == "stream",
        )

        if mode == "stream":
            assert rejected.status_code == 200, rejected.text
            if protocol == "responses":
                payloads = sse_payloads(rejected.text)
                assert any(
                    payload.get("type") == "response.completed" for payload in payloads
                ), rejected.text
                assert not any(
                    payload.get("type") == "response.failed" for payload in payloads
                ), rejected.text
            else:
                assert "event: done" in rejected.text, rejected.text
                assert "event: error" not in rejected.text, rejected.text
        else:
            assert rejected.status_code in {200, 202}, rejected.text
            payload = rejected.json()
            if mode == "background":
                for _attempt in range(100):
                    fetched = client.get(f"/{protocol}/{payload['id']}")
                    assert fetched.status_code == 200, fetched.text
                    payload = fetched.json()
                    if payload["status"] in {"completed", "failed", "cancelled"}:
                        break
                assert payload["status"] == "completed", payload
            elif protocol == "responses":
                assert payload["status"] == "completed", payload
            assert not any(
                item.get("name") == HITL_FUNCTION_NAME
                for item in payload.get("output", [])
            ), payload

        assert executed == []
        assert len(model_inputs) == 2, model_inputs
        tool_results = [
            message for message in model_inputs[1] if isinstance(message, ToolMessage)
        ]
        assert {message.tool_call_id for message in tool_results} == {
            "call-first",
            "call-other",
        }
        assert all(message.content == "Not authorized" for message in tool_results)
        assert all(message.status == "error" for message in tool_results)

        followup = _post_turn(client, protocol, conversation_id, "second")

    assert followup.status_code == 200, followup.text
    payload = followup.json()
    text = assistant_text(payload) if protocol == "responses" else payload["response"]
    assert text == "Echo: second", payload
    assert executed == []


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
def test_steering_requires_explicit_task_enablement(protocol: Protocol) -> None:
    """Missing task infrastructure must fail before running the graph."""
    graph_started = threading.Event()

    async def record_start(config: RunnableConfig) -> None:
        graph_started.set()

    host_type = (
        ResponsesHostServer if protocol == "responses" else InvocationsHostServer
    )
    host = host_type(
        _build_approval_graph(before_approval=record_start),
        options=ResponsesServerOptions(steerable_conversations=True),
    )
    assert not resilient_tasks_enabled()

    with TestClient(host.app) as client:
        response = _post_turn(client, protocol, "missing-task-manager", "first")

    assert not graph_started.is_set()
    if protocol == "responses":
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "failed"
        assert response.json()["error"]["code"] == "steering_unavailable"
    else:
        assert response.status_code == 503, response.text
        assert response.json()["code"] == "steering_unavailable"


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
def test_explicitly_enabled_steering_serializes_same_conversation(
    protocol: Protocol,
) -> None:
    """Callers may opt in after construction, before the host starts."""
    first_started = threading.Event()
    first_finished = threading.Event()
    release_first = threading.Event()

    async def respond(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        user_text = next(
            message.content
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )
        if user_text == "first":
            configurable = config["configurable"]
            cancellation_signal = (
                configurable["response_cancellation_signal"]
                if protocol == "responses"
                else configurable["invocation_context"].cancel
            )
            cancel_waiter = asyncio.create_task(cancellation_signal.wait())
            release_waiter = asyncio.create_task(
                asyncio.to_thread(release_first.wait, 5)
            )
            first_started.set()
            try:
                await asyncio.wait(
                    {cancel_waiter, release_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                cancel_waiter.cancel()
                release_waiter.cancel()
                await asyncio.gather(
                    cancel_waiter, release_waiter, return_exceptions=True
                )
                first_finished.set()
        elif not first_finished.is_set():
            return {"messages": [AIMessage(content="Overlapping conversation turns")]}
        return {"messages": [AIMessage(content=f"Echo: {user_text}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    host_type = (
        ResponsesHostServer if protocol == "responses" else InvocationsHostServer
    )
    host = host_type(
        builder.compile(checkpointer=InMemorySaver()),
        options=ResponsesServerOptions(steerable_conversations=True),
    )
    assert not resilient_tasks_enabled()
    set_resilient_tasks_enabled(True)
    conversation_id = f"steering-only-{uuid.uuid4().hex}"

    with TestClient(host.app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        first_future = executor.submit(
            _post_turn,
            client,
            protocol,
            conversation_id,
            "first",
            background=protocol == "responses",
        )
        try:
            assert first_started.wait(timeout=5), "The first turn did not start."
            second = _post_turn(client, protocol, conversation_id, "second")
        finally:
            release_first.set()
        first = first_future.result(timeout=5)
        assert first_finished.wait(timeout=5), "The first turn did not finish."

    if protocol == "responses":
        assert first.status_code in {200, 202}, first.text
    else:
        assert first.status_code == 409, first.text
        assert "steered" in first.json()["error"]
    assert second.status_code == 200, second.text
    payload = second.json()
    text = assistant_text(payload) if protocol == "responses" else payload["response"]
    assert text == "Echo: second", payload
