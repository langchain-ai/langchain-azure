# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Full-batch continuation admission across both hosting protocols."""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("azure.ai.agentserver.invocations")
pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("starlette")

from azure.ai.agentserver.core.tasks import TaskContext
from azure.ai.agentserver.responses import ResponsesServerOptions
from azure.ai.agentserver.responses.hosting import _endpoint_handler
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt
from starlette.testclient import TestClient
from starlette.types import ASGIApp

from langchain_azure_ai.agents.hosting import (
    InvocationsHostServer,
    ResponsesHostServer,
    _invoke_host,
    _responses_host,
)
from langchain_azure_ai.agents.hosting._converters import (
    IncompleteResumeError,
    detect_pending_interrupts,
    parse_resume_command,
    validate_resume_command,
    validate_resume_inputs,
)
from langchain_azure_ai.agents.hosting._invoke_host import _internal_session_id

from .conftest import (
    approval_requests,
    emitted_items,
    pending_interrupt,
    resume_item,
    sentinels,
)
from .graphs import build_parallel_interrupt_graph, build_simple_interrupt_graph


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
@pytest.mark.parametrize("mode", ["json", "stream", "task", "background"])
@pytest.mark.parametrize(
    "case", ["empty", "message", "function", "approval", "rejection"]
)
def test_incomplete_resume_returns_400_without_advancing_checkpoint(
    protocol: str, mode: str, case: str
) -> None:
    graph = build_parallel_interrupt_graph()
    options = ResponsesServerOptions(
        resilient_background=mode == "background",
        steerable_conversations=mode == "task",
    )
    session_id = f"full-batch-{protocol}-{mode}-{case}"
    app: ASGIApp
    if protocol == "responses":
        host = ResponsesHostServer(graph, options=options, prefix="/v1")
        app = host.app
        path = "/v1/responses"
        input_key = "input"
        continuation: dict[str, Any] = {"conversation": {"id": session_id}}
        thread_id = session_id
    else:
        server = InvocationsHostServer(graph, options=options)
        app = server.app
        path = f"/invocations?agent_session_id={session_id}"
        input_key = "message"
        continuation = {}
        thread_id = _internal_session_id(session_id, "")

    with TestClient(app) as client:
        first = client.post(path, json={input_key: "Ask both.", **continuation})
        assert first.status_code == 200, first.text
        payload = first.json()
        pending = sentinels(payload)
        approvals = approval_requests(payload)
        assert len(pending) == len(approvals) == 2
        if protocol == "responses":
            continuation["previous_response_id"] = payload["id"]
            del continuation["conversation"]

        assert client.portal is not None
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        before = client.portal.call(graph.aget_state, config)

        if case == "empty":
            incomplete: str | list[dict[str, Any]] = []
        elif case == "message":
            incomplete = "Continue without supplying the pending results."
        elif case == "function":
            incomplete = [resume_item(pending[0]["call_id"], "A")]
        else:
            incomplete = [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approvals[0]["id"],
                    "approve": case == "approval",
                }
            ]
        body = {input_key: incomplete, **continuation}
        if mode == "stream":
            body["stream"] = True
        elif mode == "background":
            body["background"] = True
        rejected = client.post(path, json=body)

        assert rejected.status_code == 400, rejected.text
        assert rejected.headers["content-type"].startswith("application/json")
        if case in {"approval", "rejection"}:
            message = (
                "The following MCP approval requests do not have an approval: "
                f"{approvals[1]['id']}"
            )
            param = None
        else:
            missing = pending[1] if case == "function" else pending[0]
            message = f"No tool output found for function call {missing['call_id']}."
            param = "input"
        assert rejected.json() == {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": None,
            }
        }

        after = client.portal.call(graph.aget_state, config)
        assert after.config == before.config
        assert after.values == before.values
        assert after.tasks == before.tasks

        if case in {"approval", "rejection"}:
            complete = [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval["id"],
                    "approve": True,
                }
                for approval in reversed(approvals)
            ]
        else:
            complete = [
                resume_item(item["call_id"], value)
                for item, value in zip(reversed(pending), ("B", "A"), strict=True)
            ]
        retried = client.post(path, json={input_key: complete, **continuation})
        assert retried.status_code == 200, retried.text
        result = retried.json()
        assert not result.get("error"), result
        assert not [
            item
            for item in result.get("output", [])
            if item.get("type") == "mcp_approval_request"
        ]
        final_state = client.portal.call(graph.aget_state, config)
        assert not final_state.next


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
@pytest.mark.parametrize("mode", ["json", "stream", "task", "background"])
@pytest.mark.parametrize("case", ["empty", "stale"])
def test_single_interrupt_id_map_cannot_apply_updates_without_an_answer(
    protocol: str, mode: str, case: str
) -> None:
    graph = build_simple_interrupt_graph()
    options = ResponsesServerOptions(
        resilient_background=mode == "background",
        steerable_conversations=mode == "task",
    )
    session_id = f"single-map-{protocol}-{mode}-{case}"
    app: ASGIApp
    if protocol == "responses":
        app = ResponsesHostServer(graph, options=options).app
        path = "/responses"
        input_key = "input"
        continuation: dict[str, Any] = {"conversation": {"id": session_id}}
        thread_id = session_id
    else:
        app = InvocationsHostServer(graph, options=options).app
        path = f"/invocations?agent_session_id={session_id}"
        input_key = "message"
        continuation = {}
        thread_id = _internal_session_id(session_id, "")

    with TestClient(app) as client:
        first = client.post(path, json={input_key: "Ask my name.", **continuation})
        assert first.status_code == 200, first.text
        pending = sentinels(first.json())
        assert len(pending) == 1
        call_id = pending[0]["call_id"]
        assert call_id != "0" * 32
        assert client.portal is not None
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        before = client.portal.call(graph.aget_state, config)
        resume = {} if case == "empty" else {"0" * 32: "stale answer"}
        rejected = client.post(
            path,
            json={
                input_key: [
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(
                            {
                                "resume": resume,
                                "update": {
                                    "messages": [
                                        {
                                            "type": "human",
                                            "content": "must not be saved",
                                        }
                                    ]
                                },
                            }
                        ),
                    }
                ],
                "stream": mode == "stream",
                "background": mode == "background",
                **continuation,
            },
        )
        assert rejected.status_code == 400, rejected.text
        assert rejected.json()["error"] == {
            "message": f"No tool output found for function call {call_id}.",
            "type": "invalid_request_error",
            "param": "input",
            "code": None,
        }
        after = client.portal.call(graph.aget_state, config)
        assert after.config == before.config
        assert after.values == before.values
        assert after.tasks == before.tasks
        answer: str | dict[str, Any] = {} if case == "empty" else "Alice"
        retried = client.post(
            path,
            json={
                input_key: [resume_item(call_id, {call_id: answer})],
                **continuation,
            },
        )
        assert retried.status_code == 200, retried.text
        final_state = client.portal.call(graph.aget_state, config)
        assert not final_state.next
        assert final_state.values["messages"][-1].content == f"ok:{answer}"


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
@pytest.mark.parametrize("mode", ["json", "stream", "task", "background"])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("output", ["", "   "])
def test_empty_function_outputs_resume_the_complete_batch(
    protocol: str, mode: str, parallel: bool, output: str
) -> None:
    graph = (
        build_parallel_interrupt_graph() if parallel else build_simple_interrupt_graph()
    )
    options = ResponsesServerOptions(
        resilient_background=mode == "background",
        steerable_conversations=mode == "task",
    )
    session_id = f"empty-output-{protocol}-{mode}-{parallel}-{len(output)}"
    app: ASGIApp
    if protocol == "responses":
        app = ResponsesHostServer(graph, options=options).app
        path = "/responses"
        input_key = "input"
        continuation: dict[str, Any] = {"conversation": {"id": session_id}}
        thread_id = session_id
    else:
        app = InvocationsHostServer(graph, options=options).app
        path = f"/invocations?agent_session_id={session_id}"
        input_key = "message"
        continuation = {}
        thread_id = _internal_session_id(session_id, "")

    with TestClient(app) as client:
        first = client.post(path, json={input_key: "Ask.", **continuation})
        assert first.status_code == 200, first.text
        pending = sentinels(first.json())
        assert len(pending) == (2 if parallel else 1)
        response = client.post(
            path,
            json={
                input_key: [
                    {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": output,
                    }
                    for item in pending
                ],
                "stream": mode == "stream",
                "background": mode == "background",
                **continuation,
            },
        )
        expected_status = (
            202 if protocol == "invocations" and mode == "background" else 200
        )
        assert response.status_code == expected_status, response.text
        if mode == "background":
            result_path = f"/{protocol}/{response.json()['id']}"
            result = client.get(result_path)
            for _ in range(20):
                if result.json().get("status") == "completed":
                    break
                result = client.get(result_path)
            assert result.status_code == 200, result.text
            assert result.json()["status"] == "completed", result.text
        elif mode == "stream":
            terminal_event = (
                "response.completed" if protocol == "responses" else "event: done"
            )
            assert terminal_event in response.text
        assert client.portal is not None
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        final_state = client.portal.call(graph.aget_state, config)
        assert not final_state.next
        assert not final_state.interrupts
        answers = {
            message.content
            for message in final_state.values["messages"]
            if message.type == "ai"
        }
        expected_answers = (
            {f"a={output}", f"b={output}"} if parallel else {f"ok:{output}"}
        )
        assert answers == expected_answers


@pytest.mark.parametrize(
    "outputs",
    [
        ['{"resume": "A"}'],
        ['{"update": {"marker": "set"}}', '{"resume": "B"}'],
        ['{"resume": "A"}', '{"goto": "finish"}'],
    ],
)
def test_parser_rejects_partial_and_control_only_responses(
    outputs: list[str],
) -> None:
    pending = (
        pending_interrupt(id="int-a"),
        pending_interrupt(id="int-b"),
    )
    items = [
        {
            "type": "function_call_output",
            "call_id": interrupt.id,
            "output": output,
        }
        for interrupt, output in zip(pending, outputs)
    ]

    with pytest.raises(
        IncompleteResumeError, match="No tool output found for function call"
    ):
        parse_resume_command(items, pending)


@pytest.mark.parametrize("other_id", ["int-a", "unknown"])
def test_duplicate_or_unknown_ids_do_not_complete_a_batch(other_id: str) -> None:
    pending = (pending_interrupt(id="int-a"), pending_interrupt(id="int-b"))
    items = [resume_item("int-a", "A"), resume_item(other_id, "B")]

    with pytest.raises(IncompleteResumeError) as error:
        validate_resume_inputs(items, pending)

    assert str(error.value) == "No tool output found for function call int-b."
    assert error.value.param == "input"


async def test_missing_mcp_error_lists_emitted_ids_in_pending_order() -> None:
    pending = tuple(pending_interrupt(id=f"int-{suffix}") for suffix in "abc")
    output = await emitted_items(pending)
    approvals = approval_requests({"output": output})
    items = [
        {
            "type": "mcp_approval_response",
            "approval_request_id": approval_id,
            "approve": True,
        }
        for approval_id in (approvals[2]["id"], approvals[2]["id"], "mcpr_unknown")
    ]

    with pytest.raises(IncompleteResumeError) as error:
        validate_resume_inputs(items, pending, output_items=output)

    assert error.value.as_error() == {
        "message": (
            "The following MCP approval requests do not have an approval: "
            f"{approvals[0]['id']}, {approvals[1]['id']}"
        ),
        "type": "invalid_request_error",
        "param": None,
        "code": None,
    }


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
@pytest.mark.parametrize("stream", [False, True])
def test_execution_revalidates_after_admission(
    protocol: str, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = build_parallel_interrupt_graph()
    options = ResponsesServerOptions(steerable_conversations=True)
    session_id = f"revalidate-{protocol}-{stream}"
    app: ASGIApp
    module: ModuleType
    if protocol == "responses":
        app = ResponsesHostServer(graph, options=options).app
        path = "/responses"
        input_key = "input"
        continuation: dict[str, Any] = {"conversation": {"id": session_id}}
        thread_id = session_id
        module = _responses_host
    else:
        app = InvocationsHostServer(graph, options=options).app
        path = f"/invocations?agent_session_id={session_id}"
        input_key = "message"
        continuation = {}
        thread_id = _internal_session_id(session_id, "")
        module = _invoke_host

    with TestClient(app) as client:
        first = client.post(path, json={input_key: "Ask both.", **continuation})
        assert first.status_code == 200, first.text
        pending = sentinels(first.json())
        assert len(pending) == 2
        assert client.portal is not None
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        before = client.portal.call(graph.aget_state, config)
        lookups = 0

        async def pending_after_admission(
            candidate_graph: CompiledStateGraph, config: RunnableConfig
        ) -> tuple[Interrupt, ...]:
            nonlocal lookups
            lookups += 1
            if lookups == 1:
                # Model a batch that appears after the request is admitted.
                return ()
            return await detect_pending_interrupts(candidate_graph, config)

        monkeypatch.setattr(
            module, "detect_pending_interrupts", pending_after_admission
        )
        response = client.post(
            path,
            json={
                input_key: [resume_item(pending[0]["call_id"], "A")],
                "stream": stream,
                **continuation,
            },
        )

        assert lookups >= 2
        missing = f"No tool output found for function call {pending[1]['call_id']}."
        if stream:
            assert response.status_code == 200, response.text
            assert "invalid_request_error" in response.text
            assert missing in response.text
        elif protocol == "responses":
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "failed"
            assert response.json()["error"] == {
                "code": "invalid_request_error",
                "message": missing,
            }
        else:
            assert response.status_code == 400, response.text
            assert response.json()["error"] == {
                "message": missing,
                "type": "invalid_request_error",
                "param": "input",
                "code": None,
            }
        after = client.portal.call(graph.aget_state, config)
        assert after.config == before.config
        assert after.values == before.values
        assert after.tasks == before.tasks


def test_custom_responses_resume_protocol_remains_reachable() -> None:
    class CustomResumeHost(ResponsesHostServer):
        calls = 0

        async def build_resume_command(
            self, request: Any, context: Any, pending: Sequence[Interrupt]
        ) -> tuple[Command | None, frozenset[str]]:
            self.calls += 1
            return (
                Command(resume={item.id: "approved" for item in pending}),
                frozenset(item.id for item in pending),
            )

    host = CustomResumeHost(build_parallel_interrupt_graph())
    continuation = {"conversation": {"id": "custom-complete-resume"}}
    with TestClient(host.app) as client:
        first = client.post("/responses", json={"input": "Ask both.", **continuation})
        assert first.status_code == 200, first.text
        assert len(sentinels(first.json())) == 2
        response = client.post(
            "/responses", json={"input": "Approve the entire batch.", **continuation}
        )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"
    assert not sentinels(response.json())
    assert host.calls == 2


@pytest.mark.parametrize("mode", ["ordinary", "json", "stream", "background"])
def test_invocations_checkpoint_hook_is_shared_by_admission_and_execution(
    mode: str,
) -> None:
    class CustomCheckpointHost(InvocationsHostServer):
        def build_checkpoint_config(self, thread_id: str) -> RunnableConfig:
            config = super().build_checkpoint_config(thread_id)
            config["configurable"]["thread_id"] = f"custom:{thread_id}"
            return config

    graph = build_parallel_interrupt_graph()
    host = CustomCheckpointHost(
        graph,
        options=ResponsesServerOptions(
            resilient_background=mode == "background",
            steerable_conversations=mode != "ordinary",
        ),
    )
    session_id = f"custom-checkpoint-{mode}"
    path = f"/invocations?agent_session_id={session_id}"
    config: RunnableConfig = {
        "configurable": {"thread_id": f"custom:{_internal_session_id(session_id, '')}"}
    }
    with TestClient(host.app) as client:
        first = client.post(path, json={"message": "Ask both."})
        assert first.status_code == 200, first.text
        pending = sentinels(first.json())
        assert len(pending) == 2
        assert client.portal is not None
        before = client.portal.call(graph.aget_state, config)
        assert len(before.interrupts) == 2
        incomplete = client.post(
            path,
            json={
                "message": [resume_item(pending[0]["call_id"], "A")],
                "stream": mode == "stream",
                "background": mode == "background",
            },
        )
        assert incomplete.status_code == 400, incomplete.text
        after = client.portal.call(graph.aget_state, config)
        assert after.config == before.config
        assert after.values == before.values
        assert after.tasks == before.tasks
        complete = client.post(
            path,
            json={
                "message": [
                    resume_item(item["call_id"], "approved") for item in pending
                ]
            },
        )
        assert complete.status_code == 200, complete.text
        assert not complete.json().get("output")
        assert not client.portal.call(graph.aget_state, config).next


@pytest.mark.parametrize("key", ["thread_id", "checkpoint_ns", "checkpoint_id"])
def test_task_config_checkpoint_override_requires_explicit_migration(
    key: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class LegacyTaskThreadHost(InvocationsHostServer):
        def build_task_runnable_config(
            self, session_id: str, context: Any
        ) -> RunnableConfig:
            config = super().build_task_runnable_config(session_id, context)
            config["configurable"][key] = f"legacy:{context.task_id}"
            return config

    host = LegacyTaskThreadHost(
        build_parallel_interrupt_graph(),
        options=ResponsesServerOptions(steerable_conversations=True),
    )
    with TestClient(host.app) as client:
        response = client.post(
            "/invocations?agent_session_id=legacy-task-config",
            json={"message": "Ask both."},
        )
        invocation_id = response.headers["x-agent-invocation-id"]
        stored = client.get(f"/invocations/{invocation_id}")

    assert response.status_code == 500, response.text
    assert "build_checkpoint_config" in caplog.text
    assert stored.status_code == 200, stored.text
    assert stored.json()["status"] == "failed"
    assert stored.json()["error"] == {
        "code": "internal_error",
        "message": "Internal server error.",
    }
    assert invocation_id not in host._cancel_requests


def test_task_execution_fields_do_not_mutate_checkpoint_hook_result() -> None:
    selected: RunnableConfig = {"configurable": {"thread_id": "custom-thread"}}

    class CachedCheckpointHost(InvocationsHostServer):
        def build_checkpoint_config(self, thread_id: str) -> RunnableConfig:
            return selected

    host = CachedCheckpointHost(build_parallel_interrupt_graph())
    context: TaskContext[dict[str, Any]] = TaskContext(
        task_id="internal-session", session_id="public-session", input={}
    )
    config = host.build_task_runnable_config("public-session", context)

    assert selected == {"configurable": {"thread_id": "custom-thread"}}
    assert config["configurable"]["thread_id"] == "custom-thread"
    assert config["configurable"]["invocation_context"] is context
    assert config["configurable"]["invocation_cancellation_signal"] is context.cancel


@pytest.mark.parametrize("source", ["checkpoint", "selector", "hook"])
def test_responses_preflight_failure_returns_json_and_closes_span(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = build_parallel_interrupt_graph()
    host = ResponsesHostServer(graph)
    continuation = {"conversation": {"id": f"preflight-outage-{source}"}}
    with TestClient(host.app, raise_server_exceptions=False) as client:
        first = client.post("/responses", json={"input": "Ask both.", **continuation})
        assert first.status_code == 200, first.text
        pending = sentinels(first.json())
        failure = RuntimeError(f"{source} unavailable")
        if source == "checkpoint":
            monkeypatch.setattr(graph, "aget_state", AsyncMock(side_effect=failure))
        elif source == "selector":
            monkeypatch.setattr(
                host, "build_runnable_config", AsyncMock(side_effect=failure)
            )
        else:
            monkeypatch.setattr(
                host, "validate_resume_input", AsyncMock(side_effect=failure)
            )
        span = MagicMock()
        monkeypatch.setattr(
            _endpoint_handler, "start_create_span", lambda *a, **k: span
        )
        response = client.post(
            "/responses",
            json={
                "input": [resume_item(item["call_id"], "approved") for item in pending],
                **continuation,
            },
        )

    assert response.status_code == 500, response.text
    assert response.json()["error"] == {
        "message": "Internal server error.",
        "type": "server_error",
        "param": None,
        "code": "server_error",
    }
    span.end.assert_called_once_with(failure)


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
def test_validation_hook_runs_at_admission_and_execution(protocol: str) -> None:
    checks: list[tuple[str, ...]] = []

    class RecordingResponsesHost(ResponsesHostServer):
        async def validate_resume_input(
            self,
            request: Any,
            context: Any,
            pending: Sequence[Interrupt],
            resume_command: Command | None,
        ) -> None:
            checks.append(tuple(item.id for item in pending))
            await super().validate_resume_input(
                request, context, pending, resume_command
            )

    class RecordingInvocationsHost(InvocationsHostServer):
        async def validate_resume_input(
            self,
            message: Any,
            pending: Sequence[Interrupt],
            resume_command: Command | None,
        ) -> None:
            checks.append(tuple(item.id for item in pending))
            await super().validate_resume_input(message, pending, resume_command)

    graph = build_parallel_interrupt_graph()
    options = ResponsesServerOptions(steerable_conversations=True)
    app: ASGIApp
    if protocol == "responses":
        app = RecordingResponsesHost(graph, options=options).app
        path = "/responses"
        input_key = "input"
        continuation: dict[str, Any] = {"conversation": {"id": "record-validation"}}
    else:
        app = RecordingInvocationsHost(graph, options=options).app
        path = "/invocations?agent_session_id=record-validation"
        input_key = "message"
        continuation = {}

    with TestClient(app) as client:
        first = client.post(path, json={input_key: "Ask both.", **continuation})
        assert first.status_code == 200, first.text
        pending = sentinels(first.json())
        checks.clear()
        response = client.post(
            path,
            json={
                input_key: [
                    resume_item(item["call_id"], "approved") for item in pending
                ],
                **continuation,
            },
        )

    assert response.status_code == 200, response.text
    expected = tuple(item["call_id"] for item in pending)
    assert checks == [expected, expected]


@pytest.mark.parametrize(
    "command",
    [
        Command(resume={"int-a": None, "int-b": False}),
        Command(resume={"int-b": "", "int-a": {}}),
    ],
)
def test_normalized_command_preserves_explicit_falsy_answers(command: Command) -> None:
    pending = (pending_interrupt(id="int-a"), pending_interrupt(id="int-b"))
    validate_resume_command(command, pending)


@pytest.mark.parametrize("resume", [{}, {"0" * 32: "stale answer"}])
@pytest.mark.parametrize("validator", ["command", "inputs", "parser"])
def test_single_interrupt_id_map_requires_the_pending_id(
    resume: dict[str, Any], validator: str
) -> None:
    interrupt_id = "1" * 32
    pending = (pending_interrupt(id=interrupt_id),)
    with pytest.raises(IncompleteResumeError, match=f"function call {interrupt_id}"):
        if validator == "command":
            validate_resume_command(Command(resume=resume), pending)
        elif validator == "inputs":
            validate_resume_inputs([resume_item(interrupt_id, resume)], pending)
        else:
            parse_resume_command([resume_item(interrupt_id, resume)], pending)


@pytest.mark.parametrize(
    "resume", ["", "   ", False, 0, [], {"approved": True}, {"F" * 32: "payload"}]
)
@pytest.mark.parametrize("mapped", [False, True])
def test_single_interrupt_preserves_scalar_and_id_mapped_values(
    resume: Any, mapped: bool
) -> None:
    interrupt_id = "1" * 32
    pending = (pending_interrupt(id=interrupt_id),)
    value = {interrupt_id: resume} if mapped else resume
    validate_resume_command(Command(resume=value), pending)
    command, _ = parse_resume_command([resume_item(interrupt_id, value)], pending)
    assert command is not None
    assert command.resume == value


@pytest.mark.parametrize(
    "command",
    [
        Command(resume={"int-a": "A"}),
        Command(resume={"int-a": "A", "unknown": "B"}),
        Command(resume="not-an-id-map"),
        Command(update={"marker": "set"}),
        Command(goto="next"),
    ],
)
def test_normalized_command_cannot_partially_resume_parallel_interrupts(
    command: Command,
) -> None:
    pending = (pending_interrupt(id="int-a"), pending_interrupt(id="int-b"))
    with pytest.raises(IncompleteResumeError):
        validate_resume_command(command, pending)


def test_normalized_rejections_answer_mcp_requests_without_a_command() -> None:
    pending = (pending_interrupt(id="int-a"), pending_interrupt(id="int-b"))
    items = [
        {
            "type": "mcp_approval_response",
            "approval_request_id": item.id,
            "approve": False,
        }
        for item in pending
    ]
    validate_resume_command(None, pending, items=items)


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
@pytest.mark.parametrize("task_backed", [False, True])
@pytest.mark.parametrize("source", ["hook", "selector", "checkpoint"])
def test_preflight_hook_errors_return_server_errors_for_both_protocols(
    protocol: str,
    task_backed: bool,
    source: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = build_parallel_interrupt_graph()
    options = ResponsesServerOptions(steerable_conversations=task_backed)
    app: ASGIApp
    if protocol == "responses":
        host = ResponsesHostServer(graph, options=options)
        app = host.app
        path = "/responses"
        input_key = "input"
        continuation: dict[str, Any] = {"conversation": {"id": "hook-error"}}
    else:
        server = InvocationsHostServer(graph, options=options)
        app = server.app
        path = "/invocations?agent_session_id=hook-error"
        input_key = "message"
        continuation = {}
    target = host if protocol == "responses" else server
    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post(path, json={input_key: "Ask both.", **continuation})
        assert first.status_code == 200, first.text
        pending = sentinels(first.json())
        failure = RuntimeError(f"custom {source} unavailable")
        if source == "hook":
            monkeypatch.setattr(
                target, "validate_resume_input", AsyncMock(side_effect=failure)
            )
        elif source == "checkpoint":
            monkeypatch.setattr(graph, "aget_state", AsyncMock(side_effect=failure))
        elif protocol == "responses":
            monkeypatch.setattr(
                target, "build_runnable_config", AsyncMock(side_effect=failure)
            )
        else:
            monkeypatch.setattr(
                target, "build_checkpoint_config", MagicMock(side_effect=failure)
            )
        response = client.post(
            path,
            json={
                input_key: [
                    resume_item(item["call_id"], "approved") for item in pending
                ],
                **continuation,
            },
        )
    assert response.status_code == 500, response.text
    assert response.json()["error"]["type"] == "server_error"
    assert response.json()["error"]["message"] == "Internal server error."


@pytest.mark.parametrize("mode", ["json", "background", "stream"])
@pytest.mark.parametrize("source", ["hook", "selector", "checkpoint"])
def test_execution_hook_errors_persist_terminal_failed_invocations(
    mode: str, source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = build_parallel_interrupt_graph()
    server = InvocationsHostServer(
        graph,
        options=ResponsesServerOptions(
            resilient_background=True, steerable_conversations=True
        ),
    )
    session_id = f"execution-failure-{mode}-{source}"
    path = f"/invocations?agent_session_id={session_id}"
    config: RunnableConfig = {
        "configurable": {"thread_id": _internal_session_id(session_id, "")}
    }
    get_state = graph.aget_state
    with TestClient(server.app) as client:
        first = client.post(path, json={"message": "Ask both."})
        assert first.status_code == 200, first.text
        pending = sentinels(first.json())
        assert len(pending) == 2
        assert client.portal is not None
        before = client.portal.call(get_state, config)
        failure = RuntimeError(f"{source} unavailable during execution")
        if source == "hook":
            monkeypatch.setattr(
                server,
                "validate_resume_input",
                AsyncMock(side_effect=[None, failure]),
            )
        elif source == "selector":
            selected = server.build_checkpoint_config(
                _internal_session_id(session_id, "")
            )
            monkeypatch.setattr(
                server,
                "build_checkpoint_config",
                MagicMock(side_effect=[selected, failure]),
            )
        else:
            monkeypatch.setattr(
                graph, "aget_state", AsyncMock(side_effect=[before, failure])
            )
        response = client.post(
            path,
            json={
                "message": [
                    resume_item(item["call_id"], "approved") for item in pending
                ],
                "background": mode == "background",
                "stream": mode == "stream",
            },
        )
        expected_status = {"json": 500, "background": 202, "stream": 200}[mode]
        assert response.status_code == expected_status, response.text
        if mode == "stream":
            assert "event: error\n" in response.text
            streamed_errors = [
                json.loads(line.removeprefix("data: "))
                for line in response.text.splitlines()
                if line.startswith("data: {")
            ]
            assert streamed_errors == [
                {"code": "internal_error", "message": "Internal server error."}
            ]
        invocation_id = response.headers["x-agent-invocation-id"]
        stored = client.get(f"/invocations/{invocation_id}")
        for _ in range(20):
            if stored.json().get("status") == "failed":
                break
            stored = client.get(f"/invocations/{invocation_id}")
        assert stored.status_code == 200, stored.text
        assert stored.json()["status"] == "failed"
        assert stored.json()["error"] == {
            "code": "internal_error",
            "message": "Internal server error.",
        }
        assert invocation_id not in server._cancel_requests
        after = client.portal.call(get_state, config)
        assert after.config == before.config
        assert after.values == before.values
        assert after.tasks == before.tasks
