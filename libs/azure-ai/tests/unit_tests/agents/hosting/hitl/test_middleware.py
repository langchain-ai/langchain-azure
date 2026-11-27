# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Host integration tests for LangChain HITL middleware decisions."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("azure.ai.agentserver.invocations")
pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("starlette")

from azure.ai.agentserver.responses import InMemoryResponseProvider  # noqa: E402
from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import HumanInTheLoopMiddleware  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.base import CheckpointTuple  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from langchain_azure_ai.agents.hosting import (  # noqa: E402
    InvocationsHostServer,
    ResponsesHostServer,
)

from .conftest import (  # noqa: E402
    REAL_INTERRUPT_ASYNC_XFAIL,
    ScriptRegistrar,
    approval_requests,
    assistant_text,
    sse_payloads,
)
from .graphs import ScriptedModel  # noqa: E402


class ScriptedToolCallingModel(BaseChatModel):
    """Scripted chat model compatible with ``create_agent``."""

    key: str

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedToolCallingModel:
        del tools, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        return ChatResult(
            generations=[
                ChatGeneration(message=ScriptedModel(self.key).invoke(messages))
            ]
        )


def _middleware_graph(
    key: str, tool_calls: list[str], allowed_decisions: list[str] | None = None
) -> CompiledStateGraph:
    config: Any = {"allowed_decisions": allowed_decisions or ["approve", "reject"]}

    @tool
    def risky_tool(value: str) -> str:
        """Record a risky tool execution."""
        tool_calls.append(value)
        return f"executed:{value}"

    return create_agent(
        model=ScriptedToolCallingModel(key=key),
        tools=[risky_tool],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"risky_tool": config})],
        checkpointer=InMemorySaver(),
    )


def _register_script(
    script: ScriptRegistrar, key: str, expected_text: str
) -> list[AIMessage]:
    return script(
        key,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_risky",
                        "name": "risky_tool",
                        "args": {"value": "approved"},
                    }
                ],
            ),
            AIMessage(content=f"The action was {expected_text}."),
        ],
    )


def _approval(approval_id: str, approve: bool) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "mcp_approval_response",
        "approval_request_id": approval_id,
        "approve": approve,
    }
    if not approve:
        item["reason"] = "Denied"
    return item


def _assert_result(key: str, tool_calls: list[str], approve: bool) -> None:
    assert tool_calls == (["approved"] if approve else [])
    if not approve:
        rejections = [
            message
            for turn in ScriptedModel.seen[key]
            for message in turn
            if isinstance(message, ToolMessage) and message.status == "error"
        ]
        assert rejections
        assert all("Denied" in str(message.content) for message in rejections)


@pytest.mark.parametrize(
    ("approve", "expected_text"), [(True, "approved"), (False, "rejected")]
)
@REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_resumes_hitl_middleware(
    script: ScriptRegistrar, approve: bool, expected_text: str
) -> None:
    key = f"responses-middleware-{approve}"
    _register_script(script, key, expected_text)
    tool_calls: list[str] = []
    host = ResponsesHostServer(_middleware_graph(key, tool_calls))
    conversation_id = key

    with TestClient(host.app) as client:
        first = client.post(
            "/responses",
            json={"input": "do it", "conversation": {"id": conversation_id}},
        )
        approval_id = approval_requests(first.json())[0]["id"]
        resumed = client.post(
            "/responses",
            json={
                "input": [_approval(approval_id, approve)],
                "conversation": {"id": conversation_id},
            },
        )

    payload = resumed.json()
    assert payload["status"] == "completed", payload
    assert expected_text in assistant_text(payload)
    _assert_result(key, tool_calls, approve)


@pytest.mark.parametrize("stream", [False, True])
@REAL_INTERRUPT_ASYNC_XFAIL
def test_response_linked_approval_cannot_execute_twice(
    script: ScriptRegistrar, stream: bool
) -> None:
    key = f"response-linked-middleware-{stream}"
    script_queue = _register_script(script, key, "approved")
    script_queue.append(AIMessage(content="Unexpected second execution."))
    tool_calls: list[str] = []
    host = ResponsesHostServer(_middleware_graph(key, tool_calls, ["approve"]))

    with TestClient(host.app) as client:
        paused = client.post("/responses", json={"input": "do it"}).json()
        approval_id = approval_requests(paused)[0]["id"]
        body = {"previous_response_id": paused["id"], "stream": stream}
        rejected = client.post(
            "/responses", json={**body, "input": [_approval(approval_id, False)]}
        )
        accepted = client.post(
            "/responses", json={**body, "input": [_approval(approval_id, True)]}
        )
        duplicate = client.post(
            "/responses", json={**body, "input": [_approval(approval_id, True)]}
        )

    assert tool_calls == ["approved"]
    assert len(script_queue) == 1
    for result, status in (
        (rejected, "failed"),
        (accepted, "completed"),
        (duplicate, "failed"),
    ):
        assert result.status_code == 200, result.text
        payload = sse_payloads(result.text)[-1]["response"] if stream else result.json()
        assert payload["status"] == status, payload
    assert payload["error"]["code"] == "interrupt_not_pending"


@pytest.mark.parametrize("stream", [False, True])
@REAL_INTERRUPT_ASYNC_XFAIL
def test_response_linked_retry_from_rejected_approval(
    script: ScriptRegistrar, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = f"response-linked-rejected-{stream}"
    _register_script(script, key, "approved")
    tool_calls: list[str] = []
    host = ResponsesHostServer(
        _middleware_graph(key, tool_calls, ["approve"]),
        store=InMemoryResponseProvider(),
    )
    execute = Mock(wraps=host.graph.astream)
    chain_write = AsyncMock()

    with TestClient(host.app) as client:
        paused = client.post("/responses", json={"input": "do it"}).json()
        approval_id = approval_requests(paused)[0]["id"]
        monkeypatch.setattr(host.graph, "astream", execute)
        monkeypatch.setattr(host._conversation_chain_store, "set", chain_write)
        rejected = client.post(
            "/responses",
            json={
                "previous_response_id": paused["id"],
                "input": [_approval(approval_id, False)],
                "stream": stream,
            },
        )
        assert rejected.status_code == 200, rejected.text
        failed = (
            sse_payloads(rejected.text)[-1]["response"] if stream else rejected.json()
        )
        assert failed["status"] == "failed"
        assert failed["error"]["code"] == "interrupt_rejected"
        execute.assert_not_called()
        chain_write.assert_not_awaited()

        body = {
            "previous_response_id": failed["id"],
            "input": [_approval(approval_id, True)],
            "stream": stream,
        }
        retried = client.post("/responses", json=body)
        assert retried.status_code == 200, retried.text
        result = (
            sse_payloads(retried.text)[-1]["response"] if stream else retried.json()
        )
        assert result["status"] == "completed", result
        assert assistant_text(result) == "The action was approved."

        duplicate = client.post("/responses", json=body)
        assert duplicate.status_code == 200, duplicate.text
        stale = (
            sse_payloads(duplicate.text)[-1]["response"] if stream else duplicate.json()
        )
        assert stale["status"] == "failed", stale
        assert stale["error"]["code"] == "interrupt_not_pending"

    execute.assert_called_once()
    chain_write.assert_not_awaited()
    assert tool_calls == ["approved"]


@pytest.mark.parametrize("stream", [False, True])
@REAL_INTERRUPT_ASYNC_XFAIL
def test_response_linked_approval_reports_live_checkpoint_read_failure(
    script: ScriptRegistrar, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = f"response-linked-read-failure-{stream}"
    _register_script(script, key, "approved")
    tool_calls: list[str] = []
    graph = _middleware_graph(key, tool_calls, ["approve"])
    saver = graph.checkpointer
    assert isinstance(saver, InMemorySaver)
    host = ResponsesHostServer(graph, store=InMemoryResponseProvider())
    get_tuple = saver.aget_tuple

    async def fail_live_read(config: RunnableConfig) -> CheckpointTuple | None:
        if "checkpoint_id" not in (config.get("configurable") or {}):
            raise ConnectionError("Cannot read the live checkpoint")
        return await get_tuple(config)

    with TestClient(host.app) as client:
        paused = client.post("/responses", json={"input": "do it"}).json()
        approval_id = approval_requests(paused)[0]["id"]
        monkeypatch.setattr(saver, "aget_tuple", fail_live_read)
        execute = Mock(wraps=host.graph.astream)
        monkeypatch.setattr(host.graph, "astream", execute)
        response = client.post(
            "/responses",
            json={
                "previous_response_id": paused["id"],
                "input": [_approval(approval_id, True)],
                "stream": stream,
            },
        )

    assert response.status_code == 200, response.text
    failed = sse_payloads(response.text)[-1]["response"] if stream else response.json()
    assert failed["status"] == "failed", failed
    assert failed["error"]["code"] == "server_error", failed
    execute.assert_not_called()
    assert tool_calls == []


@pytest.mark.parametrize(
    ("approve", "expected_text"), [(True, "approved"), (False, "rejected")]
)
@REAL_INTERRUPT_ASYNC_XFAIL
def test_invocations_host_resumes_hitl_middleware(
    script: ScriptRegistrar, approve: bool, expected_text: str
) -> None:
    key = f"invocations-middleware-{approve}"
    _register_script(script, key, expected_text)
    tool_calls: list[str] = []
    server = InvocationsHostServer(_middleware_graph(key, tool_calls))

    with TestClient(server.app) as client:
        first = client.post(
            f"/invocations?agent_session_id={key}", json={"message": "do it"}
        )
        approval_id = next(
            item["id"]
            for item in first.json()["output"]
            if item.get("type") == "mcp_approval_request"
        )
        resumed = client.post(
            f"/invocations?agent_session_id={key}",
            json={"message": [_approval(approval_id, approve)]},
        )

    assert resumed.status_code == 200, resumed.text
    assert expected_text in resumed.json()["response"]
    _assert_result(key, tool_calls, approve)


@pytest.mark.parametrize(
    ("allowed_decisions", "invalid_approve"),
    [(["approve"], False), (["reject"], True)],
)
@REAL_INTERRUPT_ASYNC_XFAIL
def test_disallowed_shortcut_keeps_interrupt_pending(
    script: ScriptRegistrar,
    allowed_decisions: list[str],
    invalid_approve: bool,
) -> None:
    key = f"middleware-retry-{invalid_approve}"
    expected_text = "approved" if not invalid_approve else "rejected"
    _register_script(script, key, expected_text)
    tool_calls: list[str] = []
    host = ResponsesHostServer(_middleware_graph(key, tool_calls, allowed_decisions))

    with TestClient(host.app) as client:
        first = client.post(
            "/responses",
            json={"input": "do it", "conversation": {"id": key}},
        )
        approval_id = approval_requests(first.json())[0]["id"]
        failed = client.post(
            "/responses",
            json={
                "input": [_approval(approval_id, invalid_approve)],
                "conversation": {"id": key},
            },
        )
        retried = client.post(
            "/responses",
            json={
                "input": [_approval(approval_id, not invalid_approve)],
                "conversation": {"id": key},
            },
        )

    assert failed.json()["status"] == "failed"
    assert retried.json()["status"] == "completed"
    assert expected_text in assistant_text(retried.json())
    _assert_result(key, tool_calls, not invalid_approve)


@pytest.mark.parametrize("decisions", [(True, False), (False, True)])
@REAL_INTERRUPT_ASYNC_XFAIL
def test_conflicting_approvals_keep_interrupt_pending(
    script: ScriptRegistrar, decisions: tuple[bool, bool]
) -> None:
    key = f"middleware-conflict-{decisions[0]}"
    _register_script(script, key, "approved")
    tool_calls: list[str] = []
    host = ResponsesHostServer(_middleware_graph(key, tool_calls))

    with TestClient(host.app) as client:
        first = client.post(
            "/responses",
            json={"input": "do it", "conversation": {"id": key}},
        )
        approval_id = approval_requests(first.json())[0]["id"]
        failed = client.post(
            "/responses",
            json={
                "input": [_approval(approval_id, decision) for decision in decisions],
                "conversation": {"id": key},
            },
        )
        retried = client.post(
            "/responses",
            json={
                "input": [_approval(approval_id, True)],
                "conversation": {"id": key},
            },
        )

    assert failed.json()["status"] == "failed"
    assert "conflicting" in failed.json()["error"]["message"]
    assert retried.json()["status"] == "completed"
    assert tool_calls == ["approved"]
