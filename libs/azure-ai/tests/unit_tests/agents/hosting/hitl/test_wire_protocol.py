# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end tests for the HITL wire protocol between client and host.

Covers what the client actually sees and sends: the paired sentinel items a
pause emits, the two resume channels, recovery from a mismatched resume id,
streaming parity, and thread scoping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("starlette")

from azure.ai.agentserver.responses import (
    InMemoryResponseProvider,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from starlette.testclient import TestClient

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.agents.hosting._converters import (
    HITL_FUNCTION_NAME,
    HITL_MCP_SERVER_LABEL,
)

from ..conftest import make_echo_graph
from .conftest import (
    REAL_INTERRUPT_ASYNC_XFAIL,
    ScriptRegistrar,
    approval_requests,
    assistant_text,
    client_for,
    hitl_items_in,
    interrupt_value,
    resume_item,
    sentinel_item,
    sentinels,
    sse_payloads,
)
from .graphs import (
    MessagesState,
    ScriptedModel,
    build_ask_human_graph,
    build_parallel_empty_update_interrupt_graph,
    build_parallel_interrupt_graph,
    build_sequential_interrupt_graph,
    build_simple_interrupt_graph,
)


class TestInterruptEmission:
    """A pause must surface as a resumable pair of output items."""

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_emits_both_channels_and_resumes(self, script: ScriptRegistrar) -> None:
        key = "hitl-test"
        script(
            key,
            [
                # Turn 1: model decides to ask the user for their location.
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_ask_1",
                            "name": "AskHuman",
                            "args": {"question": "Where are you?"},
                        }
                    ],
                ),
                # Turn 2 (after resume): model produces the final answer.
                AIMessage(content="It's sunny in Seattle."),
            ],
        )
        host = ResponsesHostServer(build_ask_human_graph(key))
        conversation_id = "conv-hitl-1"

        with client_for(host) as client:
            # 1. Initial turn — graph should pause and the response should
            #    expose a __hosted_agent_adapter_interrupt__ function_call item.
            first = client.post(
                "/responses",
                json={
                    "input": "Look up the weather where I am.",
                    "conversation": {"id": conversation_id},
                },
            )
            assert first.status_code == 200, first.text
            first_payload = first.json()
            assert first_payload["status"] == "completed"
            interrupts = sentinels(first_payload)
            assert len(interrupts) == 1, first_payload
            envelope = json.loads(interrupts[0]["arguments"])
            assert envelope["value"] == "Where are you?"
            call_id = interrupts[0]["call_id"]
            assert call_id  # LangGraph interrupt id
            assert envelope["interrupt_id"] == call_id

            # The host should ALSO have emitted a paired mcp_approval_request
            # item with a storage-compatible id and the same arguments envelope.
            approvals = approval_requests(first_payload)
            assert len(approvals) == 1, first_payload
            assert approvals[0]["id"].startswith("mcpr_")
            assert approvals[0]["server_label"] == HITL_MCP_SERVER_LABEL
            assert approvals[0]["arguments"] == interrupts[0]["arguments"]
            assert json.loads(approvals[0]["arguments"])["interrupt_id"] == call_id

            # 2. Resume turn — submit a function_call_output keyed by the
            #    interrupt id. The host should resume the graph and return
            #    the assistant's final message.
            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(call_id, "Seattle")],
                },
            )
            assert second.status_code == 200, second.text
            second_payload = second.json()
            assert second_payload["status"] == "completed"
            # No new pending interrupt this time.
            assert not sentinels(second_payload), second_payload
            # And we should see the final assistant message text.
            assert "Seattle" in assistant_text(second_payload)


class TestEchoedSentinelItems:
    """Clients that replay output items must not poison the model context.

    The stateless Responses pattern is to resend the previous turn's
    output items alongside the new input. That echoes the HITL sentinel
    back to us on every later turn, long after the interrupt was consumed
    and the resume path stopped filtering it by consumed id.
    """

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_echoed_sentinel_never_reaches_the_model(
        self, script: ScriptRegistrar
    ) -> None:
        key = "hitl-echo"
        script(
            key,
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_ask_1",
                            "name": "AskHuman",
                            "args": {"question": "Where are you?"},
                        }
                    ],
                ),
                AIMessage(content="It's sunny in Seattle."),
                AIMessage(content="Happy to help."),
            ],
        )
        host = ResponsesHostServer(build_ask_human_graph(key))
        conversation_id = "conv-hitl-echo"

        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={
                    "input": "Look up the weather where I am.",
                    "conversation": {"id": conversation_id},
                },
            )
            assert first.status_code == 200, first.text
            call_id = sentinels(first.json())[0]["call_id"]

            # Resume, echoing the sentinel the way a stateless client
            # would. The pause is still open here, so the consumed-id set
            # is what keeps the pair out of the message channel.
            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [
                        sentinel_item(call_id, "Where are you?"),
                        resume_item(call_id, "Seattle"),
                    ],
                },
            )
            assert second.status_code == 200, second.text
            assert "Seattle" in assistant_text(second.json())

            # A later turn still carrying the echo. Nothing is pending now,
            # so only the reserved-name filter stands between the sentinel
            # and the model.
            third = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [
                        sentinel_item(call_id, "Where are you?"),
                        resume_item(call_id, "Seattle"),
                    ],
                },
            )
            assert third.status_code == 200, third.text
            assert third.json()["status"] == "completed", third.text
            assert not sentinels(third.json()), third.text

        turns = ScriptedModel.seen[key]
        assert len(turns) == 3, turns
        for turn in turns:
            reserved = [
                call
                for message in turn
                for call in getattr(message, "tool_calls", None) or ()
                if call["name"] == HITL_FUNCTION_NAME
            ]
            assert not reserved, turn


class TestResumeCallIdMismatch:
    """Recovery when the client answers with the wrong id."""

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_rejects_wrong_call_id_without_advancing_a_pause(
        self, script: ScriptRegistrar
    ) -> None:
        """A wrong call ID returns 400; the original pending ID stays valid."""
        key = "hitl-bad-resume"
        remaining = script(
            key,
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_ask_bad",
                            "name": "AskHuman",
                            "args": {"question": "Where?"},
                        }
                    ],
                ),
                AIMessage(content="Resumed after the correct reply."),
            ],
        )
        host = ResponsesHostServer(build_ask_human_graph(key))
        conversation_id = "conv-bad-resume"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={
                    "input": "ask me",
                    "conversation": {"id": conversation_id},
                },
            )
            assert first.status_code == 200, first.text
            interrupt_items = sentinels(first.json())
            assert len(interrupt_items) == 1
            sentinel_call_id = interrupt_items[0]["call_id"]

            # Client mistakenly echoes the AskHuman call_id instead of
            # the sentinel.
            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_ask_bad",
                            "output": '{"resume": "Seattle"}',
                        }
                    ],
                },
            )
            assert second.status_code == 400, second.text
            assert "call_ask_bad" in second.json()["error"]["message"]
            assert len(remaining) == 1

            resumed = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(sentinel_call_id, "Seattle")],
                },
            )
            assert resumed.status_code == 200, resumed.text
            assert "Resumed after the correct reply." in assistant_text(resumed.json())
        assert not remaining


class TestResumeChannels:
    """Complete replies through either of the paired resume channels."""

    @pytest.mark.parametrize("channel", ["mcp", "function"])
    def test_repaused_parallel_interrupt_requires_complete_replies(
        self,
        channel: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        graph = build_parallel_empty_update_interrupt_graph(follow_up=True)
        host = ResponsesHostServer(graph)
        conversation_id = "conv-mixed-partial-results"
        with client_for(host) as client:
            first = client.post(
                "/responses", json={"input": "start", "conversation": conversation_id}
            )
            assert first.status_code == 200, first.text
            calls = {
                interrupt_value(item): item["call_id"]
                for item in sentinels(first.json())
            }
            approvals = approval_requests(first.json())
            assert client.portal is not None
            config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}
            client.portal.call(
                graph.ainvoke, Command(resume={calls["question_b"]: "B1"}), config
            )
            before = client.portal.call(graph.aget_state, config)
            rejected = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
                    "input": (
                        [
                            {
                                "type": "mcp_approval_response",
                                "approval_request_id": approvals[0]["id"],
                                "approve": True,
                            }
                        ]
                        if channel == "mcp"
                        else [resume_item(calls["question_a"], "A")]
                    ),
                },
            )
            assert rejected.status_code == 400, rejected.text
            after = client.portal.call(graph.aget_state, config)
            assert after.config == before.config
            assert after.values == before.values
            assert after.tasks == before.tasks
            complete = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
                    "input": [
                        resume_item(calls["question_a"], "A"),
                        resume_item(calls["question_b"], "B2"),
                    ],
                },
            )
            assert complete.status_code == 200, complete.text
            assert not approval_requests(complete.json())
            assert "b=B1:B2" in assistant_text(complete.json())

    @pytest.mark.parametrize(
        "scenario", ["repaused", "completed-empty", "completed-output"]
    )
    @pytest.mark.parametrize("channel", ["mcp", "function"])
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("background", [False, True])
    def test_legacy_partial_resume_requires_new_conversation(
        self,
        scenario: str,
        channel: str,
        stream: bool,
        background: bool,
        foundry_state_stores: dict[str, dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        graph = (
            build_parallel_interrupt_graph()
            if scenario == "completed-output"
            else build_parallel_empty_update_interrupt_graph(
                follow_up=scenario == "repaused"
            )
        )
        host = ResponsesHostServer(graph)
        conversation_id = "conv-legacy-partial"
        with client_for(host) as client:
            first = client.post(
                "/responses", json={"input": "start", "conversation": conversation_id}
            )
            assert first.status_code == 200, first.text
            calls = {
                interrupt_value(item): item["call_id"]
                for item in sentinels(first.json())
            }
            assert client.portal is not None
            config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}
            answered = "question_b" if scenario == "repaused" else "question_a"
            client.portal.call(
                graph.ainvoke, Command(resume={calls[answered]: "old-answer"}), config
            )
            for records in foundry_state_stores.values():
                for key in tuple(records):
                    if key.startswith("langgraph_pending_interrupts:"):
                        del records[key]
            checkpointer = graph.checkpointer
            assert isinstance(checkpointer, InMemorySaver)
            before = client.portal.call(checkpointer.aget_tuple, config)
            assert before is not None
            graph_stream = MagicMock(wraps=graph.astream)
            monkeypatch.setattr(graph, "astream", graph_stream)
            replies = (
                [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": item["id"],
                        "approve": True,
                    }
                    for item in approval_requests(first.json())
                ]
                if channel == "mcp"
                else [resume_item(call_id, "new-answer") for call_id in calls.values()]
            )
            for index, inputs in enumerate((replies[:1], replies)):
                response_id = "caresp_" + str(index + 5) * 50
                rejected = client.post(
                    "/responses",
                    headers={"x-ms-response-id": response_id},
                    json={
                        "conversation": conversation_id,
                        "stream": stream,
                        "background": background,
                        "input": inputs,
                    },
                )
                assert rejected.status_code == 400, rejected.text
                assert rejected.headers["content-type"].startswith("application/json")
                error = rejected.json()["error"]
                assert error["type"] == "invalid_request_error", error
                assert "legacy" in error["message"].lower(), error
                assert "new conversation" in error["message"].lower(), error
                assert client.get(f"/responses/{response_id}").status_code == 404
            graph_stream.assert_not_called()
            after = client.portal.call(checkpointer.aget_tuple, config)
            assert after == before
            stored = client.get(f"/responses/{first.json()['id']}")
            assert stored.status_code == 200, stored.text
            assert {item["call_id"] for item in sentinels(stored.json())} == set(
                calls.values()
            )

            new_conversation_id = "conv-restarted"
            restarted = client.post(
                "/responses",
                json={"input": "start", "conversation": new_conversation_id},
            )
            assert restarted.status_code == 200, restarted.text
            assert {item["call_id"] for item in sentinels(restarted.json())}.isdisjoint(
                calls.values()
            )
            for _ in range(2):
                new_calls = sentinels(restarted.json())
                if not new_calls:
                    break
                restarted = client.post(
                    "/responses",
                    json={
                        "conversation": new_conversation_id,
                        "input": [
                            resume_item(item["call_id"], f"new-{interrupt_value(item)}")
                            for item in new_calls
                        ],
                    },
                )
                assert restarted.status_code == 200, restarted.text
            assert not sentinels(restarted.json())
            assert not approval_requests(restarted.json())
            assert "new-question_b" in assistant_text(restarted.json())
            after_restart = client.portal.call(checkpointer.aget_tuple, config)
            assert after_restart == before

    def test_large_full_batch_fits_minimum_dictionary_capacity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))

        class CapacityStore:
            def __init__(self) -> None:
                self.data: dict[tuple[str, str], dict[str, str]] = {}

            async def get(
                self, conversation_chain_id: str, key: str
            ) -> dict[str, str] | None:
                value = self.data.get((conversation_chain_id, key))
                return None if value is None else dict(value)

            async def set(
                self, conversation_chain_id: str, key: str, data: dict[str, str]
            ) -> None:
                assert sum(len(k) + len(v) for k, v in data.items()) <= 512
                self.data[conversation_chain_id, key] = dict(data)

        def ask(state: MessagesState) -> dict[str, list[AIMessage]]:
            return {"messages": [AIMessage(content=f"ok:{interrupt('question')}")]}

        builder = StateGraph(MessagesState)
        for index in range(16):
            name = f"ask_{index}"
            builder.add_node(name, ask)
            builder.add_edge(START, name)
            builder.add_edge(name, END)
        graph = builder.compile(checkpointer=InMemorySaver())
        host = ResponsesHostServer(graph, conversation_chain_store=CapacityStore())
        with client_for(host) as client:
            first = client.post(
                "/responses", json={"input": "start", "conversation": "conv-large"}
            )
            assert first.status_code == 200, first.text
            assert first.json()["status"] == "completed", first.text
            approvals = approval_requests(first.json())
            assert len(approvals) == 16
            assert (
                sum(
                    len(json.loads(item["arguments"])["interrupt_id"]) + 1
                    for item in approvals
                )
                > 512
            )
            complete = client.post(
                "/responses",
                json={
                    "conversation": "conv-large",
                    "input": [
                        {
                            "type": "mcp_approval_response",
                            "approval_request_id": item["id"],
                            "approve": True,
                        }
                        for item in approvals
                    ],
                },
            )
            assert complete.status_code == 200, complete.text
            assert complete.json()["status"] == "completed", complete.text
            assert not approval_requests(complete.json())

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("checkpointer", [None, False])
    def test_unpaused_input_with_disabled_checkpointer_is_unchanged(
        self,
        stream: bool,
        checkpointer: None | bool,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        graph = make_echo_graph().builder.compile(checkpointer=checkpointer)
        host = ResponsesHostServer(graph)
        with client_for(host) as client:
            response = client.post(
                "/responses", json={"input": "hello", "stream": stream}
            )
            assert response.status_code == 200, response.text
            payload = (
                next(
                    event["response"]
                    for event in sse_payloads(response.text)
                    if event.get("type") == "response.completed"
                )
                if stream
                else response.json()
            )
            assert "Echo: hello" in assistant_text(payload)

    @pytest.mark.parametrize("indexed", [False, True])
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("channel", ["mcp", "function"])
    def test_partial_reply_returns_400_without_advancing_graph(
        self,
        indexed: bool,
        stream: bool,
        channel: str,
        foundry_state_stores: dict[str, dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        graph = build_parallel_interrupt_graph()
        host = ResponsesHostServer(graph)
        conversation_id = "conv-complete-approvals"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "start", "conversation": conversation_id},
            )
            assert first.status_code == 200, first.text
            if not indexed:
                for records in foundry_state_stores.values():
                    for key in tuple(records):
                        if key.startswith("langgraph_pending_interrupts:"):
                            del records[key]
            approvals = approval_requests(first.json())
            assert len(approvals) == 2
            calls = sentinels(first.json())
            replies = (
                [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": item["id"],
                        "approve": True,
                    }
                    for item in approvals
                ]
                if channel == "mcp"
                else [
                    resume_item(item["call_id"], answer)
                    for item, answer in zip(calls, ("A", "B"), strict=True)
                ]
            )
            assert client.portal is not None
            config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}
            before = client.portal.call(graph.aget_state, config)

            partial = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
                    "stream": stream,
                    "input": replies[:1],
                },
            )

            assert partial.status_code == 400, partial.text
            assert partial.headers["content-type"].startswith("application/json")
            error = partial.json()["error"]
            assert error["type"] == "invalid_request_error"
            assert error["param"] == "input"
            assert "pending" in error["message"].lower()
            after = client.portal.call(graph.aget_state, config)
            assert after.config == before.config
            assert after.values == before.values
            assert after.tasks == before.tasks

            complete = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
                    "stream": stream,
                    "input": replies,
                },
            )
            assert complete.status_code == 200, complete.text
            payload = (
                next(
                    event["response"]
                    for event in sse_payloads(complete.text)
                    if event.get("type") == "response.completed"
                )
                if stream
                else complete.json()
            )
            assert payload["status"] == "completed", payload
            assert not approval_requests(payload)
            expected = ("question_a", "question_b") if channel == "mcp" else ("A", "B")
            assert f"a={expected[0]}" in assistant_text(payload)
            assert f"b={expected[1]}" in assistant_text(payload)

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("background", [False, True])
    @pytest.mark.parametrize("input_value", ["waiting", []])
    def test_zero_replies_return_400_without_creating_response(
        self,
        stream: bool,
        background: bool,
        input_value: str | list[object],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        graph = build_parallel_interrupt_graph()
        host = ResponsesHostServer(graph)
        conversation_id = "conv-zero-replies"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "start", "conversation": conversation_id},
            )
            assert first.status_code == 200, first.text
            assert len(approval_requests(first.json())) == 2
            assert client.portal is not None
            config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}
            before = client.portal.call(graph.aget_state, config)
            response_id = "caresp_" + "3" * 50

            rejected = client.post(
                "/responses",
                headers={"x-agent-response-id": response_id},
                json={
                    "conversation": conversation_id,
                    "stream": stream,
                    "background": background,
                    "input": input_value,
                },
            )

            assert rejected.status_code == 400, rejected.text
            assert rejected.headers["content-type"].startswith("application/json")
            assert rejected.json()["error"]["type"] == "invalid_request_error"
            assert rejected.json()["error"]["param"] == "input"
            assert client.get(f"/responses/{response_id}").status_code == 404
            after = client.portal.call(graph.aget_state, config)
            assert after.config == before.config
            assert after.values == before.values
            assert after.tasks == before.tasks

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "reply_kind", ["deny-one", "duplicate", "unknown", "update"]
    )
    def test_incomplete_approval_variants_return_400(
        self,
        stream: bool,
        reply_kind: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        host = ResponsesHostServer(build_parallel_interrupt_graph())
        conversation_id = "conv-incomplete-approvals"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "start", "conversation": conversation_id},
            )
            assert first.status_code == 200, first.text
            approvals = approval_requests(first.json())
            calls = sentinels(first.json())
            reply = {
                "type": "mcp_approval_response",
                "approval_request_id": approvals[0]["id"],
                "approve": reply_kind != "deny-one",
            }
            replies = [reply]
            if reply_kind == "duplicate":
                replies.append(dict(reply))
            elif reply_kind == "unknown":
                replies.append(
                    {
                        **reply,
                        "approval_request_id": "mcpr_unknown",
                    }
                )
            elif reply_kind == "update":
                replies.append(
                    {
                        "type": "function_call_output",
                        "call_id": calls[1]["call_id"],
                        "output": json.dumps({"update": {"messages": []}}),
                    }
                )
            response_id = "caresp_" + "2" * 50
            rejected = client.post(
                "/responses",
                headers={"x-agent-response-id": response_id},
                json={
                    "conversation": conversation_id,
                    "stream": stream,
                    "background": True,
                    "input": replies,
                },
            )

            assert rejected.status_code == 400, rejected.text
            assert rejected.headers["content-type"].startswith("application/json")
            assert rejected.json()["error"]["type"] == "invalid_request_error"
            assert rejected.json()["error"]["param"] == "input"
            assert client.get(f"/responses/{response_id}").status_code == 404

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("duplicate_output", [False, True])
    def test_update_only_override_cannot_bypass_complete_approval_requirement(
        self,
        stream: bool,
        duplicate_output: bool,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        host = ResponsesHostServer(build_parallel_interrupt_graph())
        conversation_id = "conv-approval-override"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "start", "conversation": conversation_id},
            )
            assert first.status_code == 200, first.text
            approvals = approval_requests(first.json())
            call_id = sentinels(first.json())[0]["call_id"]
            replies = [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": item["id"],
                    "approve": True,
                }
                for item in approvals
            ]
            replies.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({"update": {"messages": []}}),
                }
            )
            if duplicate_output:
                replies.append(resume_item(call_id, "A"))
            rejected = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
                    "stream": stream,
                    "input": replies,
                },
            )

            assert rejected.status_code == 400, rejected.text
            assert call_id in rejected.json()["error"]["message"]

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("function_override", [False, True])
    def test_complete_mixed_replies_preserve_function_output_precedence(
        self,
        stream: bool,
        function_override: bool,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        host = ResponsesHostServer(build_parallel_interrupt_graph())
        conversation_id = "conv-mixed-approvals"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "start", "conversation": conversation_id},
            )
            assert first.status_code == 200, first.text
            approvals = approval_requests(first.json())
            calls = sentinels(first.json())
            replies = [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approvals[0]["id"],
                    "approve": True,
                },
                resume_item(calls[1]["call_id"], "B"),
            ]
            if function_override:
                replies.append(resume_item(calls[0]["call_id"], "A"))
            resumed = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
                    "stream": stream,
                    "input": replies,
                },
            )

            assert resumed.status_code == 200, resumed.text
            payload = (
                next(
                    event["response"]
                    for event in sse_payloads(resumed.text)
                    if event.get("type") == "response.completed"
                )
                if stream
                else resumed.json()
            )
            assert payload["status"] == "completed", payload
            assert not approval_requests(payload)
            expected_a = "A" if function_override else "question_a"
            assert f"a={expected_a}" in assistant_text(payload)
            assert "b=B" in assistant_text(payload)

    @pytest.mark.parametrize("stream", [False, True])
    def test_complete_denial_is_not_treated_as_incomplete(
        self, stream: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        host = ResponsesHostServer(build_parallel_interrupt_graph())
        conversation_id = "conv-complete-denial"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "start", "conversation": conversation_id},
            )
            assert first.status_code == 200, first.text
            resumed = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
                    "stream": stream,
                    "input": [
                        {
                            "type": "mcp_approval_response",
                            "approval_request_id": item["id"],
                            "approve": False,
                        }
                        for item in approval_requests(first.json())
                    ],
                },
            )

            assert resumed.status_code == 200, resumed.text
            payload = (
                next(
                    event["response"]
                    for event in sse_payloads(resumed.text)
                    if event.get("type") == "response.failed"
                )
                if stream
                else resumed.json()
            )
            assert payload["error"]["code"] == "interrupt_rejected", payload

    def test_mcp_reply_only_requires_interrupts_still_pending(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        graph = build_parallel_interrupt_graph()
        host = ResponsesHostServer(graph)
        conversation_id = "conv-function-partial"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "start", "conversation": conversation_id},
            )
            assert first.status_code == 200, first.text
            calls = sentinels(first.json())
            approvals = approval_requests(first.json())
            assert client.portal is not None
            config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}
            # Legacy or Invocations turns can leave an already-partial checkpoint.
            client.portal.call(
                graph.ainvoke, Command(resume={calls[0]["call_id"]: "A"}), config
            )
            complete = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
                    "input": [
                        {
                            "type": "mcp_approval_response",
                            "approval_request_id": approvals[1]["id"],
                            "approve": True,
                        }
                    ],
                },
            )

            assert complete.status_code == 200, complete.text
            assert complete.json()["status"] == "completed"
            assert not approval_requests(complete.json())
            assert "b=question_b" in assistant_text(complete.json())

    def test_sequential_approvals_remain_separate_batches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        host = ResponsesHostServer(build_sequential_interrupt_graph())
        conversation_id = "conv-sequential-approvals"
        with client_for(host) as client:
            response = client.post(
                "/responses",
                json={"input": "start", "conversation": conversation_id},
            )
            for _ in range(2):
                assert response.status_code == 200, response.text
                approvals = approval_requests(response.json())
                assert len(approvals) == 1
                response = client.post(
                    "/responses",
                    json={
                        "conversation": conversation_id,
                        "input": [
                            {
                                "type": "mcp_approval_response",
                                "approval_request_id": approvals[0]["id"],
                                "approve": True,
                            }
                        ],
                    },
                )

            assert response.status_code == 200, response.text
            assert not approval_requests(response.json())
            assert "name?@city?" in assistant_text(response.json())

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("background", [False, True])
    @pytest.mark.parametrize("failure_type", [OSError, ValueError])
    def test_checkpoint_read_failure_is_not_reported_as_invalid_approval(
        self,
        stream: bool,
        background: bool,
        failure_type: type[Exception],
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        graph = build_simple_interrupt_graph()
        hook = MagicMock(spec=["on_span_start", "on_span_end"])
        host = ResponsesHostServer(
            graph, options=ResponsesServerOptions(create_span_hook=hook)
        )
        conversation_id = "conv-approval-read-failure"
        with TestClient(host.app, raise_server_exceptions=False) as client:
            first = client.post(
                "/responses",
                json={"input": "start", "conversation": conversation_id},
            )
            assert first.status_code == 200, first.text
            assert hook.on_span_start.call_count == 1
            assert hook.on_span_end.call_count == 1
            approval_id = approval_requests(first.json())[0]["id"]
            failure = failure_type("checkpoint read unavailable")
            monkeypatch.setattr(
                graph.checkpointer,
                "aget_tuple",
                AsyncMock(side_effect=failure),
            )
            response_id = "caresp_" + "4" * 50
            failed = client.post(
                "/responses",
                headers={"x-agent-response-id": response_id},
                json={
                    "conversation": conversation_id,
                    "stream": stream,
                    "background": background,
                    "input": [
                        {
                            "type": "mcp_approval_response",
                            "approval_request_id": approval_id,
                            "approve": True,
                        }
                    ],
                },
            )

            assert failed.status_code == 500, failed.text
            assert failed.headers["content-type"].startswith("application/json")
            assert failed.json()["error"]["type"] == "server_error"
            assert failed.json()["error"]["code"] == "server_error"
            assert (
                failed.headers["x-agent-session-id"]
                == first.headers["x-agent-session-id"]
            )
            assert hook.on_span_start.call_count == 2
            assert hook.on_span_end.call_count == 2
            span_error = hook.on_span_end.call_args.args[2]
            assert span_error is failure or span_error.__cause__ is failure
            assert "checkpoint read unavailable" in caplog.text
            assert client.get(f"/responses/{response_id}").status_code == 404

    @pytest.mark.parametrize("linkage", ["conversation", "previous_response_id"])
    def test_preflight_uses_the_execution_user_and_linkage(
        self, linkage: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        host = ResponsesHostServer(build_parallel_interrupt_graph())
        alice = {"x-agent-user-id": "alice"}
        with client_for(host) as client:
            first = client.post(
                "/responses",
                headers=alice,
                json={
                    "input": "start",
                    **(
                        {"conversation": "conv-scoped-approvals"}
                        if linkage == "conversation"
                        else {}
                    ),
                },
            )
            assert first.status_code == 200, first.text
            link = {
                linkage: (
                    "conv-scoped-approvals"
                    if linkage == "conversation"
                    else first.json()["id"]
                )
            }
            replies = [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": item["id"],
                    "approve": True,
                }
                for item in approval_requests(first.json())
            ]
            wrong_user = client.post(
                "/responses",
                headers={"x-agent-user-id": "bob"},
                json={**link, "input": replies},
            )
            assert wrong_user.status_code in {400, 404}, wrong_user.text
            partial = client.post(
                "/responses", headers=alice, json={**link, "input": replies[:1]}
            )
            assert partial.status_code == 400, partial.text
            complete = client.post(
                "/responses", headers=alice, json={**link, "input": replies}
            )
            assert complete.status_code == 200, complete.text
            assert not approval_requests(complete.json())

    def test_preflight_applies_to_resolved_approval_item_references(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        host = ResponsesHostServer(build_parallel_interrupt_graph())
        conversation_id = "conv-referenced-approvals"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "start", "conversation": conversation_id},
            )
            assert first.status_code == 200, first.text
            denied = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
                    "input": [
                        {
                            "type": "mcp_approval_response",
                            "approval_request_id": item["id"],
                            "approve": False,
                        }
                        for item in approval_requests(first.json())
                    ],
                },
            )
            assert denied.status_code == 200, denied.text
            assert denied.json()["status"] == "failed"
            stored = client.get(f"/responses/{denied.json()['id']}/input_items")
            assert stored.status_code == 200, stored.text
            references = [
                {"type": "item_reference", "id": item["id"]}
                for item in stored.json()["data"]
                if item["type"] == "mcp_approval_response"
            ]
            assert len(references) == 2

            partial = client.post(
                "/responses",
                json={"conversation": conversation_id, "input": references[:1]},
            )
            assert partial.status_code == 400, partial.text
            complete = client.post(
                "/responses",
                json={"conversation": conversation_id, "input": references},
            )
            assert complete.status_code == 200, complete.text
            assert complete.json()["error"]["code"] == "interrupt_rejected"

    def test_attached_prefixed_app_uses_only_the_current_graph_validator(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        app = ResponsesAgentServerHost(prefix="/v1", store=InMemoryResponseProvider())
        ResponsesHostServer(build_simple_interrupt_graph(), app=app)
        host = ResponsesHostServer(build_parallel_interrupt_graph(), app=app)
        conversation_id = "conv-attached-approvals"
        with client_for(host) as client:
            first = client.post(
                "/v1/responses",
                json={"input": "start", "conversation": conversation_id},
            )
            assert first.status_code == 200, first.text
            replies = [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": item["id"],
                    "approve": True,
                }
                for item in approval_requests(first.json())
            ]
            partial = client.post(
                "/v1/responses",
                json={"conversation": conversation_id, "input": replies[:1]},
            )
            assert partial.status_code == 400, partial.text
            complete = client.post(
                "/v1/responses",
                json={"conversation": conversation_id, "input": replies},
            )
            assert complete.status_code == 200, complete.text
            assert not approval_requests(complete.json())

    def test_custom_resume_hook_runs_once_during_preflight(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))

        class CustomResumeHost(ResponsesHostServer):
            config_calls = 0
            resume_calls = 0

            async def build_runnable_config(
                self, request: Any, context: Any
            ) -> RunnableConfig:
                self.config_calls += 1
                return await super().build_runnable_config(request, context)

            async def build_resume_command(
                self, request: Any, context: Any, pending: Any
            ) -> tuple[Command | None, frozenset[str]]:
                del request, context
                self.resume_calls += 1
                return (
                    Command(
                        resume={
                            pending[0].id: "A",
                            pending[1].id: "B",
                        }
                    ),
                    frozenset(),
                )

        host = CustomResumeHost(build_parallel_interrupt_graph())
        conversation_id = "conv-custom-resume"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "start", "conversation": conversation_id},
            )
            assert first.status_code == 200, first.text
            assert host.config_calls == 1

            resumed = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
                    "input": "custom protocol response",
                },
            )

            assert resumed.status_code == 200, resumed.text
            assert resumed.json()["status"] == "completed"
            assert "a=A" in assistant_text(resumed.json())
            assert "b=B" in assistant_text(resumed.json())
            assert host.config_calls == 2
            assert host.resume_calls == 1

    def test_custom_rejection_hook_runs_once_during_preflight(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))

        class CustomRejectionHost(ResponsesHostServer):
            rejection_calls = 0

            async def detect_rejection(
                self, request: Any, context: Any, pending: Any
            ) -> str | None:
                del request, context, pending
                self.rejection_calls += 1
                return "blocked by custom rejection policy"

        host = CustomRejectionHost(build_simple_interrupt_graph())
        conversation_id = "conv-custom-rejection"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "start", "conversation": conversation_id},
            )
            assert first.status_code == 200, first.text

            rejected = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
                    "input": "custom protocol rejection",
                },
            )

            assert rejected.status_code == 200, rejected.text
            assert rejected.json()["status"] == "failed"
            assert rejected.json()["error"] == {
                "code": "interrupt_rejected",
                "message": "blocked by custom rejection policy",
            }
            assert host.rejection_calls == 1

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_approve_resumes_the_graph(self, script: ScriptRegistrar) -> None:
        """Client resumes a paused graph via ``mcp_approval_response{approve:true}``;
        the host should drive the graph with ``Command(resume=interrupt.value)``
        (echoing the original interrupt value back, per design)."""
        key = "hitl-approve"
        script(
            key,
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_ask_approve",
                            "name": "AskHuman",
                            "args": {"question": "Confirm: run weather lookup?"},
                        }
                    ],
                ),
                AIMessage(content="OK, lookup completed."),
            ],
        )
        host = ResponsesHostServer(build_ask_human_graph(key))
        conversation_id = "conv-approve"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={
                    "input": "do the thing",
                    "conversation": {"id": conversation_id},
                },
            )
            assert first.status_code == 200, first.text
            approvals = approval_requests(first.json())
            assert len(approvals) == 1
            approval_id = approvals[0]["id"]

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [
                        {
                            "type": "mcp_approval_response",
                            "approval_request_id": approval_id,
                            "approve": True,
                        }
                    ],
                },
            )
            assert second.status_code == 200, second.text
            payload = second.json()
            assert payload["status"] == "completed"
            # No new pending interrupt this time.
            assert not sentinels(payload), payload
            assert not approval_requests(payload), payload
            assert "lookup completed" in assistant_text(payload)

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_reject_fails_the_turn(self, script: ScriptRegistrar) -> None:
        """``mcp_approval_response{approve:false}`` short-circuits the turn into
        ``response.failed(code='interrupt_rejected', …)``; the graph is NOT
        driven on the rejection turn."""
        key = "hitl-reject"
        remaining = script(
            key,
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_ask_reject",
                            "name": "AskHuman",
                            "args": {"question": "Confirm: irreversible action?"},
                        }
                    ],
                ),
                # MUST NOT be consumed — the host should not drive the graph on
                # the rejection turn.
                AIMessage(content="should not be reached"),
            ],
        )
        host = ResponsesHostServer(build_ask_human_graph(key))
        conversation_id = "conv-reject"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={
                    "input": "do something risky",
                    "conversation": {"id": conversation_id},
                },
            )
            assert first.status_code == 200, first.text
            approvals = approval_requests(first.json())
            assert len(approvals) == 1
            approval_id = approvals[0]["id"]

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [
                        {
                            "type": "mcp_approval_response",
                            "approval_request_id": approval_id,
                            "approve": False,
                            "reason": "user said no",
                        }
                    ],
                },
            )
            # The agentserver Responses lifecycle still returns 200 with
            # a ``failed`` status payload (mirrors how other failures are
            # surfaced).
            assert second.status_code == 200, second.text
            payload = second.json()
            assert payload["status"] == "failed", payload
            err = payload.get("error") or {}
            assert err.get("code") == "interrupt_rejected", payload
            assert approval_id in (err.get("message") or "")
            assert "user said no" in (err.get("message") or "")
        # The second scripted AIMessage must remain un-consumed because
        # the graph was not driven on the rejection turn.
        assert len(remaining) == 1


class TestThreadScoping:
    """Resume is scoped to the thread that paused.

    https://docs.langchain.com/oss/python/langgraph/interrupts#resuming-interrupts
    """

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_does_not_resume_a_pause_from_another_conversation(self) -> None:
        """An interrupt id is only meaningful on the thread that produced it.

        Conversation id maps to LangGraph's ``thread_id``, so replaying a
        resume item against a different conversation must not reach into the
        original thread. The second conversation starts its own run (and its
        own pause), and the first stays resumable.
        """
        host = ResponsesHostServer(build_simple_interrupt_graph())
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "hi", "conversation": {"id": "conv-thread-a"}},
            )
            assert first.status_code == 200, first.text
            pending = sentinels(first.json())
            assert len(pending) == 1, first.json()
            call_id = pending[0]["call_id"]

            # Same resume item, wrong thread: nothing is pending there, so it
            # is treated as ordinary input and that thread pauses on its own.
            other = client.post(
                "/responses",
                json={
                    "conversation": {"id": "conv-thread-b"},
                    "input": [resume_item(call_id, "Ada")],
                },
            )
            assert other.status_code == 200, other.text
            other_payload = other.json()
            other_sentinels = sentinels(other_payload)
            assert len(other_sentinels) == 1, other_payload
            assert other_sentinels[0]["call_id"] != call_id, other_payload
            assert "ok:Ada" not in assistant_text(other_payload), other_payload

            # The original thread is untouched and still resumable.
            resumed = client.post(
                "/responses",
                json={
                    "conversation": {"id": "conv-thread-a"},
                    "input": [resume_item(call_id, "Ada")],
                },
            )
            assert resumed.status_code == 200, resumed.text
            payload = resumed.json()
            assert payload["status"] == "completed", payload
            assert not sentinels(payload), payload
            assert "ok:Ada" in assistant_text(payload), payload


class TestStreaming:
    """Streaming with human-in-the-loop interrupts.

    https://docs.langchain.com/oss/python/langgraph/interrupts#stream-with-human-in-the-loop-hitl-interrupts
    """

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_streams_both_interrupt_channels_and_resumes(self) -> None:
        """Streaming clients must see the same two channels, and be able to
        resume over the streaming endpoint too.

        The buffered counterpart is ``TestInterruptEmission``; this pins that
        both item types survive SSE framing and that a streamed resume
        completes the run.
        """
        host = ResponsesHostServer(build_simple_interrupt_graph())
        conversation_id = "conv-stream-hitl"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={
                    "input": "hi",
                    "conversation": {"id": conversation_id},
                    "stream": True,
                },
            )
            assert first.status_code == 200, first.text
            items = hitl_items_in(sse_payloads(first.text))
            assert items, first.text
            assert {"function_call", "mcp_approval_request"} <= {
                item["type"] for item in items
            }, first.text
            call_ids = {
                item["call_id"] for item in items if item["type"] == "function_call"
            }
            assert len(call_ids) == 1, first.text
            call_id = call_ids.pop()

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(call_id, "Ada")],
                    "stream": True,
                },
            )
            assert second.status_code == 200, second.text
            assert not hitl_items_in(sse_payloads(second.text)), second.text
            assert "ok:Ada" in second.text, second.text
