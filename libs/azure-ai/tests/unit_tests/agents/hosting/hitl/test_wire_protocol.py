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
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("starlette")

from azure.ai.agentserver.responses import (
    InMemoryResponseProvider,
    ResponsesAgentServerHost,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from starlette.testclient import TestClient

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.agents.hosting._converters import (
    HITL_FUNCTION_NAME,
    HITL_MCP_SERVER_LABEL,
)

from .conftest import (
    REAL_INTERRUPT_ASYNC_XFAIL,
    ScriptRegistrar,
    approval_requests,
    assistant_text,
    client_for,
    hitl_items_in,
    resume_item,
    sentinel_item,
    sentinels,
    sse_payloads,
)
from .graphs import (
    ScriptedModel,
    build_ask_human_graph,
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
    def test_reemits_sentinel_when_a_pause_is_outstanding(
        self, script: ScriptRegistrar
    ) -> None:
        """Pending interrupt + wrong-call_id resume → host re-emits the
        sentinel instead of driving the graph with a malformed message list.

        This is the recovery path for a client that echoed the wrong
        function_call's call_id on resume (e.g. the LLM's ``AskHuman`` id
        instead of the interrupt sentinel id).
        """
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
                # Second AIMessage should NEVER be consumed — the host must not
                # drive the graph on the bad-resume turn.
                AIMessage(content="should not be reached"),
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
            assert second.status_code == 200, second.text
            payload = second.json()
            assert payload["status"] == "completed"
            # Host re-emits the SAME pending sentinel (both channels) so
            # the client can retry with the correct call_id.
            reemitted = sentinels(payload)
            assert len(reemitted) == 1
            assert reemitted[0]["call_id"] == sentinel_call_id
            approvals = approval_requests(payload)
            assert len(approvals) == 1
            assert approvals[0]["id"].startswith("mcpr_")
            assert (
                json.loads(approvals[0]["arguments"])["interrupt_id"]
                == sentinel_call_id
            )
            # And no spurious assistant message from a second LLM call.
            assert not [it for it in payload["output"] if it.get("type") == "message"]
        # The second scripted AIMessage must remain un-consumed because
        # the graph was not driven on the bad-resume turn.
        assert len(remaining) == 1


class TestMcpApprovalChannel:
    """Resuming (or failing) a turn via ``mcp_approval_response``."""

    @pytest.mark.parametrize("stream", [False, True])
    def test_partial_approval_returns_400_without_advancing_graph(
        self, stream: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
            approvals = approval_requests(first.json())
            assert len(approvals) == 2
            assert client.portal is not None
            config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}
            before = client.portal.call(graph.aget_state, config)

            partial = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
                    "stream": stream,
                    "input": [
                        {
                            "type": "mcp_approval_response",
                            "approval_request_id": approvals[0]["id"],
                            "approve": True,
                        }
                    ],
                },
            )

            assert partial.status_code == 400, partial.text
            assert partial.headers["content-type"].startswith("application/json")
            error = partial.json()["error"]
            assert error["type"] == "invalid_request_error"
            assert error["param"] == "input"
            assert "approval" in error["message"].lower()
            after = client.portal.call(graph.aget_state, config)
            assert after.config == before.config
            assert after.values == before.values
            assert after.tasks == before.tasks

            complete = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
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
            assert "a=question_a" in assistant_text(complete.json())
            assert "b=question_b" in assistant_text(complete.json())

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
    def test_complete_mixed_replies_preserve_function_output_precedence(
        self, stream: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
            resumed = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
                    "stream": stream,
                    "input": [
                        {
                            "type": "mcp_approval_response",
                            "approval_request_id": approvals[0]["id"],
                            "approve": True,
                        },
                        resume_item(calls[0]["call_id"], "A"),
                        resume_item(calls[1]["call_id"], "B"),
                    ],
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
            assert "a=A" in assistant_text(payload)
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
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        host = ResponsesHostServer(build_parallel_interrupt_graph())
        conversation_id = "conv-function-partial"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "start", "conversation": conversation_id},
            )
            assert first.status_code == 200, first.text
            calls = sentinels(first.json())
            partial = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
                    "input": [resume_item(calls[0]["call_id"], "A")],
                },
            )
            assert partial.status_code == 200, partial.text
            approvals = approval_requests(partial.json())
            assert len(approvals) == 1
            complete = client.post(
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

    def test_checkpoint_read_failure_is_not_reported_as_invalid_approval(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        graph = build_simple_interrupt_graph()
        host = ResponsesHostServer(graph)
        conversation_id = "conv-approval-read-failure"
        with TestClient(host.app, raise_server_exceptions=False) as client:
            first = client.post(
                "/responses",
                json={"input": "start", "conversation": conversation_id},
            )
            assert first.status_code == 200, first.text
            approval_id = approval_requests(first.json())[0]["id"]
            monkeypatch.setattr(
                graph.checkpointer,
                "aget_tuple",
                AsyncMock(side_effect=OSError("checkpoint read unavailable")),
            )
            failed = client.post(
                "/responses",
                json={
                    "conversation": conversation_id,
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
