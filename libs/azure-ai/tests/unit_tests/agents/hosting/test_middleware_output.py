# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Middleware output must be checked before being published by a host."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

import pytest
from azure.ai.agentserver.responses import ResponseEventStream, ResponsesServerOptions
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, SummarizationMiddleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from starlette.testclient import TestClient

from langchain_azure_ai.agents.hosting import (
    InvocationsHostServer,
    ResponsesHostServer,
)
from langchain_azure_ai.agents.hosting._converters._stream import stream_graph_to_events
from langchain_azure_ai.agents.hosting._converters._utils import is_internal_message


def _events(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data:").strip())
        for line in body.splitlines()
        if line.startswith("data:") and line.removeprefix("data:").strip() != "[DONE]"
    ]


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
@pytest.mark.parametrize("strategy", ["redact", "block"])
@pytest.mark.parametrize("task_backed", [False, True])
def test_final_output_checks_pii_before_publishing(
    protocol: str, strategy: Literal["redact", "block"], task_backed: bool
) -> None:
    graph = create_agent(
        FakeListChatModel(responses=["Contact alice@example.com"]),
        middleware=[
            PIIMiddleware(
                "email", strategy=strategy, apply_to_input=False, apply_to_output=True
            )
        ],
        checkpointer=InMemorySaver() if task_backed else None,
    )
    options = ResponsesServerOptions(steerable_conversations=task_backed)
    server: Any = (
        ResponsesHostServer(graph, output_mode="final", options=options)
        if protocol == "responses"
        else InvocationsHostServer(graph, output_mode="final", options=options)
    )
    request = (
        {"input": "hello", "stream": True}
        if protocol == "responses"
        else {"message": "hello", "stream": True}
    )
    with TestClient(server.app) as client:
        response = client.post(f"/{protocol}", json=request)
    assert response.status_code == 200, response.text
    assert "alice@example.com" not in response.text
    events = _events(response.text)
    text = "".join(
        event.get("token", "")
        + (
            event.get("delta", "")
            if event.get("type") == "response.output_text.delta"
            else ""
        )
        for event in events
    )
    if strategy == "redact":
        assert text == "Contact [REDACTED_EMAIL]"
    else:
        assert text == ""
        assert (
            "response.failed" in response.text
            or "event: error" in response.text
            or any(event.get("status") == "failed" for event in events)
        )


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
async def test_summary_tokens_are_not_published(protocol: str) -> None:
    graph = create_agent(
        FakeListChatModel(responses=["PUBLIC_ANSWER"]),
        middleware=[
            SummarizationMiddleware(
                FakeListChatModel(responses=["INTERNAL_SUMMARY"]),
                trigger=("messages", 3),
                keep=("messages", 1),
            )
        ],
    )
    if protocol == "responses":
        with TestClient(ResponsesHostServer(graph).app) as client:
            response = client.post(
                "/responses",
                json={
                    "input": [
                        {"role": "user", "content": "old question"},
                        {"role": "assistant", "content": "old answer"},
                        {"role": "user", "content": "new question"},
                    ],
                    "stream": True,
                },
            )
        assert response.status_code == 200, response.text
        body = response.text
    else:
        server = InvocationsHostServer(graph)
        body = b"".join(
            [
                chunk
                async for chunk in server._stream_tokens(
                    {
                        "messages": [
                            HumanMessage("old question"),
                            AIMessage("old answer"),
                            HumanMessage("new question"),
                        ]
                    },
                    {},
                )
            ]
        ).decode()
    events = _events(body)
    text = "".join(
        event.get("token", "")
        + (
            event.get("delta", "")
            if event.get("type") == "response.output_text.delta"
            else ""
        )
        for event in events
    )
    assert text == "PUBLIC_ANSWER"
    assert "INTERNAL_SUMMARY" not in body


@pytest.mark.parametrize("termination", ["cancel", "shutdown", "interrupt", "error"])
async def test_final_response_never_publishes_unapproved_state(
    termination: str,
) -> None:
    cancel = asyncio.Event()
    shutdown = asyncio.Event()

    async def graph_stream() -> Any:
        yield "values", {"messages": [AIMessage("UNAPPROVED")]}
        if termination == "cancel":
            cancel.set()
        elif termination == "shutdown":
            shutdown.set()
        elif termination == "interrupt":
            yield "updates", {"__interrupt__": [object()]}
        else:
            raise ValueError("Output rejected")

    stream = ResponseEventStream(response_id="resp-test")
    stream.emit_created()
    stream.emit_in_progress()
    events = []
    try:
        async for event in stream_graph_to_events(
            graph_stream(),
            stream,
            cancellation_signal=cancel,
            shutdown_signal=shutdown,
            output_mode="final",
        ):
            events.append(event)
    except ValueError:
        assert termination == "error"
    assert not events


def test_summary_filter_uses_call_metadata_only() -> None:
    message = AIMessage("visible", additional_kwargs={"lc_source": "summarization"})
    assert not is_internal_message(message)
    assert not is_internal_message((message, {"lc_source": "another_source"}))
    assert not is_internal_message((message, {"lc_internal_call": "caller_supplied"}))
    assert is_internal_message((message, {"lc_source": "summarization"}))


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_final_output_waits_for_approval_and_resumes(
    protocol: str, streaming: bool
) -> None:
    def model(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage("UNAPPROVED", id="answer")]}

    def review(state: MessagesState) -> dict[str, Any]:
        interrupt("Approve output?")
        return {"messages": [AIMessage("APPROVED", id="answer")]}

    builder = StateGraph(MessagesState)
    builder.add_node("model", model)
    builder.add_node("review", review)
    builder.add_edge(START, "model")
    builder.add_edge("model", "review")
    builder.add_edge("review", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    if protocol == "responses":
        server: Any = ResponsesHostServer(graph, output_mode="final")
        request = {"input": "hello", "stream": streaming}
    else:
        server = InvocationsHostServer(graph, output_mode="final")
        request = {"message": "hello", "stream": streaming}
    with TestClient(server.app) as client:
        response = client.post(f"/{protocol}", json=request)
    assert response.status_code == 200, response.text
    assert "UNAPPROVED" not in response.text
    assert "Approve output?" in response.text

    # Verify the checked state replaces the earlier message after resuming.
    config: Any = {"configurable": {"thread_id": "resume-test"}}
    await graph.ainvoke({"messages": [HumanMessage("hello")]}, config)
    if protocol == "invocations":
        body = b"".join(
            [
                chunk
                async for chunk in server._stream_tokens(Command(resume=True), config)
            ]
        ).decode()
    else:
        stream = ResponseEventStream(response_id="resp-resume")
        stream.emit_created()
        stream.emit_in_progress()
        events = [
            event
            async for event in stream_graph_to_events(
                graph.astream(
                    Command(resume=True),
                    config,
                    stream_mode=["values", "updates", "messages"],
                ),
                stream,
                cancellation_signal=asyncio.Event(),
                output_mode="final",
            )
        ]
        body = json.dumps(events)
    assert "UNAPPROVED" not in body
    assert "APPROVED" in body
