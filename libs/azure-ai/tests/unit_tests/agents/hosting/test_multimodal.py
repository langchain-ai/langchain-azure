# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Lossless Responses content conversion, using real SDK event builders."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

import pytest
from azure.ai.agentserver.responses import ResponseEventStream
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import _construct_lc_result_from_responses_api
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from openai.types.responses import Response
from pydantic import SecretStr
from starlette.testclient import TestClient

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.agents.hosting._converters import (
    build_messages_input,
    items_to_messages,
    state_to_events,
    stream_graph_to_events,
)

IMAGE_URL = {
    "type": "input_image",
    "image_url": "https://example.com/image.png",
    "detail": "high",
}
IMAGE_DATA = {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="}
IMAGE_ID = {"type": "input_image", "file_id": "file-image", "detail": "auto"}
FILE_URL = {
    "type": "input_file",
    "file_url": "https://example.com/doc.pdf",
    "filename": "doc.pdf",
}
FILE_ID = {"type": "input_file", "file_id": "file-doc", "filename": "doc.pdf"}
FILE_DATA = {
    "type": "input_file",
    "file_data": "data:text/plain;base64,aGVsbG8=",
    "filename": "hello.txt",
}
RICH = [IMAGE_URL, IMAGE_DATA, IMAGE_ID, FILE_URL, FILE_ID, FILE_DATA]
CITATION = {
    "type": "url_citation",
    "url": "https://example.com",
    "title": "Source",
    "start_index": 0,
    "end_index": 5,
}


@pytest.mark.parametrize("block", RICH)
def test_input_and_tool_history_preserve_rich_blocks(block: dict[str, Any]) -> None:
    content = [
        {"type": "input_text", "text": "before"},
        block,
        {"type": "input_text", "text": "after"},
    ]
    original = deepcopy(content)
    items = [
        {"type": "message", "role": "user", "content": content},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "tool",
            "arguments": "{}",
        },
        {"type": "function_call_output", "call_id": "call-1", "output": content},
    ]
    messages = build_messages_input(items)["messages"]
    expected = [
        {"type": "text", "text": "before"},
        block,
        {"type": "text", "text": "after"},
    ]
    assert messages[0].content == expected
    assert messages[2].content == expected
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].tool_call_id == "call-1"
    assert isinstance(messages[0].content, list)
    assert isinstance(messages[0].content[1], dict)
    messages[0].content[1]["mutated"] = True
    assert content == original
    assert messages[2].content == expected


def test_assistant_history_keeps_annotations_and_refusal() -> None:
    content = [
        {"type": "output_text", "text": "hello", "annotations": [CITATION]},
        {"type": "refusal", "refusal": "Cannot do that."},
    ]
    messages = items_to_messages(
        [{"type": "message", "role": "assistant", "content": content}]
    )
    assert messages[0].content == [
        {"type": "text", "text": "hello", "annotations": [CITATION]},
        content[1],
    ]


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "input_audio", "data": "secret"}],
        [{"type": "input_image"}],
        [{"type": "input_file", "file_id": "one", "file_url": "two"}],
        [{"type": "input_text", "text": 123}],
        {"untyped": "secret"},
    ],
)
def test_unsupported_or_invalid_input_fails_explicitly(content: Any) -> None:
    with pytest.raises(ValueError, match="Responses"):
        items_to_messages([{"type": "message", "role": "user", "content": content}])


async def _events(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


async def _output(
    messages: list[Any],
    mode: str,
    *,
    chunks: list[Any] | None = None,
) -> tuple[list[Any], list[Any]]:
    stream = ResponseEventStream(response_id="resp-multimodal")
    stream.emit_created()
    stream.emit_in_progress()
    if mode == "state":
        events = [e async for e in state_to_events({"messages": messages}, stream)]
    else:
        graph_events = (
            chunks if chunks is not None else [("messages", (m, {})) for m in messages]
        )
        graph_events += [
            ("updates", {"agent": {"messages": messages}}),
            ("values", {"messages": messages}),
        ]
        events = [
            e
            async for e in stream_graph_to_events(
                _events(graph_events),
                stream,
                cancellation_signal=asyncio.Event(),
            )
        ]
    completed = stream.emit_completed()
    return completed["response"]["output"], events


@pytest.mark.parametrize("mode", ["state", "tokens"])
async def test_assistant_parts_preserve_order_annotations_and_refusal(
    mode: str,
) -> None:
    message = AIMessage(
        content=[
            {"type": "text", "text": "hello", "annotations": [CITATION]},
            {"type": "refusal", "refusal": "Cannot do that."},
            {"type": "text", "text": "goodbye"},
        ]
    )
    output, events = await _output([message], mode)
    expected = [
        {
            "type": "output_text",
            "text": "hello",
            "annotations": [CITATION],
            "logprobs": [],
        },
        {"type": "refusal", "refusal": "Cannot do that."},
        {"type": "output_text", "text": "goodbye", "annotations": [], "logprobs": []},
    ]
    assert len(output) == 1
    assert output[0]["content"] == expected
    assert [
        e["item"]["content"] for e in events if e["type"] == "response.output_item.done"
    ] == [expected]
    replay = items_to_messages(output)
    assert isinstance(replay[0].content, list)
    assert isinstance(replay[0].content[0], dict)
    assert replay[0].content[0]["annotations"] == [CITATION]


@pytest.mark.parametrize("mode", ["state", "tokens"])
@pytest.mark.parametrize("block", RICH)
async def test_tool_output_preserves_content_and_replays(
    mode: str,
    block: dict[str, Any],
) -> None:
    content: list[str | dict[str, Any]] = [
        {"type": "text", "text": "before"},
        block,
        {"type": "text", "text": "after"},
    ]
    output, _ = await _output(
        [ToolMessage(content=content, tool_call_id="call-1")], mode
    )
    assert output[0]["output"] == [
        {"type": "input_text", "text": "before"},
        block,
        {"type": "input_text", "text": "after"},
    ]
    assert items_to_messages(output)[0].content == content


@pytest.mark.parametrize("mode", ["state", "tokens"])
async def test_tool_standard_image_and_file_keep_mime_filename(mode: str) -> None:
    message = ToolMessage(
        tool_call_id="call-1",
        content=[
            {"type": "image", "base64": "aGVsbG8=", "mime_type": "image/png"},
            {
                "type": "file",
                "base64": "aGVsbG8=",
                "mime_type": "application/pdf",
                "filename": "hello.pdf",
            },
        ],
    )
    output, _ = await _output([message], mode)
    assert output[0]["output"] == [
        IMAGE_DATA,
        {
            "type": "input_file",
            "file_data": "data:application/pdf;base64,aGVsbG8=",
            "filename": "hello.pdf",
        },
    ]


@pytest.mark.parametrize("mode", ["state", "tokens"])
async def test_unrepresentable_assistant_content_is_not_silently_lost(
    mode: str,
) -> None:
    with pytest.raises(ValueError, match="Responses.*image"):
        await _output([AIMessage(content=[{"type": "image", "url": "private"}])], mode)


@pytest.mark.parametrize("mode", ["state", "tokens"])
async def test_artifact_is_not_stringified_or_inserted_into_content(mode: str) -> None:
    with pytest.raises(ValueError, match="artifact"):
        await _output(
            [
                ToolMessage(
                    content="public",
                    tool_call_id="call-1",
                    artifact={"private": b"secret"},
                )
            ],
            mode,
        )


async def test_indexed_stream_fragments_merge_without_duplicate_blocks() -> None:
    chunks: list[Any] = [
        (
            "messages",
            (
                AIMessageChunk(
                    id="answer",
                    content=[
                        {"type": "text", "text": "hel", "index": 0},
                    ],
                ),
                {},
            ),
        ),
        (
            "messages",
            (
                AIMessageChunk(
                    id="answer",
                    content=[
                        {
                            "type": "text",
                            "text": "lo",
                            "index": 0,
                            "annotations": [CITATION],
                        },
                        {"type": "refusal", "refusal": "No.", "index": 1},
                    ],
                ),
                {},
            ),
        ),
    ]
    output, _ = await _output(
        [AIMessage(id="answer", content="ignored update")], "tokens", chunks=chunks
    )
    assert len(output) == 1
    assert output[0]["content"] == [
        {
            "type": "output_text",
            "text": "hello",
            "annotations": [CITATION],
            "logprobs": [],
        },
        {"type": "refusal", "refusal": "No."},
    ]


@pytest.mark.parametrize("mode", ["state", "tokens"])
async def test_typed_text_retains_text_delta_compatibility(mode: str) -> None:
    output, events = await _output(
        [
            AIMessage(
                content=[
                    {"type": "text", "text": "one"},
                    {"type": "text", "text": "two"},
                ]
            )
        ],
        mode,
    )
    assert (
        "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
        == "onetwo"
    )
    assert [p["text"] for p in output[0]["content"]] == ["one", "two"]


@pytest.mark.parametrize("mode", ["state", "tokens"])
async def test_standard_url_citation_maps_to_native_annotation(mode: str) -> None:
    citation = {**CITATION, "type": "citation"}
    output, _ = await _output(
        [
            AIMessage(
                content=[
                    {"type": "text", "text": "hello", "annotations": [citation]},
                ]
            )
        ],
        mode,
    )
    assert output[0]["content"][0]["annotations"] == [CITATION]


@pytest.mark.parametrize("mode", ["state", "tokens"])
@pytest.mark.parametrize(
    "part",
    [
        {"type": "refusal", "refusal": "Cannot help."},
        {
            "type": "output_text",
            "text": "A file",
            "annotations": [
                {
                    "type": "file_citation",
                    "file_id": "file-1",
                    "filename": "report.pdf",
                    "index": 2,
                },
            ],
            "logprobs": [],
        },
    ],
)
async def test_actual_langchain_responses_blocks_round_trip(
    mode: str,
    part: dict[str, Any],
) -> None:
    response = Response.model_validate(
        {
            "id": "resp-model",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "message",
                    "id": "msg-model",
                    "role": "assistant",
                    "status": "completed",
                    "content": [part],
                }
            ],
        }
    )
    message = (
        _construct_lc_result_from_responses_api(
            response,
            output_version="responses/v1",
        )
        .generations[0]
        .message
    )
    output, _ = await _output([message], mode)
    assert output[0]["content"] == [part]


@pytest.mark.parametrize("mode", ["state", "tokens"])
async def test_provider_file_wrapper_preserves_all_file_fields(mode: str) -> None:
    output, _ = await _output(
        [
            ToolMessage(
                content=[
                    {
                        "type": "file",
                        "file": {k: v for k, v in FILE_ID.items() if k != "type"},
                    },
                ],
                tool_call_id="call-1",
            )
        ],
        mode,
    )
    assert output[0]["output"] == [FILE_ID]


async def test_conflicting_file_names_are_not_silently_overwritten() -> None:
    with pytest.raises(ValueError, match="filename"):
        await _output(
            [
                ToolMessage(
                    content=[
                        {
                            "type": "file",
                            "file_id": "file-1",
                            "filename": "one",
                            "extras": {"filename": "two"},
                        },
                    ],
                    tool_call_id="call-1",
                )
            ],
            "tokens",
        )


@pytest.mark.parametrize("block", RICH)
def test_preserved_input_reaches_langchain_responses_payload(
    block: dict[str, Any],
) -> None:
    model = ChatOpenAI(
        model="test", api_key=SecretStr("test-key"), use_responses_api=True
    )
    messages = items_to_messages(
        [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe"},
                    block,
                ],
            },
        ]
    )
    payload = model._get_request_payload(messages)
    assert payload["input"][0]["content"] == [
        {"type": "input_text", "text": "describe"},
        block,
    ]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_http_rich_content_and_second_turn_history(
    streaming: bool,
    checkpointed: bool,
) -> None:
    captured: list[list[Any]] = []

    def plan(state: MessagesState) -> dict[str, Any]:
        captured.append(deepcopy(state["messages"]))
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "document", "id": f"call-{len(captured)}", "args": {}},
                    ],
                )
            ]
        }

    def tool(state: MessagesState) -> dict[str, Any]:
        return {
            "messages": [
                ToolMessage(
                    content=[{"type": "text", "text": "document"}, *RICH],
                    tool_call_id=f"call-{len(captured)}",
                )
            ]
        }

    def answer(state: MessagesState) -> dict[str, Any]:
        return {
            "messages": [
                AIMessage(
                    content=[
                        {"type": "text", "text": "hello", "annotations": [CITATION]},
                        {"type": "refusal", "refusal": "Cannot do that."},
                    ]
                )
            ]
        }

    graph = StateGraph(MessagesState)
    graph.add_node("plan", plan)
    graph.add_node("tool", tool)
    graph.add_node("answer", answer)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "tool")
    graph.add_edge("tool", "answer")
    graph.add_edge("answer", END)
    server = ResponsesHostServer(
        graph.compile(checkpointer=InMemorySaver() if checkpointed else None),
    )

    def completed(response: Any) -> dict[str, Any]:
        assert response.status_code == 200, response.text
        if not streaming:
            return response.json()
        events = [
            json.loads(line.removeprefix("data:").strip())
            for line in response.text.splitlines()
            if line.startswith("data:")
            and line.removeprefix("data:").strip() != "[DONE]"
        ]
        return next(
            event["response"]
            for event in events
            if event.get("type") == "response.completed"
        )

    with TestClient(server.app) as client:
        first = completed(
            client.post(
                "/responses",
                json={
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "describe"},
                                *RICH,
                            ],
                        }
                    ],
                    "stream": streaming,
                },
            )
        )
        assert first["status"] == "completed"
        assert first["output"][-1]["content"][0]["annotations"] == [CITATION]
        assert captured[0][0].content == [{"type": "text", "text": "describe"}, *RICH]
        assert [item["type"] for item in first["output"]] == [
            "function_call",
            "function_call_output",
            "message",
        ]
        assert first["output"][1]["output"] == [
            {"type": "input_text", "text": "document"},
            *RICH,
        ]
        second = completed(
            client.post(
                "/responses",
                json={
                    "input": "continue",
                    "previous_response_id": first["id"],
                    "stream": streaming,
                },
            )
        )
    assert second["status"] == "completed"
    history = captured[1]
    assert sum(m.content == captured[0][0].content for m in history) == 1
    annotated = [
        m for m in history if isinstance(m, AIMessage) and isinstance(m.content, list)
    ]
    assert len(annotated) == 1
    assert isinstance(annotated[0].content, list)
    assert isinstance(annotated[0].content[0], dict)
    assert annotated[0].content[0]["annotations"] == [CITATION]
    tools = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tools) == 1
    assert tools[0].content == [{"type": "text", "text": "document"}, *RICH]


@pytest.mark.parametrize("streaming", [False, True])
def test_unsupported_output_returns_failure_not_empty_success(streaming: bool) -> None:
    graph = StateGraph(MessagesState)
    graph.add_node(
        "answer",
        lambda _: {
            "messages": [
                AIMessage(content=[{"type": "audio", "url": "private"}]),
            ]
        },
    )
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    with TestClient(ResponsesHostServer(graph.compile()).app) as client:
        response = client.post(
            "/responses", json={"input": "hello", "stream": streaming}
        )
    if streaming:
        assert "response.failed" in response.text
        assert "response.completed" not in response.text
    else:
        assert response.json()["status"] == "failed"
    assert "private" not in response.text
