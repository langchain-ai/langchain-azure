# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Citations survive Responses output, concurrent streams, and history."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from copy import deepcopy
from typing import Any

import httpx
import pytest

pytest.importorskip("azure.ai.agentserver.responses")

from azure.ai.agentserver.responses import ResponseEventStream  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    AIMessageChunk,
    HumanMessage,
)
from langchain_core.messages.content import create_citation  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from openai.types.responses import ResponseOutputMessage  # noqa: E402
from pydantic import SecretStr  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from langchain_azure_ai.agents.hosting import ResponsesHostServer  # noqa: E402
from langchain_azure_ai.agents.hosting._converters import (  # noqa: E402
    items_to_messages,
    state_to_events,
)
from langchain_azure_ai.agents.hosting._converters._stream import (  # noqa: E402
    StreamConverter,
)

URL: dict[str, Any] = {
    "type": "url_citation",
    "url": "https://example.com/source",
    "title": "Source",
    "start_index": 0,
    "end_index": 5,
}
FILE: dict[str, Any] = {
    "type": "file_citation",
    "file_id": "file-report",
    "filename": "report.pdf",
    "index": 0,
}
CONTAINER_FILE: dict[str, Any] = {
    "type": "container_file_citation",
    "container_id": "cntr-test",
    "file_id": "file-report",
    "filename": "report.pdf",
    "start_index": 0,
    "end_index": 5,
}
FILE_PATH: dict[str, Any] = {"type": "file_path", "file_id": "file-report", "index": 0}


def _citation(name: str) -> dict[str, Any]:
    return {
        "type": "url_citation",
        "url": f"https://example.com/{name}",
        "title": name,
        "start_index": 0,
        "end_index": 1,
    }


def _model_response(
    parts: list[dict[str, Any]], name: str = "provider"
) -> dict[str, Any]:
    return {
        "id": f"resp-{name}",
        "object": "response",
        "created_at": 0,
        "model": "test",
        "status": "completed",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": f"msg-{name}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": parts,
            }
        ],
    }


def _model_events(response: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield {
        "type": "response.created",
        "response": {**response, "status": "in_progress", "output": []},
    }
    for output_index, item in enumerate(response["output"]):
        yield {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {**item, "status": "in_progress", "content": []},
        }
        for content_index, part in enumerate(item["content"]):
            location = {
                "item_id": item["id"],
                "output_index": output_index,
                "content_index": content_index,
            }
            yield {
                **location,
                "type": "response.content_part.added",
                "part": {**part, "text": "", "annotations": []},
            }
            yield {
                **location,
                "type": "response.output_text.delta",
                "delta": part["text"],
                "logprobs": [],
            }
            for annotation_index, annotation in enumerate(part["annotations"]):
                yield {
                    **location,
                    "type": "response.output_text.annotation.added",
                    "annotation_index": annotation_index,
                    "annotation": annotation,
                }
            yield {
                **location,
                "type": "response.output_text.done",
                "text": part["text"],
                "logprobs": [],
            }
            yield {**location, "type": "response.content_part.done", "part": part}
        yield {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": item,
        }
    yield {"type": "response.completed", "response": response}


def _sse_event(event: dict[str, Any], sequence: int) -> bytes:
    return f"data: {json.dumps({**event, 'sequence_number': sequence})}\n\n".encode()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "annotation,expected",
    [
        pytest.param(create_citation(url=URL["url"]), None, id="url-only"),
        *[
            pytest.param(
                {k: v for k, v in URL.items() if k != missing},
                None,
                id=f"missing-{missing}",
            )
            for missing in ("url", "title", "start_index", "end_index")
        ],
        pytest.param({**URL, "title": None}, None, id="null-title"),
        pytest.param({**URL, "start_index": "0"}, None, id="string-index"),
        pytest.param({**URL, "end_index": True}, None, id="bool-index"),
        pytest.param(
            {"type": "file_citation", "file_id": "file-report"},
            None,
            id="incomplete-file",
        ),
        pytest.param({"type": "provider_citation"}, None, id="unsupported"),
        pytest.param("invalid", None, id="non-object"),
        pytest.param(
            create_citation(
                url=URL["url"],
                title="Source",
                start_index=0,
                end_index=5,
                cited_text="source excerpt",
            ),
            None,
            id="unsupported-standard-citation",
        ),
        pytest.param(
            {
                "type": "file_citation",
                "file_id": "file-report",
                "filename": "report.pdf",
                "file_index": 0,
            },
            FILE,
            id="langchain-file",
        ),
        pytest.param(FILE_PATH, FILE_PATH, id="file-path"),
        pytest.param(CONTAINER_FILE, CONTAINER_FILE, id="container-file"),
    ],
)
async def test_only_protocol_valid_annotations_reach_output_and_events(
    streaming: bool, annotation: Any, expected: dict[str, Any] | None
) -> None:
    stream = ResponseEventStream(response_id="resp-validation")
    stream.emit_created()
    stream.emit_in_progress()
    content: list[Any] = [
        {
            "type": "text",
            "text": "hello",
            "index": 0,
            "annotations": [URL, annotation, FILE],
        }
    ]
    original = deepcopy(content)
    if streaming:
        converter = StreamConverter(stream)
        events = [
            e
            async for e in converter.handle_message_chunk(
                (AIMessageChunk(id="answer", content=content), {})
            )
        ]
        events.extend([e async for e in converter.flush()])
    else:
        events = [
            e
            async for e in state_to_events(
                {"messages": [AIMessage(content=content)]}, stream
            )
        ]
    output = stream.emit_completed()["response"]["output"]
    item = ResponseOutputMessage.model_validate(output[0], strict=True)
    wanted = [URL, *([expected] if expected is not None else []), FILE]
    assert item.content[0].model_dump()["annotations"] == wanted
    assert item.content[0].model_dump()["text"] == "hello"
    annotation_events = [
        e for e in events if e["type"] == "response.output_text.annotation.added"
    ]
    assert [e["annotation"] for e in annotation_events] == wanted
    assert [e["annotation_index"] for e in annotation_events] == list(
        range(len(wanted))
    )
    assert content == original


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("annotation", [URL, FILE])
async def test_citations_preserve_parts_events_and_history(
    streaming: bool, annotation: dict[str, Any]
) -> None:
    stream = ResponseEventStream(response_id="resp-citations")
    events: list[Any] = [stream.emit_created(), stream.emit_in_progress()]
    content: list[Any] = [
        {"type": "text", "text": "hello", "annotations": [annotation]},
        {"type": "text", "text": "world", "annotations": [URL]},
    ]
    if streaming:
        converter = StreamConverter(stream)
        for part in content:
            events.extend(
                [
                    e
                    async for e in converter.handle_message_chunk(
                        (AIMessageChunk(id="answer", content=[part]), {})
                    )
                ]
            )
        events.extend([e async for e in converter.flush()])
    else:
        events.extend(
            [
                e
                async for e in state_to_events(
                    {"messages": [AIMessage(content=content)]}, stream
                )
            ]
        )
    completed = stream.emit_completed()
    output = completed["response"]["output"]
    assert len(output) == 1
    item = ResponseOutputMessage.model_validate(output[0], strict=True)
    assert [p.model_dump()["annotations"] for p in item.content] == [
        [annotation],
        [URL],
    ]
    assert [p.model_dump()["text"] for p in item.content] == ["hello", "world"]
    added = [e for e in events if e["type"] == "response.output_item.added"]
    done = [e for e in events if e["type"] == "response.output_item.done"]
    assert len(added) == len(done) == 1
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    assert added[0]["item"]["content"] == []
    assert added[0]["item"]["status"] == "in_progress"
    assert done[0]["item"]["status"] == "completed"
    assert added[0]["item"]["id"] == done[0]["item"]["id"] == item.id
    assert output[0]["type"] == "message"
    assert done[0]["item"]["content"] == output[0]["content"]
    part_added = [e for e in events if e["type"] == "response.content_part.added"]
    assert [e["content_index"] for e in part_added] == [0, 1]
    assert all(e["part"]["annotations"] == [] for e in part_added)
    assert [
        e["content_index"]
        for e in events
        if e["type"] == "response.output_text.annotation.added"
    ] == [0, 1]
    assert [
        e["part"]["annotations"]
        for e in events
        if e["type"] == "response.content_part.done"
    ] == [[annotation], [URL]]
    assert [
        e["annotation"]
        for e in events
        if e["type"] == "response.output_text.annotation.added"
    ] == [annotation, URL]
    for index in range(2):
        part_events = [e for e in events if e.get("content_index") == index]
        assert all(
            e["item_id"] == item.id and e["output_index"] == 0 for e in part_events
        )
        assert [e["type"] for e in part_events] == [
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.output_text.annotation.added",
            "response.content_part.done",
        ]
        assert part_events[3]["annotation_index"] == 0
        assert (
            part_events[1]["delta"] == part_events[2]["text"] == content[index]["text"]
        )
    replay = items_to_messages(output)
    assert isinstance(replay[0].content, list)
    assert isinstance(replay[0].content[0], dict)
    assert replay[0].content[0]["annotations"] == [annotation]
    original = deepcopy(output)
    replay[0].content[0]["annotations"][0]["type"] = "changed"
    assert output == original


async def test_indexed_text_streams_before_late_citation_and_real_client_reads_it() -> (
    None
):
    stream = ResponseEventStream(response_id="resp-stream-citation")
    events: list[Any] = [stream.emit_created(), stream.emit_in_progress()]
    converter = StreamConverter(stream)
    chunks = [
        {"type": "text", "text": "hel", "index": 0},
        {"type": "text", "text": "lo", "index": 0},
        {
            "type": "text",
            "annotations": [create_citation(url=URL["url"]), URL],
            "index": 0,
        },
    ]
    for part in chunks:
        emitted = [
            e
            async for e in converter.handle_message_chunk(
                (AIMessageChunk(id="answer", content=[part]), {})
            )
        ]
        assert [
            e["delta"] for e in emitted if e["type"] == "response.output_text.delta"
        ] == ([part["text"]] if "text" in part else [])
        assert not any(e["type"].endswith(".done") for e in emitted)
        events.extend(emitted)
    assert (
        "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
        == "hello"
    )
    events.extend([e async for e in converter.flush()])
    events.append(stream.emit_completed())
    data = "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=data
            )
        )
    ) as client:
        model = ChatOpenAI(
            model="test",
            api_key=SecretStr("test"),
            use_responses_api=True,
            output_version="responses/v1",
            http_client=client,
        )
        received = list(model.stream("hello"))
    message = received[0]
    for chunk in received[1:]:
        message += chunk
    assert message.text == "hello"
    assert isinstance(message.content, list)
    assert isinstance(message.content[0], dict)
    assert message.content[0]["annotations"] == [URL]


async def test_citations_do_not_leak_to_the_next_message() -> None:
    stream = ResponseEventStream(response_id="resp-boundary")
    stream.emit_created()
    stream.emit_in_progress()
    converter = StreamConverter(stream)
    for message in [
        AIMessage(
            id="first",
            content=[{"type": "text", "text": "hello", "annotations": [URL]}],
        ),
        AIMessage(id="second", content="uncited"),
    ]:
        _ = [e async for e in converter.handle_message_chunk((message, {}))]
    _ = [e async for e in converter.flush()]
    output: Any = stream.emit_completed()["response"]["output"]
    assert len(output) == 2
    assert output[0]["content"][0]["annotations"] == [URL]
    assert output[1]["content"][0]["annotations"] == []


@pytest.mark.parametrize("with_text", [False, True])
async def test_invalid_annotation_only_chunk_does_not_open_or_split_a_part(
    with_text: bool,
) -> None:
    stream = ResponseEventStream(response_id="resp-invalid-only")
    stream.emit_created()
    stream.emit_in_progress()
    converter = StreamConverter(stream)
    if with_text:
        _ = [
            e
            async for e in converter.handle_message_chunk(
                (
                    AIMessageChunk(
                        id="answer",
                        content=[{"type": "text", "text": "hel", "index": 0}],
                    ),
                    {},
                )
            )
        ]
    events = [
        e
        async for e in converter.handle_message_chunk(
            (
                AIMessageChunk(
                    id="answer",
                    content=[
                        {
                            "type": "text",
                            "index": 1,
                            "annotations": [{"type": "url_citation"}],
                        }
                    ],
                ),
                {},
            )
        )
    ]
    assert events == []
    if with_text:
        _ = [
            e
            async for e in converter.handle_message_chunk(
                (
                    AIMessageChunk(
                        id="answer",
                        content=[
                            {
                                "type": "text",
                                "text": "lo",
                                "index": 0,
                                "annotations": [URL],
                            }
                        ],
                    ),
                    {},
                )
            )
        ]
    _ = [e async for e in converter.flush()]
    output = stream.emit_completed()["response"]["output"]
    if with_text:
        assert len(output) == 1
        item = ResponseOutputMessage.model_validate(output[0], strict=True)
        assert [part.model_dump() for part in item.content] == [
            {
                "type": "output_text",
                "text": "hello",
                "annotations": [URL],
                "logprobs": [],
            }
        ]
    else:
        assert output == []


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_http_citation_reaches_client_and_second_turn(
    streaming: bool, checkpointed: bool
) -> None:
    captured: list[list[Any]] = []
    content: list[Any] = [
        {
            "type": "text",
            "text": "hello",
            "annotations": [create_citation(url=URL["url"]), URL],
        }
    ]

    def answer(state: MessagesState) -> dict[str, Any]:
        captured.append(deepcopy(state["messages"]))
        return {"messages": [AIMessage(content=deepcopy(content))]}

    graph = StateGraph(MessagesState)
    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    server = ResponsesHostServer(
        graph.compile(checkpointer=InMemorySaver() if checkpointed else None)
    )

    def completed(result: Any) -> dict[str, Any]:
        assert result.status_code == 200
        if not streaming:
            return result.json()
        events = [
            json.loads(line[5:].strip())
            for line in result.text.splitlines()
            if line.startswith("data:") and line[5:].strip() != "[DONE]"
        ]
        return next(e["response"] for e in events if e["type"] == "response.completed")

    with TestClient(server.app) as client:
        first = completed(
            client.post("/responses", json={"input": "hello", "stream": streaming})
        )
        second = completed(
            client.post(
                "/responses",
                json={
                    "input": "continue",
                    "stream": streaming,
                    "previous_response_id": first["id"],
                },
            )
        )
    assert first["status"] == second["status"] == "completed"
    assert first["output"][0]["content"][0]["annotations"] == [URL]
    assert second["output"][0]["content"][0]["annotations"] == [URL]
    previous = next(m for m in captured[1] if isinstance(m, AIMessage))
    assert isinstance(previous.content, list)
    assert len(previous.content) == 1
    assert isinstance(previous.content[0], dict)
    assert previous.content[0]["text"] == "hello"
    # A checkpointer owns the original graph state; only wire history is normalized.
    expected = content[0]["annotations"] if checkpointed else [URL]
    assert previous.content[0]["annotations"] == expected


async def test_invalid_citation_does_not_end_reasoning() -> None:
    stream = ResponseEventStream(response_id="resp-invalid-reasoning")
    stream.emit_created()
    stream.emit_in_progress()
    converter = StreamConverter(stream)
    blocks: list[dict[str, Any]] = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think "}]},
        {"type": "text", "index": 0, "annotations": [{"type": "url_citation"}]},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "more"}]},
    ]
    for block in blocks:
        events = [
            event
            async for event in converter.handle_message_chunk(
                (AIMessageChunk(id="answer", content=[block]), {})
            )
        ]
        if block["type"] == "text":
            assert events == []
    _ = [event async for event in converter.flush()]
    output = stream.emit_completed()["response"]["output"]
    assert len(output) == 1
    assert output[0]["type"] == "reasoning"
    assert output[0]["summary"] == [{"type": "summary_text", "text": "think more"}]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("model_streaming", [False, True])
def test_model_client_citations_reach_host_response(
    streaming: bool,
    model_streaming: bool,
) -> None:
    """Exercise provider result parsing and LangGraph callbacks before the host."""
    parts = [
        {"type": "output_text", "text": "hello", "annotations": [URL], "logprobs": []},
        {
            "type": "output_text",
            "text": "report",
            "annotations": [FILE, CONTAINER_FILE, FILE_PATH],
            "logprobs": [],
        },
    ]
    response_fixture = _model_response(parts)
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if model_streaming:
            data = b"".join(
                _sse_event(event, i)
                for i, event in enumerate(_model_events(response_fixture))
            )
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=data
            )
        return httpx.Response(200, json=response_fixture)

    async def answer(state: MessagesState) -> dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            model = ChatOpenAI(
                model="test",
                api_key=SecretStr("test"),
                use_responses_api=True,
                output_version="responses/v1",
                streaming=model_streaming,
                disable_streaming=not model_streaming,
                http_async_client=client,
            )
            return {"messages": [await model.ainvoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", answer)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    with TestClient(ResponsesHostServer(graph.compile()).app) as client:
        result = client.post(
            "/responses",
            json={
                "input": "Find the source",
                "stream": streaming,
                "store": False,
            },
        )
    assert result.status_code == 200
    assert len(requests) == 1
    if streaming:
        events = [
            json.loads(line[5:].strip())
            for line in result.text.splitlines()
            if line.startswith("data:") and line[5:].strip() != "[DONE]"
        ]
        response = next(
            e["response"] for e in events if e["type"] == "response.completed"
        )
        assert [
            e["annotation"]
            for e in events
            if e["type"] == "response.output_text.annotation.added"
        ] == [URL, FILE, CONTAINER_FILE, FILE_PATH]
    else:
        response = result.json()
    assert response["status"] == "completed"
    assert response["output"][0]["content"] == parts


@pytest.mark.parametrize("finish_first", ["A", "B"])
async def test_parallel_model_citations_stay_with_text_and_replayed_history(
    finish_first: str,
) -> None:
    """Force A text, B text, then late citations, without timing-based sleeps."""
    text_seen = {name: asyncio.Event() for name in ("A", "B")}
    node_done = {name: asyncio.Event() for name in ("A", "B")}
    requests: list[dict[str, Any]] = []

    class ModelStream(httpx.AsyncByteStream):
        def __init__(self, name: str) -> None:
            self.name = name

        async def __aiter__(self) -> AsyncIterator[bytes]:
            name = self.name
            part = {
                "type": "output_text",
                "text": name,
                "annotations": [_citation(name)],
                "logprobs": [],
            }
            response = _model_response([part], name)
            for i, event in enumerate(_model_events(response)):
                if event["type"] == "response.output_text.delta" and name == "B":
                    await asyncio.wait_for(text_seen["A"].wait(), 5)
                if event["type"] == "response.output_text.annotation.added":
                    await asyncio.wait_for(text_seen["B"].wait(), 5)
                    if name != finish_first:
                        await asyncio.wait_for(node_done[finish_first].wait(), 5)
                yield _sse_event(event, i)

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        name = body["input"][-1]["content"]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=ModelStream(name)
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = ChatOpenAI(
            model="test",
            api_key=SecretStr("test"),
            use_responses_api=True,
            output_version="responses/v1",
            streaming=True,
            http_async_client=client,
        )

        async def a(state: MessagesState) -> dict[str, Any]:
            return {"messages": [await model.ainvoke("A")]}

        async def b(state: MessagesState) -> dict[str, Any]:
            return {"messages": [await model.ainvoke("B")]}

        builder = StateGraph(MessagesState)
        builder.add_node("A", a)
        builder.add_node("B", b)
        for name in ("A", "B"):
            builder.add_edge(START, name)
            builder.add_edge(name, END)
        stream = ResponseEventStream(response_id="resp-parallel")
        events: list[Any] = [stream.emit_created(), stream.emit_in_progress()]
        converter = StreamConverter(stream)
        async for mode, payload in builder.compile().astream(
            {"messages": [HumanMessage(content="start")]},
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                emitted = [e async for e in converter.handle_message_chunk(payload)]
                events.extend(emitted)
                for e in emitted:
                    if e["type"] == "response.output_text.delta":
                        text_seen[e["delta"]].set()
            else:
                events.extend([e async for e in converter.handle_update(payload)])
                for name in payload:
                    node_done[name].set()
        events.extend([e async for e in converter.flush()])
        output = stream.emit_completed()["response"]["output"]

    assert len(requests) == 2
    assert [
        e["delta"] for e in events if e["type"] == "response.output_text.delta"
    ] == ["A", "B"]
    assert len(output) == 2
    assert [
        e["output_index"] for e in events if e["type"] == "response.output_item.done"
    ] == ([0, 1] if finish_first == "A" else [1, 0])
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    for index, name in enumerate(("A", "B")):
        item = ResponseOutputMessage.model_validate(output[index], strict=True)
        assert [p.model_dump() for p in item.content] == [
            {
                "type": "output_text",
                "text": name,
                "annotations": [_citation(name)],
                "logprobs": [],
            }
        ]
        owned = [e for e in events if e.get("item_id") == item.id]
        assert all(
            e["output_index"] == index and e["content_index"] == 0 for e in owned
        )
        assert [
            e["annotation"]
            for e in owned
            if e["type"] == "response.output_text.annotation.added"
        ] == [_citation(name)]
        done = [
            e
            for e in events
            if e["type"] == "response.output_item.done" and e["output_index"] == index
        ]
        assert len(done) == 1
        assert done[0]["item"] == output[index]
    replay = items_to_messages(output)
    assert [m.content for m in replay] == [
        [
            {
                "type": "text",
                "text": name,
                "annotations": [_citation(name)],
                "logprobs": [],
            }
        ]
        for name in ("A", "B")
    ]


async def test_late_citation_returns_to_its_original_content_part() -> None:
    stream = ResponseEventStream(response_id="resp-parts")
    stream.emit_created()
    stream.emit_in_progress()
    converter = StreamConverter(stream)
    for block in [
        {"type": "text", "text": "A", "index": 0},
        {"type": "text", "text": "B", "index": 1},
        {"type": "text", "annotations": [_citation("A")], "index": 0},
    ]:
        _ = [
            e
            async for e in converter.handle_message_chunk(
                (AIMessageChunk(id="answer", content=[block]), {})
            )
        ]
    _ = [
        e
        async for e in converter.handle_update(
            {"node": {"messages": [AIMessage(id="answer", content="AB")]}}
        )
    ]
    item = ResponseOutputMessage.model_validate(
        stream.emit_completed()["response"]["output"][0], strict=True
    )
    assert [p.model_dump() for p in item.content] == [
        {
            "type": "output_text",
            "text": "A",
            "annotations": [_citation("A")],
            "logprobs": [],
        },
        {"type": "output_text", "text": "B", "annotations": [], "logprobs": []},
    ]


@pytest.mark.parametrize("completion", ["update", "last_chunk"])
async def test_completing_b_leaves_a_open_until_its_own_completion(
    completion: str,
) -> None:
    stream = ResponseEventStream(response_id="resp-completion")
    stream.emit_created()
    stream.emit_in_progress()
    converter = StreamConverter(stream)
    for name in ("A", "B"):
        _ = [
            e
            async for e in converter.handle_message_chunk(
                (
                    AIMessageChunk(
                        id=name, content=[{"type": "text", "text": name, "index": 0}]
                    ),
                    {},
                )
            )
        ]
    if completion == "update":
        events = [
            e
            async for e in converter.handle_update(
                {"B": {"messages": [AIMessage(id="B", content="B")]}}
            )
        ]
    else:
        events = [
            e
            async for e in converter.handle_message_chunk(
                (AIMessageChunk(id="B", content="", chunk_position="last"), {})
            )
        ]
    assert [
        e["output_index"] for e in events if e["type"] == "response.output_item.done"
    ] == [1]
    assert all(e["output_index"] == 1 for e in events)
    events = [
        e
        async for e in converter.handle_message_chunk(
            (
                AIMessageChunk(
                    id="A",
                    content=[
                        {"type": "text", "annotations": [_citation("A")], "index": 0}
                    ],
                ),
                {},
            )
        )
    ]
    assert events == []
    events = [e async for e in converter.checkpoint()]
    assert [
        e["output_index"]
        for e in events
        if isinstance(e, dict) and e["type"] == "response.output_item.done"
    ] == [0]
    assert [e async for e in converter.flush()] == []
    output = stream.emit_completed()["response"]["output"]
    assert len(output) == 2
    item = ResponseOutputMessage.model_validate(output[0], strict=True)
    assert item.content[0].model_dump()["annotations"] == [_citation("A")]
