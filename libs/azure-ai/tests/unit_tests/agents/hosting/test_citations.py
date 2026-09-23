# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Citations survive Responses output, streaming clients, and history."""

import json
from copy import deepcopy
from typing import Any

import httpx
import pytest
from azure.ai.agentserver.responses import ResponseEventStream
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.content import create_citation
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from openai.types.responses import ResponseOutputMessage
from pydantic import SecretStr
from starlette.testclient import TestClient

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.agents.hosting._converters import (
    items_to_messages,
    state_to_events,
)
from langchain_azure_ai.agents.hosting._converters._stream import StreamConverter

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
    model_response: dict[str, Any] = {
        "id": "resp-provider",
        "object": "response",
        "created_at": 0,
        "model": "test",
        "status": "completed",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "msg-provider",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": parts,
            }
        ],
    }
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if model_streaming:
            item = model_response["output"][0]
            events: list[dict[str, Any]] = [
                {
                    "type": "response.created",
                    "response": {
                        **model_response,
                        "status": "in_progress",
                        "output": [],
                    },
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {**item, "status": "in_progress", "content": []},
                },
            ]
            for index, part in enumerate(parts):
                location = {
                    "item_id": "msg-provider",
                    "output_index": 0,
                    "content_index": index,
                }
                events.extend(
                    [
                        {
                            **location,
                            "type": "response.content_part.added",
                            "part": {**part, "text": "", "annotations": []},
                        },
                        {
                            **location,
                            "type": "response.output_text.delta",
                            "delta": part["text"],
                            "logprobs": [],
                        },
                    ]
                )
                for annotation_index, annotation in enumerate(part["annotations"]):
                    events.append(
                        {
                            **location,
                            "type": "response.output_text.annotation.added",
                            "annotation_index": annotation_index,
                            "annotation": annotation,
                        }
                    )
                events.extend(
                    [
                        {
                            **location,
                            "type": "response.output_text.done",
                            "text": part["text"],
                            "logprobs": [],
                        },
                        {
                            **location,
                            "type": "response.content_part.done",
                            "part": part,
                        },
                    ]
                )
            events.extend(
                [
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": item,
                    },
                    {"type": "response.completed", "response": model_response},
                ]
            )
            data = "".join(
                f"data: {json.dumps({**event, 'sequence_number': i})}\n\n"
                for i, event in enumerate(events)
            )
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=data
            )
        return httpx.Response(200, json=model_response)

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
