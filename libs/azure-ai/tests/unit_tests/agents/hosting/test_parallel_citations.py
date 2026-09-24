# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Citation ownership across interleaved LangGraph model streams."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from azure.ai.agentserver.responses import ResponseEventStream
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from openai.types.responses import ResponseOutputMessage
from pydantic import SecretStr

from langchain_azure_ai.agents.hosting._converters import items_to_messages
from langchain_azure_ai.agents.hosting._converters._stream import StreamConverter

from .responses_fixtures import model_events, model_response, sse_event


def citation(name: str) -> dict[str, Any]:
    return {
        "type": "url_citation",
        "url": f"https://example.com/{name}",
        "title": name,
        "start_index": 0,
        "end_index": 1,
    }


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
                "annotations": [citation(name)],
                "logprobs": [],
            }
            response = model_response([part], name)
            for i, event in enumerate(model_events(response)):
                if event["type"] == "response.output_text.delta" and name == "B":
                    await asyncio.wait_for(text_seen["A"].wait(), 5)
                if event["type"] == "response.output_text.annotation.added":
                    await asyncio.wait_for(text_seen["B"].wait(), 5)
                    if name != finish_first:
                        await asyncio.wait_for(node_done[finish_first].wait(), 5)
                yield sse_event(event, i)

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
                "annotations": [citation(name)],
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
        ] == [citation(name)]
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
                "annotations": [citation(name)],
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
        {"type": "text", "annotations": [citation("A")], "index": 0},
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
            "annotations": [citation("A")],
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
                        {"type": "text", "annotations": [citation("A")], "index": 0}
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
    assert item.content[0].model_dump()["annotations"] == [citation("A")]
