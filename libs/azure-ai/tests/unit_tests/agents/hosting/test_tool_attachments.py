# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tool attachments reach Responses clients without exposing private artifacts."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, cast

import pytest

pytest.importorskip("azure.ai.agentserver.responses")

from azure.ai.agentserver.responses import ResponseEventStream  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.messages.content import (  # noqa: E402
    create_file_block,
    create_image_block,
    create_text_block,
)
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from openai.types.responses import ResponseOutputItem  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from langchain_azure_ai.agents.hosting import ResponsesHostServer  # noqa: E402
from langchain_azure_ai.agents.hosting._converters import state_to_events  # noqa: E402

IMAGE = {
    "type": "input_image",
    "image_url": "https://example.com/chart.png",
    "detail": "high",
}
FILE = {"type": "input_file", "file_id": "file-report", "filename": "report.pdf"}
CASES = [
    (IMAGE, IMAGE),
    (FILE, FILE),
    (
        {"type": "input_image", "file_id": "file-image"},
        {"type": "input_image", "file_id": "file-image", "detail": "auto"},
    ),
    (create_image_block(url=IMAGE["image_url"], detail="high"), IMAGE),
    (
        create_image_block(file_id="file-image"),
        {"type": "input_image", "file_id": "file-image", "detail": "auto"},
    ),
    (
        create_image_block(base64="aGVsbG8=", mime_type="image/png"),
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,aGVsbG8=",
            "detail": "auto",
        },
    ),
    (create_file_block(file_id="file-report", filename="report.pdf"), FILE),
    (
        create_file_block(url="https://example.com/report.pdf", filename="report.pdf"),
        {
            "type": "input_file",
            "file_url": "https://example.com/report.pdf",
            "filename": "report.pdf",
        },
    ),
    (
        create_file_block(
            base64="aGVsbG8=", mime_type="application/pdf", filename="report.pdf"
        ),
        {
            "type": "input_file",
            "file_data": "data:application/pdf;base64,aGVsbG8=",
            "filename": "report.pdf",
        },
    ),
    (
        {
            "type": "image_url",
            "image_url": {"url": IMAGE["image_url"], "detail": "high"},
        },
        IMAGE,
    ),
    (
        {"type": "file", "file": {"file_id": "file-report", "filename": "report.pdf"}},
        FILE,
    ),
]


@pytest.mark.parametrize("block,expected", CASES)
@pytest.mark.parametrize("mode", ["state", "json", "sse"])
async def test_tool_attachments_reach_client(
    block: dict[str, Any], expected: dict[str, Any], mode: str
) -> None:
    message = ToolMessage(
        content=[dict(create_text_block(text="before")), block, "after"],
        tool_call_id="call-1",
        artifact={"private": "not-for-the-client"},
    )
    original = message.model_copy(deep=True)
    messages = [
        AIMessage(
            content="", tool_calls=[{"id": "call-1", "name": "report", "args": {}}]
        ),
        message,
        AIMessage(content="Report ready"),
    ]
    events: list[Any] = []
    if mode == "state":
        stream = ResponseEventStream(response_id="resp-tools")
        stream.emit_created()
        stream.emit_in_progress()
        events = [e async for e in state_to_events({"messages": messages}, stream)]
        response = stream.emit_completed()["response"]
    else:
        graph = StateGraph(MessagesState)
        graph.add_node("report", lambda _: {"messages": deepcopy(messages)})
        graph.add_edge(START, "report")
        graph.add_edge("report", END)
        with TestClient(ResponsesHostServer(graph.compile()).app) as client:
            result = client.post(
                "/responses",
                json={
                    "input": "Make a report",
                    "stream": mode == "sse",
                    "store": False,
                },
            )
            assert result.status_code == 200, result.text
            if mode == "sse":
                events = [
                    json.loads(line[5:].strip())
                    for line in result.text.splitlines()
                    if line.startswith("data:") and line[5:].strip() != "[DONE]"
                ]
                response = next(
                    e["response"] for e in events if e["type"] == "response.completed"
                )
            else:
                response = result.json()
    assert response["status"] == "completed"
    outputs = [
        item for item in response["output"] if item["type"] == "function_call_output"
    ]
    assert len(outputs) == 1
    parsed: ResponseOutputItem = TypeAdapter(ResponseOutputItem).validate_python(
        outputs[0], strict=True
    )
    assert parsed.type == "function_call_output"
    assert parsed.model_dump(exclude_none=True)["output"] == outputs[0]["output"]
    expected_parts = [
        {"type": "input_text", "text": "before"},
        expected,
        {"type": "input_text", "text": "after"},
    ]
    assert outputs[0]["call_id"] == "call-1"
    assert outputs[0]["output"] == expected_parts
    assert "not-for-the-client" not in json.dumps(response)
    assert message == original
    if events:
        done = [
            e["item"]
            for e in events
            if e["type"] == "response.output_item.done"
            and e["item"]["type"] == "function_call_output"
        ]
        assert len(done) == 1
        assert done[0]["output"] == expected_parts
    cast(dict[str, Any], outputs[0]["output"][1])["mutated"] = True
    assert message == original


@pytest.mark.parametrize(
    "content,expected",
    [
        ("plain", "plain"),
        (["one", {"type": "text", "text": "two"}], "onetwo"),
        ([], ""),
    ],
)
async def test_text_only_tool_output_is_unchanged(content: Any, expected: str) -> None:
    stream = ResponseEventStream(response_id="resp-text")
    stream.emit_created()
    stream.emit_in_progress()
    message = ToolMessage(
        content=content, tool_call_id="call-1", artifact={"private": "data"}
    )
    _ = [e async for e in state_to_events({"messages": [message]}, stream)]
    item = stream.emit_completed()["response"]["output"][0]
    assert item["type"] == "function_call_output"
    assert item["output"] == expected


@pytest.mark.parametrize("block", [IMAGE, FILE])
async def test_attachment_only_tool_result_is_not_empty(block: dict[str, Any]) -> None:
    stream = ResponseEventStream(response_id="resp-attachment")
    stream.emit_created()
    stream.emit_in_progress()
    message = ToolMessage(content=[block], tool_call_id="call-1")
    _ = [e async for e in state_to_events({"messages": [message]}, stream)]
    item = stream.emit_completed()["response"]["output"][0]
    assert item["type"] == "function_call_output"
    assert item["output"] == [block]
