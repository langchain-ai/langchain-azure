# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end tests for ``LangGraphResponsesHostServer`` via Starlette TestClient."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from langchain_azure_ai.agents.hosting import LangGraphResponsesHostServer

from .conftest import (
    make_custom_state_graph,
    make_echo_graph,
    make_streaming_graph,
)


def _client(server: LangGraphResponsesHostServer) -> TestClient:
    return TestClient(server.app)


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    current_type = ""
    for line in body.splitlines():
        if line.startswith("event:"):
            current_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data = line.split(":", 1)[1].strip()
            if not data:
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                payload = {"raw": data}
            events.append((current_type, payload))
    return events


def test_non_streaming_request_returns_completed_response() -> None:
    server = LangGraphResponsesHostServer(make_echo_graph())
    with _client(server) as client:
        resp = client.post(
            "/responses", json={"input": "hello", "model": "test"}
        )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["status"] == "completed"
    output = payload["output"]
    assert any(item.get("type") == "message" for item in output)
    text = "".join(
        part.get("text", "")
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", [])
    )
    assert "Echo: hello" in text


def test_streaming_request_emits_sse_lifecycle_events() -> None:
    server = LangGraphResponsesHostServer(make_streaming_graph())
    with _client(server) as client:
        resp = client.post(
            "/responses",
            json={"input": "ignored", "stream": True, "model": "test"},
        )
    assert resp.status_code == 200, resp.text
    events = _parse_sse(resp.text)
    types = [t for t, _ in events]
    assert "response.created" in types
    assert "response.in_progress" in types
    assert "response.output_text.delta" in types
    assert "response.completed" in types
    deltas = [p["delta"] for t, p in events if t == "response.output_text.delta"]
    assert "".join(deltas) == "Hello, world!"


def test_health_endpoint_is_available() -> None:
    server = LangGraphResponsesHostServer(make_echo_graph())
    with _client(server) as client:
        resp = client.get("/healthy")
    assert resp.status_code == 200


def test_constructor_rejects_non_messages_state_schema() -> None:
    with pytest.raises(ValueError, match="messages"):
        LangGraphResponsesHostServer(make_custom_state_graph())
