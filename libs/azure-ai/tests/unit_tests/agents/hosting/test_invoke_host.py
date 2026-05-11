# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end tests for ``AzureAIInvokeAgentHost`` via Starlette TestClient."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from langchain_azure_ai.agents.hosting import AzureAIInvokeAgentHost

from .conftest import (
    make_custom_state_graph,
    make_echo_graph,
    make_streaming_graph,
)


def _client(server: AzureAIInvokeAgentHost) -> TestClient:
    return TestClient(server.app)


def test_non_streaming_invocation_returns_response_text() -> None:
    server = AzureAIInvokeAgentHost(make_echo_graph())
    with _client(server) as client:
        resp = client.post("/invocations", json={"message": "hi"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"response": "Echo: hi"}


def test_streaming_invocation_emits_sse_tokens_and_done() -> None:
    server = AzureAIInvokeAgentHost(make_streaming_graph())
    with _client(server) as client:
        resp = client.post(
            "/invocations", json={"message": "ignored", "stream": True}
        )
    assert resp.status_code == 200, resp.text
    body = resp.text
    tokens: list[str] = []
    saw_done = False
    for line in body.splitlines():
        if line.startswith("data:"):
            data = line.split(":", 1)[1].strip()
            if not data or data == "{}":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "token" in payload:
                tokens.append(payload["token"])
        elif line.startswith("event:") and line.split(":", 1)[1].strip() == "done":
            saw_done = True
    assert "".join(tokens) == "Hello, world!"
    assert saw_done


def test_session_id_is_propagated_to_response_headers() -> None:
    server = AzureAIInvokeAgentHost(make_echo_graph())
    with _client(server) as client:
        resp = client.post("/invocations", json={"message": "hi"})
    assert resp.status_code == 200
    assert resp.headers.get("x-agent-session-id")
    assert resp.headers.get("x-agent-invocation-id")


def test_missing_message_returns_400() -> None:
    server = AzureAIInvokeAgentHost(make_echo_graph())
    with _client(server) as client:
        resp = client.post("/invocations", json={})
    assert resp.status_code == 400
    assert "message" in resp.json()["error"].lower()


def test_constructor_rejects_non_messages_state_schema() -> None:
    with pytest.raises(ValueError, match="messages"):
        AzureAIInvokeAgentHost(make_custom_state_graph())
