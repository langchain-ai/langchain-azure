# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Real end-to-end tests for every sample under ``samples/hosting/``.

Each test launches the sample exactly the way its module docstring
advertises (``python sample_XX_*.py``), then drives the running process
through HTTP the same way the curl snippets do. The full stack is
exercised: ``DefaultAzureCredential`` → Azure AI Foundry → real Azure
OpenAI deployment → LangGraph → host → HTTP.

Run them locally with:

.. code-block:: pwsh

    cd samples/hosting
    az login
    cp .env.example .env  # then fill in AZURE_AI_PROJECT_ENDPOINT
    python -m pip install -r requirements-dev.txt
    python -m pytest tests

Tests skip themselves automatically when the required environment
variables are not set, so this suite is safe to run in CI on PRs that
don't have Azure credentials provisioned.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx
import pytest

from .conftest import (
    SampleServer,
    requires_foundry_endpoint,
    requires_foundry_toolbox,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse a Server-Sent-Events body into ``(event_type, data)`` pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
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


def _response_text(payload: dict[str, Any]) -> str:
    """Concatenate the assistant text across all ``message`` items."""
    return "".join(
        part.get("text", "")
        for item in payload.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
    )


def _output_types(payload: dict[str, Any]) -> list[str]:
    return [item.get("type") for item in payload.get("output", [])]


def _post(
    server: SampleServer,
    path: str,
    *,
    json_body: dict[str, Any],
    stream: bool = False,
    timeout: float = 120.0,
) -> httpx.Response:
    """POST to the running sample and return the response."""
    url = f"{server.url}{path}"
    if stream:
        # Real SSE — read the full body.
        with httpx.stream(
            "POST", url, json=json_body, timeout=timeout
        ) as resp:
            body = "".join(resp.iter_text())
        # Build a synthetic Response with the collected text for assertions.
        resp_obj = httpx.Response(
            status_code=resp.status_code,
            headers=resp.headers,
            content=body.encode("utf-8"),
            request=resp.request,
        )
        return resp_obj
    return httpx.post(url, json=json_body, timeout=timeout)


# ---------------------------------------------------------------------------
# sample_01_responses_basic.py
# ---------------------------------------------------------------------------


class TestSample01ResponsesBasic:
    """Smallest case: Responses API + no-tool ``create_agent``."""

    SCRIPT = "sample_01_responses_basic.py"

    def test_non_streaming_returns_assistant_message(self, start_sample):
        requires_foundry_endpoint()
        server = start_sample(self.SCRIPT)
        resp = _post(
            server,
            "/responses",
            json_body={"input": "Say hello in one short sentence.", "model": "gpt-4o"},
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] == "completed"
        assert "message" in _output_types(payload)
        assert _response_text(payload).strip()  # non-empty model reply

    def test_streaming_emits_completed_lifecycle(self, start_sample):
        requires_foundry_endpoint()
        server = start_sample(self.SCRIPT)
        resp = _post(
            server,
            "/responses",
            json_body={
                "input": "Say hi.",
                "model": "gpt-4o",
                "stream": True,
            },
            stream=True,
        )
        assert resp.status_code == 200, resp.text
        types = [t for t, _ in _parse_sse(resp.text)]
        assert "response.created" in types
        assert "response.completed" in types


# ---------------------------------------------------------------------------
# sample_02_responses_tools.py
# ---------------------------------------------------------------------------


class TestSample02ResponsesTools:
    """Responses API surfaces function_call / function_call_output items."""

    SCRIPT = "sample_02_responses_tools.py"

    def test_non_streaming_surfaces_full_tool_round_trip(self, start_sample):
        requires_foundry_endpoint()
        server = start_sample(self.SCRIPT)
        resp = _post(
            server,
            "/responses",
            json_body={
                "input": "What is the weather in Seattle?",
                "model": "gpt-4o",
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        types = _output_types(payload)
        assert "function_call" in types, payload
        assert "function_call_output" in types, payload
        assert "message" in types, payload

    def test_streaming_emits_function_call_items(self, start_sample):
        requires_foundry_endpoint()
        server = start_sample(self.SCRIPT)
        resp = _post(
            server,
            "/responses",
            json_body={
                "input": "What is the weather in Tokyo?",
                "model": "gpt-4o",
                "stream": True,
            },
            stream=True,
        )
        assert resp.status_code == 200, resp.text
        events = _parse_sse(resp.text)
        types = [t for t, _ in events]
        assert "response.created" in types
        assert "response.completed" in types
        # At least one function_call appears in the streamed output.
        function_call_items = [
            payload
            for event_type, payload in events
            if event_type == "response.output_item.done"
            and payload.get("item", {}).get("type") == "function_call"
        ]
        assert function_call_items, events


# ---------------------------------------------------------------------------
# sample_03_invocations_basic.py
# ---------------------------------------------------------------------------


class TestSample03InvocationsBasic:
    """Invocations API + MemorySaver checkpointer for multi-turn continuity."""

    SCRIPT = "sample_03_invocations_basic.py"

    def test_multi_turn_session_remembers_name(self, start_sample):
        requires_foundry_endpoint()
        server = start_sample(self.SCRIPT)

        r1 = _post(
            server,
            "/invocations",
            json_body={"message": "My name is Alice. Just say OK."},
        )
        assert r1.status_code == 200, r1.text
        session_id = r1.headers.get("x-agent-session-id")
        assert session_id

        r2 = _post(
            server,
            f"/invocations?agent_session_id={session_id}",
            json_body={"message": "What is my name?"},
        )
        assert r2.status_code == 200, r2.text
        body = r2.json()
        assert "response" in body
        assert "alice" in body["response"].lower()

    def test_streaming_emits_done_event(self, start_sample):
        requires_foundry_endpoint()
        server = start_sample(self.SCRIPT)
        resp = _post(
            server,
            "/invocations",
            json_body={"message": "Count from 1 to 3.", "stream": True},
            stream=True,
        )
        assert resp.status_code == 200, resp.text
        body = resp.text
        # Some token frames should have arrived.
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
            elif line.startswith("event:") and (
                line.split(":", 1)[1].strip() == "done"
            ):
                saw_done = True
        assert tokens, body
        assert saw_done, body


# ---------------------------------------------------------------------------
# sample_04_invocations_tools.py
# ---------------------------------------------------------------------------


class TestSample04InvocationsTools:
    """Invocations API: tool runs server-side, response is final text only."""

    SCRIPT = "sample_04_invocations_tools.py"

    def test_returns_final_assistant_text(self, start_sample):
        requires_foundry_endpoint()
        server = start_sample(self.SCRIPT)
        resp = _post(
            server,
            "/invocations",
            json_body={"message": "What is the weather in Seattle?"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert set(body.keys()) == {"response"}, body
        assert "seattle" in body["response"].lower()


# ---------------------------------------------------------------------------
# sample_05_workflow_all_in_one.py
# ---------------------------------------------------------------------------


class TestSample05WorkflowAllInOne:
    """One process, two protocols on the same port."""

    SCRIPT = "sample_05_workflow_all_in_one.py"

    def test_responses_endpoint_includes_tool_round_trip(self, start_sample):
        requires_foundry_endpoint()
        server = start_sample(self.SCRIPT)
        resp = _post(
            server,
            "/responses",
            json_body={
                "input": "What is the weather in Seattle?",
                "model": "gpt-4o",
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        types = _output_types(payload)
        assert "function_call" in types, payload
        assert "function_call_output" in types, payload
        assert "message" in types, payload

    def test_invocations_endpoint_returns_final_text(self, start_sample):
        requires_foundry_endpoint()
        server = start_sample(self.SCRIPT)
        resp = _post(
            server,
            "/invocations",
            json_body={"message": "What is 17 plus 25?"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "response" in body
        # The workflow's `add` tool returns 42; the synthesizer must
        # surface the answer in its final paragraph.
        assert "42" in body["response"]


# ---------------------------------------------------------------------------
# sample_06_responses_hitl.py
# ---------------------------------------------------------------------------


class TestSample06ResponsesHitl:
    """Human-in-the-loop pause + resume over the Responses API."""

    SCRIPT = "sample_06_responses_hitl.py"

    def test_pause_then_resume(self, start_sample):
        requires_foundry_endpoint()
        server = start_sample(self.SCRIPT)
        conversation_id = f"e2e-hitl-{uuid.uuid4().hex[:8]}"

        # Turn 1: model is told to ask the user where they are. The
        # response must contain a `function_call` whose name is the
        # interrupt sentinel that callers resume against.
        first = _post(
            server,
            "/responses",
            json_body={
                "input": (
                    "Ask me where I am, then look up the weather there."
                ),
                "conversation": {"id": conversation_id},
            },
            timeout=180.0,
        )
        assert first.status_code == 200, first.text
        first_payload = first.json()
        assert first_payload["status"] == "completed"
        interrupts = [
            item
            for item in first_payload["output"]
            if item.get("type") == "function_call"
            and item.get("name") == "__hosted_agent_adapter_interrupt__"
        ]
        assert interrupts, first_payload
        call_id = interrupts[0]["call_id"]
        assert call_id

        # Turn 2: resume with a function_call_output keyed by the
        # interrupt's call_id.
        second = _post(
            server,
            "/responses",
            json_body={
                "conversation": {"id": conversation_id},
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps({"resume": "Seattle"}),
                    }
                ],
            },
            timeout=300.0,
        )
        assert second.status_code == 200, second.text
        second_payload = second.json()
        assert second_payload["status"] == "completed"
        # No new interrupt this time, and the final assistant text
        # references Seattle.
        assert not [
            it
            for it in second_payload["output"]
            if it.get("type") == "function_call"
            and it.get("name") == "__hosted_agent_adapter_interrupt__"
        ]
        assert "seattle" in _response_text(second_payload).lower()


# ---------------------------------------------------------------------------
# sample_07_responses_toolbox.py
# ---------------------------------------------------------------------------


class TestSample07ResponsesToolbox:
    """Responses API + Azure AI Foundry Toolbox tools."""

    SCRIPT = "sample_07_responses_toolbox.py"

    def test_serves_responses_after_loading_toolbox(self, start_sample):
        requires_foundry_toolbox()
        # Toolbox bootstrap can take a moment; give the process room.
        server = start_sample(self.SCRIPT, health_timeout=180.0)
        resp = _post(
            server,
            "/responses",
            json_body={
                "input": "What tools do you have available?",
                "model": "gpt-4o",
            },
            timeout=180.0,
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] == "completed"
        assert _response_text(payload).strip()


# ---------------------------------------------------------------------------
# Common smoke check: every sample serves /healthy.
# ---------------------------------------------------------------------------


SAMPLES_HEALTH = [
    "sample_01_responses_basic.py",
    "sample_02_responses_tools.py",
    "sample_03_invocations_basic.py",
    "sample_04_invocations_tools.py",
    "sample_05_workflow_all_in_one.py",
    "sample_06_responses_hitl.py",
]


@pytest.mark.parametrize("script", SAMPLES_HEALTH)
def test_sample_health_endpoint(start_sample, script: str) -> None:
    """Every documented sample boots and serves ``/readiness``."""
    requires_foundry_endpoint()
    server = start_sample(script)
    # ``start_sample`` already waited for /readiness — re-hit it to confirm
    # idempotency and a fast steady-state response.
    start = time.monotonic()
    resp = httpx.get(f"{server.url}/readiness", timeout=5.0)
    elapsed = time.monotonic() - start
    assert resp.status_code == 200, resp.text
    assert elapsed < 5.0
