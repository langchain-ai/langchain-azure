# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared scaffolding for the human-in-the-loop Responses-host tests.

The mock graphs live in ``graphs.py``; this module holds the test-side
helpers: the ``script`` fixture that feeds ``ScriptedModel``, an HTTP client
factory, and the accessors used to pick HITL items out of a response payload.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("starlette")

from azure.ai.agentserver.responses import ResponseEventStream  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.types import Interrupt  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from langchain_azure_ai.agents.hosting import ResponsesHostServer  # noqa: E402
from langchain_azure_ai.agents.hosting._converters import (  # noqa: E402
    HITL_FUNCTION_NAME,
    emit_interrupts,
)

from .graphs import ScriptedModel  # noqa: E402

# LangGraph's ``interrupt()`` in a synchronous graph node reads the active
# runnable config from a context var that is not propagated across the async
# boundaries used by the host on older interpreters. ToolNode passes that
# config explicitly, so tool interrupts do not need this marker.
REAL_INTERRUPT_ASYNC_XFAIL = pytest.mark.xfail(
    sys.version_info < (3, 11),
    reason=(
        "LangGraph interrupt() loses runnable config in async graph execution "
        "on Python < 3.11."
    ),
    strict=True,
)

#: Registers a script and returns the live queue, so a test can assert on
#: how many scripted turns were left un-consumed.
ScriptRegistrar = Callable[[str, list[AIMessage]], list[AIMessage]]


@pytest.fixture
def script() -> Iterator[ScriptRegistrar]:
    """Register scripted assistant turns for ``ScriptedModel``.

    Registrations are torn down after the test so keys never leak between
    tests sharing the class-level script table.
    """
    keys: list[str] = []

    def register(key: str, messages: list[AIMessage]) -> list[AIMessage]:
        queue = list(messages)
        ScriptedModel.script[key] = queue
        keys.append(key)
        return queue

    yield register

    for key in keys:
        ScriptedModel.script.pop(key, None)
        ScriptedModel.seen.pop(key, None)


def client_for(host: ResponsesHostServer) -> TestClient:
    """Return an HTTP test client bound to a host's ASGI app."""
    return TestClient(host.app)


# ---------------------------------------------------------------------------
# Converter-level helpers
# ---------------------------------------------------------------------------


def pending_interrupt(*, id: str = "int-1", value: Any = "Q?") -> Interrupt:
    """Build a pending ``Interrupt`` the way LangGraph would report one."""
    return Interrupt(value=value, id=id)


async def emitted_items(interrupts: Any) -> list[Any]:
    """Run ``emit_interrupts`` over a stream and return the output items."""
    stream = ResponseEventStream(response_id="resp-emit", request=MagicMock())
    # The stream's state machine requires the lifecycle prologue before any
    # output item may be emitted.
    stream.emit_created()
    stream.emit_in_progress()
    events = [event async for event in emit_interrupts(interrupts, stream)]
    assert events  # emission must not be silent
    return list(stream.response.get("output") or [])


# ---------------------------------------------------------------------------
# Response payload accessors
# ---------------------------------------------------------------------------


def sentinels(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ``function_call`` HITL sentinels in a response payload."""
    return [
        item
        for item in payload["output"]
        if item.get("type") == "function_call"
        and item.get("name") == HITL_FUNCTION_NAME
    ]


def approval_requests(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ``mcp_approval_request`` HITL sentinels in a payload."""
    return [
        item
        for item in payload["output"]
        if item.get("type") == "mcp_approval_request"
        and item.get("name") == HITL_FUNCTION_NAME
    ]


def assistant_text(payload: dict[str, Any]) -> str:
    """Concatenate the text of every assistant message in a payload."""
    return "".join(
        part.get("text", "")
        for item in payload["output"]
        if item.get("type") == "message"
        for part in item.get("content", [])
    )


def interrupt_value(item: dict[str, Any]) -> Any:
    """Return the ``value`` carried by a sentinel's arguments envelope."""
    return json.loads(item["arguments"])["value"]


def resume_item(call_id: str, value: Any) -> dict[str, Any]:
    """Build the ``function_call_output`` input item that resumes a pause."""
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps({"resume": value}),
    }


def sentinel_item(call_id: str, value: Any = "Q?") -> dict[str, Any]:
    """Build the ``function_call`` sentinel as a client would echo it back.

    Stateless Responses clients resend the previous turn's output items
    alongside their new input, so the host sees its own HITL sentinel
    arriving as request input.
    """
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": HITL_FUNCTION_NAME,
        "arguments": json.dumps({"interrupt_id": call_id, "value": value}),
    }


# ---------------------------------------------------------------------------
# Streaming (SSE) helpers
# ---------------------------------------------------------------------------


def sse_payloads(body: str) -> list[Any]:
    """Decode the JSON payloads carried by an SSE response body."""
    payloads: list[Any] = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        chunk = line[len("data:") :].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            payloads.append(json.loads(chunk))
        except json.JSONDecodeError:
            continue
    return payloads


def hitl_items_in(payloads: Any) -> list[dict[str, Any]]:
    """Collect every HITL item nested anywhere in the decoded SSE events.

    Event *names* are the SDK's business and change between previews, so
    we scan structurally for the item shapes we are contractually
    required to emit instead of hard-coding event types.
    """
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("name") == HITL_FUNCTION_NAME and "type" in node:
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payloads)
    return found
