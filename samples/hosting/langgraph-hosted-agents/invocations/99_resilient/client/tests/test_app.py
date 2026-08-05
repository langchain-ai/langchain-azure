from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from textual.containers import Horizontal
from textual.widgets import Button, Static

from app import InvocationsCuiApp, WrappingComposer, WrappingLog
from conversation import Conversation

_STREAM_END = object()


@dataclass(frozen=True)
class CapturedRequest:
    method: str
    url: httpx.URL
    body: dict[str, Any]


class QueueSSEStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self._chunks: asyncio.Queue[bytes | object] = asyncio.Queue()
        self.requested = asyncio.Event()

    async def __aiter__(self):
        while (chunk := await self._chunks.get()) is not _STREAM_END:
            assert isinstance(chunk, bytes)
            yield chunk

    async def aclose(self) -> None:
        return None

    async def send(self, event_type: str, data: dict[str, Any]) -> None:
        payload = json.dumps(data, separators=(",", ":"))
        await self._chunks.put(
            f"event: {event_type}\ndata: {payload}\n\n".encode()
        )

    async def finish(self) -> None:
        await self._chunks.put(_STREAM_END)


class FakeInvocationsServer:
    def __init__(self) -> None:
        self.requests: list[CapturedRequest] = []
        self.streams: list[QueueSSEStream] = []

    def add_stream(self) -> QueueSSEStream:
        stream = QueueSSEStream()
        self.streams.append(stream)
        return stream

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        stream_index = len(self.requests)
        if stream_index >= len(self.streams):
            return httpx.Response(500, json={"error": "No fake stream queued"})

        body = json.loads((await request.aread()).decode())
        assert isinstance(body, dict)
        self.requests.append(CapturedRequest(request.method, request.url, body))
        stream = self.streams[stream_index]
        stream.requested.set()
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=stream,
        )


def _build_app(
    server: FakeInvocationsServer,
) -> tuple[httpx.AsyncClient, InvocationsCuiApp]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    conversation = Conversation(
        client,
        "http://agent.test/invocations",
        session_id="trip-demo",
        reconnect_delay=0,
        reconnect_timeout=1,
    )
    return client, InvocationsCuiApp(conversation)


async def _complete(stream: QueueSSEStream, text: str = "") -> None:
    if text:
        await stream.send("message", {"token": text})
    await stream.send("done", {})
    await stream.finish()


@pytest.mark.asyncio
async def test_composer_submits_invocation_and_renders_stream() -> None:
    server = FakeInvocationsServer()
    stream = server.add_stream()
    client, app = _build_app(server)

    async with client, app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        message = "a long trip request " * 20
        composer.value = message

        assert composer.soft_wrap
        await pilot.press("enter")
        await asyncio.wait_for(stream.requested.wait(), timeout=1)
        await pilot.pause()

        request = server.requests[0]
        assert request.method == "POST"
        assert request.url.path == "/invocations"
        assert request.url.params["agent_session_id"] == "trip-demo"
        assert request.body == {"message": message.strip(), "stream": True}
        assert composer.value == ""
        assert app.query_one("#send", Button).disabled

        await _complete(stream, "Searching options...")
        await pilot.pause()

        transcript = app.query_one("#transcript", WrappingLog)
        assert (
            f"You: {message.strip()}\n\n"
            f"{InvocationsCuiApp.MESSAGE_SEPARATOR}\n\n"
            "Assistant: Searching options..."
        ) in transcript.text
        assert "[completed]" in transcript.text
        assert str(app.query_one("#status", Static).render()) == "Completed"
        assert not app.query_one("#send", Button).disabled


@pytest.mark.asyncio
async def test_active_invocation_rejects_a_second_message() -> None:
    server = FakeInvocationsServer()
    stream = server.add_stream()
    client, app = _build_app(server)

    async with client, app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "first"
        await pilot.press("enter")
        await asyncio.wait_for(stream.requested.wait(), timeout=1)

        composer.value = "replacement"
        await pilot.press("enter")
        await pilot.pause()

        assert len(server.requests) == 1
        assert composer.value == "replacement"
        assert app.query_one("#send", Button).disabled

        await _complete(stream)
        await pilot.pause()


@pytest.mark.asyncio
async def test_stream_events_do_not_steal_focus() -> None:
    server = FakeInvocationsServer()
    stream = server.add_stream()
    client, app = _build_app(server)

    async with client, app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "book a trip"
        await pilot.press("enter")
        await asyncio.wait_for(stream.requested.wait(), timeout=1)
        await pilot.press("tab")
        focused = app.focused
        assert focused is not composer

        await stream.send("message", {"token": "Searching options..."})
        await pilot.pause()

        assert app.focused is focused
        await _complete(stream)
        await pilot.pause()


@pytest.mark.asyncio
async def test_recovery_retries_same_invocation_session() -> None:
    server = FakeInvocationsServer()
    first_stream = server.add_stream()
    recovery_stream = server.add_stream()
    client, app = _build_app(server)

    async with client, app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "simulate a crash"
        await pilot.press("enter")
        await asyncio.wait_for(first_stream.requested.wait(), timeout=1)

        await first_stream.send("message", {"token": "Starting work..."})
        await first_stream.finish()
        await asyncio.wait_for(recovery_stream.requested.wait(), timeout=1)
        await pilot.pause()

        assert len(server.requests) == 2
        assert server.requests[0].body == server.requests[1].body
        assert (
            server.requests[0].url.params["agent_session_id"]
            == server.requests[1].url.params["agent_session_id"]
            == "trip-demo"
        )

        transcript = app.query_one("#transcript", WrappingLog)
        assert "Starting work..." in transcript.text
        assert "[Connection lost. Retrying invocation...]" in transcript.text
        assert str(app.query_one("#status", Static).render()) == "Receiving output..."

        await _complete(recovery_stream, "Recovered successfully.")
        await pilot.pause()

        assert "Recovered successfully." in transcript.text
        assert transcript.text.count(
            "[Connection lost. Retrying invocation...]"
        ) == 1


@pytest.mark.asyncio
async def test_approval_event_shows_controls_and_posts_decision() -> None:
    server = FakeInvocationsServer()
    request_stream = server.add_stream()
    approval_stream = server.add_stream()
    client, app = _build_app(server)

    async with client, app.run_test() as pilot:
        approval_actions = app.query_one("#approval-actions", Horizontal)
        assert not approval_actions.has_class("visible")

        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "book a trip"
        await pilot.press("enter")
        await asyncio.wait_for(request_stream.requested.wait(), timeout=1)
        await request_stream.send(
            "approval_required",
            {
                "id": "approval-1",
                "action": "book_trip",
                "arguments": {"city": "Paris"},
                "prompt": "Approve this sensitive tool call?",
            },
        )
        await _complete(request_stream)
        await pilot.pause()

        transcript = app.query_one("#transcript", WrappingLog)
        assert "[Approval required]" in transcript.text
        assert "Action: book_trip" in transcript.text
        assert 'Arguments: {"city": "Paris"}' in transcript.text
        assert approval_actions.has_class("visible")
        approve_button = app.query_one("#approve", Button)
        assert not approve_button.disabled
        assert not app.query_one("#deny", Button).disabled
        assert app.focused is approve_button
        assert str(app.query_one("#status", Static).render()) == "Approval required"

        await pilot.press("right")
        assert app.focused is app.query_one("#deny", Button)
        await pilot.press("left")
        assert app.focused is approve_button

        await pilot.click("#approve")
        await asyncio.wait_for(approval_stream.requested.wait(), timeout=1)
        await pilot.pause()

        assert not approval_actions.has_class("visible")
        assert server.requests[1].body == {"message": "approve", "stream": True}
        assert server.requests[1].url.params["agent_session_id"] == "trip-demo"

        await _complete(approval_stream, "Trip booked.")
        await pilot.pause()


@pytest.mark.asyncio
async def test_first_ctrl_c_cancels_local_invocation_and_second_exits() -> None:
    server = FakeInvocationsServer()
    stream = server.add_stream()
    client, app = _build_app(server)

    async with client, app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "first"
        await pilot.press("enter")
        await asyncio.wait_for(stream.requested.wait(), timeout=1)

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app._exit_armed
        assert app.is_running
        assert "[cancelled]" in app.query_one("#transcript", WrappingLog).text
        assert str(app.query_one("#status", Static).render()) == "Cancelled"

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert not app.is_running