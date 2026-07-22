from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from textual.containers import Horizontal
from textual.widgets import Button, Static

from app import ResponsesCuiApp, WrappingComposer, WrappingLog
from conversation import Conversation

_STREAM_END = object()


@dataclass(frozen=True)
class FakeOpenAIEvent:
    type: str
    payload: dict[str, Any]

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return self.payload


class FakeResponses:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.streams: list[asyncio.Queue[FakeOpenAIEvent | object]] = []
        self.retrieve_stream: asyncio.Queue[FakeOpenAIEvent | object] = asyncio.Queue()
        self.cancelled: list[str] = []
        self.cancel_gate: asyncio.Event | None = None

    def add_stream(self) -> asyncio.Queue[FakeOpenAIEvent | object]:
        stream: asyncio.Queue[FakeOpenAIEvent | object] = asyncio.Queue()
        self.streams.append(stream)
        return stream

    async def create(self, **request: Any) -> AsyncIterator[FakeOpenAIEvent]:
        async def events() -> AsyncIterator[FakeOpenAIEvent]:
            index = len(self.requests)
            self.requests.append(request)
            while (event := await self.streams[index].get()) is not _STREAM_END:
                assert isinstance(event, FakeOpenAIEvent)
                yield event

        return events()

    async def retrieve(
        self,
        response_id: str,
        *,
        stream: bool,
        starting_after: int | None = None,
    ) -> AsyncIterator[FakeOpenAIEvent]:
        assert stream

        async def events() -> AsyncIterator[FakeOpenAIEvent]:
            while (event := await self.retrieve_stream.get()) is not _STREAM_END:
                assert isinstance(event, FakeOpenAIEvent)
                yield event

        return events()

    async def cancel(self, response_id: str) -> dict[str, Any]:
        self.cancelled.append(response_id)
        if self.cancel_gate is not None:
            await self.cancel_gate.wait()
        return {"id": response_id, "status": "cancelling"}


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def _created(response_id: str) -> FakeOpenAIEvent:
    return FakeOpenAIEvent(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": response_id,
                "status": "in_progress",
                "metadata": {"foundry.agent.steerable_conversation": "true"},
            },
        },
    )


@pytest.mark.asyncio
async def test_composer_remains_editable_while_response_is_starting() -> None:
    client = FakeClient()
    client.responses.add_stream()
    conversation = Conversation(client, reconnect_delay=0)  # type: ignore[arg-type]
    app = ResponsesCuiApp(conversation)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "first"
        await pilot.press("enter")
        await pilot.pause()

        assert not composer.disabled
        assert composer.has_focus

        await pilot.press("n", "e", "x", "t")
        assert composer.value == "next"


@pytest.mark.asyncio
async def test_enter_during_output_automatically_steers() -> None:
    client = FakeClient()
    first_stream = client.responses.add_stream()
    client.responses.add_stream()
    conversation = Conversation(
        client,  # type: ignore[arg-type]
        reconnect_delay=0,
    )
    app = ResponsesCuiApp(conversation)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "first"
        await pilot.press("enter")
        await pilot.pause()

        await first_stream.put(_created("resp-1"))
        await pilot.pause()
        assert not composer.disabled

        composer.value = "replacement"
        await pilot.press("enter")
        await pilot.pause()

        assert client.responses.requests[1]["previous_response_id"] == "resp-1"


@pytest.mark.asyncio
async def test_stream_events_do_not_steal_focus() -> None:
    client = FakeClient()
    stream = client.responses.add_stream()
    conversation = Conversation(client, reconnect_delay=0)  # type: ignore[arg-type]
    app = ResponsesCuiApp(conversation)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "first"
        await pilot.press("enter")
        await stream.put(_created("resp-1"))
        await pilot.pause()

        await pilot.press("tab")
        focused = app.focused
        assert focused is not composer

        await stream.put(
            FakeOpenAIEvent(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "delta": "hello",
                },
            )
        )
        await pilot.pause()

        assert app.focused is focused


@pytest.mark.asyncio
async def test_recovery_retries_are_visible_in_wrapped_transcript() -> None:
    client = FakeClient()
    stream = client.responses.add_stream()
    conversation = Conversation(client, reconnect_delay=0)  # type: ignore[arg-type]
    app = ResponsesCuiApp(conversation)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "book a trip"
        await pilot.press("enter")
        await stream.put(_created("resp-1"))
        await stream.put(
            FakeOpenAIEvent(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "delta": "Searching options...",
                },
            )
        )
        await stream.put(_STREAM_END)
        await pilot.pause()

        transcript = app.query_one("#transcript", WrappingLog)
        assert transcript.soft_wrap
        assert (
            "Searching options...\n[Connection lost. Retrying response...]\n"
            in transcript.text
        )
        assert str(app.query_one("#status", Static).render()) == "Receiving output..."

        await client.responses.retrieve_stream.put(
            FakeOpenAIEvent(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 2,
                    "delta": "Recovered and booking.",
                },
            )
        )
        await client.responses.retrieve_stream.put(
            FakeOpenAIEvent(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {"id": "resp-1", "status": "completed"},
                },
            )
        )
        await pilot.pause()
        assert (
            "[Connection lost. Retrying response...]\nRecovered and booking."
            in transcript.text
        )
        assert transcript.text.count("[Connection lost. Retrying response...]") == 1


@pytest.mark.asyncio
async def test_composer_soft_wraps_and_enter_submits() -> None:
    client = FakeClient()
    client.responses.add_stream()
    conversation = Conversation(client, reconnect_delay=0)  # type: ignore[arg-type]
    app = ResponsesCuiApp(conversation)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "a long trip request " * 20

        assert composer.soft_wrap
        await pilot.press("enter")
        await pilot.pause()

        assert client.responses.requests[0]["input"] == ("a long trip request " * 20).strip()
        assert composer.value == ""


@pytest.mark.asyncio
async def test_transcript_separates_user_and_assistant_messages() -> None:
    client = FakeClient()
    client.responses.add_stream()
    conversation = Conversation(client, reconnect_delay=0)  # type: ignore[arg-type]
    app = ResponsesCuiApp(conversation)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "book a trip"
        await pilot.press("enter")
        await pilot.pause()

        transcript = app.query_one("#transcript", WrappingLog)
        assert (
            "You: book a trip\n\n"
            f"{ResponsesCuiApp.MESSAGE_SEPARATOR}\n\n"
            "Assistant: "
        ) in transcript.text


@pytest.mark.asyncio
async def test_transcript_shows_each_tool_call_name_once() -> None:
    client = FakeClient()
    stream = client.responses.add_stream()
    conversation = Conversation(client, reconnect_delay=0)  # type: ignore[arg-type]
    app = ResponsesCuiApp(conversation)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "book a trip"
        await pilot.press("enter")
        await stream.put(_created("resp-1"))
        tool_event = FakeOpenAIEvent(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "item": {
                    "type": "function_call",
                    "id": "fc-1",
                    "name": "search_flights",
                },
            },
        )
        await stream.put(tool_event)
        await stream.put(tool_event)
        await pilot.pause()

        transcript = app.query_one("#transcript", WrappingLog)
        assert transcript.text.count("[Tool: search_flights]") == 1


@pytest.mark.asyncio
async def test_approval_request_shows_inline_decision_controls_on_demand() -> None:
    client = FakeClient()
    stream = client.responses.add_stream()
    client.responses.add_stream()
    conversation = Conversation(client, reconnect_delay=0)  # type: ignore[arg-type]
    app = ResponsesCuiApp(conversation)

    async with app.run_test() as pilot:
        approval_actions = app.query_one("#approval-actions", Horizontal)
        assert not approval_actions.has_class("visible")

        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "book a trip"
        await pilot.press("enter")
        await stream.put(_created("resp-1"))
        await stream.put(
            FakeOpenAIEvent(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "item": {
                        "type": "mcp_approval_request",
                        "id": "approval-1",
                        "arguments": (
                            '{"interrupt_id":"approval-1","value":'
                            '{"action":"book_trip","arguments":{"city":"Paris"},'
                            '"prompt":"Approve this sensitive tool call?"}}'
                        ),
                    },
                },
            )
        )
        await stream.put(
            FakeOpenAIEvent(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {"id": "resp-1", "status": "completed"},
                },
            )
        )
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
        await pilot.pause()

        assert not approval_actions.has_class("visible")
        assert client.responses.requests[1]["input"] == [
            {
                "type": "mcp_approval_response",
                "approval_request_id": "approval-1",
                "approve": True,
            }
        ]


@pytest.mark.asyncio
async def test_recovered_approval_is_actionable_before_terminal_event() -> None:
    client = FakeClient()
    stream = client.responses.add_stream()
    client.responses.add_stream()
    conversation = Conversation(client, reconnect_delay=0)  # type: ignore[arg-type]
    app = ResponsesCuiApp(conversation)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "book a trip after recovery"
        await pilot.press("enter")
        await stream.put(_created("resp-1"))
        await stream.put(_STREAM_END)
        await pilot.pause()

        await client.responses.retrieve_stream.put(
            FakeOpenAIEvent(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "item": {
                        "type": "mcp_approval_request",
                        "id": "approval-1",
                        "arguments": (
                            '{"interrupt_id":"approval-1","value":'
                            '{"action":"book_trip","arguments":{"city":"Tokyo"},'
                            '"prompt":"Approve this sensitive tool call?"}}'
                        ),
                    },
                },
            )
        )
        await pilot.pause()

        approval_actions = app.query_one("#approval-actions", Horizontal)
        approve_button = app.query_one("#approve", Button)
        assert approval_actions.has_class("visible")
        assert not approve_button.disabled
        assert app.focused is approve_button

        await pilot.press("enter")
        await pilot.pause()
        assert len(client.responses.requests) == 1

        await client.responses.retrieve_stream.put(
            FakeOpenAIEvent(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {"id": "resp-1", "status": "completed"},
                },
            )
        )
        await pilot.pause()

        assert client.responses.requests[1]["input"] == [
            {
                "type": "mcp_approval_response",
                "approval_request_id": "approval-1",
                "approve": True,
            }
        ]


@pytest.mark.asyncio
async def test_first_ctrl_c_cancels_and_second_exits() -> None:
    client = FakeClient()
    client.responses.cancel_gate = asyncio.Event()
    stream = client.responses.add_stream()
    conversation = Conversation(client, reconnect_delay=0)  # type: ignore[arg-type]
    app = ResponsesCuiApp(conversation)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "first"
        await pilot.press("enter")
        await stream.put(_created("resp-1"))
        await pilot.pause()

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert client.responses.cancelled == ["resp-1"]
        assert app._exit_armed

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert not app.is_running
