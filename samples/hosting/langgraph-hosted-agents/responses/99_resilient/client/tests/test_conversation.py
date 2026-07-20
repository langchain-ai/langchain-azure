from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from conversation import Conversation, ConversationError, TurnSnapshot

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
        self.create_streams: list[asyncio.Queue[FakeOpenAIEvent | object]] = []
        self.get_events: list[FakeOpenAIEvent] = []
        self.get_calls: list[tuple[str, int | None]] = []
        self.cancelled: list[str] = []

    def add_create_stream(self) -> asyncio.Queue[FakeOpenAIEvent | object]:
        stream: asyncio.Queue[FakeOpenAIEvent | object] = asyncio.Queue()
        self.create_streams.append(stream)
        return stream

    async def create(self, **request: Any) -> AsyncIterator[FakeOpenAIEvent]:
        async def events() -> AsyncIterator[FakeOpenAIEvent]:
            index = len(self.requests)
            self.requests.append(request)
            stream = self.create_streams[index]
            while (event := await stream.get()) is not _STREAM_END:
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
            self.get_calls.append((response_id, starting_after))
            for event in self.get_events:
                yield event

        return events()

    async def cancel(self, response_id: str) -> dict[str, Any]:
        self.cancelled.append(response_id)
        return {"id": response_id, "status": "cancelling"}


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def _event(event_type: str, sequence_number: int, **data: Any) -> FakeOpenAIEvent:
    payload = {"type": event_type, "sequence_number": sequence_number, **data}
    return FakeOpenAIEvent(event_type, payload)


def _created(response_id: str, *, steerable: bool) -> FakeOpenAIEvent:
    return _event(
        "response.created",
        0,
        response={
            "id": response_id,
            "status": "in_progress",
            "metadata": {
                "foundry.agent.steerable_conversation": str(steerable).lower()
            },
        },
    )


async def _wait_for_turn(
    conversation: Conversation,
    predicate,
) -> TurnSnapshot:
    while True:
        event = await asyncio.wait_for(conversation.next_event(), timeout=1)
        if predicate(event.turn):
            return event.turn


@pytest.mark.asyncio
async def test_runs_first_turn_and_recovers_stream_by_cursor() -> None:
    client = FakeClient()
    stream = client.responses.add_create_stream()
    conversation = Conversation(
        client,  # type: ignore[arg-type]
        reconnect_delay=0,
    )
    client.responses.get_events = [
        _event("response.output_text.delta", 1, delta="hello"),
        _event(
            "response.completed",
            2,
            response={"id": "resp-1", "status": "completed"},
        ),
    ]

    conversation.send("first")
    await stream.put(_created("resp-1", steerable=True))
    await stream.put(_STREAM_END)
    terminal = await _wait_for_turn(
        conversation,
        lambda turn: turn.status == "completed",
    )

    assert client.responses.requests == [
        {
            "input": "first",
            "background": True,
            "stream": True,
            "store": True,
        }
    ]
    assert client.responses.get_calls == [("resp-1", 0)]
    assert terminal.output_text == "hello"
    await conversation.close()


@pytest.mark.asyncio
async def test_send_during_output_automatically_uses_active_response_as_parent() -> (
    None
):
    client = FakeClient()
    first_stream = client.responses.add_create_stream()
    second_stream = client.responses.add_create_stream()
    conversation = Conversation(
        client,  # type: ignore[arg-type]
        reconnect_delay=0,
    )

    first = conversation.send("first")
    await first_stream.put(_created("resp-1", steerable=True))
    await _wait_for_turn(
        conversation,
        lambda turn: turn.id == first.id and turn.response_id == "resp-1",
    )

    second = conversation.send("replacement")
    await second_stream.put(_created("resp-2", steerable=True))
    await second_stream.put(
        _event(
            "response.completed",
            1,
            response={"id": "resp-2", "status": "completed"},
        )
    )
    await _wait_for_turn(
        conversation,
        lambda turn: turn.id == second.id and turn.status == "completed",
    )

    assert client.responses.requests[1]["previous_response_id"] == "resp-1"
    assert "conversation" not in client.responses.requests[1]
    assert conversation.current_turn is not None
    assert conversation.current_turn.id == second.id

    await first_stream.put(
        _event(
            "response.failed",
            1,
            response={"id": "resp-1", "status": "failed"},
        )
    )
    steered = await _wait_for_turn(
        conversation,
        lambda turn: turn.id == first.id and turn.connection == "terminal",
    )
    assert steered.status == "steering"
    await conversation.close()


@pytest.mark.asyncio
async def test_send_during_non_steerable_output_is_rejected() -> None:
    client = FakeClient()
    stream = client.responses.add_create_stream()
    conversation = Conversation(client, reconnect_delay=0)  # type: ignore[arg-type]

    first = conversation.send("first")
    await stream.put(_created("resp-1", steerable=False))
    await _wait_for_turn(
        conversation,
        lambda turn: turn.id == first.id and turn.response_id == "resp-1",
    )

    with pytest.raises(ConversationError, match="does not support steering"):
        conversation.send("replacement")

    await conversation.close()


@pytest.mark.asyncio
async def test_cancel_targets_current_response() -> None:
    client = FakeClient()
    stream = client.responses.add_create_stream()
    conversation = Conversation(client, reconnect_delay=0)  # type: ignore[arg-type]

    first = conversation.send("first")
    await stream.put(_created("resp-1", steerable=True))
    await _wait_for_turn(
        conversation,
        lambda turn: turn.id == first.id and turn.response_id == "resp-1",
    )

    await conversation.cancel_current()

    assert client.responses.cancelled == ["resp-1"]
    await conversation.close()


@pytest.mark.asyncio
async def test_failed_event_after_cancel_is_reported_as_cancelled() -> None:
    client = FakeClient()
    stream = client.responses.add_create_stream()
    conversation = Conversation(client, reconnect_delay=0)  # type: ignore[arg-type]

    conversation.send("first")
    await stream.put(_created("resp-1", steerable=True))
    await _wait_for_turn(
        conversation,
        lambda turn: turn.response_id == "resp-1",
    )

    await conversation.cancel_current()
    await stream.put(
        _event(
            "response.failed",
            1,
            response={"id": "resp-1", "status": "failed"},
        )
    )
    terminal = await _wait_for_turn(
        conversation,
        lambda turn: turn.connection == "terminal",
    )

    assert terminal.status == "cancelled"
    assert terminal.error is None
    await conversation.close()


@pytest.mark.asyncio
async def test_failed_turn_does_not_advance_next_turn_parent() -> None:
    client = FakeClient()
    first_stream = client.responses.add_create_stream()
    second_stream = client.responses.add_create_stream()
    conversation = Conversation(client, reconnect_delay=0)  # type: ignore[arg-type]

    conversation.send("first")
    await first_stream.put(_created("resp-1", steerable=True))
    await first_stream.put(
        _event(
            "response.failed",
            1,
            response={"id": "resp-1", "status": "failed"},
        )
    )
    await _wait_for_turn(conversation, lambda turn: turn.status == "failed")

    conversation.send("second")
    await asyncio.sleep(0)

    assert "previous_response_id" not in client.responses.requests[1]
    await second_stream.put(_STREAM_END)
    await conversation.close()
