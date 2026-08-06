from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from conversation import Conversation, ConversationError, TurnSnapshot
from openai import APIStatusError, NotFoundError

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
        self.create_streams: list[
            asyncio.Queue[FakeOpenAIEvent | BaseException | object]
        ] = []
        self.create_errors: list[BaseException] = []
        self.get_events: list[FakeOpenAIEvent] = []
        self.get_streams: list[list[FakeOpenAIEvent | BaseException]] = []
        self.get_errors: list[BaseException] = []
        self.get_calls: list[tuple[str, int | None]] = []
        self.cancelled: list[str] = []

    def add_create_stream(
        self,
    ) -> asyncio.Queue[FakeOpenAIEvent | BaseException | object]:
        stream: asyncio.Queue[FakeOpenAIEvent | BaseException | object] = (
            asyncio.Queue()
        )
        self.create_streams.append(stream)
        return stream

    async def create(self, **request: Any) -> AsyncIterator[FakeOpenAIEvent]:
        self.requests.append(request)
        if self.create_errors:
            raise self.create_errors.pop(0)
        stream = self.create_streams.pop(0)

        async def events() -> AsyncIterator[FakeOpenAIEvent]:
            while (event := await stream.get()) is not _STREAM_END:
                if isinstance(event, BaseException):
                    raise event
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
        self.get_calls.append((response_id, starting_after))
        if self.get_errors:
            raise self.get_errors.pop(0)
        stream_events = (
            self.get_streams.pop(0) if self.get_streams else self.get_events
        )

        async def events() -> AsyncIterator[FakeOpenAIEvent]:
            for event in stream_events:
                if isinstance(event, BaseException):
                    raise event
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


def _not_found() -> NotFoundError:
    request = httpx.Request("GET", "http://test/responses/missing")
    response = httpx.Response(404, request=request)
    return NotFoundError("Response not found", response=response, body=None)


def _status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("GET", "http://test/responses/test")
    response = httpx.Response(status_code, request=request)
    return APIStatusError(
        f"Server returned {status_code}",
        response=response,
        body={"error": "temporary server error"},
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
        conversation_id="trip-demo",
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

    first = conversation.send("first")
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
            "conversation": "trip-demo",
            "extra_headers": {"x-agent-response-id": first.id},
        }
    ]
    assert client.responses.get_calls == [("resp-1", 0)]
    assert terminal.output_text == "hello"
    await conversation.close()


@pytest.mark.asyncio
async def test_recovers_after_create_stream_transport_error() -> None:
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
    await stream.put(httpx.RemoteProtocolError("server disconnected"))
    terminal = await _wait_for_turn(
        conversation,
        lambda turn: turn.connection == "terminal",
    )

    assert terminal.status == "completed"
    assert terminal.output_text == "hello"
    assert client.responses.get_calls == [("resp-1", 0)]
    await conversation.close()


@pytest.mark.asyncio
async def test_retries_transport_error_during_recovery_from_latest_cursor() -> None:
    client = FakeClient()
    stream = client.responses.add_create_stream()
    conversation = Conversation(
        client,  # type: ignore[arg-type]
        reconnect_delay=0,
    )
    client.responses.get_streams = [
        [
            _event("response.output_text.delta", 1, delta="hel"),
            httpx.RemoteProtocolError("server disconnected again"),
        ],
        [
            _event("response.output_text.delta", 2, delta="lo"),
            _event(
                "response.completed",
                3,
                response={"id": "resp-1", "status": "completed"},
            ),
        ],
    ]

    conversation.send("first")
    await stream.put(_created("resp-1", steerable=True))
    await stream.put(_STREAM_END)
    terminal = await _wait_for_turn(
        conversation,
        lambda turn: turn.connection == "terminal",
    )

    assert terminal.status == "completed"
    assert terminal.output_text == "hello"
    assert client.responses.get_calls == [("resp-1", 0), ("resp-1", 1)]
    await conversation.close()


@pytest.mark.asyncio
async def test_retries_create_with_stable_id_when_initial_request_was_not_found() -> (
    None
):
    client = FakeClient()
    first_stream = client.responses.add_create_stream()
    second_stream = client.responses.add_create_stream()
    conversation = Conversation(
        client,  # type: ignore[arg-type]
        reconnect_delay=0,
    )
    client.responses.get_streams = [[_not_found()]]

    first = conversation.send("first")
    await first_stream.put(httpx.RemoteProtocolError("server disconnected"))
    await second_stream.put(_created(first.id, steerable=True))
    await second_stream.put(
        _event(
            "response.completed",
            1,
            response={"id": first.id, "status": "completed"},
        )
    )
    terminal = await _wait_for_turn(
        conversation,
        lambda turn: turn.connection == "terminal",
    )

    assert terminal.status == "completed"
    assert client.responses.get_calls == [(first.id, None)]
    assert [request["extra_headers"] for request in client.responses.requests] == [
        {"x-agent-response-id": first.id},
        {"x-agent-response-id": first.id},
    ]
    await conversation.close()


@pytest.mark.asyncio
async def test_known_response_not_found_is_retrieved_again_without_create() -> None:
    client = FakeClient()
    stream = client.responses.add_create_stream()
    conversation = Conversation(
        client,  # type: ignore[arg-type]
        reconnect_delay=0,
    )
    client.responses.get_streams = [
        [_not_found()],
        [
            _event(
                "response.completed",
                1,
                response={"id": "resp-1", "status": "completed"},
            )
        ],
    ]

    conversation.send("first")
    await stream.put(_created("resp-1", steerable=True))
    await stream.put(httpx.RemoteProtocolError("server disconnected"))
    terminal = await _wait_for_turn(
        conversation,
        lambda turn: turn.connection == "terminal",
    )

    assert terminal.status == "completed"
    assert len(client.responses.requests) == 1
    assert client.responses.get_calls == [("resp-1", 0), ("resp-1", 0)]
    await conversation.close()


@pytest.mark.asyncio
async def test_create_server_error_polls_then_retries_with_stable_id() -> None:
    client = FakeClient()
    stream = client.responses.add_create_stream()
    client.responses.create_errors = [_status_error(500)]
    client.responses.get_streams = [[_not_found()]]
    conversation = Conversation(
        client,  # type: ignore[arg-type]
        reconnect_delay=0,
    )

    first = conversation.send("first")
    await stream.put(_created(first.id, steerable=True))
    await stream.put(
        _event(
            "response.completed",
            1,
            response={"id": first.id, "status": "completed"},
        )
    )
    terminal = await _wait_for_turn(
        conversation,
        lambda turn: turn.connection == "terminal",
    )

    assert terminal.status == "completed"
    assert client.responses.get_calls == [(first.id, None)]
    assert [request["extra_headers"] for request in client.responses.requests] == [
        {"x-agent-response-id": first.id},
        {"x-agent-response-id": first.id},
    ]
    await conversation.close()


@pytest.mark.asyncio
async def test_retrieve_server_error_keeps_polling_same_response() -> None:
    client = FakeClient()
    stream = client.responses.add_create_stream()
    client.responses.get_errors = [_status_error(503)]
    client.responses.get_events = [
        _event(
            "response.completed",
            1,
            response={"id": "resp-1", "status": "completed"},
        )
    ]
    conversation = Conversation(
        client,  # type: ignore[arg-type]
        reconnect_delay=0,
    )

    conversation.send("first")
    await stream.put(_created("resp-1", steerable=True))
    await stream.put(_STREAM_END)
    terminal = await _wait_for_turn(
        conversation,
        lambda turn: turn.connection == "terminal",
    )

    assert terminal.status == "completed"
    assert len(client.responses.requests) == 1
    assert client.responses.get_calls == [("resp-1", 0), ("resp-1", 0)]
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
        conversation_id="trip-demo",
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
    assert [request["conversation"] for request in client.responses.requests] == [
        "trip-demo",
        "trip-demo",
    ]
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
