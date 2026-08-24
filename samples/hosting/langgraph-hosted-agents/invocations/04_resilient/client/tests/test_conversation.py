from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from conversation import Conversation, TurnSnapshot


@dataclass(frozen=True)
class CapturedRequest:
    headers: httpx.Headers
    body: dict[str, Any]


class FakeInvocationsServer:
    def __init__(self) -> None:
        self.requests: list[CapturedRequest] = []
        self._create_results: list[
            tuple[str, int, asyncio.Event, asyncio.Event | None]
        ] = []
        self._retrievals: dict[str, list[tuple[int, dict[str, Any]]]] = {}

    def add_create(
        self,
        invocation_id: str,
        *,
        status_code: int = 202,
        release: asyncio.Event | None = None,
    ) -> asyncio.Event:
        created = asyncio.Event()
        self._create_results.append((invocation_id, status_code, created, release))
        return created

    def add_retrieval(
        self,
        invocation_id: str,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
    ) -> None:
        self._retrievals.setdefault(invocation_id, []).append(
            (status_code, payload)
        )

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            invocation_id = request.url.path.rstrip("/").rsplit("/", 1)[-1]
            retrievals = self._retrievals.get(invocation_id, [])
            if retrievals:
                status_code, payload = retrievals.pop(0)
                return httpx.Response(status_code, json=payload)
            return httpx.Response(200, json={"status": "in_progress"})

        body = json.loads((await request.aread()).decode())
        assert isinstance(body, dict)
        self.requests.append(CapturedRequest(request.headers, body))
        invocation_id, status_code, created, release = self._create_results.pop(0)
        created.set()
        if release is not None:
            await release.wait()
        if status_code != 202:
            return httpx.Response(
                status_code,
                json={"error": "Conversation already has an active invocation."},
            )
        return httpx.Response(
            202,
            json={
                "id": invocation_id,
                "status": "queued",
                "agent_session_id": request.url.params["agent_session_id"],
            },
        )


def _conversation(
    server: FakeInvocationsServer,
) -> tuple[httpx.AsyncClient, Conversation]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    conversation = Conversation(
        client,
        "http://agent.test/invocations",
        session_id="trip-demo",
        reconnect_delay=0.001,
        reconnect_timeout=1,
    )
    return client, conversation


async def _wait_for_turn(
    conversation: Conversation,
    predicate,
) -> TurnSnapshot:
    while True:
        for turn in conversation.turns:
            if predicate(turn):
                return turn
        await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_completed_turn_uses_canonical_invocation_id() -> None:
    server = FakeInvocationsServer()
    created = server.add_create("inv-canonical")
    server.add_retrieval(
        "inv-canonical",
        {"status": "completed", "response": "hello"},
    )
    client, conversation = _conversation(server)

    async with client:
        turn = conversation.send("first")
        await asyncio.wait_for(created.wait(), timeout=1)
        terminal = await _wait_for_turn(
            conversation,
            lambda candidate: candidate.id == turn.id
            and candidate.connection == "terminal",
        )

    assert terminal.status == "completed"
    assert terminal.output_text == "hello"
    assert terminal.accepted
    await conversation.close()


@pytest.mark.asyncio
async def test_steering_uses_active_canonical_invocation_as_parent() -> None:
    server = FakeInvocationsServer()
    first_created = server.add_create("inv-first")
    second_created = server.add_create("inv-second")
    client, conversation = _conversation(server)

    async with client:
        first = conversation.send("first")
        await asyncio.wait_for(first_created.wait(), timeout=1)
        await _wait_for_turn(
            conversation,
            lambda turn: turn.id == first.id and turn.accepted,
        )

        second = conversation.send("replacement")
        await asyncio.wait_for(second_created.wait(), timeout=1)

        assert server.requests[1].body["previous_invocation_id"] == "inv-first"
        assert conversation.current_turn is not None
        assert conversation.current_turn.id == second.id

        server.add_retrieval(
            "inv-first",
            {
                "status": "cancelled",
                "error": {
                    "code": "steered",
                    "message": "Invocation was superseded by a steered turn.",
                },
            },
        )
        server.add_retrieval(
            "inv-second",
            {"status": "completed", "response": "replacement accepted"},
        )
        steered = await _wait_for_turn(
            conversation,
            lambda turn: turn.id == first.id and turn.connection == "terminal",
        )
        completed = await _wait_for_turn(
            conversation,
            lambda turn: turn.id == second.id and turn.connection == "terminal",
        )

    assert steered.status == "steering"
    assert steered.error is None
    assert completed.status == "completed"
    assert conversation.current_turn is not None
    assert conversation.current_turn.id == second.id
    await conversation.close()


@pytest.mark.asyncio
async def test_steering_queues_until_active_invocation_is_created() -> None:
    server = FakeInvocationsServer()
    release_first = asyncio.Event()
    first_created = server.add_create("inv-first", release=release_first)
    second_created = server.add_create("inv-second")
    client, conversation = _conversation(server)

    async with client:
        conversation.send("first")
        await asyncio.wait_for(first_created.wait(), timeout=1)

        second = conversation.send("immediate steering")
        await asyncio.sleep(0)

        assert len(server.requests) == 1
        assert conversation.current_turn is not None
        assert conversation.current_turn.id == second.id

        release_first.set()
        await asyncio.wait_for(second_created.wait(), timeout=1)

        assert server.requests[1].body["previous_invocation_id"] == "inv-first"

    await conversation.close()


@pytest.mark.asyncio
async def test_queued_steering_fails_when_parent_is_not_accepted() -> None:
    server = FakeInvocationsServer()
    release_first = asyncio.Event()
    first_created = server.add_create(
        "inv-first",
        status_code=409,
        release=release_first,
    )
    client, conversation = _conversation(server)

    async with client:
        first = conversation.send("first")
        await asyncio.wait_for(first_created.wait(), timeout=1)
        second = conversation.send("immediate steering")

        release_first.set()
        first_failed = await _wait_for_turn(
            conversation,
            lambda turn: turn.id == first.id and turn.connection == "terminal",
        )
        second_failed = await _wait_for_turn(
            conversation,
            lambda turn: turn.id == second.id and turn.connection == "terminal",
        )

    assert first_failed.status == "failed"
    assert second_failed.status == "failed"
    assert second_failed.error == (
        "Cannot steer because the previous invocation was not accepted"
    )
    assert len(server.requests) == 1
    await conversation.close()


@pytest.mark.asyncio
async def test_rejected_steering_restores_active_turn() -> None:
    server = FakeInvocationsServer()
    first_created = server.add_create("inv-first")
    rejected = server.add_create("inv-rejected", status_code=409)
    client, conversation = _conversation(server)

    async with client:
        first = conversation.send("first")
        await asyncio.wait_for(first_created.wait(), timeout=1)
        await _wait_for_turn(
            conversation,
            lambda turn: turn.id == first.id and turn.accepted,
        )

        second = conversation.send("replacement")
        await asyncio.wait_for(rejected.wait(), timeout=1)
        failed = await _wait_for_turn(
            conversation,
            lambda turn: turn.id == second.id and turn.connection == "terminal",
        )

        assert failed.status == "failed"
        assert conversation.current_turn is not None
        assert conversation.current_turn.id == first.id
        assert conversation.current_turn.connection != "terminal"

        server.add_retrieval(
            "inv-first",
            {"status": "completed", "response": "original completed"},
        )
        await _wait_for_turn(
            conversation,
            lambda turn: turn.id == first.id and turn.connection == "terminal",
        )

    await conversation.close()


@pytest.mark.asyncio
async def test_completed_follow_up_chains_from_latest_canonical_id() -> None:
    server = FakeInvocationsServer()
    first_created = server.add_create("inv-first")
    second_created = server.add_create("inv-second")
    server.add_retrieval("inv-first", {"status": "completed", "response": "one"})
    client, conversation = _conversation(server)

    async with client:
        first = conversation.send("first")
        await asyncio.wait_for(first_created.wait(), timeout=1)
        await _wait_for_turn(
            conversation,
            lambda turn: turn.id == first.id and turn.connection == "terminal",
        )

        conversation.send("second")
        await asyncio.wait_for(second_created.wait(), timeout=1)

    assert server.requests[1].body["previous_invocation_id"] == "inv-first"
    await conversation.close()