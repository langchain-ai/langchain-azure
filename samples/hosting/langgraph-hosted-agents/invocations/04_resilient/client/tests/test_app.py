from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from textual.containers import Horizontal
from textual.widgets import Button, Static

import conversation as conversation_module
from app import InvocationsCuiApp, WrappingComposer, WrappingLog
from conversation import Conversation, _is_retryable_error


@dataclass(frozen=True)
class CapturedRequest:
    method: str
    url: httpx.URL
    headers: httpx.Headers
    body: dict[str, Any]


class FakeInvocationsServer:
    def __init__(self) -> None:
        self.requests: list[CapturedRequest] = []
        self.post_attempts: list[CapturedRequest] = []
        self.post_statuses: list[int] = []
        self.response_invocation_ids: list[str] = []
        self.create_events: list[asyncio.Event] = []
        self.retrievals: list[tuple[int, dict[str, Any]]] = []
        self.retrieval_urls: list[httpx.URL] = []
        self.retrieval_requested = asyncio.Event()

    def add_create(self) -> asyncio.Event:
        requested = asyncio.Event()
        self.create_events.append(requested)
        return requested

    def add_post_status(self, status_code: int) -> None:
        self.post_statuses.append(status_code)

    def add_response_invocation_id(self, invocation_id: str) -> None:
        self.response_invocation_ids.append(invocation_id)

    def add_retrieval(
        self,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
    ) -> None:
        self.retrievals.append((status_code, payload))

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.retrieval_urls.append(request.url)
            self.retrieval_requested.set()
            if not self.retrievals:
                return httpx.Response(404, json={"error": "Invocation not found"})
            status_code, payload = self.retrievals.pop(0)
            return httpx.Response(status_code, json=payload)

        body = json.loads((await request.aread()).decode())
        assert isinstance(body, dict)
        captured = CapturedRequest(request.method, request.url, request.headers, body)
        self.post_attempts.append(captured)
        if self.post_statuses:
            status_code = self.post_statuses.pop(0)
            if status_code != 202:
                payload: dict[str, Any] = {"error": "Temporary server error"}
                if status_code == 424:
                    payload = {
                        "error": {
                            "code": "session_not_ready",
                            "message": "Session is restarting",
                        }
                    }
                return httpx.Response(
                    status_code,
                    json=payload,
                )

        request_index = len(self.requests)
        self.requests.append(captured)
        if request_index < len(self.create_events):
            self.create_events[request_index].set()
        invocation_id = (
            self.response_invocation_ids.pop(0)
            if self.response_invocation_ids
            else request.headers["x-agent-invocation-id"]
        )
        return httpx.Response(
            202,
            headers={"x-agent-invocation-id": invocation_id},
            json={
                "id": invocation_id,
                "status": "queued",
                "agent_session_id": request.url.params["agent_session_id"],
            },
        )


def _build_app(
    server: FakeInvocationsServer,
    invocations_url: str = "http://agent.test/invocations",
) -> tuple[httpx.AsyncClient, InvocationsCuiApp]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    conversation = Conversation(
        client,
        invocations_url,
        session_id="trip-demo",
        reconnect_delay=0.01,
        reconnect_timeout=1,
    )
    return client, InvocationsCuiApp(conversation)


def _completed(text: str = "", **extra: Any) -> dict[str, Any]:
    return {"status": "completed", "response": text, **extra}


async def _wait_for_terminal(conversation: Conversation):
    while True:
        turn = conversation.current_turn
        if turn is not None and turn.connection == "terminal":
            return turn
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_composer_submits_background_invocation_and_renders_result() -> None:
    server = FakeInvocationsServer()
    created = server.add_create()
    server.add_retrieval(_completed("Searching options..."))
    client, app = _build_app(server)

    async with client, app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        message = "a long trip request " * 20
        composer.value = message

        assert composer.soft_wrap
        await pilot.press("enter")
        await asyncio.wait_for(created.wait(), timeout=1)
        await _wait_for_terminal(app._conversation)
        await pilot.pause()

        request = server.requests[0]
        assert request.method == "POST"
        assert request.url.path == "/invocations"
        assert request.url.params["agent_session_id"] == "trip-demo"
        assert request.body == {"message": message.strip(), "background": True}
        assert composer.value == ""

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
    created = server.add_create()
    client, app = _build_app(server)

    async with client, app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "first"
        await pilot.press("enter")
        await asyncio.wait_for(created.wait(), timeout=1)

        composer.value = "replacement"
        await pilot.press("enter")
        await pilot.pause()

        assert len(server.requests) == 1
        assert composer.value == "replacement"
        assert app.query_one("#send", Button).disabled

        server.add_retrieval(_completed())
        await _wait_for_terminal(app._conversation)
        await pilot.pause()


@pytest.mark.asyncio
async def test_next_turn_uses_canonical_response_invocation_id() -> None:
    server = FakeInvocationsServer()
    first_created = server.add_create()
    second_created = server.add_create()
    server.add_response_invocation_id("inv-canonical-first")
    server.add_response_invocation_id("inv-canonical-second")
    server.add_retrieval(_completed())
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    conversation = Conversation(
        client,
        "http://agent.test/invocations",
        session_id="trip-demo",
        reconnect_delay=0,
        reconnect_timeout=1,
    )

    async with client:
        first = conversation.send("first")
        await asyncio.wait_for(first_created.wait(), timeout=1)
        await _wait_for_terminal(conversation)

        server.add_retrieval(_completed())
        second = conversation.send("second")
        await asyncio.wait_for(second_created.wait(), timeout=1)

        assert first.id != "inv-canonical-first"
        assert second.id != "inv-canonical-second"
        assert server.requests[1].body["previous_invocation_id"] == (
            "inv-canonical-first"
        )
        await _wait_for_terminal(conversation)

    await conversation.close()


@pytest.mark.asyncio
async def test_poll_events_do_not_steal_focus() -> None:
    server = FakeInvocationsServer()
    created = server.add_create()
    client, app = _build_app(server)

    async with client, app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "book a trip"
        await pilot.press("enter")
        await asyncio.wait_for(created.wait(), timeout=1)
        await pilot.press("tab")
        focused = app.focused
        assert focused is not composer

        retrieval_count = len(server.retrieval_urls)
        server.add_retrieval({"status": "in_progress"})
        while len(server.retrieval_urls) == retrieval_count:
            await asyncio.sleep(0.01)
        await pilot.pause()

        assert app.focused is focused

        server.add_retrieval(_completed("Searching options..."))
        await _wait_for_terminal(app._conversation)
        await pilot.pause()


@pytest.mark.asyncio
async def test_recovery_polls_same_admitted_invocation() -> None:
    server = FakeInvocationsServer()
    created = server.add_create()
    server.add_response_invocation_id("inv-canonical-recovery")
    server.add_retrieval(
        {
            "error": {
                "code": "session_not_ready",
                "message": "Session is restarting",
            }
        },
        status_code=424,
    )
    server.add_retrieval(
        {
            "status": "completed",
            "response": "Starting work...Recovered successfully.",
        }
    )
    client, app = _build_app(
        server,
        "https://agent.test/endpoint/protocols/invocations?api-version=v1",
    )

    async with client, app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "simulate a crash"
        await pilot.press("enter")
        await asyncio.wait_for(created.wait(), timeout=1)
        await asyncio.wait_for(server.retrieval_requested.wait(), timeout=1)
        await _wait_for_terminal(app._conversation)
        await pilot.pause()

        assert len(server.requests) == 1
        requested_id = server.requests[0].headers["x-agent-invocation-id"]
        assert requested_id != "inv-canonical-recovery"
        assert requested_id.startswith("inv_")
        assert len(requested_id) == 54
        assert requested_id.removeprefix("inv_").isalnum()
        assert server.requests[0].url.params["agent_session_id"] == "trip-demo"
        assert server.requests[0].url.params["api-version"] == "v1"
        assert server.retrieval_urls[0].path.endswith(
            "/invocations/inv-canonical-recovery"
        )
        assert server.retrieval_urls[0].params["api-version"] == "v1"
        assert server.retrieval_urls[0].params["agent_session_id"] == "trip-demo"

        transcript = app.query_one("#transcript", WrappingLog)
        assert "Starting work..." in transcript.text
        assert "[Connection lost. Retrying invocation...]" not in transcript.text
        assert "Recovered successfully." in transcript.text
        assert transcript.text.count("Starting work...") == 1


@pytest.mark.asyncio
async def test_missing_admitted_invocation_keeps_polling_without_repost() -> None:
    server = FakeInvocationsServer()
    created = server.add_create()
    server.add_response_invocation_id("inv-canonical-recovery")
    server.add_retrieval({}, status_code=404)
    server.add_retrieval(
        {
            "status": "completed",
            "response": "Recovered successfully.",
        }
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    conversation = Conversation(
        client,
        "http://agent.test/invocations",
        session_id="trip-demo",
        reconnect_delay=0,
        reconnect_timeout=1,
    )

    async with client:
        conversation.send("simulate a crash")
        await asyncio.wait_for(created.wait(), timeout=1)
        terminal = await _wait_for_terminal(conversation)

    assert terminal.status == "completed"
    assert terminal.output_text == "Recovered successfully."
    assert len(server.post_attempts) == 1
    assert len(server.retrieval_urls) == 2
    assert all(
        url.path.endswith("/invocations/inv-canonical-recovery")
        for url in server.retrieval_urls
    )
    assert all(
        url.params["agent_session_id"] == "trip-demo"
        for url in server.retrieval_urls
    )
    await conversation.close()


@pytest.mark.asyncio
async def test_pending_recovery_times_out_after_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeInvocationsServer()
    created = server.add_create()
    server.add_response_invocation_id("inv-canonical-recovery")
    server.add_retrieval({"error": "Temporary server error"}, status_code=503)
    server.add_retrieval({}, status_code=404)
    times = iter((100.0, 100.5, 101.0))
    monkeypatch.setattr(conversation_module, "monotonic", lambda: next(times))
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    conversation = Conversation(
        client,
        "http://agent.test/invocations",
        session_id="trip-demo",
        reconnect_delay=0,
        reconnect_timeout=1,
    )

    async with client:
        conversation.send("simulate a crash")
        await asyncio.wait_for(created.wait(), timeout=1)
        terminal = await _wait_for_terminal(conversation)

    assert terminal.status == "failed"
    assert terminal.error is not None
    assert terminal.error.startswith("Timed out retrying invocation:")
    assert len(server.retrieval_urls) == 2
    await conversation.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [424, 500])
async def test_create_server_error_polls_then_retries_same_invocation(
    status_code: int,
) -> None:
    server = FakeInvocationsServer()
    server.add_post_status(status_code)
    server.add_retrieval({}, status_code=404)
    server.add_create()
    server.add_retrieval(_completed("Recovered successfully."))
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    conversation = Conversation(
        client,
        "http://agent.test/invocations",
        session_id="trip-demo",
        reconnect_delay=0,
        reconnect_timeout=1,
    )

    async with client:
        conversation.send("simulate a crash")
        terminal = await _wait_for_terminal(conversation)

    assert terminal.status == "completed"
    assert terminal.output_text == "Recovered successfully."
    assert len(server.post_attempts) == 2
    assert {
        request.headers["x-agent-invocation-id"] for request in server.post_attempts
    } == {terminal.id}
    await conversation.close()


@pytest.mark.asyncio
async def test_recovered_invocation_becomes_next_turn_parent() -> None:
    server = FakeInvocationsServer()
    server.add_post_status(424)
    server.add_retrieval(
        {
            "status": "completed",
            "response": "Recovered successfully.",
        }
    )
    next_created = server.add_create()
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    conversation = Conversation(
        client,
        "http://agent.test/invocations",
        session_id="trip-demo",
        reconnect_delay=0,
        reconnect_timeout=1,
    )

    async with client:
        recovered = conversation.send("simulate a crash")
        terminal = await _wait_for_terminal(conversation)
        assert terminal.status == "completed"

        server.add_retrieval(_completed())
        conversation.send("continue")
        await asyncio.wait_for(next_created.wait(), timeout=1)
        assert server.requests[0].body["previous_invocation_id"] == recovered.id
        await _wait_for_terminal(conversation)

    await conversation.close()


def test_unrelated_failed_dependency_is_not_retryable() -> None:
    request = httpx.Request("POST", "https://agent.test/invocations")
    response = httpx.Response(
        424,
        request=request,
        json={"error": {"code": "unrelated_dependency"}},
    )
    error = httpx.HTTPStatusError(
        "failed dependency",
        request=request,
        response=response,
    )

    assert not _is_retryable_error(error)


@pytest.mark.asyncio
async def test_retrieve_server_error_keeps_polling_same_invocation() -> None:
    server = FakeInvocationsServer()
    server.add_create()
    server.add_retrieval({"error": "Temporary server error"}, status_code=503)
    server.add_retrieval(
        {
            "status": "completed",
            "response": "Recovered successfully.",
        }
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    conversation = Conversation(
        client,
        "http://agent.test/invocations",
        session_id="trip-demo",
        reconnect_delay=0,
        reconnect_timeout=1,
    )

    async with client:
        conversation.send("simulate a crash")
        terminal = await _wait_for_terminal(conversation)

    assert terminal.status == "completed"
    assert terminal.output_text == "Recovered successfully."
    assert len(server.requests) == 1
    assert len(server.retrieval_urls) == 2
    await conversation.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("button_id", "expected_decision"),
    [
        (
            "approve",
            {
                "type": "mcp_approval_response",
                "approval_request_id": "mcpr_approval-1",
                "approve": True,
            },
        ),
        (
            "deny",
            {
                "type": "function_call_output",
                "call_id": "interrupt-1",
                "output": json.dumps({"resume": False}),
            },
        ),
    ],
)
async def test_approval_event_shows_controls_and_posts_decision(
    button_id: str, expected_decision: dict[str, Any]
) -> None:
    server = FakeInvocationsServer()
    request_created = server.add_create()
    approval_created = server.add_create()
    server.add_retrieval(
        _completed(
            output=[
                {
                    "type": "mcp_approval_request",
                    "id": "mcpr_approval-1",
                    "server_label": "langgraph",
                    "name": "__hosted_agent_adapter_interrupt__",
                    "arguments": json.dumps(
                        {
                            "interrupt_id": "interrupt-1",
                            "value": {
                                "action": "book_trip",
                                "arguments": {"city": "Paris"},
                                "prompt": "Approve this sensitive tool call?",
                            },
                        }
                    ),
                }
            ]
        )
    )
    client, app = _build_app(server)

    async with client, app.run_test() as pilot:
        approval_actions = app.query_one("#approval-actions", Horizontal)
        assert not approval_actions.has_class("visible")

        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "book a trip"
        await pilot.press("enter")
        await asyncio.wait_for(request_created.wait(), timeout=1)
        await _wait_for_terminal(app._conversation)
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

        server.add_retrieval(_completed("Trip booked."))
        await pilot.click(f"#{button_id}")
        await asyncio.wait_for(approval_created.wait(), timeout=1)
        await pilot.pause()

        assert not approval_actions.has_class("visible")
        assert server.requests[1].body == {
            "message": [expected_decision],
            "background": True,
            "previous_invocation_id": server.requests[0].headers[
                "x-agent-invocation-id"
            ],
        }
        assert server.requests[1].url.params["agent_session_id"] == "trip-demo"

        await _wait_for_terminal(app._conversation)
        await pilot.pause()


@pytest.mark.asyncio
async def test_first_ctrl_c_cancels_local_invocation_and_second_exits() -> None:
    server = FakeInvocationsServer()
    created = server.add_create()
    client, app = _build_app(server)

    async with client, app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "first"
        await pilot.press("enter")
        await asyncio.wait_for(created.wait(), timeout=1)

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app._exit_armed
        assert app.is_running
        assert "[cancelled]" in app.query_one("#transcript", WrappingLog).text
        assert str(app.query_one("#status", Static).render()) == "Cancelled"

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert not app.is_running


@pytest.mark.asyncio
async def test_turn_after_local_cancel_chains_from_admitted_invocation() -> None:
    server = FakeInvocationsServer()
    first_created = server.add_create()
    second_created = server.add_create()
    client, app = _build_app(server)

    async with client, app.run_test() as pilot:
        composer = app.query_one("#composer", WrappingComposer)
        composer.value = "first"
        await pilot.press("enter")
        await asyncio.wait_for(first_created.wait(), timeout=1)
        first_invocation_id = server.requests[0].headers["x-agent-invocation-id"]

        await app.action_cancel_invocation()
        await pilot.pause()

        composer.value = "second"
        server.add_retrieval(_completed())
        await pilot.press("enter")
        await asyncio.wait_for(second_created.wait(), timeout=1)

        assert server.requests[1].body["previous_invocation_id"] == (
            first_invocation_id
        )
        await _wait_for_terminal(app._conversation)
        await pilot.pause()
