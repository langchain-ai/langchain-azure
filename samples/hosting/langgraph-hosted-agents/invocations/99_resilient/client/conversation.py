"""Linear Invocations conversation with durable same-session recovery."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Literal
from uuid import uuid4

import httpx


def _is_retryable_error(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return 500 <= error.response.status_code < 600
    return isinstance(
        error,
        (httpx.TransportError, httpx.TimeoutException),
    )


class ConversationError(RuntimeError):
    """Raised when an action is invalid for the current conversation state."""


@dataclass(frozen=True)
class ApprovalRequest:
    """A sensitive tool call waiting for a human decision."""

    id: str
    interrupt_id: str
    action: str
    arguments: dict[str, Any]
    prompt: str


@dataclass(frozen=True)
class TurnSnapshot:
    """Immutable view of one invocation turn."""

    id: str
    user_text: str
    status: str
    connection: Literal["sending", "streaming", "recovering", "terminal"]
    output_text: str
    approval: ApprovalRequest | None
    error: str | None


@dataclass(frozen=True)
class ConversationEvent:
    """A turn state change published to the UI."""

    turn: TurnSnapshot
    protocol_event: dict[str, Any] | None = None


@dataclass
class _Turn:
    id: str
    user_text: str
    request_message: str | list[dict[str, Any]]
    previous_invocation_id: str | None
    status: str = "queued"
    connection: Literal["sending", "streaming", "recovering", "terminal"] = "sending"
    output_chunks: list[str] = field(default_factory=list)
    approval: ApprovalRequest | None = None
    error: str | None = None

    def snapshot(self) -> TurnSnapshot:
        return TurnSnapshot(
            id=self.id,
            user_text=self.user_text,
            status=self.status,
            connection=self.connection,
            output_text="".join(self.output_chunks),
            approval=self.approval,
            error=self.error,
        )


class Conversation:
    """Run one Invocations session and retry interrupted turns in place."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        invocations_url: str,
        *,
        session_id: str,
        reconnect_delay: float = 0.5,
        reconnect_timeout: float = 120.0,
    ) -> None:
        self._client = client
        self._invocations_url = httpx.URL(invocations_url)
        self._session_id = session_id
        self._reconnect_delay = reconnect_delay
        self._reconnect_timeout = reconnect_timeout
        self._turns: list[_Turn] = []
        self._current_turn: _Turn | None = None
        self._last_invocation_id: str | None = None
        self._events: asyncio.Queue[ConversationEvent] = asyncio.Queue()
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @property
    def session_id(self) -> str:
        """Return the stable Agent Server session ID."""
        return self._session_id

    @property
    def turns(self) -> tuple[TurnSnapshot, ...]:
        """Return immutable snapshots of all turns."""
        return tuple(turn.snapshot() for turn in self._turns)

    @property
    def current_turn(self) -> TurnSnapshot | None:
        """Return the turn currently controlled by the composer."""
        return self._current_turn.snapshot() if self._current_turn else None

    async def next_event(self) -> ConversationEvent:
        """Wait for the next state change."""
        return await self._events.get()

    def send(self, text: str) -> TurnSnapshot:
        """Start a user turn when no invocation is active."""
        normalized = text.strip()
        if not normalized:
            raise ConversationError("Message cannot be empty")
        if (
            self._current_turn is not None
            and self._current_turn.connection != "terminal"
        ):
            raise ConversationError("Wait for the active invocation to finish")
        return self._start_turn(user_text=normalized, request_message=normalized)

    def approve_current(self) -> TurnSnapshot:
        """Approve the sensitive tool call on the current turn."""
        return self._decide_current(approve=True)

    def deny_current(self) -> TurnSnapshot:
        """Reject the sensitive tool call on the current turn."""
        return self._decide_current(approve=False)

    def _decide_current(self, *, approve: bool) -> TurnSnapshot:
        turn = self._current_turn
        if turn is None or turn.connection != "terminal" or turn.approval is None:
            raise ConversationError("There is no tool call waiting for approval")
        if approve:
            decision = {
                "type": "mcp_approval_response",
                "approval_request_id": turn.approval.id,
                "approve": True,
            }
        else:
            decision = {
                "type": "function_call_output",
                "call_id": turn.approval.interrupt_id,
                "output": json.dumps({"resume": False}),
            }
        label = f"{'Approve' if approve else 'Deny'} {turn.approval.action}"
        return self._start_turn(user_text=label, request_message=[decision])

    async def cancel_current(self) -> None:
        """Cancel the local HTTP request for the active invocation."""
        turn = self._current_turn
        if turn is None or turn.connection == "terminal":
            raise ConversationError("There is no active invocation to cancel")
        task = self._tasks.get(turn.id)
        if task is None:
            raise ConversationError("The active invocation has no local task")
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def close(self) -> None:
        """Cancel and await all local turn tasks."""
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _start_turn(
        self,
        *,
        user_text: str,
        request_message: str | list[dict[str, Any]],
    ) -> TurnSnapshot:
        turn = _Turn(
            id=f"turn-{uuid4().hex}",
            user_text=user_text,
            request_message=request_message,
            previous_invocation_id=self._last_invocation_id,
        )
        self._turns.append(turn)
        self._current_turn = turn
        self._publish(turn)
        task = asyncio.create_task(self._run_turn(turn))
        self._tasks[turn.id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(turn.id, None))
        return turn.snapshot()

    async def _run_turn(self, turn: _Turn) -> None:
        deadline = monotonic() + self._reconnect_timeout
        recovering = False
        try:
            while True:
                turn.connection = "recovering" if recovering else "sending"
                turn.status = "recovering" if recovering else "in_progress"
                self._publish(turn)
                try:
                    if recovering:
                        recovery_status = await self._recover_once(turn)
                        if recovery_status == "pending":
                            await asyncio.sleep(self._reconnect_delay)
                            continue
                        if recovery_status == "missing":
                            recovering = False
                            await asyncio.sleep(self._reconnect_delay)
                            continue
                        completed = True
                    else:
                        completed = await self._stream_once(turn)
                    if not completed:
                        raise httpx.RemoteProtocolError(
                            "Invocation stream ended before event: done"
                        )
                    turn.status = "approval_required" if turn.approval else "completed"
                    turn.connection = "terminal"
                    turn.error = None
                    self._publish(turn)
                    return
                except (
                    httpx.HTTPStatusError,
                    httpx.TransportError,
                    httpx.TimeoutException,
                ) as exc:
                    detail = str(exc)
                    if isinstance(exc, httpx.HTTPStatusError):
                        with suppress(httpx.ResponseNotRead):
                            detail = exc.response.text or detail
                    if not _is_retryable_error(exc):
                        self._fail(turn, detail)
                        return
                    if monotonic() >= deadline:
                        self._fail(turn, f"Timed out retrying invocation: {detail}")
                        return
                    recovering = True
                    turn.error = detail
                    self._publish(turn)
                    await asyncio.sleep(self._reconnect_delay)
                except (ConversationError, json.JSONDecodeError) as exc:
                    self._fail(turn, str(exc))
                    return
        except asyncio.CancelledError:
            turn.status = "cancelled"
            turn.connection = "terminal"
            turn.error = None
            self._publish(turn)
            raise

    async def _stream_once(self, turn: _Turn) -> bool:
        request_body: dict[str, Any] = {
            "message": turn.request_message,
            "stream": True,
        }
        if turn.previous_invocation_id is not None:
            request_body["previous_invocation_id"] = turn.previous_invocation_id
        async with self._client.stream(
            "POST",
            self._invocations_url.copy_set_param(
                "agent_session_id", self._session_id
            ),
            headers={"x-agent-invocation-id": turn.id},
            json=request_body,
            timeout=None,
        ) as response:
            response.raise_for_status()
            self._last_invocation_id = turn.id
            turn.connection = "streaming"
            turn.status = "in_progress"
            turn.error = None
            self._publish(turn)

            completed = False
            async for event_type, data in _iter_sse_events(response.aiter_lines()):
                protocol_event = {"event": event_type, "data": data}
                if event_type == "message":
                    token = data.get("token")
                    if isinstance(token, str):
                        turn.output_chunks.append(token)
                elif event_type == "output_item":
                    self._apply_output_item(turn, data)
                elif event_type == "error":
                    raise ConversationError(str(data.get("error") or data))
                elif event_type == "done":
                    completed = True
                self._publish(turn, protocol_event)
            return completed

    async def _recover_once(
        self,
        turn: _Turn,
    ) -> Literal["completed", "pending", "missing"]:
        invocation_url = self._invocations_url.copy_with(
            path=f"{self._invocations_url.path.rstrip('/')}/{turn.id}"
        )
        response = await self._client.get(
            invocation_url,
            timeout=None,
        )
        if response.status_code == 404:
            return "missing"
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ConversationError("Invocation recovery returned an invalid payload")

        status = payload.get("status")
        if status in {"queued", "in_progress"}:
            self._publish(turn, {"event": "recovery", "data": payload})
            return "pending"
        if status == "completed":
            response_text = payload.get("response")
            if isinstance(response_text, str):
                streamed_text = "".join(turn.output_chunks)
                if response_text.startswith(streamed_text):
                    recovered_text = response_text[len(streamed_text) :]
                elif streamed_text:
                    recovered_text = f"\n{response_text}"
                else:
                    recovered_text = response_text
                turn.output_chunks[:] = [response_text]
                if recovered_text:
                    self._publish(
                        turn,
                        {"event": "message", "data": {"token": recovered_text}},
                    )
            turn.approval = None
            for item in payload.get("output") or []:
                if isinstance(item, dict):
                    self._apply_output_item(turn, item)
                    self._publish(turn, {"event": "output_item", "data": item})
            return "completed"
        if status in {"failed", "cancelled"}:
            error = payload.get("error")
            if isinstance(error, dict):
                error = error.get("message") or error.get("code")
            raise ConversationError(str(error or f"Invocation {status}"))
        raise ConversationError(f"Invocation recovery returned status {status!r}")

    @staticmethod
    def _apply_output_item(turn: _Turn, data: dict[str, Any]) -> None:
        if data.get("type") != "mcp_approval_request":
            return
        envelope = json.loads(str(data.get("arguments") or "{}"))
        interrupt_id = envelope.get("interrupt_id")
        if not isinstance(interrupt_id, str) or not interrupt_id:
            raise ConversationError(
                "Approval item is missing its LangGraph interrupt ID"
            )
        value = envelope.get("value")
        value = value if isinstance(value, dict) else {}
        arguments = value.get("arguments")
        turn.approval = ApprovalRequest(
            id=str(data.get("id") or "approval"),
            interrupt_id=interrupt_id,
            action=str(value.get("action") or "sensitive tool"),
            arguments=arguments if isinstance(arguments, dict) else {},
            prompt=str(value.get("prompt") or "Approve this tool call?"),
        )

    def _fail(self, turn: _Turn, message: str) -> None:
        turn.status = "failed"
        turn.connection = "terminal"
        turn.error = message
        self._publish(turn)

    def _publish(
        self,
        turn: _Turn,
        protocol_event: dict[str, Any] | None = None,
    ) -> None:
        self._events.put_nowait(ConversationEvent(turn.snapshot(), protocol_event))


async def _iter_sse_events(
    lines: AsyncIterator[str],
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    event_type = "message"
    data_lines: list[str] = []
    async for line in lines:
        if not line:
            if data_lines:
                raw_data = "\n".join(data_lines)
                payload = json.loads(raw_data)
                if not isinstance(payload, dict):
                    payload = {"value": payload}
                yield event_type, payload
            event_type = "message"
            data_lines = []
        elif line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip() or "message"
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())

    if data_lines:
        payload = json.loads("\n".join(data_lines))
        if not isinstance(payload, dict):
            payload = {"value": payload}
        yield event_type, payload
