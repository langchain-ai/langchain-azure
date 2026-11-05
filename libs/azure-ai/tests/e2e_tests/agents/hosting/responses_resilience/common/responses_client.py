# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Resilient Responses client shared by local and Foundry E2E tests."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from time import monotonic
from typing import Any, cast
from urllib.parse import parse_qsl, urlsplit, urlunsplit
from uuid import uuid4

import httpx
import httpx2
from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    AsyncStream,
    NotFoundError,
    OpenAIError,
)
from openai.types.responses import ResponseStreamEvent

TERMINAL_STATUSES = {"cancelled", "completed", "failed", "incomplete"}
RESPONSE_ID_HEADER = "x-agent-response-id"
UNCREATED_RESPONSE_TIMEOUT_SECONDS = 5.0
TRANSPORT_ERRORS = (
    httpx.TransportError,
    httpx.TimeoutException,
    httpx2.TransportError,
    httpx2.TimeoutException,
)


@dataclass(frozen=True)
class TurnResult:
    """Terminal result of one resilient Responses turn."""

    requested_response_id: str
    response_id: str
    status: str
    output_text: str
    pre_recovery_output_text: str
    pre_reset_output_text: str
    recovery_started_seconds: float | None


@dataclass(frozen=True)
class StoredResponseResult:
    """Terminal status and output read from the persisted response snapshot."""

    status: str
    output_text: str


def create_response_id(partition_hint: str | None = None) -> str:
    """Create a canonical Agent Server response ID."""

    partition_key: str | None = None
    if partition_hint:
        _, separator, body = partition_hint.partition("_")
        if separator and len(body) == 50:
            partition_key = body[:18]
        elif separator and len(body) == 48:
            partition_key = f"{body[-16:]}00"
    if partition_key is None:
        partition_key = f"{uuid4().hex[:16]}00"
    return f"caresp_{partition_key}{uuid4().hex}"


def create_openai_client(endpoint: str, api_key: str | Any) -> AsyncOpenAI:
    """Create an OpenAI client for a local or hosted Responses endpoint."""

    parsed = urlsplit(endpoint)
    path = parsed.path.rstrip("/").removesuffix("/responses")
    base_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("api-version", "v1")
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        default_query=query,
        max_retries=0,
    )


def _is_retryable(error: BaseException) -> bool:
    if isinstance(error, APIStatusError):
        if error.status_code == 424:
            return error.code == "session_not_ready"
        return 500 <= error.status_code < 600
    return isinstance(error, (APIConnectionError, *TRANSPORT_ERRORS))


def _snapshot_output_text(response: dict[str, Any]) -> str:
    """Assemble message output text from a Responses reset snapshot."""

    texts: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if (
                isinstance(content, dict)
                and content.get("type") == "output_text"
                and isinstance(content.get("text"), str)
            ):
                texts.append(content["text"])
    return "".join(texts)


def final_result(output_text: str) -> dict[str, Any]:
    """Decode the final JSON line from deterministic workflow output."""

    return json.loads(output_text.rstrip().splitlines()[-1])


async def retrieve_stored_response(
    client: AsyncOpenAI,
    response_id: str,
    *,
    timeout: float = 60.0,
    retry_delay: float = 0.5,
) -> StoredResponseResult:
    """Retrieve a terminal response snapshot across a host replacement."""

    deadline = monotonic() + timeout
    while monotonic() < deadline:
        try:
            response = await client.responses.retrieve(response_id)
            payload = response.model_dump(mode="json")
            status = payload.get("status")
            if isinstance(status, str) and status in TERMINAL_STATUSES:
                return StoredResponseResult(
                    status=status,
                    output_text=_snapshot_output_text(payload),
                )
        except (OpenAIError, *TRANSPORT_ERRORS) as exc:
            if not isinstance(exc, NotFoundError) and not _is_retryable(exc):
                raise
        await asyncio.sleep(retry_delay)
    raise TimeoutError(
        f"Stored response {response_id} was not terminal within {timeout:.1f}s"
    )


async def run_resilient_turn(
    client: AsyncOpenAI,
    input_text: str,
    *,
    conversation_id: str | None,
    metadata: dict[str, str] | None = None,
    reconnect_timeout: float = 60.0,
    reconnect_delay: float = 0.5,
) -> TurnResult:
    """Create one turn and retrieve it by cursor until terminal."""

    requested_response_id = create_response_id(conversation_id)
    response_id: str | None = None
    cursor: int | None = None
    output_chunks: list[str] = []
    status = "queued"
    recovering = False
    recovering_at: float | None = None
    pre_recovery_output_text = ""
    pre_reset_output_text = ""
    recovery_started_seconds: float | None = None
    deadline = monotonic() + reconnect_timeout
    last_error = "no response event"

    async def consume(events: AsyncStream[ResponseStreamEvent]) -> bool:
        nonlocal cursor, output_chunks, pre_reset_output_text
        nonlocal recovery_started_seconds, response_id, status
        async for event in events:
            payload = event.model_dump(mode="json")
            sequence_number = payload.get("sequence_number")
            if isinstance(sequence_number, int) and sequence_number >= 0:
                cursor = sequence_number

            response = payload.get("response")
            if isinstance(response, dict):
                candidate = response.get("id") or response.get("response_id")
                if isinstance(candidate, str) and candidate:
                    response_id = candidate
                candidate_status = response.get("status")
                if isinstance(candidate_status, str) and candidate_status:
                    status = candidate_status
                if event.type == "response.in_progress":
                    # A recovered attempt emits an authoritative reset snapshot.
                    # Save replayed pre-crash deltas, then replace them before
                    # appending output from rerun work.
                    if recovering_at is not None:
                        pre_reset_output_text = "".join(output_chunks)
                    output_chunks = [_snapshot_output_text(response)]

            delta = payload.get("delta")
            if event.type == "response.output_text.delta" and isinstance(delta, str):
                output_chunks.append(delta)
            if (
                recovering_at is not None
                and recovery_started_seconds is None
                and event.type == "response.in_progress"
            ):
                recovery_started_seconds = monotonic() - recovering_at
            if event.type.startswith("response."):
                event_status = event.type.removeprefix("response.")
                if event_status in TERMINAL_STATUSES:
                    status = event_status
                    return True
        return False

    while monotonic() < deadline:
        try:
            if recovering:
                target_id = response_id or requested_response_id
                try:
                    if response_id is None:
                        await client.responses.retrieve(target_id)
                    if cursor is None:
                        stream = await client.responses.retrieve(target_id, stream=True)
                    else:
                        stream = await client.responses.retrieve(
                            target_id,
                            starting_after=cursor,
                            stream=True,
                        )
                except NotFoundError:
                    if response_id is None:
                        requested_response_id = create_response_id(conversation_id)
                        recovering = False
                    await asyncio.sleep(reconnect_delay)
                    continue
            else:
                request: dict[str, Any] = {
                    "input": input_text,
                    "background": True,
                    "stream": True,
                    "store": True,
                    "extra_headers": {RESPONSE_ID_HEADER: requested_response_id},
                }
                if conversation_id is not None:
                    request["conversation"] = conversation_id
                if metadata is not None:
                    request["metadata"] = metadata
                stream = cast(
                    AsyncStream[ResponseStreamEvent],
                    await client.responses.create(**request),
                )

            if await consume(cast(AsyncStream[ResponseStreamEvent], stream)):
                if status != "completed":
                    raise AssertionError(f"Response ended with status {status}")
                return TurnResult(
                    requested_response_id=requested_response_id,
                    response_id=response_id or requested_response_id,
                    status=status,
                    output_text="".join(output_chunks),
                    pre_recovery_output_text=pre_recovery_output_text,
                    pre_reset_output_text=pre_reset_output_text,
                    recovery_started_seconds=recovery_started_seconds,
                )

            if not recovering:
                pre_recovery_output_text = "".join(output_chunks)
                recovering = True
                recovering_at = monotonic()
            await asyncio.sleep(reconnect_delay)
        except (OpenAIError, *TRANSPORT_ERRORS) as exc:
            last_error = str(exc)
            if not _is_retryable(exc):
                raise
            if not recovering:
                pre_recovery_output_text = "".join(output_chunks)
                recovering = True
                recovering_at = monotonic()
            await asyncio.sleep(reconnect_delay)

    raise TimeoutError(
        f"Response {response_id or requested_response_id} did not complete within "
        f"{reconnect_timeout:.1f}s; last={last_error}"
    )
