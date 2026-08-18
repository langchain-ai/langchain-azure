# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for durable Invocations latest-state storage."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from langchain_azure_ai.agents.hosting import _invocation_store
from langchain_azure_ai.agents.hosting._invocation_store import (
    INVOCATION_STATE_RETENTION_SECONDS,
    FileInvocationStateStore,
    FoundryInvocationStateStore,
)


def _envelope(*, status: str, sequence_number: int) -> dict[str, Any]:
    return {
        "id": "invocation-1",
        "status": status,
        "agent_session_id": "session-1",
        "sequence_number": sequence_number,
    }


@pytest.mark.asyncio
async def test_file_store_persists_across_instances(tmp_path: Path) -> None:
    first = FileInvocationStateStore(tmp_path)
    await first.set(_envelope(status="completed", sequence_number=1))

    second = FileInvocationStateStore(tmp_path)

    assert await second.get("invocation-1") == _envelope(
        status="completed",
        sequence_number=1,
    )


@pytest.mark.asyncio
async def test_file_store_persists_private_recovery_state(tmp_path: Path) -> None:
    first = FileInvocationStateStore(tmp_path)
    envelope = _envelope(status="in_progress", sequence_number=1)
    recovery_state = {
        "langgraph_thread_id": "thread-1",
        "langgraph_checkpoint_id": "checkpoint-1",
    }
    await first.set(envelope)
    await first.set_recovery_state("invocation-1", recovery_state)

    second = FileInvocationStateStore(tmp_path)

    assert await second.get("invocation-1") == envelope
    assert await second.get_recovery_state("invocation-1") == recovery_state


@pytest.mark.asyncio
async def test_terminal_state_clears_private_recovery_state(tmp_path: Path) -> None:
    store = FileInvocationStateStore(tmp_path)
    await store.set(_envelope(status="in_progress", sequence_number=1))
    await store.set_recovery_state(
        "invocation-1",
        {"langgraph_checkpoint_id": "checkpoint-1"},
    )

    await store.set(_envelope(status="completed", sequence_number=2))

    assert await store.get_recovery_state("invocation-1") is None


@pytest.mark.asyncio
async def test_active_state_does_not_expire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_000.0
    monkeypatch.setattr(_invocation_store.time, "time", lambda: now)
    store = FileInvocationStateStore(tmp_path)
    await store.set(_envelope(status="in_progress", sequence_number=1))

    now += INVOCATION_STATE_RETENTION_SECONDS + 1

    assert await store.get("invocation-1") == _envelope(
        status="in_progress",
        sequence_number=1,
    )


@pytest.mark.asyncio
async def test_terminal_state_expires_after_retention_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_000.0
    monkeypatch.setattr(_invocation_store.time, "time", lambda: now)
    store = FileInvocationStateStore(tmp_path)
    await store.set(_envelope(status="completed", sequence_number=1))

    now += INVOCATION_STATE_RETENTION_SECONDS + 1

    assert await store.get("invocation-1") is None


@pytest.mark.asyncio
async def test_terminal_state_cannot_regress(tmp_path: Path) -> None:
    store = FileInvocationStateStore(tmp_path)
    terminal = _envelope(status="completed", sequence_number=2)
    await store.set(terminal)

    await store.set(_envelope(status="cancelling", sequence_number=3))

    assert await store.get("invocation-1") == terminal


@pytest.mark.asyncio
async def test_foundry_store_uses_current_user_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeFoundryStateStore:
        @classmethod
        async def get_or_create(
            cls,
            name: str,
            **kwargs: Any,
        ) -> "FakeFoundryStateStore":
            captured["name"] = name
            captured.update(kwargs)
            return cls()

        async def __aenter__(self) -> "FakeFoundryStateStore":
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def get_item(self, _key: str) -> None:
            return None

        async def set_item(self, key: str, value: dict[str, Any]) -> None:
            captured["key"] = key
            captured["value"] = value

    monkeypatch.setattr(
        _invocation_store,
        "get_request_context",
        lambda: SimpleNamespace(user_id="user-a"),
    )
    monkeypatch.setattr(
        _invocation_store,
        "FoundryStateStore",
        FakeFoundryStateStore,
    )

    await FoundryInvocationStateStore().set(
        _envelope(status="completed", sequence_number=1)
    )

    assert captured["user_isolation"] is True
    assert captured["user_id"] == "user-a"
    assert captured["item_ttl_seconds"] == -1
    assert captured["key"] == "invocation-1"
