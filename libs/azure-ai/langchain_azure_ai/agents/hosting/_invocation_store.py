# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Persistence for the latest Invocations API status envelope."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from azure.ai.agentserver.core import get_request_context, resolve_state_subdir
from azure.ai.agentserver.core.storage import FoundryStateStore

INVOCATION_STATE_RETENTION_SECONDS = 30 * 24 * 60 * 60
"""Post-completion invocation retrieval window (30 days)."""

_INVOCATION_STORE_NAME = "langchain_azure_ai.agents.hosting/invocations"
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


class InvocationStateStore(Protocol):
    """Store the latest public status envelope for an invocation."""

    async def get(self, invocation_id: str) -> dict[str, Any] | None: ...

    async def set(self, envelope: dict[str, Any]) -> None: ...


def _record(envelope: dict[str, Any]) -> dict[str, Any]:
    expires_at = (
        time.time() + INVOCATION_STATE_RETENTION_SECONDS
        if envelope.get("status") in _TERMINAL_STATUSES
        else None
    )
    return {
        "envelope": deepcopy(envelope),
        "expires_at": expires_at,
    }


def _read_record(value: Any) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(value, dict):
        return None, False
    envelope = value.get("envelope")
    if not isinstance(envelope, dict):
        return None, False
    expires_at = value.get("expires_at")
    if isinstance(expires_at, (int, float)) and time.time() >= expires_at:
        return None, True
    return deepcopy(envelope), False


def _sequence_number(envelope: dict[str, Any]) -> int:
    value = envelope.get("sequence_number")
    return value if isinstance(value, int) else -1


def _should_replace(
    current: dict[str, Any] | None,
    incoming: dict[str, Any],
) -> bool:
    if current is None:
        return True
    if current.get("status") in _TERMINAL_STATUSES:
        return False
    return _sequence_number(incoming) > _sequence_number(current)


class FileInvocationStateStore:
    """Local file-backed invocation state with atomic envelope replacement."""

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self._root = Path(storage_dir or resolve_state_subdir("invocations"))
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _path(self, invocation_id: str) -> Path:
        digest = hashlib.sha256(invocation_id.encode("utf-8")).hexdigest()
        return self._root / f"{digest}.json"

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if not isinstance(value, dict):
            raise RuntimeError(f"Invalid invocation state record: {path}")
        return value

    @staticmethod
    def _write(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, separators=(",", ":"), ensure_ascii=False)
            os.replace(temporary_path, path)
        except BaseException:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    async def get(self, invocation_id: str) -> dict[str, Any] | None:
        path = self._path(invocation_id)
        async with self._lock:
            value = await asyncio.to_thread(self._read, path)
            envelope, expired = _read_record(value)
            if expired:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            return envelope

    async def set(self, envelope: dict[str, Any]) -> None:
        invocation_id = envelope.get("id")
        if not isinstance(invocation_id, str) or not invocation_id:
            raise ValueError("Invocation state envelope requires a non-empty 'id'.")
        path = self._path(invocation_id)
        async with self._lock:
            current = await asyncio.to_thread(self._read, path)
            current_envelope, _ = _read_record(current)
            if not _should_replace(current_envelope, envelope):
                return
            await asyncio.to_thread(self._write, path, _record(envelope))


class FoundryInvocationStateStore:
    """Hosted invocation state backed by a Foundry state-store item."""

    async def _store(self) -> FoundryStateStore:
        user_id = get_request_context().user_id
        return await FoundryStateStore.get_or_create(
            _INVOCATION_STORE_NAME,
            user_isolation=True,
            item_ttl_seconds=-1,
            description="Latest state for LangChain hosted invocations",
            user_id=user_id,
        )

    async def get(self, invocation_id: str) -> dict[str, Any] | None:
        store = await self._store()
        async with store:
            item = await store.get_item(invocation_id)
            envelope, expired = _read_record(item.value if item is not None else None)
            if expired:
                await store.delete_item(invocation_id)
            return envelope

    async def set(self, envelope: dict[str, Any]) -> None:
        invocation_id = envelope.get("id")
        if not isinstance(invocation_id, str) or not invocation_id:
            raise ValueError("Invocation state envelope requires a non-empty 'id'.")
        store = await self._store()
        async with store:
            item = await store.get_item(invocation_id)
            current, _ = _read_record(item.value if item is not None else None)
            if not _should_replace(current, envelope):
                return
            await store.set_item(invocation_id, _record(envelope))


def create_invocation_state_store(*, hosted: bool) -> InvocationStateStore:
    """Select the same hosted/local durability split used by Responses."""
    if hosted:
        return FoundryInvocationStateStore()
    return FileInvocationStateStore()
