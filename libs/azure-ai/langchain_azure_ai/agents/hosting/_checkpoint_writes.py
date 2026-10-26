# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Preserve write indices omitted by LangGraph's public pending-write tuples."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any, NamedTuple

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver

_SQLITE_WRITES_QUERY = """
SELECT task_id, idx, channel, type, value FROM writes
WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
ORDER BY task_id, idx
"""


class _IndexedWrite(NamedTuple):
    task_id: str
    write_index: int
    channel: str
    value: Any
    task_path: str = ""


class _IndexedPendingWrites(list[tuple[str, str, Any]]):
    """Keep the public triple shape while retaining original write identities."""

    def __init__(self, writes: Iterable[_IndexedWrite]) -> None:
        self.indexed = tuple(writes)
        super().__init__(
            (write.task_id, write.channel, write.value) for write in self.indexed
        )


def _available_writes(
    source: BaseCheckpointSaver, saved: CheckpointTuple
) -> _IndexedPendingWrites | None:
    if isinstance(saved.pending_writes, _IndexedPendingWrites):
        return saved.pending_writes
    if not saved.pending_writes:
        return _IndexedPendingWrites(())
    if isinstance(source, InMemorySaver):
        configurable = saved.config["configurable"]
        records = source.writes.get(
            (
                configurable["thread_id"],
                configurable.get("checkpoint_ns", ""),
                saved.checkpoint["id"],
            ),
            {},
        )
        return _IndexedPendingWrites(
            _IndexedWrite(
                task_id, index, channel, source.serde.loads_typed(value), task_path
            )
            for (_, index), (task_id, channel, value, task_path) in records.items()
        )
    if all(channel in WRITES_IDX_MAP for _, channel, _ in saved.pending_writes):
        return _IndexedPendingWrites(
            _IndexedWrite(task_id, WRITES_IDX_MAP[channel], channel, value)
            for task_id, channel, value in saved.pending_writes
        )
    return None


def _sqlite_params(saved: CheckpointTuple) -> tuple[str, str, str]:
    configurable = saved.config["configurable"]
    return (
        str(configurable["thread_id"]),
        configurable.get("checkpoint_ns", ""),
        saved.checkpoint["id"],
    )


def _deserialize_sqlite_writes(
    source: BaseCheckpointSaver,
    rows: Iterable[Iterable[Any]],
) -> _IndexedPendingWrites:
    return _IndexedPendingWrites(
        _IndexedWrite(task_id, index, channel, source.serde.loads_typed((kind, value)))
        for task_id, index, channel, kind, value in rows
    )


def _get_indexed_writes(
    source: BaseCheckpointSaver, saved: CheckpointTuple
) -> _IndexedPendingWrites:
    available = _available_writes(source, saved)
    if available is not None:
        return available
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ModuleNotFoundError as error:
        if error.name != "langgraph.checkpoint.sqlite":
            raise
    else:
        if isinstance(source, SqliteSaver):
            with source.cursor(transaction=False) as cursor:
                cursor.execute(_SQLITE_WRITES_QUERY, _sqlite_params(saved))
                return _deserialize_sqlite_writes(source, cursor.fetchall())
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ModuleNotFoundError as error:
        if error.name != "langgraph.checkpoint.sqlite":
            raise
    else:
        if isinstance(source, AsyncSqliteSaver):
            return asyncio.run_coroutine_threadsafe(
                _aget_indexed_writes(source, saved), source.loop
            ).result()
    raise NotImplementedError(
        f"store=false cannot import ordinary pending writes from "
        f"{type(source).__qualname__} without their original write indices. "
        "Use FoundryCheckpointSaver, InMemorySaver, SqliteSaver, or AsyncSqliteSaver."
    )


async def _aget_indexed_writes(
    source: BaseCheckpointSaver, saved: CheckpointTuple
) -> _IndexedPendingWrites:
    available = _available_writes(source, saved)
    if available is not None:
        return available
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ModuleNotFoundError as error:
        if error.name != "langgraph.checkpoint.sqlite":
            raise
    else:
        if isinstance(source, AsyncSqliteSaver):
            async with (
                source.lock,
                source.conn.execute(
                    _SQLITE_WRITES_QUERY, _sqlite_params(saved)
                ) as cursor,
            ):
                return _deserialize_sqlite_writes(source, await cursor.fetchall())
    return _get_indexed_writes(source, saved)
