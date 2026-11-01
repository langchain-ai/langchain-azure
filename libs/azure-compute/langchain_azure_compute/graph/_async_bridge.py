# Copyright (c) Microsoft. All rights reserved.

"""Persistent background event-loop bridge for async LangGraph checkpointers and tools.

Durable Task Python worker activities (`run_step`, `run_tool`) execute as synchronous
functions on a thread pool. However, LangGraph graphs frequently use async nodes,
async tools (`coroutine=...`), or async checkpointers such as `AsyncSqliteSaver`
(backed by `aiosqlite`) and `AsyncPostgresSaver`.

Calling `asyncio.run()` per activity invocation breaks stateful async checkpointers
for three reasons:
  1. `asyncio.run()` creates a new event loop and closes it as soon as the activity
     returns.
  2. `aiosqlite.Connection` and `AsyncSqliteSaver` bind their background worker
     tasks, futures, and `asyncio.Lock` instances to the event loop that was active
     when the connection was opened (or first used).
  3. On the very next superstep activity call, a fresh `asyncio.run()` loop either
     hangs waiting on a closed loop or raises `RuntimeError: Task got Future attached
     to a different loop` / `ValueError: no active connection`.

To keep async checkpointers and connections alive across every superstep activity:
  - If the checkpointer already owns a running event loop (`saver.loop`), we submit
    coroutines directly to that loop via `asyncio.run_coroutine_threadsafe()`.
  - Otherwise, we lazily start a single daemon event-loop thread
    (`langgraph-durable-async-loop`)
    and rebind `saver.loop` / `saver.lock` to it so all supersteps share one
    long-lived event loop.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine, Sequence
from typing import Any, TypeVar

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from langchain_azure_compute.graph._config import Checkpointer, CompiledGraph

_T = TypeVar("_T")

_BG_LOCK = threading.Lock()
_BG_LOOP: asyncio.AbstractEventLoop | None = None
_BG_THREAD: threading.Thread | None = None


def _get_bg_loop() -> asyncio.AbstractEventLoop:
    """Return (or lazily start) the process-wide background `asyncio` event loop."""
    global _BG_LOOP, _BG_THREAD
    with _BG_LOCK:
        if _BG_LOOP is None or not _BG_LOOP.is_running():
            ready = threading.Event()

            def _runner() -> None:
                global _BG_LOOP
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                _BG_LOOP = loop
                ready.set()
                loop.run_forever()

            _BG_THREAD = threading.Thread(
                target=_runner, name="langgraph-durable-async-loop", daemon=True
            )
            _BG_THREAD.start()
            ready.wait()
        return _BG_LOOP  # type: ignore[return-value]


DEFAULT_ASYNC_TIMEOUT: float = 300.0


def _run_async(
    graph: CompiledGraph,
    coro: Coroutine[Any, Any, _T],
    *,
    timeout: float | None = None,
) -> _T:
    """Execute `coro` on the checkpointer's running loop or the persistent background.

    loop.
    """
    import concurrent.futures

    eff_timeout = (
        float(timeout)
        if timeout is not None
        else float(getattr(graph, "_durable_async_timeout", DEFAULT_ASYNC_TIMEOUT))
    )
    saver = getattr(graph, "checkpointer", None)
    saver_loop = getattr(saver, "loop", None)
    if isinstance(saver_loop, asyncio.AbstractEventLoop) and saver_loop.is_running():
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not saver_loop:
            fut = asyncio.run_coroutine_threadsafe(coro, saver_loop)
            try:
                return fut.result(timeout=eff_timeout)
            except (
                TimeoutError,
                concurrent.futures.TimeoutError,
            ) as exc:  # pragma: no cover
                fut.cancel()  # pragma: no cover
                raise TimeoutError(  # pragma: no cover
                    f"async operation timed out after {eff_timeout}s"
                ) from exc
    target_loop = _get_bg_loop()
    if saver is not None and hasattr(saver, "loop"):
        with _BG_LOCK:
            try:
                cur_loop = getattr(saver, "loop", None)
                if not (
                    isinstance(cur_loop, asyncio.AbstractEventLoop)
                    and cur_loop.is_running()
                ):
                    saver.loop = target_loop
                    if hasattr(saver, "lock"):  # pragma: no cover
                        saver.lock = asyncio.Lock()
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
    fut = asyncio.run_coroutine_threadsafe(coro, target_loop)
    try:
        return fut.result(timeout=eff_timeout)
    except (TimeoutError, concurrent.futures.TimeoutError) as exc:
        fut.cancel()
        raise TimeoutError(f"async operation timed out after {eff_timeout}s") from exc


def _is_async_saver(saver: Checkpointer | None) -> bool:
    """Return whether the saver requires asynchronous checkpoint operations.

    Covers async SQLite, PostgreSQL, MongoDB, and savers without sync get_tuple.
    """
    if saver is None:  # pragma: no cover
        return False  # pragma: no cover
    cls = type(saver)
    if cls.__name__.startswith("Async"):
        return True
    loop = getattr(saver, "loop", None)
    if isinstance(loop, asyncio.AbstractEventLoop):  # pragma: no cover
        return True  # pragma: no cover
    # Any BaseCheckpointSaver subclass that overrides aget_tuple but leaves sync
    # get_tuple
    # as the base NotImplementedError stub is an async-only checkpointer.
    if issubclass(cls, BaseCheckpointSaver):  # pragma: no cover
        if (
            cls.get_tuple is BaseCheckpointSaver.get_tuple
            and cls.aget_tuple is not BaseCheckpointSaver.aget_tuple
        ):  # pragma: no cover
            return True  # pragma: no cover
    return False


def _ensure_async_saver(saver: Checkpointer | None) -> None:
    """Attach async checkpoint shims to a synchronous saver.

    Supports any synchronous
    `BaseCheckpointSaver` (such as `SqliteSaver`, `PostgresSaver`, `MongoDBSaver`, or
    `CosmosDBSaver`) when an async node or tool forces fallback from `graph.invoke()` to
    `graph.ainvoke()`.
    """
    if (
        saver is None
        or getattr(saver, "_durable_async_patched", False)
        or _is_async_saver(saver)
    ):
        return
    cls = type(saver)
    needs_shim = cls.__name__ == "SqliteSaver" or (
        issubclass(cls, BaseCheckpointSaver)
        and cls.aget_tuple is BaseCheckpointSaver.aget_tuple
    )
    if needs_shim:

        async def aget_tuple(config: RunnableConfig) -> CheckpointTuple | None:
            return saver.get_tuple(config)

        async def aput(
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> RunnableConfig:
            return saver.put(config, checkpoint, metadata, new_versions)

        async def aput_writes(
            config: RunnableConfig,
            writes: Sequence[tuple[str, Any]],
            task_id: str,
            task_path: str = "",
        ) -> None:
            saver.put_writes(config, writes, task_id, task_path)

        setattr(saver, "aget_tuple", aget_tuple)
        setattr(saver, "aput", aput)
        setattr(saver, "aput_writes", aput_writes)
        setattr(saver, "_durable_async_patched", True)
