"""Integration tests for SQLServerSaver."""

import os
import uuid
from typing import Generator, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from sqlalchemy import create_engine, text

from langchain_sqlserver import SQLServerSaver

_CONNECTION_STRING = str(os.environ.get("TEST_AZURESQLSERVER_TRUSTED_CONNECTION"))
_PYODBC_CONNECTION_STRING = str(os.environ.get("TEST_PYODBC_CONNECTION_STRING"))

_BASE_METADATA: CheckpointMetadata = cast(
    CheckpointMetadata, {"source": "input", "step": 1, "parents": {}}
)


def _unique_suffix() -> str:
    """Build a short table-name suffix unique per test run."""
    return uuid.uuid4().hex[:8]


@pytest.fixture
def saver() -> Generator[SQLServerSaver, None, None]:
    """Build a saver bound to per-test table names so concurrent test runs
    against the same DB do not collide. Tables are dropped on teardown."""
    suffix = _unique_suffix()
    checkpoints_table = f"lc_test_checkpoints_{suffix}"
    writes_table = f"lc_test_writes_{suffix}"
    saver = SQLServerSaver(
        connection_string=_CONNECTION_STRING,
        checkpoints_table=checkpoints_table,
        writes_table=writes_table,
    )
    yield saver

    try:
        conn = create_engine(_PYODBC_CONNECTION_STRING).connect()
        conn.execute(text(f"drop table if exists {checkpoints_table}"))
        conn.execute(text(f"drop table if exists {writes_table}"))
        conn.commit()
        conn.close()
    except Exception:
        pass


def _config(thread_id: str, checkpoint_id: str = "") -> RunnableConfig:
    configurable = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    return cast(RunnableConfig, {"configurable": configurable})


def _meta(step: int, source: str = "loop") -> CheckpointMetadata:
    return cast(
        CheckpointMetadata, {"source": source, "step": step, "parents": {}}
    )


def test_put_then_get_tuple_returns_checkpoint(saver: SQLServerSaver) -> None:
    """A checkpoint round-trips through `put` + `get_tuple` with metadata
    preserved and parent linkage absent on a root checkpoint."""
    thread_id = f"thread-{_unique_suffix()}"
    checkpoint: Checkpoint = empty_checkpoint()
    metadata: CheckpointMetadata = {"source": "input", "step": 1, "parents": {}}

    saved = saver.put(_config(thread_id), checkpoint, metadata, {})
    assert saved["configurable"]["checkpoint_id"] == checkpoint["id"]

    tup = saver.get_tuple(_config(thread_id))
    assert tup is not None
    assert tup.checkpoint["id"] == checkpoint["id"]
    assert tup.metadata["source"] == "input"
    assert tup.parent_config is None


def test_get_tuple_returns_latest_when_checkpoint_id_omitted(
    saver: SQLServerSaver,
) -> None:
    """With no `checkpoint_id` in the config, `get_tuple` returns the newest
    checkpoint for the thread (ordered by descending checkpoint_id)."""
    thread_id = f"thread-{_unique_suffix()}"
    cp1 = empty_checkpoint()
    saver.put(_config(thread_id), cp1, _BASE_METADATA, {})
    cp2 = create_checkpoint(cp1, {}, 1)
    saver.put(
        _config(thread_id, cp1["id"]),
        cp2,
        _meta(2),
        {},
    )

    latest = saver.get_tuple(_config(thread_id))
    assert latest is not None
    assert latest.checkpoint["id"] == cp2["id"]
    assert latest.parent_config is not None
    assert latest.parent_config["configurable"]["checkpoint_id"] == cp1["id"]


def test_list_orders_newest_first_and_respects_limit(
    saver: SQLServerSaver,
) -> None:
    """`list` should yield checkpoints newest-first and honor `limit`."""
    thread_id = f"thread-{_unique_suffix()}"
    cp1 = empty_checkpoint()
    cp2 = create_checkpoint(cp1, {}, 1)
    cp3 = create_checkpoint(cp2, {}, 2)
    saver.put(_config(thread_id), cp1, _BASE_METADATA, {})
    saver.put(
        _config(thread_id, cp1["id"]),
        cp2,
        _meta(2),
        {},
    )
    saver.put(
        _config(thread_id, cp2["id"]),
        cp3,
        _meta(3),
        {},
    )

    ids = [t.checkpoint["id"] for t in saver.list(_config(thread_id))]
    assert ids == [cp3["id"], cp2["id"], cp1["id"]]

    capped = list(saver.list(_config(thread_id), limit=2))
    assert [t.checkpoint["id"] for t in capped] == [cp3["id"], cp2["id"]]


def test_list_before_excludes_target_and_newer(saver: SQLServerSaver) -> None:
    """The `before` argument excludes the target checkpoint and everything
    newer; older checkpoints come back in descending order."""
    thread_id = f"thread-{_unique_suffix()}"
    cp1 = empty_checkpoint()
    cp2 = create_checkpoint(cp1, {}, 1)
    cp3 = create_checkpoint(cp2, {}, 2)
    saver.put(_config(thread_id), cp1, _BASE_METADATA, {})
    saver.put(
        _config(thread_id, cp1["id"]),
        cp2,
        _meta(2),
        {},
    )
    saver.put(
        _config(thread_id, cp2["id"]),
        cp3,
        _meta(3),
        {},
    )

    older = list(
        saver.list(_config(thread_id), before=_config(thread_id, cp3["id"]))
    )
    assert [t.checkpoint["id"] for t in older] == [cp2["id"], cp1["id"]]


def test_put_writes_are_returned_via_get_tuple(saver: SQLServerSaver) -> None:
    """Pending writes attached to a checkpoint via `put_writes` are returned
    on the corresponding `get_tuple.pending_writes`."""
    thread_id = f"thread-{_unique_suffix()}"
    cp = empty_checkpoint()
    saver.put(_config(thread_id), cp, {"source": "input", "step": 1, "parents": {}}, {})

    saver.put_writes(
        _config(thread_id, cp["id"]),
        [("my_channel", "hello"), ("my_channel", "world")],
        task_id="task-1",
    )

    tup = saver.get_tuple(_config(thread_id, cp["id"]))
    assert tup is not None
    assert tup.pending_writes is not None
    values = [v for _task, _ch, v in tup.pending_writes]
    assert values == ["hello", "world"]


def test_delete_thread_removes_checkpoints_and_writes(
    saver: SQLServerSaver,
) -> None:
    """`delete_thread` clears both tables for the given thread, leaving any
    other thread's data intact."""
    keep_thread = f"thread-{_unique_suffix()}"
    drop_thread = f"thread-{_unique_suffix()}"
    cp = empty_checkpoint()
    for tid in (keep_thread, drop_thread):
        saver.put(_config(tid), cp, _BASE_METADATA, {})
        saver.put_writes(
            _config(tid, cp["id"]),
            [("ch", "v")],
            task_id="t",
        )

    saver.delete_thread(drop_thread)

    assert saver.get_tuple(_config(drop_thread)) is None
    surviving = saver.get_tuple(_config(keep_thread))
    assert surviving is not None
    assert surviving.pending_writes is not None
    assert [v for _t, _c, v in surviving.pending_writes] == ["v"]
