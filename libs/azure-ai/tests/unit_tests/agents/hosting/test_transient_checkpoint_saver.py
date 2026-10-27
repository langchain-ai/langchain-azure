# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for checkpoint branches that never write to their source saver."""

from __future__ import annotations

import operator
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Annotated, Any, cast
from uuid import UUID

import pytest
from pytest_mock import MockerFixture

pytest.importorskip("azure.ai.agentserver.responses")

from azure.core.credentials_async import AsyncTokenCredential
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    CheckpointTuple,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from langchain_azure_ai.agents.hosting import FoundryCheckpointSaver
from langchain_azure_ai.agents.hosting._checkpoint_writes import (
    _aget_indexed_writes,
    _get_indexed_writes,
)
from langchain_azure_ai.agents.hosting._responses.transient_checkpoint_saver import (
    TransientCheckpointSaver,
)


class _ReplayState(TypedDict):
    count: int
    stop_at: int


class _ReplaySubgraphState(_ReplayState, total=False):
    private_value: int


class _ReplayHistoryState(_ReplayState):
    private_history: list[str]


def _append_history(current: list[str], batches: Sequence[list[str]]) -> list[str]:
    return current + [item for batch in batches for item in batch]


@pytest.fixture(params=["memory", "sqlite"])
async def source_saver(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[BaseCheckpointSaver]:
    if request.param == "memory":
        yield InMemorySaver()
    elif request.param == "foundry":
        monkeypatch.delenv("FOUNDRY_HOSTING_ENVIRONMENT", raising=False)
        monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
        async with FoundryCheckpointSaver(
            cast(AsyncTokenCredential, object())
        ) as saver:
            yield saver
    else:
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            yield saver


@pytest.fixture(params=["memory", "sqlite"])
def sync_source_saver(
    request: pytest.FixtureRequest,
) -> Iterator[BaseCheckpointSaver]:
    if request.param == "memory":
        yield InMemorySaver()
    else:
        with SqliteSaver.from_conn_string(":memory:") as saver:
            yield saver


@pytest.mark.parametrize("write_count", [1, 12])
@pytest.mark.parametrize("imported", [False, True], ids=["local", "imported"])
async def test_transient_pending_writes_do_not_reload_cached_checkpoints(
    source_saver: BaseCheckpointSaver,
    imported: bool,
    write_count: int,
    mocker: MockerFixture,
) -> None:
    branch = TransientCheckpointSaver(source_saver)
    config: RunnableConfig = {
        "configurable": {"thread_id": "saved", "checkpoint_ns": ""}
    }
    checkpoint = empty_checkpoint()
    saver = source_saver if imported else branch
    config = await saver.aput(config, checkpoint, {}, {})
    if imported:
        assert await branch.aget_tuple(config) is not None
    source_before = [saved async for saved in source_saver.alist(config)]
    decode = mocker.spy(branch.serde, "loads_typed")

    for number in range(32):
        await branch.aput_writes(
            config,
            [("history", [str(number), str(index)]) for index in range(write_count)],
            f"task-{number}",
        )

    assert decode.call_count == 0
    saved = await branch.aget_tuple(config)
    assert saved is not None
    assert saved.pending_writes == [
        (f"task-{number}", "history", [str(number), str(index)])
        for number in sorted(range(32), key=lambda number: f"task-{number}")
        for index in range(write_count)
    ]
    assert [saved async for saved in source_saver.alist(config)] == source_before


@pytest.mark.parametrize("write_count", [1, 12])
@pytest.mark.parametrize("imported", [False, True], ids=["local", "imported"])
def test_transient_sync_pending_writes_do_not_reload_cached_checkpoints(
    sync_source_saver: BaseCheckpointSaver,
    imported: bool,
    write_count: int,
    mocker: MockerFixture,
) -> None:
    branch = TransientCheckpointSaver(sync_source_saver)
    config: RunnableConfig = {
        "configurable": {"thread_id": "saved", "checkpoint_ns": ""}
    }
    checkpoint = empty_checkpoint()
    saver = sync_source_saver if imported else branch
    config = saver.put(config, checkpoint, {}, {})
    if imported:
        assert branch.get_tuple(config) is not None
    source_before = list(sync_source_saver.list(config))
    decode = mocker.spy(branch.serde, "loads_typed")

    for number in range(32):
        branch.put_writes(
            config,
            [("history", [str(number), str(index)]) for index in range(write_count)],
            f"task-{number}",
        )

    assert decode.call_count == 0
    saved = branch.get_tuple(config)
    assert saved is not None
    assert saved.pending_writes == [
        (f"task-{number}", "history", [str(number), str(index)])
        for number in sorted(range(32), key=lambda number: f"task-{number}")
        for index in range(write_count)
    ]
    assert list(sync_source_saver.list(config)) == source_before


@pytest.mark.parametrize("preload", [False, True])
@pytest.mark.parametrize(
    ("initial_batches", "retried"),
    [
        (
            [[("__resume__", ["yes"])], [("history", ["answer"])]],
            [("history", ["answer"])],
        ),
        (
            [
                [("__interrupt__", ["question"])],
                [("__resume__", ["yes"]), ("history", ["answer"])],
            ],
            [("__resume__", ["yes"]), ("history", ["answer"])],
        ),
        (
            [
                [
                    ("__resume__", ["first"]),
                    ("__resume__", ["first", "second"]),
                    ("history", ["same"]),
                    ("history", ["same"]),
                ]
            ],
            [("history", ["new-at-zero"])],
        ),
        (
            [
                [("__resume__", ["yes"]), ("history", ["one"])],
                [("history", ["zero"])],
            ],
            [("history", ["zero"])],
        ),
        (
            [[("history", ["same"]), ("history", ["same"])]],
            [("history", ["same"]), ("history", ["same"])],
        ),
        (
            [[("history", ["first"]), ("other", ["second"])]],
            [("other", ["replacement-at-zero"])],
        ),
        (
            [[(channel, ["old"]) for channel in WRITES_IDX_MAP]],
            [(channel, ["new"]) for channel in WRITES_IDX_MAP],
        ),
    ],
    ids=[
        "separate-resume",
        "interrupt-then-mixed-resume",
        "multiple-resumes-and-equal-values",
        "out-of-order-indices",
        "equal-values-at-distinct-indices",
        "first-write-wins-across-channels",
        "control-writes-overwrite",
    ],
)
async def test_transient_pending_writes_preserve_source_indices(
    source_saver: BaseCheckpointSaver,
    initial_batches: list[list[tuple[str, Any]]],
    retried: list[tuple[str, Any]],
    preload: bool,
) -> None:
    config: RunnableConfig = {
        "configurable": {"thread_id": "saved", "checkpoint_ns": ""}
    }
    checkpoint = empty_checkpoint()
    saved_config = await source_saver.aput(config, checkpoint, {}, {})
    for batch in initial_batches:
        await source_saver.aput_writes(saved_config, batch, "task")
    source_before = await source_saver.aget_tuple(saved_config)
    assert source_before is not None

    branch = TransientCheckpointSaver(source_saver)
    if preload:
        assert await branch.aget_tuple(saved_config) is not None
    await branch.aput_writes(saved_config, retried, "task")
    local = await branch.aget_tuple(saved_config)
    assert local is not None
    assert await source_saver.aget_tuple(saved_config) == source_before

    await source_saver.aput_writes(saved_config, retried, "task")
    expected = await source_saver.aget_tuple(saved_config)
    assert expected is not None
    assert local.pending_writes == expected.pending_writes
    history = [saved async for saved in branch.alist(config)]
    assert history == [local]


def test_transient_pending_writes_preserve_sync_indices(
    sync_source_saver: BaseCheckpointSaver,
) -> None:
    checkpoint = empty_checkpoint()
    config = sync_source_saver.put(
        {"configurable": {"thread_id": "saved", "checkpoint_ns": ""}},
        checkpoint,
        {},
        {},
    )
    sync_source_saver.put_writes(config, [("__interrupt__", ["question"])], "task")
    writes = [("__resume__", ["yes"]), ("history", ["answer"])]
    sync_source_saver.put_writes(config, writes, "task")
    source_before = sync_source_saver.get_tuple(config)
    assert source_before is not None

    branch = TransientCheckpointSaver(sync_source_saver)
    branch.put_writes(config, writes, "task")
    local = branch.get_tuple(config)
    assert local is not None
    assert local.pending_writes == source_before.pending_writes
    assert list(branch.list(config)) == [local]
    assert sync_source_saver.get_tuple(config) == source_before


@pytest.mark.parametrize("source_saver", ["sqlite"], indirect=True)
async def test_transient_imports_legacy_async_sqlite_writes(
    source_saver: BaseCheckpointSaver,
) -> None:
    assert isinstance(source_saver, AsyncSqliteSaver)
    config = await source_saver.aput(
        {"configurable": {"thread_id": "saved", "checkpoint_ns": ""}},
        empty_checkpoint(),
        {},
        {},
    )
    await source_saver.aput_writes(config, [("history", ["answer"])], "task", "node-a")
    saved = await source_saver.aget_tuple(config)
    assert saved is not None
    async with source_saver.lock, source_saver.conn.cursor() as cursor:
        await cursor.execute("PRAGMA table_info(writes)")
        if any(column[1] == "task_path" for column in await cursor.fetchall()):
            await cursor.execute("ALTER TABLE writes DROP COLUMN task_path")

    imported = await _aget_indexed_writes(source_saver, saved)

    assert imported == saved.pending_writes
    assert imported.indexed[0].write_index == 0
    assert imported.indexed[0].task_path == ""


@pytest.mark.parametrize("sync_source_saver", ["sqlite"], indirect=True)
def test_transient_imports_legacy_sync_sqlite_writes(
    sync_source_saver: BaseCheckpointSaver,
) -> None:
    assert isinstance(sync_source_saver, SqliteSaver)
    config = sync_source_saver.put(
        {"configurable": {"thread_id": "saved", "checkpoint_ns": ""}},
        empty_checkpoint(),
        {},
        {},
    )
    sync_source_saver.put_writes(config, [("history", ["answer"])], "task", "node-a")
    saved = sync_source_saver.get_tuple(config)
    assert saved is not None
    with sync_source_saver.cursor() as cursor:
        cursor.execute("PRAGMA table_info(writes)")
        if any(column[1] == "task_path" for column in cursor.fetchall()):
            cursor.execute("ALTER TABLE writes DROP COLUMN task_path")

    imported = _get_indexed_writes(sync_source_saver, saved)

    assert imported == saved.pending_writes
    assert imported.indexed[0].write_index == 0
    assert imported.indexed[0].task_path == ""


async def test_transient_mixed_resume_keeps_memory_overwrite_semantics(
    source_saver: BaseCheckpointSaver,
) -> None:
    config = await source_saver.aput(
        {"configurable": {"thread_id": "saved", "checkpoint_ns": ""}},
        empty_checkpoint(),
        {},
        {},
    )
    await source_saver.aput_writes(
        config, [("__resume__", ["old"]), ("history", ["answer"])], "task"
    )
    source_before = await source_saver.aget_tuple(config)
    branch = TransientCheckpointSaver(source_saver)

    await branch.aput_writes(
        config, [("__resume__", ["new"]), ("history", ["answer"])], "task"
    )
    saved = await branch.aget_tuple(config)

    assert saved is not None
    assert saved.pending_writes == [
        ("task", "__resume__", ["new"]),
        ("task", "history", ["answer"]),
    ]
    assert await source_saver.aget_tuple(config) == source_before


@pytest.mark.parametrize("write_count", [1, 12])
async def test_transient_pending_writes_preserve_ordinary_reducer_order(
    source_saver: BaseCheckpointSaver,
    write_count: int,
) -> None:
    class HistoryState(TypedDict):
        history: Annotated[list[str], operator.add]

    def writer(state: HistoryState) -> dict[str, list[str]]:
        return {}

    builder = StateGraph(HistoryState)
    builder.add_node("writer", writer)
    builder.add_edge(START, "writer")
    builder.add_edge("writer", END)
    graph = builder.compile(checkpointer=source_saver, interrupt_before=["writer"])
    config: RunnableConfig = {"configurable": {"thread_id": "saved"}}
    await graph.ainvoke({"history": ["seed"]}, config)
    state = await graph.aget_state(config)
    task_id = state.tasks[0].id
    await source_saver.aput_writes(
        state.config,
        [("__resume__", ["yes"])]
        + [("history", [str(number)]) for number in range(1, write_count + 1)],
        task_id,
    )
    await source_saver.aput_writes(state.config, [("history", ["zero"])], task_id)
    expected = await graph.aget_state(config)
    source_before = await source_saver.aget_tuple(state.config)

    transient = TransientCheckpointSaver(source_saver)
    branch = graph.copy(update={"checkpointer": transient})
    assert (await branch.aget_state(config)).values == expected.values
    assert (await transient.aget_tuple(state.config)) == source_before


def _delta_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    delta = pytest.importorskip("langgraph.channels.delta")
    HistoryState = TypedDict(
        "HistoryState",
        {
            "history": Annotated[
                list[str], delta.DeltaChannel(_append_history, snapshot_frequency=1000)
            ]
        },
    )
    builder = StateGraph(HistoryState)
    builder.add_edge(START, END)
    return builder.compile(checkpointer=checkpointer)


@pytest.mark.parametrize(
    ("source_saver", "task_paths", "expected_order"),
    [
        ("memory", ("", ""), ["a", "z"]),
        ("sqlite", ("", ""), ["a", "z"]),
        ("foundry", ("", ""), ["a", "z"]),
        ("memory", ("node-a", "node-z"), ["z", "a"]),
        ("sqlite", ("node-a", "node-z"), ["z", "a"]),
        ("foundry", ("node-a", "node-z"), ["z", "a"]),
    ],
    indirect=["source_saver"],
)
async def test_transient_delta_pending_writes_follow_native_order(
    source_saver: BaseCheckpointSaver,
    task_paths: tuple[str, str],
    expected_order: list[str],
) -> None:
    checkpoint = empty_checkpoint()
    checkpoint["id"] = str(UUID(int=1))
    checkpoint["channel_values"] = {"history": ["seed"]}
    checkpoint["channel_versions"] = {"history": 1}
    parent = await source_saver.aput(
        {"configurable": {"thread_id": "saved", "checkpoint_ns": ""}},
        checkpoint,
        {},
        checkpoint["channel_versions"],
    )
    await source_saver.aput_writes(
        parent, [("history", ["z"])], "task-z", task_paths[0]
    )
    await source_saver.aput_writes(
        parent, [("history", ["a"])], "task-a", task_paths[1]
    )
    child = empty_checkpoint()
    child["id"] = str(UUID(int=2))
    child["channel_versions"] = {"history": 2}
    config = await source_saver.aput(parent, child, {}, child["channel_versions"])
    source_before = [saved async for saved in source_saver.alist(config)]
    graph = _delta_graph(source_saver)
    expected = await graph.aget_state(config)

    branch = graph.copy(update={"checkpointer": TransientCheckpointSaver(source_saver)})
    actual = await branch.aget_state(config)

    assert expected.values == {"history": ["seed", *expected_order]}
    assert actual.values == expected.values
    assert [saved async for saved in source_saver.alist(config)] == source_before


@pytest.mark.parametrize(
    ("sync_source_saver", "task_paths", "expected_order"),
    [
        ("memory", ("", ""), ["a", "z"]),
        ("sqlite", ("", ""), ["a", "z"]),
        ("memory", ("node-a", "node-z"), ["z", "a"]),
        ("sqlite", ("node-a", "node-z"), ["z", "a"]),
    ],
    indirect=["sync_source_saver"],
)
def test_transient_delta_pending_writes_follow_sync_native_order(
    sync_source_saver: BaseCheckpointSaver,
    task_paths: tuple[str, str],
    expected_order: list[str],
) -> None:
    checkpoint = empty_checkpoint()
    checkpoint["id"] = str(UUID(int=1))
    checkpoint["channel_values"] = {"history": ["seed"]}
    checkpoint["channel_versions"] = {"history": 1}
    parent = sync_source_saver.put(
        {"configurable": {"thread_id": "saved", "checkpoint_ns": ""}},
        checkpoint,
        {},
        checkpoint["channel_versions"],
    )
    sync_source_saver.put_writes(parent, [("history", ["z"])], "task-z", task_paths[0])
    sync_source_saver.put_writes(parent, [("history", ["a"])], "task-a", task_paths[1])
    child = empty_checkpoint()
    child["id"] = str(UUID(int=2))
    child["channel_versions"] = {"history": 2}
    config = sync_source_saver.put(parent, child, {}, child["channel_versions"])
    graph = _delta_graph(sync_source_saver)
    expected = graph.get_state(config)
    branch = graph.copy(
        update={"checkpointer": TransientCheckpointSaver(sync_source_saver)}
    )

    assert expected.values == {"history": ["seed", *expected_order]}
    assert branch.get_state(config).values == expected.values


async def test_transient_delta_retry_does_not_duplicate_history(
    source_saver: BaseCheckpointSaver,
) -> None:
    checkpoint = empty_checkpoint()
    checkpoint["id"] = str(UUID(int=1))
    checkpoint["channel_values"] = {"history": ["seed"]}
    checkpoint["channel_versions"] = {"history": 1}
    parent = await source_saver.aput(
        {"configurable": {"thread_id": "saved", "checkpoint_ns": ""}},
        checkpoint,
        {},
        checkpoint["channel_versions"],
    )
    await source_saver.aput_writes(parent, [("__resume__", ["yes"])], "task")
    await source_saver.aput_writes(parent, [("history", ["answer"])], "task")
    child = empty_checkpoint()
    child["id"] = str(UUID(int=2))
    child["channel_versions"] = {"history": 2}
    config = await source_saver.aput(parent, child, {}, child["channel_versions"])
    source_before = [saved async for saved in source_saver.alist(config)]
    graph = _delta_graph(source_saver)

    transient = TransientCheckpointSaver(source_saver)
    assert await transient.aget_tuple(parent) is not None
    await transient.aput_writes(parent, [("history", ["answer"])], "task")
    branch = graph.copy(update={"checkpointer": transient})
    assert (await graph.aget_state(config)).values == {"history": ["seed", "answer"]}
    assert (await branch.aget_state(config)).values == {"history": ["seed", "answer"]}
    assert [saved async for saved in source_saver.alist(config)] == source_before


class _UnindexedSaver(BaseCheckpointSaver):
    def __init__(self, source: BaseCheckpointSaver) -> None:
        super().__init__(serde=source.serde)
        self.source = source

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.source.get_tuple(config)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await self.source.aget_tuple(config)


@pytest.mark.parametrize("ordinary_write", [False, True])
async def test_transient_unindexed_source_never_guesses_write_indices(
    ordinary_write: bool,
) -> None:
    source = InMemorySaver()
    config = source.put(
        {"configurable": {"thread_id": "saved", "checkpoint_ns": ""}},
        empty_checkpoint(),
        {},
        {},
    )
    writes: list[tuple[str, Any]] = [("__resume__", ["yes"])]
    if ordinary_write:
        writes.append(("history", ["answer"]))
    source.put_writes(config, writes, "task")
    source_before = source.get_tuple(config)
    branch = TransientCheckpointSaver(_UnindexedSaver(source))

    if ordinary_write:
        for _ in range(2):
            with pytest.raises(NotImplementedError, match="original write indices"):
                branch.get_tuple(config)
            with pytest.raises(NotImplementedError, match="original write indices"):
                await branch.aget_tuple(config)
    else:
        assert branch.get_tuple(config) == source_before
        await branch.aput_writes(config, [("__resume__", ["local"])], "task")
        saved = await branch.aget_tuple(config)
        assert saved is not None
        assert saved.pending_writes == [("task", "__resume__", ["local"])]
    assert source.get_tuple(config) == source_before


def _graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    def echo(state: MessagesState) -> dict[str, list[AIMessage]]:
        return {"messages": [AIMessage(content="echo")]}

    builder = StateGraph(MessagesState)
    builder.add_node("echo", echo)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return builder.compile(checkpointer=checkpointer)


async def test_transient_branch_preserves_source(
    source_saver: BaseCheckpointSaver,
) -> None:
    graph = _graph(source_saver)
    config: RunnableConfig = {"configurable": {"thread_id": "saved"}}
    await graph.ainvoke({"messages": [HumanMessage(content="saved")]}, config)
    source_before = [saved async for saved in source_saver.alist(config)]

    branch = graph.copy(update={"checkpointer": TransientCheckpointSaver(source_saver)})
    result = await branch.ainvoke(
        {"messages": [HumanMessage(content="transient")]}, config
    )
    assert graph.checkpointer is source_saver
    assert [message.content for message in result["messages"]] == [
        "saved",
        "echo",
        "transient",
        "echo",
    ]
    assert [saved async for saved in source_saver.alist(config)] == source_before
    fresh_branch = graph.copy(
        update={"checkpointer": TransientCheckpointSaver(source_saver)}
    )
    following = await fresh_branch.ainvoke(
        {"messages": [HumanMessage(content="next")]}, config
    )
    assert [message.content for message in following["messages"]] == [
        "saved",
        "echo",
        "next",
        "echo",
    ]


async def test_transient_branch_respects_checkpoint_id(
    source_saver: BaseCheckpointSaver,
) -> None:
    graph = _graph(source_saver)
    config: RunnableConfig = {"configurable": {"thread_id": "saved"}}
    await graph.ainvoke({"messages": [HumanMessage(content="saved")]}, config)
    saved = await source_saver.aget_tuple(config)
    assert saved is not None
    await graph.ainvoke({"messages": [HumanMessage(content="newer")]}, config)
    source_before = [checkpoint async for checkpoint in source_saver.alist(config)]

    branch = graph.copy(update={"checkpointer": TransientCheckpointSaver(source_saver)})
    result = await branch.ainvoke(
        {"messages": [HumanMessage(content="transient")]}, saved.config
    )
    assert [message.content for message in result["messages"]] == [
        "saved",
        "echo",
        "transient",
        "echo",
    ]
    assert [
        checkpoint async for checkpoint in source_saver.alist(config)
    ] == source_before


def test_transient_branch_supports_synchronous_execution() -> None:
    source = InMemorySaver()
    graph = _graph(source)
    config: RunnableConfig = {"configurable": {"thread_id": "saved"}}
    graph.invoke({"messages": [HumanMessage(content="saved")]}, config)
    source_before = list(source.list(config))

    branch = graph.copy(update={"checkpointer": TransientCheckpointSaver(source)})
    result = branch.invoke({"messages": [HumanMessage(content="transient")]}, config)
    assert [message.content for message in result["messages"]] == [
        "saved",
        "echo",
        "transient",
        "echo",
    ]
    assert list(source.list(config)) == source_before


def test_transient_latest_read_preserves_local_pending_writes() -> None:
    source = InMemorySaver()
    config: RunnableConfig = {
        "configurable": {"thread_id": "saved", "checkpoint_ns": ""}
    }
    for checkpoint_id in ("001", "003"):
        checkpoint = empty_checkpoint()
        checkpoint["id"] = checkpoint_id
        checkpoint["channel_values"] = {
            "nullable": None,
            "enabled": False,
            "items": [],
        }
        checkpoint["channel_versions"] = {
            name: 1 for name in ("nullable", "enabled", "items", "omitted")
        }
        source.put(
            config,
            checkpoint,
            {"source": "update", "step": int(checkpoint_id)},
            checkpoint["channel_versions"],
        )
    latest_config: RunnableConfig = {
        "configurable": {**config["configurable"], "checkpoint_id": "003"}
    }
    source.put_writes(latest_config, [("__resume__", "source")], "task")
    original = source.get_tuple(latest_config)
    assert original is not None

    branch = TransientCheckpointSaver(source)
    older_config: RunnableConfig = {
        "configurable": {**config["configurable"], "checkpoint_id": "001"}
    }
    assert branch.get_tuple(older_config) is not None
    latest = branch.get_tuple(config)
    assert latest is not None
    assert latest.checkpoint == original.checkpoint
    branch.put_writes(latest.config, [("__resume__", "local")], "task")

    newer = empty_checkpoint()
    newer["id"] = "004"
    source.put(latest_config, newer, {"source": "update", "step": 4}, {})
    current = branch.get_tuple(config)
    assert current is not None
    assert current.checkpoint == original.checkpoint
    assert current.pending_writes == [("task", "__resume__", "local")]
    assert source.get_tuple(latest_config) == original
    history = list(branch.list(config))
    assert history[1] == current


@pytest.mark.parametrize("explicit_checkpoint", [False, True])
async def test_transient_branch_isolates_subgraph_checkpoints(
    source_saver: BaseCheckpointSaver,
    explicit_checkpoint: bool,
) -> None:
    child = _graph(source_saver)
    builder = StateGraph(MessagesState)
    builder.add_node("child", child)
    builder.add_edge(START, "child")
    builder.add_edge("child", END)
    graph = builder.compile(checkpointer=source_saver)
    config: RunnableConfig = {"configurable": {"thread_id": "saved"}}
    await graph.ainvoke({"messages": [HumanMessage(content="saved")]}, config)
    saved = await source_saver.aget_tuple(config)
    assert saved is not None
    await graph.ainvoke({"messages": [HumanMessage(content="newer")]}, config)
    source_before = [saved async for saved in source_saver.alist(config)]

    branch = graph.copy(update={"checkpointer": TransientCheckpointSaver(source_saver)})
    result = await branch.ainvoke(
        {"messages": [HumanMessage(content="transient")]},
        saved.config if explicit_checkpoint else config,
    )
    expected = ["saved", "echo"]
    if not explicit_checkpoint:
        expected += ["newer", "echo"]
    expected += ["transient", "echo"]
    assert [message.content for message in result["messages"]] == expected
    assert [saved async for saved in source_saver.alist(config)] == source_before


async def test_transient_branch_preserves_unchanged_subgraph_state(
    source_saver: BaseCheckpointSaver,
) -> None:
    observed: list[int | None] = []

    def child_step(state: _ReplaySubgraphState) -> dict[str, int]:
        observed.append(state.get("private_value"))
        update = {"count": state["count"] + 1}
        if "private_value" not in state:
            update["private_value"] = 42
        return update

    child_builder = StateGraph(
        _ReplaySubgraphState, input_schema=_ReplayState, output_schema=_ReplayState
    )
    child_builder.add_node("step", child_step)
    child_builder.add_edge(START, "step")
    child_builder.add_edge("step", END)
    child = child_builder.compile(checkpointer=True)

    def route(state: _ReplayState) -> str:
        return "child" if state["count"] < state["stop_at"] else END

    builder = StateGraph(_ReplayState)
    builder.add_node("child", child)
    builder.add_edge(START, "child")
    builder.add_conditional_edges("child", route)
    graph = builder.compile(checkpointer=source_saver)
    config: RunnableConfig = {"configurable": {"thread_id": "saved"}}
    await graph.ainvoke({"count": 0, "stop_at": 1}, config)
    saved = await source_saver.aget_tuple(config)
    assert saved is not None
    assert observed == [None]
    source_before = [checkpoint async for checkpoint in source_saver.alist(config)]
    observed.clear()

    branch = graph.copy(update={"checkpointer": TransientCheckpointSaver(source_saver)})
    result = await branch.ainvoke({"count": 1, "stop_at": 3}, saved.config)

    assert observed == [42, 42]
    assert result == {"count": 3, "stop_at": 3}
    assert [
        checkpoint async for checkpoint in source_saver.alist(config)
    ] == source_before


@pytest.mark.parametrize("explicit_checkpoint", [False, True])
async def test_transient_branch_preserves_omitted_delta_snapshots(
    source_saver: BaseCheckpointSaver,
    explicit_checkpoint: bool,
) -> None:
    delta = pytest.importorskip("langgraph.channels.delta")

    DeltaReplayState = TypedDict(
        "DeltaReplayState",
        {
            "count": int,
            "stop_at": int,
            "private_history": Annotated[
                list[str], delta.DeltaChannel(_append_history, snapshot_frequency=1)
            ],
        },
    )
    observed: list[list[str]] = []

    def child_step(state: _ReplayHistoryState) -> dict[str, int]:
        observed.append(list(state["private_history"]))
        return {"count": state["count"] + 1}

    child_builder = StateGraph(DeltaReplayState)
    child_builder.add_node("step", child_step)
    child_builder.add_edge(START, "step")
    child_builder.add_edge("step", END)
    child = child_builder.compile(checkpointer=True)

    def route(state: _ReplayState) -> str:
        return "child" if state["count"] < state["stop_at"] else END

    builder = StateGraph(_ReplayState)
    builder.add_node("child", child)
    builder.add_edge(START, "child")
    builder.add_conditional_edges("child", route)
    graph = builder.compile(checkpointer=source_saver)
    config: RunnableConfig = {"configurable": {"thread_id": "saved"}}
    child_config: RunnableConfig = {
        "configurable": {"thread_id": "saved", "checkpoint_ns": "child"}
    }
    await graph.aupdate_state(
        child_config,
        {"count": 0, "stop_at": 0, "private_history": ["saved"]},
    )
    assert (await graph.aget_state(child_config)).values["private_history"] == ["saved"]
    await graph.aupdate_state(config, {"count": 0, "stop_at": 0})
    saved = await source_saver.aget_tuple(config)
    assert saved is not None
    source_before = [checkpoint async for checkpoint in source_saver.alist(config)]

    branch = graph.copy(update={"checkpointer": TransientCheckpointSaver(source_saver)})
    result = await branch.ainvoke(
        {"count": 0, "stop_at": 2},
        saved.config if explicit_checkpoint else config,
    )

    assert observed == [["saved"], ["saved"]]
    assert result == {"count": 2, "stop_at": 2}
    assert (await branch.aget_state(child_config)).values["private_history"] == [
        "saved"
    ]
    assert [
        checkpoint async for checkpoint in source_saver.alist(config)
    ] == source_before


async def test_transient_delta_history_does_not_rewind_subgraph(
    source_saver: BaseCheckpointSaver,
) -> None:
    delta = pytest.importorskip("langgraph.channels.delta")
    DeltaReplayState = TypedDict(
        "DeltaReplayState",
        {
            "count": int,
            "stop_at": int,
            "private_history": Annotated[
                list[str], delta.DeltaChannel(_append_history, snapshot_frequency=1000)
            ],
        },
    )
    observed: list[list[str]] = []
    inspect_history = False
    before: RunnableConfig | None = None

    def child_step(state: _ReplayHistoryState) -> dict[str, int | list[str]]:
        observed.append(list(state["private_history"]))
        return {
            "count": state["count"] + 1,
            "private_history": [f"child{state['count']}"],
        }

    child_builder = StateGraph(DeltaReplayState)
    child_builder.add_node("step", child_step)
    child_builder.add_edge(START, "step")
    child_builder.add_edge("step", END)
    child = child_builder.compile(checkpointer=True)

    async def inspect(state: _ReplayState, config: RunnableConfig) -> dict[str, int]:
        if inspect_history:
            configurable = {**config["configurable"], "checkpoint_ns": "child"}
            configurable.pop("checkpoint_id", None)
            history = [
                saved
                async for saved in child.aget_state_history(
                    {"configurable": configurable}, before=before, limit=1
                )
            ]
            assert len(history) == 1
            assert history[0].values["private_history"] == ["saved", "child0"]
        return {}

    def route(state: _ReplayState) -> str:
        return "child" if state["count"] < state["stop_at"] else END

    builder = StateGraph(_ReplayState)
    builder.add_node("inspect", inspect)
    builder.add_node("child", child)
    builder.add_edge(START, "inspect")
    builder.add_edge("inspect", "child")
    builder.add_conditional_edges("child", route)
    graph = builder.compile(checkpointer=source_saver)
    config: RunnableConfig = {"configurable": {"thread_id": "saved"}}
    child_config: RunnableConfig = {
        "configurable": {"thread_id": "saved", "checkpoint_ns": "child"}
    }
    await graph.aupdate_state(
        child_config,
        {"count": 0, "stop_at": 0, "private_history": ["saved"]},
    )
    await graph.ainvoke({"count": 0, "stop_at": 2}, config)
    child_state = await graph.aget_state(child_config)
    assert child_state.values["private_history"] == ["saved", "child0", "child1"]
    before = child_state.config
    source_before = [saved async for saved in source_saver.alist(config)]

    branch = graph.copy(update={"checkpointer": TransientCheckpointSaver(source_saver)})
    assert (await branch.aget_state(config)).values == {"count": 2, "stop_at": 2}
    observed.clear()
    inspect_history = True
    result = await branch.ainvoke({"count": 0, "stop_at": 1}, config)

    assert observed == [["saved", "child0", "child1"]]
    assert result == {"count": 1, "stop_at": 1}
    assert (await branch.aget_state(child_config)).values["private_history"] == [
        "saved",
        "child0",
        "child1",
        "child0",
    ]
    assert [saved async for saved in source_saver.alist(config)] == source_before


async def test_transient_branch_lists_history(
    source_saver: BaseCheckpointSaver,
) -> None:
    config: RunnableConfig = {
        "configurable": {"thread_id": "saved", "checkpoint_ns": ""}
    }
    for checkpoint_id in ("001", "003"):
        checkpoint = empty_checkpoint()
        checkpoint["id"] = checkpoint_id
        await source_saver.aput(
            config, checkpoint, {"source": "update", "step": int(checkpoint_id)}, {}
        )
    source_before = [saved async for saved in source_saver.alist(config)]

    branch = TransientCheckpointSaver(source_saver)
    older_config: RunnableConfig = {
        "configurable": {**config["configurable"], "checkpoint_id": "001"}
    }
    assert await branch.aget_tuple(older_config) is not None
    await branch.aput_writes(older_config, [("note", "local write")], "local-task")
    for checkpoint_id in ("002", "004"):
        checkpoint = empty_checkpoint()
        checkpoint["id"] = checkpoint_id
        await branch.aput(
            config, checkpoint, {"source": "update", "step": int(checkpoint_id)}, {}
        )
    latest_before = await branch.aget_tuple(config)

    history = [saved async for saved in branch.alist(config)]
    assert [saved.checkpoint["id"] for saved in history] == [
        "004",
        "003",
        "002",
        "001",
    ]
    assert history[-1].pending_writes == [("local-task", "note", "local write")]
    limited = [saved async for saved in branch.alist(config, limit=2)]
    assert [saved.checkpoint["id"] for saved in limited] == ["004", "003"]
    before: RunnableConfig = {"configurable": {"checkpoint_id": "003"}}
    bounded = [saved async for saved in branch.alist(config, before=before, limit=1)]
    assert [saved.checkpoint["id"] for saved in bounded] == ["002"]
    for step in (2, 3):
        filtered = [
            saved
            async for saved in branch.alist(config, filter={"step": step}, limit=1)
        ]
        assert [saved.checkpoint["id"] for saved in filtered] == [f"{step:03}"]
    assert [saved async for saved in branch.alist(older_config)] == [history[-1]]
    assert await branch.aget_tuple(config) == latest_before
    assert [saved async for saved in source_saver.alist(config)] == source_before


def test_transient_branch_lists_history_synchronously() -> None:
    source = InMemorySaver()
    branch = TransientCheckpointSaver(source)
    config: RunnableConfig = {
        "configurable": {"thread_id": "saved", "checkpoint_ns": ""}
    }
    for saver, checkpoint_ids in ((source, ("001", "003")), (branch, ("002", "004"))):
        for checkpoint_id in checkpoint_ids:
            checkpoint = empty_checkpoint()
            checkpoint["id"] = checkpoint_id
            saver.put(
                config, checkpoint, {"source": "update", "step": int(checkpoint_id)}, {}
            )

    history = list(branch.list(config, limit=3))
    assert [saved.checkpoint["id"] for saved in history] == ["004", "003", "002"]
    assert [saved.checkpoint["id"] for saved in source.list(config)] == ["003", "001"]


async def test_transient_history_preserves_threads_and_namespaces(
    source_saver: BaseCheckpointSaver,
) -> None:
    identities = {("saved", ""), ("saved", "child"), ("other", "")}
    for thread_id, namespace in identities:
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": namespace}
        }
        checkpoint = empty_checkpoint()
        checkpoint["id"] = "001"
        await source_saver.aput(config, checkpoint, {"source": "update", "step": 1}, {})

    branch = TransientCheckpointSaver(source_saver)
    root_config: RunnableConfig = {
        "configurable": {
            "thread_id": "saved",
            "checkpoint_ns": "",
            "checkpoint_id": "001",
        }
    }
    assert await branch.aget_tuple(root_config) is not None
    history = [saved async for saved in branch.alist(None)]
    assert len(history) == len(identities)
    assert {
        (
            saved.config["configurable"]["thread_id"],
            saved.config["configurable"]["checkpoint_ns"],
        )
        for saved in history
    } == identities
    assert [saved async for saved in branch.alist(root_config)] == [
        await branch.aget_tuple(root_config)
    ]

    child_config: RunnableConfig = {
        "configurable": {"thread_id": "saved", "checkpoint_ns": "child"}
    }
    checkpoint = empty_checkpoint()
    checkpoint["id"] = "002"
    await source_saver.aput(
        child_config, checkpoint, {"source": "update", "step": 2}, {}
    )
    latest = await branch.aget_tuple(child_config)
    assert latest is not None
    assert latest.checkpoint["id"] == "002"
    latest_history = [
        saved
        async for saved in branch.alist(
            {"configurable": {"thread_id": "saved"}}, limit=1
        )
    ]
    assert [saved.checkpoint["id"] for saved in latest_history] == ["002"]
    assert latest_history[0].config["configurable"]["checkpoint_ns"] == "child"
