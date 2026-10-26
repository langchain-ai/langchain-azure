# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for checkpoint branches that never write to their source saver."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Annotated

import pytest

pytest.importorskip("azure.ai.agentserver.responses")

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

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
) -> AsyncIterator[BaseCheckpointSaver]:
    if request.param == "memory":
        yield InMemorySaver()
    else:
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            yield saver


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
