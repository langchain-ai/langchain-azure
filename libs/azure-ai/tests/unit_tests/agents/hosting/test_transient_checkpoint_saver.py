# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for checkpoint branches that never write to their source saver."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

pytest.importorskip("azure.ai.agentserver.responses")

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from langchain_azure_ai.agents.hosting._responses.transient_checkpoint_saver import (
    TransientCheckpointSaver,
)


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


async def test_transient_branch_isolates_subgraph_checkpoints(
    source_saver: BaseCheckpointSaver,
) -> None:
    child = _graph(source_saver)
    builder = StateGraph(MessagesState)
    builder.add_node("child", child)
    builder.add_edge(START, "child")
    builder.add_edge("child", END)
    graph = builder.compile(checkpointer=source_saver)
    config: RunnableConfig = {"configurable": {"thread_id": "saved"}}
    await graph.ainvoke({"messages": [HumanMessage(content="saved")]}, config)
    source_before = [saved async for saved in source_saver.alist(config)]

    branch = graph.copy(update={"checkpointer": TransientCheckpointSaver(source_saver)})
    result = await branch.ainvoke(
        {"messages": [HumanMessage(content="transient")]}, config
    )
    assert any(message.content == "transient" for message in result["messages"])
    assert [saved async for saved in source_saver.alist(config)] == source_before
