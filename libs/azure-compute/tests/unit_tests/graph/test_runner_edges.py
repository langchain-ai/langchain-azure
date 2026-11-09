# Copyright (c) Microsoft. All rights reserved.

"""Legacy commit payloads, checkpoint races and redelivered migration boundaries."""

from __future__ import annotations

import operator
from types import SimpleNamespace as NS
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from langchain_azure_compute.graph import attach_durability
from langchain_azure_compute.graph import runner as r
from tests.unit_tests.graph.test_checkpoint_edges import snapshot
from tests.unit_tests.graph.test_real_backend import FanoutState
from tests.unit_tests.graph.test_wave2 import registration


def test_lineage_parent_traversal_and_empty_inline_output():
    builder = StateGraph(FanoutState)
    builder.add_node("echo", lambda state: state)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    reg = attach_durability(builder.compile(), name="empty_inline")
    config = r._config("t")
    empty = reg.graph.get_state(config)
    reg.graph.checkpointer.delete_thread("t")
    report = r._report(reg.graph, empty, reg.limits)
    assert report["status"] == "done" and r.unpack(report["output"]) == {}
    parent = snapshot(cid="parent")
    child = snapshot(parent_config=parent.config)
    assert not r._start_lineage_matches(
        reg.graph, snapshot(), "missing", "instance", "signature"
    )
    assert not r._start_lineage_matches(
        NS(get_state=lambda c, **kw: parent, checkpointer=NS()),
        child,
        "parent",
        "instance",
        "signature",
    )


@pytest.mark.parametrize("mode", ["raw", "command", "map", "jump"])
def test_resume_accepts_user_values_commands_and_unpacked_maps(mode):
    class State(TypedDict):
        value: str

    builder = StateGraph(State)
    builder.add_node("review", lambda state: {"value": interrupt("review")})
    builder.add_edge(START, "review")
    builder.add_edge("review", END)
    reg = attach_durability(builder.compile(), name="raw_resume")
    start = r.run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "start",
            "expect": None,
            "input": r.pack({"value": "old"}),
        },
    )
    intr = start["interrupt"]["id"]
    resume = (
        "new"
        if mode == "raw"
        else Command(resume="new")
        if mode == "command"
        else r.pack("new")
        if mode == "jump"
        else {"__resume_map__": {intr: "new"}}
    )
    end = r.run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "resume",
            "expect": start["ckpt"],
            "resume": resume,
            **(
                {"as_node": "review", "migrate_values": r.pack({"value": "new"})}
                if mode == "jump"
                else {}
            ),
        },
    )
    assert end["status"] == "done"
    assert reg.graph.get_state(r._config("t")).values["value"] == "new"


def test_duplicate_bundle_inspection_and_same_version_jump():
    reg = registration()
    reg.graph.invoke({"value": "old"}, r._config("t"))
    current = reg.graph.get_state(r._config("t"))
    bundle = r._export_checkpoint_bundle(reg.graph, "t")
    reg.graph.checkpointer._durable_completed_expects = {("t", "old")}
    report = r.run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "inspect",
            "expect": "old",
            "checkpoint_bundle": bundle,
        },
    )
    assert report["replayed"]
    report = r.run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "migrate",
            "expect": r._checkpoint_id(current),
            "as_node": "echo",
        },
    )
    assert report["jumped"] is True


@pytest.mark.parametrize("jump", [False, True, "unmatched"])
def test_redelivered_resume_after_migration_does_not_repeat_callback(monkeypatch, jump):
    reg = registration()
    old = snapshot(metadata={"source": "loop"}, next=("echo",))
    after = snapshot(
        cid="new" if jump else "c",
        metadata={"source": "update"} if jump else {"migrated_to": "v1"},
        parent_config=r._config("t", "other")
        if jump == "unmatched"
        else old.config
        if jump
        else None,
        next=("echo",),
    )
    monkeypatch.setattr(r, "_get_state", lambda *a, **kw: after)
    called = []
    monkeypatch.setattr(r, "_invoke", lambda *a, **kw: called.append(a[1]))

    def unexpected(*a, **kw):
        pytest.fail("Migration ran twice")

    monkeypatch.setattr(r, "_apply_version_migration", unexpected)
    report = r.run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "resume",
            "expect": r._checkpoint_id(old),
            "version": "v1",
            "migrate_values": r.pack({"value": "new"}),
            "resume": r.pack(True),
        },
    )
    if jump == "unmatched":
        assert report["replayed"] and called == []
    else:
        assert not report["replayed"] and called == [None]


def test_unknown_tool_nodes_are_explicit_run_errors():
    reg = registration()
    with pytest.raises(r.RunError, match="unknown tool node"):
        r.run_tool(
            reg.graph,
            {
                "thread": "t",
                "node": "missing",
                "call_id": "call",
                "index": 0,
                "call": r.pack({"name": "missing", "args": {}, "id": "call"}),
                "ckpt": None,
            },
        )
    with pytest.raises(r.RunError, match="unknown tool node"):
        r.run_step(
            reg.graph,
            reg.limits,
            {
                "thread": "t",
                "kind": "tools",
                "node": "missing",
                "outputs": [],
                "expect": None,
            },
        )


def test_legacy_mixed_tool_output_list_is_matched_to_durable_calls_only():
    @tool
    def remote() -> str:
        """Read remotely."""
        return "remote"

    @tool
    def local() -> str:
        """Read locally."""
        return "local"

    builder = StateGraph(FanoutState)
    builder.add_node("tools", ToolNode([remote, local]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    reg = attach_durability(
        builder.compile(),
        name="legacy_tools",
        durable_tools={"remote": True, "local": False},
    )
    calls = [
        {"name": name, "args": {}, "id": name, "type": "tool_call"}
        for name in ("remote", "local")
    ]
    report = r.run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "start",
            "expect": None,
            "input": r.pack(
                {
                    "messages": [AIMessage(content="", tool_calls=calls)],
                    "approved": False,
                }
            ),
        },
    )
    out = r.pack(ToolMessage(content="remote", tool_call_id="remote", name="remote"))
    skipped = r.run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "nodes",
            "expect": report["ckpt"],
            "outputs": {},
        },
    )
    assert skipped["status"] == "tools"
    end = r.run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "tools",
            "expect": report["ckpt"],
            "node": "tools",
            "outputs": [out],
        },
    )
    assert end["status"] == "done"
    assert (
        len(
            [
                m
                for m in reg.graph.get_state(r._config("t")).values["messages"]
                if isinstance(m, ToolMessage)
            ]
        )
        == 2
    )


def test_legacy_mixed_node_output_list_runs_only_uncommitted_local_siblings():
    class State(TypedDict):
        values: Annotated[list[int], operator.add]

    builder = StateGraph(State)
    builder.add_node("a", lambda s: {"values": [1]})
    builder.add_node("b", lambda s: {"values": [2]})
    for name in ("a", "b"):
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    reg = attach_durability(builder.compile(), name="legacy_nodes", durable_nodes=["a"])
    report = r.run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "start",
            "expect": None,
            "input": r.pack({"values": []}),
        },
    )
    end = r.run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "nodes",
            "expect": report["ckpt"],
            "outputs": [r.pack({"values": [1]})],
        },
    )
    assert end["status"] == "done"
    assert sorted(reg.graph.get_state(r._config("t")).values["values"]) == [1, 2]


def test_tool_spec_checkpoint_disappearance_falls_back_without_unsafe_execution():
    @tool
    def remote() -> str:
        """Read remotely."""
        return "remote"

    builder = StateGraph(FanoutState)
    builder.add_node("tools", ToolNode([remote]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    reg = attach_durability(builder.compile(), name="lost_checkpoint")
    r.run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "start",
            "expect": None,
            "input": r.pack(
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "remote",
                                    "args": {},
                                    "id": "call",
                                    "type": "tool_call",
                                },
                            ],
                        )
                    ],
                    "approved": False,
                }
            ),
        },
    )
    snap = reg.graph.get_state(r._config("t"))
    reg.graph.checkpointer.delete_thread("t")
    assert r._report(reg.graph, snap, reg.limits)["status"] == "running"
    assert (
        r._report(reg.graph, snapshot(next=("missing",)), reg.limits)["status"]
        == "running"
    )
