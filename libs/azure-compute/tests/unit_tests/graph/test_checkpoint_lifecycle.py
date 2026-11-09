# Copyright (c) Microsoft. All rights reserved.

"""Checkpoint transfer, asynchronous persistence and isolated-node compatibility."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
from langchain_core.messages import ToolMessage

from langchain_azure_compute.graph import _checkpoints as cp
from tests.unit_tests.graph.test_checkpoint_edges import Row, snapshot


def test_remaining_scalar_and_policy_helpers(monkeypatch):
    assert (
        cp._try_format_tool_error(NS(_handle_tool_errors=False), ValueError()) is None
    )
    assert str(cp._StepRecursionLimit(1, 5)) == "5"
    assert cp._normalize_run_config({"recursion_limit": 3})["recursion_limit"] == 3
    for flag in (ValueError, (ValueError,)):
        assert (
            cp._format_failed_tool_message(
                NS(_handle_tool_errors=flag),
                {"id": "call", "name": "tool"},
                {"error_type": "ValueError", "message": "bad"},
            ).status
            == "error"
        )
    monkeypatch.setattr(
        cp, "_get_state", lambda *a: snapshot(parent_config=cp._config("t", "parent"))
    )
    assert cp._resolve_turn_input_step(NS(), cp._config("t")) == -1
    assert cp._tool_nodes(NS(builder=NS(nodes={}))) == {}
    assert cp._static_breakpoint_hit(NS(), snapshot()) is None
    assert (
        cp._static_breakpoint_hit(
            NS(interrupt_after_nodes=("other",)), snapshot(next=("node",))
        )
        is None
    )
    assert cp._merged_interrupt_before(NS(interrupt_before_nodes="*"), []) == "*"
    assert cp._merged_interrupt_before(
        NS(interrupt_before_nodes=("tool",)), ["tool"]
    ) == ["tool"]
    deep = snapshot(next=("tools",))
    middle = snapshot(tasks=(NS(name="child", state=deep),))
    root = snapshot(tasks=(NS(name="parent", state=middle),))
    assert (
        cp._find_subgraph_tool_target(root, {"parent:child:tools": NS()})[0]
        == "parent:child:tools"
    )


@pytest.mark.parametrize("before", ["*", ("tools",)])
def test_nested_breakpoints_do_not_duplicate_existing_or_local_tools(before):
    node = NS(
        tools_by_name={"remote": NS(), "local": NS()},
        _parse_input=lambda v: ([], "dict"),
    )
    local = NS(tools_by_name={"local": NS()}, _parse_input=lambda v: ([], "dict"))
    sub = NS(
        builder=NS(nodes={"tools": node, "local": local}), interrupt_before_nodes=before
    )
    graph = NS(get_subgraphs=lambda **kw: [("sub", sub)])
    policy = NS(resolve=lambda tn, node, name, limits: (name == "remote", 1, True))
    cp._prepare_subgraph_tool_nodes(graph, cp.Limits(), policy)
    assert sub.interrupt_before_nodes == ("*" if before == "*" else ["tools"])


def test_pending_local_calls_and_unavailable_tool_input_parser():
    def broken(value):
        raise ValueError("no message")

    policy = NS(resolve=lambda tn, node, name, limits: (name == "remote", 1, True))
    node = NS(tools_by_name={"remote": NS(), "local": NS()}, _parse_input=broken)
    graph = NS(builder=NS(nodes={"tools": node}))
    snap = snapshot(next=("tools",))
    assert cp._durable_tool_node_names(graph, cp.Limits(), policy, snap) == ["tools"]
    node._parse_input = lambda v: ([{"name": "local"}], "dict")
    assert cp._durable_tool_node_names(graph, cp.Limits(), policy, snap) == []


@pytest.mark.parametrize("mode", ["async", "fallback", "sync"])
@pytest.mark.parametrize("parent", [None, "parent"])
def test_migration_metadata_async_persistence_and_parent_coordinates(
    monkeypatch, mode, parent
):
    writes = []

    class Saver:
        def put(self, cfg, checkpoint, metadata, versions):
            if mode == "fallback":
                raise NotImplementedError()
            writes.append((cfg, metadata))

        async def aput(self, cfg, checkpoint, metadata, versions):
            writes.append((cfg, metadata))

    saver = Saver()
    graph = NS(checkpointer=saver)
    saved = NS(
        config=cp._config("t", "current"),
        metadata={},
        parent_config=cp._config("t", parent) if parent else None,
        checkpoint={},
    )
    monkeypatch.setattr(cp, "_get_saved_tuple", lambda *a: saved)
    monkeypatch.setattr(cp, "_is_async_saver", lambda s: mode == "async")
    cp._stamp_migrated_metadata(graph, snapshot(), "v2")
    assert writes[0][1]["migrated_to"] == "v2"
    assert writes[0][0]["configurable"].get("checkpoint_id") == parent
    cp._stamp_migrated_metadata(graph, None, "v2")
    cp._stamp_migrated_metadata(graph, NS(config=None), "v2")
    monkeypatch.setattr(cp, "_get_saved_tuple", lambda *a: None)
    cp._stamp_migrated_metadata(graph, snapshot(), "v2")


@pytest.mark.parametrize("transfer", ["bundle", "missing", "existing", "sync", "async"])
def test_migration_copies_separate_savers_without_duplicate_checkpoints(
    monkeypatch, transfer
):
    transferred = []

    class Saver:
        def put(self, *a):
            transferred.append(a)

        async def aput(self, *a):
            transferred.append(a)

    old, target = NS(checkpointer=NS()), NS(checkpointer=Saver())
    old._durable_previous_versions = {}
    saved = NS(config=cp._config("t"), parent_config=None, checkpoint={}, metadata={})
    monkeypatch.setattr(
        cp, "_get_state", lambda *a: snapshot(metadata={"migrated_to": "v2"})
    )
    monkeypatch.setattr(
        cp, "_export_checkpoint_bundle", lambda *a: {} if transfer == "bundle" else None
    )
    monkeypatch.setattr(
        cp, "_hydrate_checkpoint_bundle", lambda *a: transferred.append(a)
    )
    monkeypatch.setattr(
        cp,
        "_get_saved_tuple",
        lambda g, c: (
            None
            if transfer == "missing"
            else saved
            if g is old or transfer == "existing"
            else None
        ),
    )
    monkeypatch.setattr(cp, "_is_async_saver", lambda *a: transfer == "async")
    assert not cp._apply_version_migration(old, target, "t", None, "v2")
    assert bool(transferred) == (transfer in ("bundle", "sync", "async"))


@pytest.mark.parametrize("mode", ["async", "fallback", "sync"])
def test_migration_delta_async_writes_and_uninspectable_callback(monkeypatch, mode):
    writes = []

    class Saver:
        def put_writes(self, *args):
            if mode == "fallback":
                raise NotImplementedError()
            writes.append(args)

        async def aput_writes(self, *args):
            writes.append(args)

    graph = NS(checkpointer=Saver(), channels={})
    graph._durable_migrations = {"v2": lambda value: {"values": {"n": 2}}}
    monkeypatch.setattr(cp, "_get_state", lambda *a: snapshot())
    monkeypatch.setattr(cp, "_stamp_migrated_metadata", lambda *a: None)
    monkeypatch.setattr(cp, "_is_async_saver", lambda *a: mode == "async")

    def no_signature(obj):
        raise ValueError("unknown signature")

    monkeypatch.setattr(cp.inspect, "signature", no_signature)
    assert not cp._apply_version_migration(graph, graph, "t", None, "v2")
    assert writes[0][1] == [("n", 2)]


@pytest.mark.parametrize("mode", ["no_values", "unmapped", "explicit_scalar"])
def test_migration_no_delta_and_mapping_edge_shapes(monkeypatch, mode):
    graph = NS(
        checkpointer=None,
        channels={},
        _durable_migrations={
            "v2": {
                "node_map": {"other": "new"},
            }
        },
    )
    monkeypatch.setattr(cp, "_get_state", lambda *a: snapshot(next=("old",)))
    monkeypatch.setattr(cp, "_stamp_migrated_metadata", lambda *a: None)
    request = {"migrate_values": cp.pack(3)} if mode == "explicit_scalar" else {}
    assert not cp._apply_version_migration(
        graph, graph, "t", None, "v2", request=request
    )


@pytest.mark.parametrize(
    "mode", ["no_executable", "no_config", "async_fallback", "sync_error"]
)
def test_isolated_node_fallback_context_and_errors(monkeypatch, mode):
    class Bound:
        func = staticmethod(lambda: None)
        afunc = None

        def invoke(self, value, config):
            if mode == "async_fallback":
                raise NotImplementedError()
            if mode == "sync_error":
                raise TypeError("node error")
            assert config["metadata"]["trace"] == "t"
            assert config["configurable"]["custom"] == 2
            assert "__pregel_bad" not in config["configurable"]
            return {"n": 2}

        async def ainvoke(self, value, config):
            return {"n": 3}

    bound = Bound()
    if mode == "async_fallback":
        bound.afunc = lambda: None
    graph = NS(nodes={"node": NS(bound=bound)}, builder=NS(nodes={}, state_schema=Row))
    task = NS(input={"n": 1}, config=None)
    monkeypatch.setattr(
        cp,
        "_prepare_executable_tasks",
        lambda *a: {} if mode == "no_executable" else {"id": task},
    )
    if mode == "sync_error":
        with pytest.raises(TypeError, match="node error"):
            cp._exec_single_node(graph, snapshot(), "id", "node", None, "t", [])
        return
    status, output = cp._exec_single_node(
        graph,
        snapshot(),
        "id",
        "node",
        None,
        "t",
        [],
        run_config={
            "configurable": {"custom": 2, "__pregel_bad": 3},
            "metadata": {"trace": "t"},
            "recursion_limit": 3,
            "tags": ["tag"],
        },
        context={"user": "alice"},
    )
    assert status == "ok" and output["n"] == (3 if mode == "async_fallback" else 2)


def test_tool_async_fallback_scalar_reader_and_send_command(monkeypatch):
    def run_one(call, input_type, runtime):
        conf = runtime.config["configurable"]
        assert conf["__pregel_read"]("n") == 1
        conf["__pregel_send"]([("n", 2)])
        return ToolMessage(content="ok", tool_call_id="call")

    class Node:
        tools_by_name = {"tool": NS(func=lambda: None)}
        _messages_key = "messages"
        _handle_tool_errors = False

        def _run_one(self, *a):
            raise NotImplementedError()

        async def _arun_one(self, *a):
            return run_one(*a)

    result, out = cp._exec_single_tool(
        NS(builder=NS(nodes={}, state_schema=Row), store=None),
        Node(),
        Row(1),
        {"name": "tool", "id": "call", "args": {}},
        "t",
        "",
        [],
        task_input_type="dict",
    )
    assert result == "ok" and out.update["n"] == 2


def test_null_task_migration_writes_are_applied_before_task_preparation():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    saver = InMemorySaver()
    builder = StateGraph(Row)
    builder.add_node("node", lambda row: {"n": row.n})
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    graph = builder.compile(checkpointer=saver)
    graph.invoke({"n": 1}, cp._config("t"))
    snap = graph.get_state(cp._config("t"))
    saver.put_writes(snap.config, [("n", 9)], cp.NULL_TASK_ID)
    assert cp._prepare_executable_tasks(graph, snap) == {}


def test_direct_async_checkpoint_read_and_update(monkeypatch):
    class Saver:
        async def aget_tuple(self, config):
            return "saved"

    async def update(*a, **kw):
        return "updated"

    graph = NS(checkpointer=Saver(), aupdate_state=update)
    monkeypatch.setattr(cp, "_is_async_saver", lambda saver: True)
    assert cp._get_saved_tuple(graph, cp._config("t")) == "saved"
    assert cp._update_state(graph, cp._config("t"), {"n": 2}) == "updated"
