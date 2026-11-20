# Copyright (c) Microsoft. All rights reserved.

"""Checkpoint, channel-write, serialization, and migration failure contracts."""

from __future__ import annotations

import asyncio
import gc
import operator
from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph._internal._constants import (
    CONFIG_KEY_READ,
    CONFIG_KEY_SEND,
    NO_WRITES,
    NULL_TASK_ID,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Overwrite

from langchain_azure_compute.graph import _checkpoints as cp
from langchain_azure_compute.graph.runner import Limits, pack, unpack


def snapshot(cid="c", **kwargs):
    args = dict(
        config=cp._config("t", cid),
        values={"n": 1},
        metadata={},
        parent_config=None,
        next=(),
        tasks=(),
    )
    args.update(kwargs)
    return NS(**args)


@dataclass
class Row:
    n: int


@pytest.mark.parametrize(
    "policy,exception,expected",
    [
        (ValueError, ValueError("bad"), True),
        (ValueError, RuntimeError("bad"), False),
        ((ValueError,), RuntimeError("bad"), False),
        ("fixed", ValueError("bad"), True),
    ],
)
def test_error_policy_types(policy, exception, expected):
    result = cp._try_format_tool_error(NS(_handle_tool_errors=policy), exception)
    assert (result is not None) is expected


def test_error_handler_same_exception_is_unhandled_and_new_exception_propagates():
    error = ValueError("bad")

    def rethrow(exc: Exception) -> str:
        raise exc

    assert cp._try_format_tool_error(NS(_handle_tool_errors=rethrow), error) is None

    def broken(exc: Exception) -> str:
        raise LookupError("handler failed")

    with pytest.raises(LookupError, match="handler failed"):
        cp._try_format_tool_error(NS(_handle_tool_errors=broken), error)


@pytest.mark.parametrize(
    "policy,error",
    [
        (ValueError, "RuntimeError"),
        ((ValueError,), "RuntimeError"),
    ],
)
def test_exhausted_error_type_filters_rethrow(policy, error):
    with pytest.raises(RuntimeError, match="bad"):
        cp._format_failed_tool_message(
            NS(_handle_tool_errors=policy),
            {"id": "call", "name": "tool"},
            {"error_type": error, "message": "bad"},
        )


def test_exhausted_unknown_error_type_and_default_policy():
    node = NS(_handle_tool_errors=cp._default_handle_tool_errors)
    for kind in ("nonexistent.CustomError", "str"):
        msg = cp._format_failed_tool_message(
            node,
            {"id": "call", "name": "tool"},
            {"error_type": kind, "message": "bad"},
        )
        assert msg.status == "error" and "bad" in msg.content


def test_schema_config_context_and_recursion_boundaries():
    assert (
        cp._normalize_run_config({"configurable": {"thread_id": "t", "__pregel_x": 1}})
        is None
    )
    assert cp._normalize_run_config({"configurable": 7, "metadata": {}}) is None
    assert cp._normalize_run_config({"metadata": {"trace": 1}})["metadata"] == {
        "trace": 1
    }
    assert cp._coerce_with_schema({"n": "bad", "extra": 2}, Row) == {
        "n": "bad",
        "extra": 2,
    }
    assert cp._coerce_with_schema({"n": 2}, Row) == Row(2)
    assert cp._coerce_with_schema({"n": 2}, dict) == {"n": 2}
    assert cp._coerce_with_schema(Row(2), Row) == Row(2)
    assert unpack(cp._pack_context(NS(model_dump=lambda: {"n": 2}))) == {"n": 2}
    assert cp._StepRecursionLimit(1, 5).__lt__("x") is NotImplemented
    assert cp._checkpoint_id(None) is None
    assert cp._raw_checkpoint_id({"configurable": {"checkpoint_id": 7}}) == 7
    moved = snapshot(metadata={"migrated_to": "v2"})
    assert not cp._checkpoint_match(moved, "c")
    assert cp._checkpoint_match(moved, cp._checkpoint_id(moved))


def test_scoped_schema_lookup_and_lookup_failure():
    sub = NS(builder=NS(state_schema=Row))
    root = NS(
        builder=NS(schema=dict),
        get_subgraphs=lambda **kwargs: [("other", NS()), ("sub", sub)],
    )
    assert cp._state_schema_for_graph(root, "sub:node") is Row
    assert cp._state_schema_for_graph(root, "missing:node") is dict

    def fail(**kwargs):
        raise OSError("lookup failed")

    root.get_subgraphs = fail
    assert cp._state_schema_for_graph(root, "sub:node") is dict


@pytest.mark.parametrize("operation", ["state", "update", "saved"])
@pytest.mark.parametrize("error", [NotImplementedError, asyncio.InvalidStateError])
def test_sync_checkpoint_operations_bridge_after_unavailable_sync(operation, error):
    called = []

    def fail(*args, **kwargs):
        raise error()

    async def async_call(*args, **kwargs):
        called.append((args, kwargs))
        return "bridged"

    saver = NS(get_tuple=fail, aget_tuple=async_call)
    graph = NS(
        checkpointer=saver,
        get_state=fail,
        aget_state=async_call,
        update_state=fail,
        aupdate_state=async_call,
    )
    if operation == "state":
        result = cp._get_state(graph, cp._config("t"))
    elif operation == "update":
        result = cp._update_state(graph, cp._config("t"), {"n": 2}, "node")
    else:
        result = cp._get_saved_tuple(graph, cp._config("t"))
    assert result == "bridged" and len(called) == 1


@pytest.mark.parametrize(
    "error", [NotImplementedError, asyncio.InvalidStateError, TypeError]
)
@pytest.mark.parametrize("source", ["input", "loop", "other", "fails"])
def test_invoke_fallback_preserves_or_resumes_input_once(error, source):
    calls = []

    def invoke(*args, **kwargs):
        calls.append("sync")
        raise error("No synchronous function provided")

    def get_state(*args, **kwargs):
        if source == "fails":
            raise OSError("unavailable")
        return snapshot(metadata={"source": source})

    async def ainvoke(inp, cfg, **kwargs):
        calls.append(inp)
        return cfg

    graph = NS(
        checkpointer=None,
        nodes={},
        builder=NS(nodes={}),
        invoke=invoke,
        ainvoke=ainvoke,
        get_state=get_state,
    )
    cfg = cp._invoke(
        graph, {"n": 1}, cp._config("t"), max_concurrency=2, context=Row(1)
    )
    assert cfg["max_concurrency"] == 2
    assert calls == ["sync", None if source in ("input", "loop") else {"n": 1}]


@pytest.mark.parametrize("inp", [None, Command(resume=True)])
def test_invoke_fallback_does_not_replace_resume_commands(inp):
    async def ainvoke(inp, cfg, **kwargs):
        return inp

    def invoke(*args, **kwargs):
        raise NotImplementedError

    graph = NS(
        checkpointer=None,
        nodes={},
        builder=NS(nodes={}),
        invoke=invoke,
        ainvoke=ainvoke,
    )
    assert cp._invoke(graph, inp, cp._config("t")) == inp


def test_turn_input_step_and_async_node_detection():
    graph = NS(
        get_state=lambda *a, **k: snapshot(parent_config=None), checkpointer=None
    )
    assert cp._resolve_turn_input_step(graph, {}) == -1
    assert cp._resolve_turn_input_step(graph, cp._config("t"), 3) == 3
    assert cp._resolve_turn_input_step(graph, cp._config("t")) == -1
    graph.get_state = lambda *a, **k: (_ for _ in ()).throw(OSError())
    assert cp._resolve_turn_input_step(graph, cp._config("t")) == -1
    graph.nodes = {"node": NS(bound=NS(func=None, afunc=lambda: None))}
    assert cp._graph_has_async_only_node(graph)
    assert not cp._is_mid_start(
        graph, snapshot(metadata={"source": "loop"}, next=("node",)), None, "*"
    )
    graph.get_state = lambda *a, **k: snapshot(metadata={"source": "other"})
    assert not cp._is_mid_start(
        graph,
        snapshot(
            metadata={"source": "loop"},
            next=("node",),
            parent_config=cp._config("t", "p"),
        ),
        None,
        [],
    )


class Saver:
    def __init__(self, fail=False):
        self.records = []
        self.fail = fail

    def put_writes(self, cfg, writes, task):
        if self.fail:
            raise NotImplementedError
        self.records.append((writes, task))

    async def aput_writes(self, cfg, writes, task):
        self.records.append((writes, task))


class SumChannel:
    def __init__(self, value=0, broken=False):
        self.value, self.broken = value, broken

    def copy(self):
        return SumChannel(self.value, self.broken)

    def from_checkpoint(self, value):
        return SumChannel(value, self.broken)

    def update(self, values):
        if self.broken:
            raise ValueError("cannot merge")
        self.value += sum(values)

    def get(self):
        return self.value


@pytest.mark.parametrize(
    "mode", ["sum", "missing", "broken", "empty", "no_saver", "fallback", "async"]
)
def test_parallel_task_channel_writers_preserve_fresh_state_and_no_writes(mode):
    seen = []

    def writer(values, config):
        conf = config["configurable"]
        read, send = conf[CONFIG_KEY_READ], conf[CONFIG_KEY_SEND]
        assert read("__root__") == {"n": 1}
        assert read(["n", "missing"]) == {"n": 1}
        if mode != "empty":
            send([("n", 2), ("n", 3), ("new", 4)])
            seen.append(read("n", fresh=True))
            assert read("new", fresh=True) == 4

    saver = Saver(fail=mode == "fallback")
    if mode == "async":

        class AsyncSaver(Saver):
            pass

        saver = AsyncSaver()
    if mode == "no_saver":
        saver = None
    channels = (
        {}
        if mode == "missing"
        else {
            "n": SumChannel(broken=mode == "broken"),
            "new": SumChannel(),
        }
    )
    graph = NS(
        nodes={"node": NS(flat_writers=[RunnableLambda(writer)])},
        checkpointer=saver,
        channels=channels,
        builder=NS(state_schema=None),
    )
    cp._put_task_writes(graph, snapshot(), "node", "task", {"n": 1})
    if mode != "empty":
        assert seen == [3 if mode in ("missing", "broken") else 6]
    if saver:
        writes, task = saver.records[0]
        assert task == "task"
        assert writes == (
            [(NO_WRITES, None)] if mode == "empty" else [("n", 2), ("n", 3), ("new", 4)]
        )


def test_multi_writer_sequence_and_scalar_read():
    results = []

    def writer(values, config):
        read = config["configurable"][CONFIG_KEY_READ]
        results.append(read("__root__"))
        return values

    graph = NS(
        nodes={
            "node": NS(
                flat_writers=[
                    RunnableLambda(writer),
                    RunnableLambda(writer),
                ]
            )
        },
        checkpointer=Saver(),
        channels={},
        builder=NS(state_schema=None),
    )
    cp._put_task_writes(graph, snapshot(values=5), "node", "task", 5)
    assert results == [{"__root__": 5}, {"__root__": 5}]


def test_preparation_without_checkpoint_and_nested_graphs():
    assert cp._get_saved_tuple(NS(checkpointer=None), {}) is None
    assert cp._prepare_executable_tasks(NS(), None) == {}
    assert cp._prepare_executable_tasks(NS(checkpointer=None), snapshot()) == {}
    sub = NS(checkpointer=True)
    graph = NS(
        checkpointer=InMemorySaver(),
        get_subgraphs=lambda **k: [("other", NS()), ("sub", sub)],
    )
    found = cp._tool_task_graph(graph, "sub:tools")
    assert found is not sub and found.checkpointer is graph.checkpointer
    sub.checkpointer = object()
    assert cp._tool_task_graph(graph, "sub:tools") is sub
    assert cp._tool_task_graph(graph, "unknown:tools") is graph


def test_subgraph_and_breakpoint_fallbacks():
    graph = NS(
        builder=NS(nodes={"node": NS(runnable=object())}),
        get_subgraphs=lambda **k: (_ for _ in ()).throw(OSError()),
    )
    assert cp._tool_nodes(graph) == {}
    cp._prepare_subgraph_tool_nodes(NS(), Limits(), cp.ToolConfig())
    assert cp._merged_interrupt_before(
        NS(interrupt_before_nodes="*", builder=NS(nodes={"a": 1, "b": 2})),
        [],
        exclude_static=["a"],
    ) == ["b"]
    assert cp._user_interrupt_before(NS(interrupt_before_nodes="*")) == "*"
    assert cp._user_interrupt_after(NS(interrupt_after_nodes="*")) == "*"
    parent = snapshot(next=("other",))
    graph = NS(
        checkpointer=None,
        get_state=lambda *a, **k: parent,
        interrupt_before_nodes=["no"],
        interrupt_after_nodes=["no"],
    )
    snap = snapshot(next=("node",), parent_config=cp._config("t", "p"))
    assert cp._static_breakpoint_hit(graph, snap) is None
    graph.interrupt_after_nodes = ["no"]
    snap.parent_config = None
    assert cp._static_breakpoint_hit(graph, snap) is None
    nested = snapshot(next=("other",), tasks=(NS(name="inner", state=snapshot()),))
    assert (
        cp._find_subgraph_tool_target(
            snapshot(tasks=(NS(name="outer", state=nested),)),
            {},
        )
        is None
    )


def test_graph_clone_collection_nonweak_refs_and_bounded_cache(monkeypatch):
    monkeypatch.setattr(cp, "_REGISTERED_GRAPH_CLONES", {})
    cp._register_graph_clone(NS(), NS())
    assert not cp._REGISTERED_GRAPH_CLONES

    class Graph:
        pass

    original, clone = Graph(), Graph()
    cp._register_graph_clone(original, clone)
    assert cp._unwrap_registered_graph(original) is clone
    oid = id(original)
    del original
    gc.collect()
    assert oid not in cp._REGISTERED_GRAPH_CLONES
    keep = [Graph() for _ in range(257)]
    for original in keep:
        cp._register_graph_clone(original, clone)
    assert len(cp._REGISTERED_GRAPH_CLONES) == 256


@pytest.mark.parametrize(
    "kind",
    [
        "one",
        "two",
        "varargs",
        "dict",
        "node_map",
        "callable_values",
        "explicit",
        "unchanged",
        "overwrite",
    ],
)
def test_migration_forms_and_channel_deltas(kind):
    saved = NS(
        config=cp._config("t", "c"),
        parent_config=None,
        checkpoint={"id": "c"},
        metadata={},
    )
    writes, puts, updates = [], [], []
    saver = NS(
        get_tuple=lambda cfg: saved,
        put_writes=lambda cfg, values, task: writes.append((dict(values), task)),
        put=lambda *args: puts.append(args),
    )
    snap = snapshot(values={"n": 1, "items": [1]}, next=("old",))
    graph = NS(
        _durable_name="g",
        _durable_version="new",
        checkpointer=saver,
        channels={"items": NS(operator=operator.add)},
        get_state=lambda *a, **k: snap,
        update_state=lambda cfg, values, as_node=None: updates.append(
            (values, as_node)
        ),
    )
    request = {"resume": pack(True)}
    if kind == "one":

        def migrate_one(value):
            return {"n": 2}

        migration = migrate_one
    elif kind == "two":

        def migrate_two(value, snapshot):
            return {"n": 2}

        migration = migrate_two
    elif kind == "varargs":

        def migrate_varargs(*args):
            return {"n": 2}

        migration = migrate_varargs
    elif kind == "dict":
        migration = {"values": {"n": 2}}
    elif kind == "node_map":
        migration = {"node_map": {"old": "new"}, "values": {"n": 2}}
    elif kind == "callable_values":
        migration = {"values": lambda value: {"n": 2}}
    elif kind == "explicit":
        migration = {"n": 3}
        request.update(migrate_values=pack({"n": 2}), as_node="new")
    elif kind == "unchanged":
        migration = {"n": 1}
    else:
        migration = {"items": [7]}
    graph._durable_migrations = {"new": migration}
    jumped = cp._apply_version_migration(
        graph, graph, "t", None, "new", request=request
    )
    if kind in ("node_map", "explicit"):
        assert jumped and updates == [({"n": 2}, "new")]
    elif kind == "unchanged":
        assert not jumped and not writes and puts
    elif kind == "overwrite":
        assert isinstance(writes[0][0]["items"], Overwrite)
    else:
        assert not jumped and writes[0] == ({"n": 2}, NULL_TASK_ID)
    assert cp._apply_version_migration(
        graph,
        graph,
        "t",
        None,
        "new",
        request={"migrate_values": pack({"items": [1, 2]})},
    ) is (kind == "node_map")


def test_migration_callback_type_error_is_not_executed_twice():
    calls = []

    def broken(value):
        calls.append(value)
        raise TypeError("user callback error")

    graph = NS(
        _durable_name="g",
        _durable_version="new",
        _durable_migrations={"new": broken},
        checkpointer=None,
        get_state=lambda *a, **k: snapshot(),
    )
    with pytest.raises(TypeError, match="user callback"):
        cp._apply_version_migration(graph, graph, "t", None, "new")
    assert len(calls) == 1
