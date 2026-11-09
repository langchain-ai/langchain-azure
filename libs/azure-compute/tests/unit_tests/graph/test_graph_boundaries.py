# Copyright (c) Microsoft. All rights reserved.

"""Serialization and policy boundary cases previously hidden by exclusions."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import ormsgpack
import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel

from langchain_azure_compute.graph import _async_bridge as bridge
from langchain_azure_compute.graph import _config as config
from langchain_azure_compute.graph import _inline_saver as inline


class Model(BaseModel):
    value: int


@pytest.fixture
def serializer_globals(monkeypatch):
    monkeypatch.setattr(config, "_ALLOW_ALL_MSGPACK_MODULES", False)
    monkeypatch.setattr(config, "_ALLOWED_MSGPACK_MODULES", set())
    monkeypatch.setattr(config, "_EXTRA_SAFE_MSGPACK_TYPES", set())


@pytest.mark.parametrize(
    "code",
    [
        config.EXT_CONSTRUCTOR_SINGLE_ARG,
        config.EXT_CONSTRUCTOR_POS_ARGS,
        config.EXT_CONSTRUCTOR_KW_ARGS,
    ],
)
@pytest.mark.parametrize("payload", [[], [1], ["untrusted", "Dangerous", {}]])
def test_constructor_serialization_fails_closed(code, payload, serializer_globals):
    with pytest.raises(ValueError, match="Blocked"):
        config._strict_msgpack_ext_hook(code, ormsgpack.packb(payload))


@pytest.mark.parametrize(
    "code,args,expected",
    [
        (config.EXT_CONSTRUCTOR_SINGLE_ARG, ["builtins", "str", 7], "7"),
        (config.EXT_CONSTRUCTOR_POS_ARGS, ["builtins", "list", [[1, 2]]], [1, 2]),
        (config.EXT_CONSTRUCTOR_KW_ARGS, ["builtins", "dict", {"a": 1}], {"a": 1}),
    ],
)
def test_explicitly_allowed_constructor_types(code, args, expected, serializer_globals):
    config._configure_msgpack_modules([("builtins",)])
    assert config._strict_msgpack_ext_hook(code, ormsgpack.packb(args)) == expected
    config._configure_msgpack_modules(True)
    assert config._is_allowed_msgpack_type("untrusted", "Type")


@pytest.mark.parametrize("payload", [[], ["untrusted", "Type", "arg", "method"]])
def test_method_serialization_fails_closed(payload, serializer_globals):
    with pytest.raises(ValueError, match="Blocked"):
        config._strict_msgpack_ext_hook(
            config.EXT_METHOD_SINGLE_ARG, ormsgpack.packb(payload)
        )


@pytest.mark.parametrize("code", [config.EXT_PYDANTIC_V1, config.EXT_PYDANTIC_V2])
def test_pydantic_serialization_registration_and_safe_fallback(
    code, serializer_globals
):
    data = ormsgpack.packb([__name__, "Model", {"value": 7}])
    assert config._strict_msgpack_ext_hook(code, data) == {"value": 7}
    config._register_safe_msgpack_type(Model)
    assert config._strict_msgpack_ext_hook(code, data) == Model(value=7)
    with pytest.raises(ValueError, match="Blocked"):
        config._strict_msgpack_ext_hook(code, ormsgpack.packb([]))
    config._register_safe_msgpack_type(None)
    config._register_safe_msgpack_type(type("NoModule", (), {"__module__": ""}))


@pytest.mark.parametrize(
    "blob",
    [
        None,
        [],
        {"t": "unknown", "b64": ""},
        {"t": "msgpack", "b64": 7},
    ],
)
def test_malformed_packed_blobs(blob):
    with pytest.raises(ValueError):
        config.unpack(blob)


def test_empty_and_resume_map_blobs():
    assert config.unpack({"t": "empty", "b64": ""}) is None
    assert config.unpack({"__resume_map__": {"i1": config.pack(True)}}) == {"i1": True}


@pytest.mark.parametrize("entry", [[], [7, ""], ["msgpack", 7]])
def test_malformed_checkpoint_entries(entry):
    with pytest.raises(ValueError):
        inline._decode_typed(entry)


def test_checkpoint_entry_validation_and_non_memory_savers():
    assert inline._decode_typed(["empty", ""]) == ("empty", b"")
    graph = SimpleNamespace(checkpointer=object(), _durable_inline_checkpoints=True)
    assert inline._export_checkpoint_bundle(graph, "t") is None
    inline._hydrate_checkpoint_bundle(graph, "t", {"v": 1, "storage": {}})
    graph = SimpleNamespace(checkpointer=None)
    saver = inline._ensure_checkpointer(graph)
    saver.storage["t"]["empty"] = {}
    assert inline._export_checkpoint_bundle(graph, "missing") is None
    assert inline._export_checkpoint_bundle(graph, "t")["storage"] == {}


@pytest.mark.parametrize(
    "cls,keyword",
    [
        (config.ToolConfig, "durable_tools"),
        (config.NodeConfig, "durable_nodes"),
    ],
)
@pytest.mark.parametrize("option", ["durable", b"local", {"tool": 2}, ["tool"]])
def test_all_supported_policy_option_shapes(cls, keyword, option):
    obj = cls.from_options(**{keyword: option})
    assert obj.default_retries is None
    assert obj.default_durable is (option == "durable")


@pytest.mark.parametrize(
    "cls,keyword,policy,retries",
    [
        (config.ToolConfig, "durable_tools", "tool_policy", "tool_retries"),
        (config.NodeConfig, "durable_nodes", "node_policy", "node_retries"),
    ],
)
def test_policy_retry_and_selector_overrides(cls, keyword, policy, retries):
    obj = cls.from_options(
        **{
            keyword: {},
            policy: {("node", "tool"): 4},
            retries: {"tool": 3},
        }
    )
    assert obj.policy["node::tool"]
    assert obj.retries["tool"] == 3
    assert obj.retries["node::tool"] == 4
    assert cls.from_options(**{keyword: "local", retries: 2}).default_retries == 2
    with pytest.raises(ValueError, match="unsupported"):
        cls.from_options(**{keyword: object()})
    with pytest.raises(ValueError):
        cls(max_concurrency=0)


def test_selector_attributes_and_bad_policy_values():
    assert config._selector_key(SimpleNamespace(tool="t", node="n")) == "n::t"
    assert config._selector_key(SimpleNamespace(tool="t")) == "t"
    assert config._selector_key(("n", "t")) == "n::t"
    assert config._selector_key(7) == "7"
    with pytest.raises(ValueError):
        config._parse_policy_value("typo")
    assert config.mark_node(execution="local", retries=2)(Model)._durable_node is False


@pytest.mark.parametrize(
    "source", ["scoped", "bare", "tool", "marked", "default", "limits"]
)
def test_tool_policy_and_retry_precedence(source):
    tool = SimpleNamespace()
    node = SimpleNamespace(tools_by_name={"tool": tool})
    obj = config.ToolConfig()
    limits = config.Limits(tool_retries=8)
    if source == "scoped":
        obj.policy, obj.retries = {"sub:node::tool": False}, {"sub:node::tool": 2}
    elif source == "bare":
        obj.policy, obj.retries = {"node::tool": False}, {"node::tool": 2}
    elif source == "tool":
        obj.policy, obj.retries = {"tool": False}, {"tool": 2}
    elif source == "marked":
        tool._durable_tool, tool._durable_retries = False, 2
    elif source == "default":
        obj.default_durable, obj.default_retries = False, 2
    result = obj.resolve(node, "sub:node", "tool", limits)
    assert result == ((True, 8) if source == "limits" else (False, 2))
    assert config._effective_tool_config(SimpleNamespace()).default_durable is False


@pytest.mark.parametrize("source", ["scoped", "bare", "marked", "default", "limits"])
def test_node_policy_and_retry_precedence(source):
    candidate = SimpleNamespace()
    graph = SimpleNamespace(
        builder=SimpleNamespace(nodes={"node": candidate}), nodes={}
    )
    obj = config.NodeConfig()
    limits = config.Limits(node_retries=8)
    if source == "scoped":
        obj.policy, obj.retries = {"sub:node": True}, {"sub:node": 2}
    elif source == "bare":
        obj.policy, obj.retries = {"node": True}, {"node": 2}
    elif source == "marked":
        candidate._durable_node, candidate._durable_retries = True, 2
    elif source == "default":
        obj.default_durable, obj.default_retries = True, 2
    assert obj.resolve(graph, "sub:node", limits) == (
        (False, 8) if source == "limits" else (True, 2)
    )
    assert config._effective_node_config(SimpleNamespace()).default_durable is False


def test_node_candidate_wrappers_and_subgraphs():
    wrapper = SimpleNamespace(func=lambda: None, afunc=lambda: None)
    graph = SimpleNamespace(
        builder=SimpleNamespace(nodes={"node": SimpleNamespace(runnable=wrapper)}),
        nodes={"node": SimpleNamespace(bound=wrapper)},
    )
    assert len(config._node_candidates(graph, "node")) == 8
    graph.nodes["node"].subgraphs = [object()]
    assert config._is_tool_or_subgraph_node(graph, "node")
    assert config.NodeConfig().resolve(graph, "node", config.Limits()) == (False, 3)


@pytest.mark.parametrize(
    "name,version,expected",
    [
        ("", "v1", ""),
        ("step", None, "step"),
        ("step", "v1", "step@v1"),
        ("langgraph.g@old.step", None, "langgraph.g.step"),
        ("langgraph.g@old.step", "v2", "langgraph.g@v2.step"),
    ],
)
def test_versioned_activity_names(name, version, expected):
    assert config._versioned_activity_name(name, version) == expected


def test_async_saver_detection_variants():
    assert bridge._is_async_saver(None) is False
    assert bridge._is_async_saver(BaseCheckpointSaver()) is False

    class OnlyAsync(BaseCheckpointSaver):
        async def aget_tuple(self, config):
            return None

    assert bridge._is_async_saver(OnlyAsync())
    loop = asyncio.new_event_loop()
    try:
        assert bridge._is_async_saver(SimpleNamespace(loop=loop))
    finally:
        loop.close()


def test_policy_without_explicit_retries_falls_back_to_limits():
    tool = SimpleNamespace(tools_by_name={"t": SimpleNamespace()})
    cfg = config.ToolConfig.from_options(
        durable_tools={"t": True}, tool_policy={"t": False}
    )
    assert cfg.resolve(tool, "tools", "t", config.Limits()) == (False, 3)
    graph = SimpleNamespace(builder=SimpleNamespace(nodes={}), nodes={})
    cfg = config.NodeConfig.from_options(
        durable_nodes={"n": True}, node_policy={"n": False}
    )
    assert cfg.resolve(graph, "n", config.Limits()) == (False, 3)


def test_safe_method_and_unknown_extensions_delegate(serializer_globals, monkeypatch):
    monkeypatch.setattr(config, "_BASE_STRICT_EXT_HOOK", lambda code, data: "delegated")
    config._configure_msgpack_modules([("datetime",)])
    raw = ormsgpack.packb(["datetime", "datetime", "2026-01-01", "fromisoformat"])
    assert (
        config._strict_msgpack_ext_hook(config.EXT_METHOD_SINGLE_ARG, raw)
        == "delegated"
    )
    assert config._strict_msgpack_ext_hook(99, b"data") == "delegated"
    assert config._parse_policy_value(True) == (True, None)
    assert config._parse_policy_value(False) == (False, None)


def test_checkpoint_bundle_limit_and_missing_parent(monkeypatch):
    from langchain_azure_compute.graph.runner import pack, run_step
    from tests.unit_tests.graph.test_functions_auth import registration

    graph = SimpleNamespace(checkpointer=object(), _durable_inline_checkpoints=False)
    assert inline._export_checkpoint_bundle(graph, "t") is None
    inline._hydrate_checkpoint_bundle(graph, "t", {})
    reg = registration()
    run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "start",
            "expect": None,
            "input": pack({"value": "v"}),
        },
    )
    reg.graph._durable_user_interrupt_after = ["echo"]
    checkpoints = reg.graph.checkpointer.storage["t"][""]
    latest = max(checkpoints)
    for cid in list(checkpoints):
        if cid != latest:
            del checkpoints[cid]
    assert inline._export_checkpoint_bundle(reg.graph, "t")
    monkeypatch.setattr(inline, "MAX_BUNDLE_BYTES", 1)
    with pytest.raises(config.RunError, match="checkpoint_bundle size"):
        inline._export_checkpoint_bundle(reg.graph, "t")


def test_background_loop_worker_cancels_pending_tasks_and_times_out():
    bg = bridge._get_bg_loop()
    graph = SimpleNamespace(checkpointer=None)

    async def child():
        await asyncio.sleep(10)

    async def immediate_with_child():
        asyncio.create_task(child())
        return 42

    async def from_background(coro, timeout):
        return bridge._run_async(graph, coro, timeout=timeout)

    assert (
        asyncio.run_coroutine_threadsafe(
            from_background(immediate_with_child(), 2),
            bg,
        ).result(timeout=3)
        == 42
    )
    with pytest.raises(TimeoutError):
        asyncio.run_coroutine_threadsafe(
            from_background(child(), 0.01),
            bg,
        ).result(timeout=3)


def test_running_saver_loop_and_timeout():
    loop = asyncio.new_event_loop()
    ready = __import__("threading").Event()

    def serve():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = __import__("threading").Thread(target=serve, daemon=True)
    thread.start()
    ready.wait()
    graph = SimpleNamespace(checkpointer=SimpleNamespace(loop=loop))

    async def immediate():
        return 42

    async def slow():
        await asyncio.sleep(10)

    try:
        assert bridge._run_async(graph, immediate()) == 42
        with pytest.raises(TimeoutError, match="timed out"):
            bridge._run_async(graph, slow(), timeout=0.01)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
