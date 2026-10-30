# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Request-local reuse shared by all checkpoint backends."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from contextlib import closing
from copy import copy, deepcopy
from types import SimpleNamespace
from typing import Annotated, Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from azure.ai.agentserver.core import (
    FoundryAgentRequestContext,
    reset_request_context,
    set_request_context,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import ConfigurableFieldSpec, RunnableConfig
from langgraph.channels import DeltaChannel
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.constants import TASKS
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from pydantic import BaseModel, PrivateAttr
from typing_extensions import TypedDict

from langchain_azure_ai.agents.hosting import (
    FoundryCheckpointSaver,
    ResponsesHostServer,
)
from langchain_azure_ai.agents.hosting._checkpoint._checkpoint_reuse import (
    _CheckpointReadContext,
    _RequestCheckpointSaver,
)
from langchain_azure_ai.agents.hosting._instructions import _ResponseInstructions

from .conftest import make_checkpointed_echo_graph
from .test_foundry_checkpoint_saver import (
    _checkpoint,
    _config,
    _credential,
    _FakeStateStore,
)


@pytest.fixture
def saver_and_store(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_RequestCheckpointSaver, _FakeStateStore]:
    store = _FakeStateStore()
    monkeypatch.setattr(
        "langchain_azure_ai.agents.hosting._checkpoint._foundry_checkpoint_saver.FoundryStateStore.get_or_create",
        AsyncMock(return_value=store),
    )
    return _RequestCheckpointSaver(FoundryCheckpointSaver(_credential())), store


@pytest.fixture(params=["memory", "sqlite"])
async def backend(request: pytest.FixtureRequest) -> AsyncIterator[BaseCheckpointSaver]:
    if request.param == "memory":
        yield InMemorySaver()
    else:
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            yield saver


class _StateWithPrivateLock(BaseModel):
    value: str
    _lock: Any = PrivateAttr(default_factory=threading.Lock)


class _JsonWithBinaryTags:
    def dumps_typed(self, value: Any) -> tuple[str, bytes]:
        if isinstance(value, (bytes, bytearray)):
            return type(value).__name__, bytes(value)
        return "json", json.dumps(value).encode("utf-8")

    def loads_typed(self, value: tuple[str, bytes]) -> Any:
        kind, data = value
        if kind == "bytes":
            return data
        if kind == "bytearray":
            return bytearray(data)
        return json.loads(data)


@pytest.mark.parametrize("synchronous", [False, True])
async def test_request_instructions_do_not_change_raw_checkpoint_reads(
    synchronous: bool,
) -> None:
    backend = InMemorySaver()
    saver = _RequestCheckpointSaver(backend)
    instructions = _ResponseInstructions.create("request-only", "response-test")
    config = instructions.with_config(_config())
    child_config: RunnableConfig = {
        **config,
        "configurable": {
            **(config.get("configurable") or {}),
            "checkpoint_ns": "child:task",
        },
    }

    async def read(selected: RunnableConfig) -> CheckpointTuple | None:
        return (
            saver.get_tuple(selected)
            if synchronous
            else await saver.aget_tuple(selected)
        )

    assert await read(child_config) is None
    assert await read(config) is None

    checkpoint = _checkpoint("0001", "input")
    await backend.aput(_config(), checkpoint, {}, checkpoint["channel_versions"])
    with instructions.bind():
        restored = await read(config)
        assert restored is not None
        assert restored.checkpoint == checkpoint
    assert await read(child_config) is None

    other_instructions = _ResponseInstructions.create("other", "other-response")
    with other_instructions.bind():
        assert await read(config) == restored
    await backend.adelete_thread(config["configurable"]["thread_id"])
    assert await read(config) is None


@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("location", ["checkpoint", "pending_writes"])
@pytest.mark.parametrize("messages_source", ["snapshot", "replay", "child"])
async def test_send_overlay_is_isolated_from_saved_state_and_cache(
    synchronous: bool, location: str, messages_source: str
) -> None:
    backend = InMemorySaver()
    checkpoint = _checkpoint("0001", "input")
    config = _config()
    if messages_source != "snapshot":
        del checkpoint["channel_values"]["messages"]
    if messages_source == "child":
        config["configurable"]["checkpoint_ns"] = "child:task"
    messages = [
        SystemMessage(content="explicit-system", id="explicit"),
        HumanMessage(content="saved-user", id="user"),
    ]
    packet = Send("worker", {"messages": messages, "label": "keep"})
    business_packet = Send("business-data", {"messages": messages})
    checkpoint["channel_values"]["business_packet"] = business_packet
    checkpoint["channel_versions"]["business_packet"] = 1
    if location == "checkpoint":
        checkpoint["channel_values"][TASKS] = [packet]
        checkpoint["channel_versions"][TASKS] = 1
    config = await backend.aput(config, checkpoint, {}, checkpoint["channel_versions"])
    if location == "pending_writes":
        await backend.aput_writes(
            config,
            [(TASKS, packet), ("business_packet", business_packet)],
            "dispatch-task",
        )
    saver = _RequestCheckpointSaver(backend)

    async def read(instruction: str | None) -> CheckpointTuple:
        selected = _ResponseInstructions.create(
            instruction, f"response-{instruction}"
        ).with_config(config)
        result = (
            saver.get_tuple(selected)
            if synchronous
            else await saver.aget_tuple(selected)
        )
        assert result is not None
        return result

    def task(value: CheckpointTuple) -> Send:
        if location == "checkpoint":
            return value.checkpoint["channel_values"][TASKS][0]
        assert value.pending_writes is not None
        return next(
            write for _, channel, write in value.pending_writes if channel == TASKS
        )

    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        first = await read("first-instruction")
        second = await read("second-instruction")
        omitted = await read(None)
        if not synchronous:
            assert reads.cache is not None

    for value, expected in (
        (first, ["first-instruction", "explicit-system", "saved-user"]),
        (second, ["second-instruction", "explicit-system", "saved-user"]),
        (omitted, ["explicit-system", "saved-user"]),
    ):
        restored = task(value)
        assert restored.node == packet.node
        assert restored.arg["label"] == "keep"
        assert [message.content for message in restored.arg["messages"]] == expected
        assert (
            value.checkpoint["channel_values"]["business_packet"].arg["messages"]
            == messages
        )
        if location == "pending_writes":
            assert value.pending_writes is not None
            assert all(
                task_id == "dispatch-task" for task_id, _, _ in value.pending_writes
            )
            assert value.pending_writes[1][2].arg["messages"] == messages
        if messages_source != "snapshot":
            assert "messages" not in value.checkpoint["channel_values"]
    saved = await backend.aget_tuple(config)
    assert saved is not None
    assert task(saved).arg["messages"] == messages
    assert packet.arg["messages"] is messages


@pytest.mark.parametrize(
    "payload", [b"input", bytearray(b"input")], ids=["bytes", "bytearray"]
)
async def test_cache_clones_binary_channels_separately_from_header(
    monkeypatch: pytest.MonkeyPatch, payload: bytes | bytearray
) -> None:
    backend = InMemorySaver(serde=_JsonWithBinaryTags())
    checkpoint = _checkpoint("0001", "input")
    checkpoint["channel_values"]["binary"] = payload
    checkpoint["channel_versions"]["binary"] = 1
    config = await backend.aput(
        _config(), checkpoint, {}, checkpoint["channel_versions"]
    )
    await backend.aput_writes(config, [("binary", payload)], "task")
    assert await backend.aget_tuple(config) is not None
    load = AsyncMock(wraps=backend.aget_tuple)
    monkeypatch.setattr(backend, "aget_tuple", load)
    saver = _RequestCheckpointSaver(backend)
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        first = await saver.aget_tuple(config)
        assert first is not None
        first.checkpoint["channel_values"]["messages"].append("modified")
        first.checkpoint["versions_seen"]["node"] = {"binary": 2}
        first_binary = first.checkpoint["channel_values"]["binary"]
        if isinstance(first_binary, bytearray):
            first_binary.extend(b" modified")
        for selected in (_config(), config):
            cached = await saver.aget_tuple(selected)
            assert cached is not None
            assert cached.checkpoint["channel_values"] == {
                "messages": ["input"],
                "binary": payload,
            }
            assert type(cached.checkpoint["channel_values"]["binary"]) is type(payload)
            assert cached.checkpoint["versions_seen"] == {}
            assert cached.pending_writes == [("task", "binary", payload)]
            cached.checkpoint["channel_values"]["messages"].append("modified again")
            cached_binary = cached.checkpoint["channel_values"]["binary"]
            if isinstance(cached_binary, bytearray):
                cached_binary.extend(b" modified again")
        assert (
            load.await_count == 2
        )  # Exact input is cached; latest still reads storage.


async def test_cache_supports_serializable_state_that_cannot_be_deepcopied(
    backend: BaseCheckpointSaver,
) -> None:
    state = _StateWithPrivateLock(value="input")
    checkpoint = _checkpoint("0001", "input")
    checkpoint["channel_values"]["messages"] = [state]
    config = await backend.aput(
        _config(), checkpoint, {}, checkpoint["channel_versions"]
    )
    assert await backend.aget_tuple(config) is not None
    saver = _RequestCheckpointSaver(backend)
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        first = await saver.aget_tuple(config)
        assert first is not None
        first.checkpoint["channel_values"]["messages"][0].value = "modified"
        cached = await saver.aget_tuple(_config())
        assert cached is not None
        assert cached.checkpoint["channel_values"]["messages"][0].value == "input"


async def test_decorator_reuses_any_backend_without_copying_runtime_context(
    backend: BaseCheckpointSaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint("0001", "input")
    config = await backend.aput(
        _config(), checkpoint, {}, checkpoint["channel_versions"]
    )
    await backend.aput_writes(config, [("messages", {"values": [1]})], "task")
    runtime = SimpleNamespace(lock=threading.Lock())
    config["configurable"]["response_context"] = runtime
    load = AsyncMock(wraps=backend.aget_tuple)
    monkeypatch.setattr(backend, "aget_tuple", load)
    graph = make_checkpointed_echo_graph()
    graph.checkpointer = backend
    server = ResponsesHostServer(graph)
    saver = server.graph.checkpointer
    assert isinstance(saver, _RequestCheckpointSaver)
    assert graph.checkpointer is backend
    assert isinstance(saver, BaseCheckpointSaver)
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        first = await saver.aget_tuple(config)
        assert first is not None
        first.checkpoint["channel_values"]["messages"].append("modified")
        assert first.pending_writes is not None
        first.pending_writes[0][2]["values"].append("modified")
        latest = await saver.aget_tuple(_config())
        assert latest is not None
        assert latest.checkpoint["channel_values"]["messages"] == ["input"]
        assert latest.pending_writes == [("task", "messages", {"values": [1]})]
        assert await saver.aget(config) == latest.checkpoint
        if "response_context" in latest.config["configurable"]:
            assert latest.config["configurable"]["response_context"] is runtime
        assert load.await_count == 2


async def test_cache_distinguishes_savers_that_compare_equal() -> None:
    class EqualSaver(InMemorySaver):
        def __eq__(self, other: object) -> bool:
            return isinstance(other, EqualSaver)

    savers = []
    for value in ("first", "second"):
        backend = EqualSaver()
        checkpoint = _checkpoint("0001", value)
        await backend.aput(_config(), checkpoint, {}, checkpoint["channel_versions"])
        savers.append(_RequestCheckpointSaver(backend))
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(savers[0], _config("0001"))
        await savers[0].aget_tuple(_config("0001"))
        second = await savers[1].aget_tuple(_config("0001"))
        assert second is not None
        assert second.checkpoint["channel_values"]["messages"] == ["second"]


async def test_decorator_reuses_cosmos_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cosmos = pytest.importorskip("langchain_azure_cosmosdb.aio")
    container = SimpleNamespace(upsert_item=AsyncMock())
    backend = cosmos.CosmosDBSaver(container)
    config = await backend.aput(_config(), _checkpoint("0001", "input"), {}, {})
    document = container.upsert_item.call_args.args[0]

    async def lookup(query: str, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        if "TOP 1" in query:
            return [{"id": document["id"]}]
        return [document] if "c.id=@checkpoint_key" in query else []

    query = AsyncMock(side_effect=lookup)
    monkeypatch.setattr(backend, "_query_items", query)
    saver = _RequestCheckpointSaver(backend)
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        first = await saver.aget_tuple(config)
        assert first is not None
        assert await saver.aget_tuple(_config()) == first
        assert await saver.aget_tuple(config) == first
        assert (
            query.await_count == 5
        )  # Two exact-load queries plus three latest queries.
        assert sum("TOP 1" in call.args[0] for call in query.await_args_list) == 1


async def test_decorator_preserves_backend_capabilities_and_allowlist_clones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=None))
    config = await backend.aput(_config(), _checkpoint("0001", "input"), {}, {})
    saver = _RequestCheckpointSaver(backend)
    assert saver.serde is backend.serde
    assert saver.config_specs == backend.config_specs
    assert saver.get_next_version == backend.get_next_version
    assert await saver.aget_delta_channel_history(config=config, channels=[]) == {}
    load = AsyncMock(wraps=backend.aget_tuple)
    monkeypatch.setattr(backend, "aget_tuple", load)
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        first = await saver.aget_tuple(config)
        clone = saver.with_allowlist({(__name__, "Payload")})
        assert isinstance(clone, _RequestCheckpointSaver)
        assert clone._self_cache_owner is backend
        assert copy(clone)._self_cache_owner is backend
        assert await clone.aget_tuple(config) == first
        assert load.await_count == 2
    load.reset_mock()
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        assert await clone.aget_tuple(config) == first
        assert await copy(clone).aget_tuple(config) == first
        load.assert_awaited_once()


@pytest.mark.parametrize("write_count", [0, 105])
async def test_request_pins_input_and_returns_independent_snapshots(
    saver_and_store: tuple[_RequestCheckpointSaver, _FakeStateStore], write_count: int
) -> None:
    saver, store = saver_and_store
    config = await saver.aput(_config(), _checkpoint("0001", "input"), {}, {})
    await saver.aput_writes(
        config, [("messages", {"values": [i]}) for i in range(write_count)], "task"
    )
    await saver.aput(config, _checkpoint("0002", "remote"), {}, {})
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        first = await saver.aget_tuple(config)
        assert first is not None
        first.checkpoint["channel_values"]["messages"].append("modified")
        first.metadata["step"] = 999
        if first.pending_writes:
            first.pending_writes[0][2]["values"].append("modified")
        exact = await saver.aget_tuple(config)
        assert exact is not None and exact is not first
        assert exact.checkpoint["channel_values"]["messages"] == ["input"]
        assert exact.metadata.get("step") != 999
        assert exact.pending_writes == [
            ("task", "messages", {"values": [i]}) for i in range(write_count)
        ]
        assert len(store.get_item_calls) == write_count + 1
        assert all(call["tags"]["kind"] == "write" for call in store.list_keys_calls)
        latest = await saver.aget_tuple(_config())
        assert latest is not None and latest.config == _config("0002")
        assert await saver.aget_tuple(config) == exact
        assert len(store.get_item_calls) == write_count + 2
        assert reads.input_checkpoint_id == "0001"
    assert reads.cache is None
    latest = await saver.aget_tuple(_config())
    assert latest is not None and latest.config == _config("0002")


@pytest.mark.parametrize("existing", [False, True])
async def test_checkpointless_reads_always_observe_backend_head(
    saver_and_store: tuple[_RequestCheckpointSaver, _FakeStateStore], existing: bool
) -> None:
    saver, store = saver_and_store
    if existing:
        await saver.aput(_config(), _checkpoint("0001", "input"), {}, {})
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, _config())
        first = await saver.aget_tuple(_config())
        other = FoundryCheckpointSaver(_credential())
        await other.aput(_config(), _checkpoint("0002", "remote"), {}, {})
        latest = await saver.aget_tuple(_config())
        assert latest is not None and latest.config == _config("0002")
        assert first is None or first.config == _config("0001")
        assert reads.cache is None
        assert (
            sum(call["tags"]["kind"] == "checkpoint" for call in store.list_keys_calls)
            == 2
        )


@pytest.mark.parametrize("mutation", ["checkpoint", "writes", "failure", "delete"])
async def test_request_mutations_discard_cached_data_and_track_local_output(
    saver_and_store: tuple[_RequestCheckpointSaver, _FakeStateStore],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    saver, store = saver_and_store
    config = await saver.aput(_config(), _checkpoint("0001", "input"), {}, {})
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        assert await saver.aget_tuple(config) is not None
        if mutation == "checkpoint":
            await saver.aput(config, _checkpoint("0002", "output"), {}, {})
        elif mutation == "writes":
            await saver.aput_writes(config, [("messages", ["output"])], "task")
        elif mutation == "failure":
            with patch.object(store, "set_item", side_effect=RuntimeError("write")):
                with pytest.raises(RuntimeError, match="write"):
                    await saver.aput(config, _checkpoint("0002", "output"), {}, {})
        else:
            monkeypatch.setattr(saver, "_bound_store", lambda _: store)
            await saver.adelete_thread("thread-1")
        assert reads.cache is None
        result = await saver.aget_tuple(_config())
        if mutation == "delete":
            assert result is None
            assert await saver.aget_tuple(config) is None
        else:
            assert result is not None
            expected_id = "0002" if mutation == "checkpoint" else "0001"
            assert result.config == _config(expected_id)
            if mutation == "writes":
                assert result.pending_writes == [("task", "messages", ["output"])]
        assert config == _config("0001")


@pytest.mark.parametrize("boundary", ["saver", "user", "thread", "namespace"])
async def test_request_cache_respects_storage_identity(
    saver_and_store: tuple[_RequestCheckpointSaver, _FakeStateStore], boundary: str
) -> None:
    saver, store = saver_and_store
    config = await saver.aput(_config(), _checkpoint("0001", "input"), {}, {})
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        await saver.aget_tuple(config)
        target: Any = saver
        other_config = _config("0001")
        token = None
        if boundary == "saver":
            target = _RequestCheckpointSaver(FoundryCheckpointSaver(_credential()))
        elif boundary == "user":
            token = set_request_context(FoundryAgentRequestContext(user_id="other"))
        elif boundary == "thread":
            other_config["configurable"]["thread_id"] = "other"
        else:
            other_config["configurable"]["checkpoint_ns"] = "child"
        try:
            await target.aget_tuple(other_config)
        finally:
            if token is not None:
                reset_request_context(token)
        assert len(store.get_item_calls) == 2


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
async def test_failed_request_read_cannot_leave_cached_data(
    saver_and_store: tuple[_RequestCheckpointSaver, _FakeStateStore],
    error: type[BaseException],
) -> None:
    saver, store = saver_and_store
    config = await saver.aput(_config(), _checkpoint("0001", "input"), {}, {})
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        with patch.object(store, "get_item", side_effect=error):
            with pytest.raises(error):
                await saver.aget_tuple(config)
        assert reads.cache is None
        assert await saver.aget_tuple(config) is not None
        assert await saver.aget_tuple(config) is not None
        assert len(store.get_item_calls) == 2


async def test_inflight_read_cannot_refill_cache_after_mutation(
    saver_and_store: tuple[_RequestCheckpointSaver, _FakeStateStore],
) -> None:
    saver, store = saver_and_store
    config = await saver.aput(_config(), _checkpoint("0001", "input"), {}, {})
    started, release = asyncio.Event(), asyncio.Event()
    get_item = store.get_item

    async def delayed(key: str) -> Any:
        started.set()
        await release.wait()
        return await get_item(key)

    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        with patch.object(store, "get_item", delayed):
            task = asyncio.create_task(saver.aget_tuple(config))
            try:
                await asyncio.wait_for(started.wait(), 2)
                await saver.aput_writes(config, [("messages", "output")], "task")
            finally:
                release.set()
                await task
        assert reads.cache is None
        result = await saver.aget_tuple(config)
        assert result is not None
        assert result.pending_writes == [("task", "messages", "output")]


def _append_deltas(state: list[str], updates: list[list[str]]) -> list[str]:
    return state + [item for update in updates for item in update]


class _DeltaState(TypedDict):
    items: Annotated[list[str], DeltaChannel(_append_deltas)]


async def test_historical_update_preserves_existing_delta_descendant() -> None:
    backend = InMemorySaver()
    builder = StateGraph(_DeltaState)
    builder.add_node("step", lambda state: {"items": ["B"]})
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    graph = builder.compile(checkpointer=backend)
    await graph.ainvoke({"items": ["A"]}, _config())
    head = await graph.aget_state(_config())
    ancestor = head.parent_config
    assert ancestor is not None
    saver = _RequestCheckpointSaver(backend)
    wrapped = graph.copy({"checkpointer": saver})
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, ancestor)
        branch = await wrapped.aupdate_state(
            ancestor,
            {"items": ["C"]},
            as_node="step",
            task_id="00000000-0000-0000-0000-000000000001",
        )
    assert (await graph.aget_state(head.config)).values["items"] == ["A", "B"]
    assert (await graph.aget_state(branch)).values["items"] == ["A", "C"]


async def test_failed_turn_resume_replays_a_superseded_checkpoint() -> None:
    class State(TypedDict):
        approved: bool
        done: bool

    approvals = 0
    finishes = 0

    def approve(state: State) -> dict[str, bool]:
        nonlocal approvals
        answer = interrupt("Approve?")
        approvals += 1
        return {"approved": answer}

    def finish(state: State) -> dict[str, bool]:
        nonlocal finishes
        finishes += 1
        if finishes == 1:
            raise RuntimeError("failed turn")
        return {"done": True}

    backend = InMemorySaver()
    builder = StateGraph(State)
    builder.add_node("approve", approve)
    builder.add_node("finish", finish)
    builder.add_edge(START, "approve")
    builder.add_edge("approve", "finish")
    builder.add_edge("finish", END)
    graph = builder.compile(checkpointer=backend)
    await graph.ainvoke({"approved": False, "done": False}, _config())
    paused = (await graph.aget_state(_config())).config
    with pytest.raises(RuntimeError, match="failed turn"):
        await graph.ainvoke(Command(resume=True), paused)
    saver = _RequestCheckpointSaver(backend)
    wrapped = graph.copy({"checkpointer": saver})
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, paused)
        result = await wrapped.ainvoke(Command(resume=True), paused)
    assert result["done"]
    assert approvals == finishes == 2


@pytest.mark.parametrize("failure", ["encode", "decode"])
async def test_cache_copy_failure_falls_back_to_backend(
    backend: BaseCheckpointSaver, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    checkpoint = _checkpoint("0001", "input")
    config = await backend.aput(
        _config(), checkpoint, {}, checkpoint["channel_versions"]
    )
    saver = _RequestCheckpointSaver(backend)
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        if failure == "decode":
            await saver.aget_tuple(config)
            original = backend.serde.loads_typed
            failed = False

            def fail_once(value: tuple[str, bytes]) -> Any:
                nonlocal failed
                if not failed:
                    failed = True
                    raise ValueError("cache decode")
                return original(value)

            monkeypatch.setattr(backend.serde, "loads_typed", fail_once)
        else:
            monkeypatch.setattr(
                backend.serde,
                "dumps_typed",
                Mock(side_effect=ValueError("cache encode")),
            )
        for _ in range(2):
            result = await saver.aget_tuple(config)
            assert result is not None
            assert result.checkpoint["channel_values"]["messages"] == ["input"]
        assert reads.cache is None and not reads.cache_enabled


async def test_cache_hits_only_deserialize_and_use_current_config(
    backend: BaseCheckpointSaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint("0001", "input")
    config = await backend.aput(
        _config(), checkpoint, {}, checkpoint["channel_versions"]
    )
    encode = Mock(wraps=backend.serde.dumps_typed)
    monkeypatch.setattr(backend.serde, "dumps_typed", encode)
    saver = _RequestCheckpointSaver(backend)
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        first: RunnableConfig = {
            **config,
            "configurable": {**config["configurable"], "marker": "first"},
        }
        await saver.aget_tuple(first)
        calls = encode.call_count
        second: RunnableConfig = {
            **config,
            "configurable": {**config["configurable"], "marker": "second"},
        }
        cached = await saver.aget_tuple(second)
        assert cached is not None
        assert cached.config["configurable"]["marker"] == "second"
        assert encode.call_count == calls
        bare = await saver.aget_tuple(config)
        assert bare is not None and "marker" not in bare.config["configurable"]


@pytest.mark.parametrize("reference_name", ["config", "parent_config"])
@pytest.mark.parametrize(
    "locator", [{"paths": ["west"]}, ["west"]], ids=["dict", "list"]
)
async def test_restored_locator_values_do_not_alias_cached_references(
    monkeypatch: pytest.MonkeyPatch,
    reference_name: str,
    locator: dict[str, list[str]] | list[str],
) -> None:
    class LocatorSaver(InMemorySaver):
        @property
        def config_specs(self) -> list[ConfigurableFieldSpec]:
            return [ConfigurableFieldSpec(id="locator", annotation=type(locator))]

        async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
            result = await super().aget_tuple(config)
            if result is not None and result.parent_config is not None:
                result.parent_config["configurable"]["locator"] = deepcopy(
                    config["configurable"]["locator"]
                )
            return result

    backend = LocatorSaver()
    config = _config()
    for version in (1, 2):
        checkpoint = _checkpoint(f"{version:04d}", "input")
        checkpoint["channel_versions"]["messages"] = version
        config = await backend.aput(
            config, checkpoint, {}, checkpoint["channel_versions"]
        )
        config["configurable"]["locator"] = deepcopy(locator)
    runtime = SimpleNamespace(lock=threading.Lock())
    config["configurable"]["response_context"] = runtime
    load = AsyncMock(wraps=backend.aget_tuple)
    monkeypatch.setattr(backend, "aget_tuple", load)
    saver = _RequestCheckpointSaver(backend)
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        await saver.aget_tuple(config)
        first_hit = await saver.aget_tuple(config)
        assert first_hit is not None
        reference = getattr(first_hit, reference_name)
        value = reference["configurable"]["locator"]
        if isinstance(value, dict):
            value["paths"].append("changed")
        else:
            value.append("changed")
        second_hit = await saver.aget_tuple(config)
        assert second_hit is not None
        restored = getattr(second_hit, reference_name)
        assert restored["configurable"]["locator"] == locator
        assert config["configurable"]["locator"] == locator
        assert second_hit.config["configurable"]["response_context"] is runtime
        load.assert_awaited_once()


@pytest.mark.parametrize("missing_first", [False, True])
async def test_cache_preserves_backend_locator_fields_without_cross_partition_reuse(
    monkeypatch: pytest.MonkeyPatch, missing_first: bool
) -> None:
    class PartitionSaver(InMemorySaver):
        @property
        def config_specs(self) -> list[ConfigurableFieldSpec]:
            return [ConfigurableFieldSpec(id="partition", annotation=str)]

        async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
            if missing_first and config["configurable"]["partition"] == "first":
                return None
            result = await super().aget_tuple(config)
            if result is not None:
                result.checkpoint["channel_values"]["messages"] = [
                    config["configurable"]["partition"]
                ]
            return result

    backend = PartitionSaver()
    checkpoint = _checkpoint("0001", "input")
    config = await backend.aput(
        _config(), checkpoint, {}, checkpoint["channel_versions"]
    )
    load = AsyncMock(wraps=backend.aget_tuple)
    monkeypatch.setattr(backend, "aget_tuple", load)
    saver = _RequestCheckpointSaver(backend)
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        for partition in ("first", "second", "second"):
            selected: RunnableConfig = {
                **config,
                "configurable": {**config["configurable"], "partition": partition},
            }
            result = await saver.aget_tuple(selected)
            if missing_first and partition == "first":
                assert result is None
            else:
                assert result is not None
                assert result.checkpoint["channel_values"]["messages"] == [partition]
                assert result.config["configurable"]["partition"] == partition
        assert load.await_count == 2


@pytest.mark.parametrize(
    "method",
    [
        "put",
        "put_writes",
        "delete_thread",
        "delete_for_runs",
        "copy_thread",
        "prune",
        "aput",
        "aput_writes",
        "adelete_thread",
        "adelete_for_runs",
        "acopy_thread",
        "aprune",
    ],
)
async def test_all_mutators_invalidate_cached_state(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    backend = InMemorySaver()
    checkpoint = _checkpoint("0001", "input")
    config = await backend.aput(
        _config(), checkpoint, {}, checkpoint["channel_versions"]
    )
    write = backend.put_writes

    def mutate(*args: Any, **kwargs: Any) -> Any:
        write(config, [("messages", "changed")], "mutator")
        return config if method in {"put", "aput"} else None

    operation = (
        AsyncMock(side_effect=mutate)
        if method.startswith("a")
        else Mock(side_effect=mutate)
    )
    monkeypatch.setattr(backend, method, operation)
    saver = _RequestCheckpointSaver(backend)
    plain = method.removeprefix("a")
    arguments: dict[str, tuple[Any, ...]] = {
        "put": (config, checkpoint, {}, {}),
        "put_writes": (config, [("messages", "changed")], "mutator"),
        "delete_thread": ("thread-1",),
        "delete_for_runs": (["run"],),
        "copy_thread": ("other", "thread-1"),
        "prune": (["thread-1"],),
    }
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        await saver.aget_tuple(config)
        result = getattr(saver, method)(*arguments[plain])
        if method.startswith("a"):
            await result
        fresh = await saver.aget_tuple(config)
        assert fresh is not None
        assert fresh.pending_writes == [("mutator", "messages", "changed")]


@pytest.mark.parametrize("method", ["put_writes", "aput_writes"])
async def test_task_path_is_forwarded_through_keyword_arguments(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    backend = InMemorySaver()
    operation = AsyncMock() if method.startswith("a") else Mock()
    monkeypatch.setattr(backend, method, operation)
    saver = _RequestCheckpointSaver(backend)
    result = getattr(saver, method)(_config("0001"), [], "task", task_path="node")
    if method.startswith("a"):
        await result
    assert operation.call_args.kwargs == {"task_path": "node"}
