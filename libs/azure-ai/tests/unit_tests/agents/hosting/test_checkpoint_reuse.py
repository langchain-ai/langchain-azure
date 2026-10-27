# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Request-local reuse shared by all checkpoint backends."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from contextlib import closing
from copy import copy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from azure.ai.agentserver.core import (
    FoundryAgentRequestContext,
    reset_request_context,
    set_request_context,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, PrivateAttr

from langchain_azure_ai.agents.hosting import (
    FoundryCheckpointSaver,
    ResponsesHostServer,
)
from langchain_azure_ai.agents.hosting._checkpoint._checkpoint_reuse import (
    _CheckpointReadContext,
    _RequestCheckpointSaver,
)

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
        load.assert_awaited_once()


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
        load.assert_awaited_once()


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
    query = AsyncMock(side_effect=[[document], []])
    monkeypatch.setattr(backend, "_query_items", query)
    saver = _RequestCheckpointSaver(backend)
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        first = await saver.aget_tuple(config)
        assert first is not None
        assert await saver.aget_tuple(_config()) == first
        assert await saver.aget_tuple(config) == first
        assert (
            query.await_count == 2
        )  # One checkpoint query and one pending-write query.
        assert all("TOP 1" not in call.args[0] for call in query.await_args_list)


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
        assert await clone.aget_tuple(_config()) == first
        assert load.await_count == 2
    load.reset_mock()
    with closing(_CheckpointReadContext()) as reads, reads.bind():
        reads.pin(saver, config)
        assert await clone.aget_tuple(config) == first
        assert await copy(clone).aget_tuple(_config()) == first
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
        latest = await saver.aget_tuple(_config())
        assert exact == latest
        assert exact is not None and exact is not latest
        assert exact.checkpoint["channel_values"]["messages"] == ["input"]
        assert exact.metadata.get("step") != 999
        assert exact.pending_writes == [
            ("task", "messages", {"values": [i]}) for i in range(write_count)
        ]
        assert len(store.get_item_calls) == write_count + 1
        assert all(call["tags"]["kind"] == "write" for call in store.list_keys_calls)
    assert reads.cache is None
    latest = await saver.aget_tuple(_config())
    assert latest is not None and latest.config == _config("0002")


@pytest.mark.parametrize("existing", [False, True])
async def test_request_without_reference_resolves_input_only_once(
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
        assert await saver.aget_tuple(_config()) == first
        assert len(store.list_keys_calls) == (2 if existing else 1)


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
