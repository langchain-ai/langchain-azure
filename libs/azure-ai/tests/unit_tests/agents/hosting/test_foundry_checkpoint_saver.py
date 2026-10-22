# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the Foundry-backed LangGraph checkpoint saver."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from azure.core.credentials_async import AsyncTokenCredential
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from langchain_azure_ai._user_agent import get_user_agent
from langchain_azure_ai.agents.hosting import FoundryCheckpointSaver, HostingFeature
from langchain_azure_ai.agents.hosting._foundry_checkpoint_saver import (
    DEFAULT_STORE_NAME_PREFIX,
)


class _GraphState(TypedDict):
    value: int


@pytest.fixture
def state_store() -> Iterator[_FakeStateStore]:
    store = _FakeStateStore()
    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver."
        "FoundryStateStore.get_or_create",
        new=AsyncMock(return_value=store),
    ):
        yield store


def _credential() -> AsyncTokenCredential:
    return cast(AsyncTokenCredential, object())


class _FakeStateStore:
    def __init__(self) -> None:
        self.name = "langGraphCheckpoints/thread-1"
        self.user_isolation = True
        self.item_ttl_seconds = 30 * 24 * 60 * 60
        self.items: dict[str, Any] = {}
        self.tags: dict[str, dict[str, str]] = {}
        self.order: list[str] = []
        self.list_keys_calls: list[dict[str, Any]] = []
        self.closed = False
        self.deleted = False

    async def get(self) -> Any:
        return SimpleNamespace(
            user_isolation=self.user_isolation,
            item_ttl_seconds=self.item_ttl_seconds,
        )

    async def set_item(
        self,
        key: str,
        value: dict[str, Any],
        *,
        tags: dict[str, str],
    ) -> None:
        if key not in self.items:
            self.order.append(key)
            item_id = f"item-{len(self.order)}"
        else:
            item_id = self.items[key].id
        self.items[key] = SimpleNamespace(id=item_id, value=value)
        self.tags[key] = tags

    async def create_item(
        self,
        key: str,
        value: dict[str, Any],
        *,
        tags: dict[str, str],
    ) -> None:
        assert key not in self.items
        await self.set_item(key, value, tags=tags)

    async def get_item(self, key: str) -> Any | None:
        return self.items.get(key)

    async def list_keys(
        self,
        *,
        tags: dict[str, str] | None = None,
        limit: int,
        order: str,
        after: str | None = None,
        before: str | None = None,
    ) -> Any:
        assert after is None or before is None
        self.list_keys_calls.append(
            {"tags": tags, "order": order, "after": after, "before": before}
        )
        ordered_keys = list(self.order)
        if order == "desc":
            ordered_keys.reverse()
        if after is not None:
            cursor_index = next(
                index
                for index, key in enumerate(ordered_keys)
                if self.items[key].id == after
            )
            ordered_keys = ordered_keys[cursor_index + 1 :]
        if before is not None:
            cursor_index = next(
                index
                for index, key in enumerate(ordered_keys)
                if self.items[key].id == before
            )
            ordered_keys = ordered_keys[:cursor_index]
        matching_keys = [
            key
            for key in ordered_keys
            if tags is None
            or all(self.tags[key].get(name) == value for name, value in tags.items())
        ]
        keys = matching_keys[:limit]
        return SimpleNamespace(
            keys=[SimpleNamespace(id=self.items[key].id, key=key) for key in keys],
            has_more=len(matching_keys) > limit,
            last_id=self.items[keys[-1]].id if keys else None,
        )

    async def delete(self) -> None:
        self.deleted = True

    async def delete_item(self, key: str) -> None:
        self.items.pop(key, None)
        self.tags.pop(key, None)
        if key in self.order:
            self.order.remove(key)

    async def aclose(self) -> None:
        self.closed = True


def _config(
    checkpoint_id: str | None = None,
    *,
    checkpoint_ns: str = "",
) -> RunnableConfig:
    configurable = {
        "thread_id": "thread-1",
        "checkpoint_ns": checkpoint_ns,
    }
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return cast(RunnableConfig, {"configurable": configurable})


def _checkpoint(checkpoint_id: str, value: str) -> Checkpoint:
    return cast(
        Checkpoint,
        {
            "v": 2,
            "id": checkpoint_id,
            "ts": "2026-08-05T00:00:00+00:00",
            "channel_values": {"messages": [value]},
            "channel_versions": {"messages": 1},
            "versions_seen": {},
            "updated_channels": ["messages"],
        },
    )


def test_constructor_registers_foundry_checkpoint_feature() -> None:
    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver."
        "_add_process_hosting_features"
    ) as add_process_features:
        saver = FoundryCheckpointSaver(_credential())

    assert saver._hosting_features == HostingFeature.FOUNDRY_CHECKPOINT
    add_process_features.assert_called_once_with(HostingFeature.FOUNDRY_CHECKPOINT)


@pytest.mark.asyncio
async def test_put_stamps_state_store_user_agent() -> None:
    store = _FakeStateStore()
    credential = _credential()

    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver."
        "FoundryStateStore.get_or_create",
        new=AsyncMock(return_value=store),
    ) as get_or_create:
        saver = FoundryCheckpointSaver(
            credential,
            "https://example.services.ai.azure.com/api/projects/project",
        )
        result = await saver.aput(
            _config(),
            _checkpoint("checkpoint-1", "first"),
            cast(CheckpointMetadata, {"source": "input", "step": -1}),
            {},
        )

    assert result == _config("checkpoint-1")
    get_or_create.assert_awaited_once()
    assert get_or_create.call_args.args[0] == f"{DEFAULT_STORE_NAME_PREFIX}/thread-1"
    callback = get_or_create.call_args.kwargs["get_server_version"]
    assert callback is get_user_agent
    assert callback() == get_user_agent()
    assert store.tags["/checkpoint-1"] == {
        "kind": "checkpoint",
        "ns": "",
        "source": "input",
        "step": "-1",
    }


@pytest.mark.asyncio
async def test_rejects_incompatible_existing_store() -> None:
    store = _FakeStateStore()
    store.user_isolation = False
    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver."
        "FoundryStateStore.get_or_create",
        new=AsyncMock(return_value=store),
    ):
        saver = FoundryCheckpointSaver(_credential(), user_isolation=True)
        with pytest.raises(ValueError, match="user_isolation=False"):
            await saver.aget_tuple(_config())

    assert store.closed is True


@pytest.mark.asyncio
async def test_overlong_thread_id_uses_bounded_stable_store_name() -> None:
    store = _FakeStateStore()
    get_or_create = AsyncMock(return_value=store)
    long_config = cast(
        RunnableConfig,
        {"configurable": {"thread_id": "thread-" + "x" * 200}},
    )
    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver."
        "FoundryStateStore.get_or_create",
        new=get_or_create,
    ):
        saver = FoundryCheckpointSaver(_credential())
        await saver.aget_tuple(long_config)

    store_name = get_or_create.call_args.args[0]
    assert len(store_name) <= 128
    assert store_name.startswith(f"{DEFAULT_STORE_NAME_PREFIX}/")
    assert store_name == saver._store_name(long_config["configurable"]["thread_id"])


@pytest.mark.asyncio
async def test_round_trip_latest_history_and_pending_writes() -> None:
    store = _FakeStateStore()
    get_or_create = AsyncMock(return_value=store)

    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver."
        "FoundryStateStore.get_or_create",
        new=get_or_create,
    ):
        saver = FoundryCheckpointSaver(_credential())
        first_config = await saver.aput(
            _config(),
            _checkpoint("checkpoint-1", "first"),
            cast(CheckpointMetadata, {"source": "input", "step": -1}),
            {},
        )
        await saver.aput_writes(
            first_config,
            [("messages", {"content": "pending"})],
            "task-1",
        )
        second_config = await saver.aput(
            first_config,
            _checkpoint("checkpoint-2", "second"),
            cast(CheckpointMetadata, {"source": "loop", "step": 0}),
            {},
        )

        latest = await saver.aget_tuple(_config())
        exact = await saver.aget_tuple(first_config)
        history = [item async for item in saver.alist(_config())]
        before = [item async for item in saver.alist(_config(), before=second_config)]
        filtered = [
            item
            async for item in saver.alist(
                _config(),
                filter={"source": "input"},
            )
        ]

    assert latest is not None
    assert latest.config == second_config
    assert latest.checkpoint["channel_values"] == {"messages": ["second"]}
    assert latest.parent_config == first_config
    assert exact is not None
    assert exact.pending_writes == [("task-1", "messages", {"content": "pending"})]
    assert [item.config for item in history] == [second_config, first_config]
    assert [item.config for item in before] == [first_config]
    assert [item.config for item in filtered] == [first_config]
    assert get_or_create.await_count == 8


@pytest.mark.asyncio
async def test_list_before_uses_checkpoint_id_order_not_creation_order() -> None:
    store = _FakeStateStore()

    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver."
        "FoundryStateStore.get_or_create",
        new=AsyncMock(return_value=store),
    ):
        saver = FoundryCheckpointSaver(_credential())
        older_config = await saver.aput(
            _config(),
            _checkpoint("checkpoint-1", "older"),
            cast(CheckpointMetadata, {"source": "input", "step": -1}),
            {},
        )
        before_config = await saver.aput(
            older_config,
            _checkpoint("checkpoint-3", "before"),
            cast(CheckpointMetadata, {"source": "loop", "step": 0}),
            {},
        )
        middle_config = await saver.aput(
            before_config,
            _checkpoint("checkpoint-2", "middle"),
            cast(CheckpointMetadata, {"source": "loop", "step": 1}),
            {},
        )

        checkpoints = [
            item
            async for item in saver.alist(
                _config(),
                before=before_config,
            )
        ]

    assert [item.config for item in checkpoints] == [middle_config, older_config]


@pytest.mark.asyncio
async def test_latest_and_history_ignore_storage_creation_order(
    state_store: _FakeStateStore,
) -> None:
    saver = FoundryCheckpointSaver(_credential())
    configurations = {}
    for checkpoint_id in ("checkpoint-3", "checkpoint-1", "checkpoint-2"):
        configurations[checkpoint_id] = await saver.aput(
            _config(),
            _checkpoint(checkpoint_id, checkpoint_id),
            cast(CheckpointMetadata, {"source": "loop", "step": 0}),
            {},
        )

    latest = await saver.aget_tuple(_config())
    assert latest is not None
    assert latest.config == configurations["checkpoint-3"]
    history = [item async for item in saver.alist(_config())]
    assert [item.checkpoint["id"] for item in history] == [
        "checkpoint-3",
        "checkpoint-2",
        "checkpoint-1",
    ]
    assert state_store.closed


@pytest.mark.asyncio
async def test_listing_orders_across_pages_before_applying_limit(
    state_store: _FakeStateStore,
) -> None:
    saver = FoundryCheckpointSaver(_credential())
    for checkpoint_id in (
        "checkpoint-5",
        "checkpoint-2",
        "checkpoint-4",
        "checkpoint-1",
        "checkpoint-3",
    ):
        await saver.aput(
            _config(),
            _checkpoint(checkpoint_id, checkpoint_id),
            cast(CheckpointMetadata, {"source": "loop", "step": 0}),
            {},
        )
    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver._PAGE_SIZE", 2
    ):
        latest = await saver.aget_tuple(_config())
        history = [item async for item in saver.alist(_config(), limit=2)]
    assert latest is not None and latest.checkpoint["id"] == "checkpoint-5"
    assert [item.checkpoint["id"] for item in history] == [
        "checkpoint-5",
        "checkpoint-4",
    ]
    assert any(call["after"] is not None for call in state_store.list_keys_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary_exists", [False, True])
async def test_before_is_an_id_bound_not_a_storage_cursor(
    state_store: _FakeStateStore,
    boundary_exists: bool,
) -> None:
    saver = FoundryCheckpointSaver(_credential())
    for checkpoint_id in ("checkpoint-1", "checkpoint-4", "checkpoint-2"):
        await saver.aput(
            _config(),
            _checkpoint(checkpoint_id, checkpoint_id),
            cast(CheckpointMetadata, {"source": "loop", "step": 0}),
            {},
        )
    if boundary_exists:
        await saver.aput(
            _config(),
            _checkpoint("checkpoint-3", "boundary"),
            cast(CheckpointMetadata, {"source": "input", "step": 1}),
            {},
        )
    history = [
        item
        async for item in saver.alist(
            _config(),
            before=_config("checkpoint-3"),
            filter={"source": "loop"},
        )
    ]
    assert [item.checkpoint["id"] for item in history] == [
        "checkpoint-2",
        "checkpoint-1",
    ]


@pytest.mark.asyncio
async def test_listing_preserves_namespace_metadata_and_pending_writes(
    state_store: _FakeStateStore,
) -> None:
    saver = FoundryCheckpointSaver(_credential())
    root = await saver.aput(
        _config(),
        _checkpoint("checkpoint-1", "root"),
        cast(CheckpointMetadata, {"source": "input", "step": 0, "owner": "match"}),
        {},
    )
    child = await saver.aput(
        _config(checkpoint_ns="child"),
        _checkpoint("checkpoint-3", "child"),
        cast(CheckpointMetadata, {"source": "loop", "step": 1, "owner": "match"}),
        {},
    )
    await saver.aput(
        _config(),
        _checkpoint("checkpoint-2", "other"),
        cast(CheckpointMetadata, {"source": "loop", "step": 1, "owner": "other"}),
        {},
    )
    await saver.aput_writes(child, [("value", 3)], "task-child")
    all_namespaces = cast(RunnableConfig, {"configurable": {"thread_id": "thread-1"}})
    history = [
        item
        async for item in saver.alist(
            all_namespaces, filter={"owner": "match"}, limit=2
        )
    ]
    assert [item.config for item in history] == [child, root]
    assert history[0].pending_writes == [("task-child", "value", 3)]
    root_latest = await saver.aget_tuple(_config())
    assert root_latest is not None and root_latest.checkpoint["id"] == "checkpoint-2"
    exact = [item async for item in saver.alist(root)]
    assert [item.config for item in exact] == [root]


@pytest.mark.asyncio
async def test_latest_skips_a_disappearing_storage_entry(
    state_store: _FakeStateStore,
) -> None:
    saver = FoundryCheckpointSaver(_credential())
    first = await saver.aput(
        _config(),
        _checkpoint("checkpoint-1", "first"),
        cast(CheckpointMetadata, {"source": "loop", "step": 0}),
        {},
    )
    await saver.aput(
        first,
        _checkpoint("checkpoint-2", "expired"),
        cast(CheckpointMetadata, {"source": "loop", "step": 1}),
        {},
    )
    get_item = state_store.get_item

    async def missing_latest(key: str) -> Any:
        return None if key == "/checkpoint-2" else await get_item(key)

    with patch.object(state_store, "get_item", side_effect=missing_latest):
        latest = await saver.aget_tuple(_config())
    assert latest is not None and latest.config == first


@pytest.mark.asyncio
async def test_pending_writes_follow_task_path_across_pages_and_legacy_records(
    state_store: _FakeStateStore,
) -> None:
    saver = FoundryCheckpointSaver(_credential())
    configuration = await saver.aput(
        _config(),
        _checkpoint("checkpoint-1", "first"),
        cast(CheckpointMetadata, {"source": "loop", "step": 0}),
        {},
    )
    await saver.aput_writes(configuration, [("value", "right")], "task-a", "2")
    await saver.aput_writes(
        configuration,
        [("value", "left-1"), ("value", "left-2")],
        "task-z",
        "1",
    )
    await saver.aput_writes(configuration, [("value", "legacy")], "task-m")
    state_store.items["/writes/checkpoint-1/task-m/0"].value.pop("task_path")
    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver._PAGE_SIZE", 2
    ):
        checkpoint = await saver.aget_tuple(configuration)
    assert checkpoint is not None
    assert checkpoint.pending_writes == [
        ("task-m", "value", "legacy"),
        ("task-z", "value", "left-1"),
        ("task-z", "value", "left-2"),
        ("task-a", "value", "right"),
    ]


@pytest.mark.asyncio
async def test_delta_channel_restores_parallel_reducer_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channels = pytest.importorskip("langgraph.channels")
    delta_channel = getattr(channels, "DeltaChannel", None)
    if delta_channel is None:
        pytest.skip("DeltaChannel is unavailable in this LangGraph version")
    delta_channel_type = cast(Any, delta_channel)
    monkeypatch.delenv("FOUNDRY_HOSTING_ENVIRONMENT", raising=False)
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))

    def concatenate(current: list[str], updates: list[list[str]]) -> list[str]:
        return current + [value for update in updates for value in update]

    class State(TypedDict):
        values: list[str]

    State.__annotations__["values"] = Annotated[
        list[str], delta_channel_type(concatenate)
    ]

    def build(saver: FoundryCheckpointSaver) -> Any:
        builder = StateGraph(State)
        for label in ("a", "b", "c", "d", "e", "f", "g", "h"):

            async def branch(state: State, value: str = label) -> State:
                return {"values": [value]}

            builder.add_node(label, branch)
            builder.add_edge(START, label)
            builder.add_edge(label, END)
        return builder.compile(checkpointer=saver)

    configuration = cast(
        RunnableConfig, {"configurable": {"thread_id": "delta-thread"}}
    )
    async with FoundryCheckpointSaver(_credential()) as saver:
        expected = await build(saver).ainvoke({"values": []}, configuration)
    async with FoundryCheckpointSaver(_credential()) as saver:
        restored = await build(saver).aget_state(configuration)
    assert expected["values"] == list("abcdefgh")
    assert restored.values == expected


@pytest.mark.asyncio
async def test_storage_listing_errors_are_not_hidden(
    state_store: _FakeStateStore,
) -> None:
    saver = FoundryCheckpointSaver(_credential())
    with patch.object(
        state_store, "list_keys", side_effect=RuntimeError("storage failed")
    ):
        with pytest.raises(RuntimeError, match="storage failed"):
            await saver.aget_tuple(_config())
    assert state_store.closed


@pytest.mark.asyncio
async def test_local_checkpoint_order_and_resume_survive_saver_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FOUNDRY_HOSTING_ENVIRONMENT", raising=False)
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "azure.ai.agentserver.core.storage._local_state._now", lambda: 100
    )
    expected = [f"checkpoint-{index:03d}" for index in range(12)]
    async with FoundryCheckpointSaver(_credential()) as saver:
        configuration = _config()
        for checkpoint_id in expected:
            configuration = await saver.aput(
                configuration,
                _checkpoint(checkpoint_id, checkpoint_id),
                cast(CheckpointMetadata, {"source": "loop", "step": 0}),
                {},
            )
    async with FoundryCheckpointSaver(_credential()) as reopened:
        latest = await reopened.aget_tuple(_config())
        assert latest is not None and latest.checkpoint["id"] == expected[-1]
        history = [item async for item in reopened.alist(_config())]
        assert [item.checkpoint["id"] for item in history] == list(reversed(expected))

    executions = []

    async def reviewed(state: _GraphState) -> _GraphState:
        approved = interrupt("approve?")
        if approved:
            executions.append(state["value"])
        return {"value": state["value"] + int(approved)}

    def build(saver: FoundryCheckpointSaver) -> Any:
        builder = StateGraph(_GraphState)
        builder.add_node("reviewed", reviewed)
        builder.add_edge(START, "reviewed")
        builder.add_edge("reviewed", END)
        return builder.compile(checkpointer=saver)

    configuration = cast(
        RunnableConfig, {"configurable": {"thread_id": "resume-thread"}}
    )
    async with FoundryCheckpointSaver(_credential()) as saver:
        await build(saver).ainvoke({"value": 10}, configuration)
    assert executions == []
    async with FoundryCheckpointSaver(_credential()) as reopened:
        graph = build(reopened)
        paused = await graph.aget_state(configuration)
        assert paused.interrupts
        result = await graph.ainvoke(Command(resume=True), configuration)
    assert result["value"] == 11
    assert executions == [10]


@pytest.mark.asyncio
async def test_delete_user_isolated_thread_deletes_items() -> None:
    store = _FakeStateStore()
    await store.set_item(
        "/checkpoint-1",
        {"checkpoint_id": "checkpoint-1"},
        tags={"kind": "checkpoint", "ns": ""},
    )
    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver.FoundryStateStore",
        return_value=store,
    ):
        saver = FoundryCheckpointSaver(_credential())
        await saver.adelete_thread("thread-1")

    assert store.deleted is False
    assert store.items == {}
    assert store.closed is True


@pytest.mark.asyncio
async def test_delete_thread_stamps_user_agent() -> None:
    store = SimpleNamespace(delete=AsyncMock(), aclose=AsyncMock())
    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver.FoundryStateStore",
        return_value=store,
    ) as state_store:
        saver = FoundryCheckpointSaver(_credential(), user_isolation=False)
        await saver.adelete_thread("thread-1")

    assert state_store.call_args.kwargs["get_server_version"] is get_user_agent
    store.delete.assert_awaited_once()
    store.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_saver_runs_with_real_langgraph() -> None:
    store = _FakeStateStore()
    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver."
        "FoundryStateStore.get_or_create",
        new=AsyncMock(return_value=store),
    ):
        saver = FoundryCheckpointSaver(_credential())

        async def increment(state: _GraphState) -> _GraphState:
            return {"value": state["value"] + 1}

        builder = StateGraph(_GraphState)
        builder.add_node("increment", increment)
        builder.add_edge(START, "increment")
        builder.add_edge("increment", END)
        graph = builder.compile(checkpointer=saver)
        config = cast(
            RunnableConfig,
            {"configurable": {"thread_id": "thread-1"}},
        )

        result = await graph.ainvoke({"value": 1}, config)
        snapshot = await graph.aget_state(config)

    assert result == {"value": 2}
    assert snapshot.values == {"value": 2}
    assert snapshot.config["configurable"]["checkpoint_id"]


@pytest.mark.asyncio
async def test_close_closes_owned_default_credential() -> None:
    credential = SimpleNamespace(close=AsyncMock())
    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver."
        "DefaultAzureCredential",
        return_value=credential,
    ):
        saver = FoundryCheckpointSaver()
        await saver.aclose()

    credential.close.assert_awaited_once()
