# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the Foundry-backed LangGraph checkpoint saver."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from azure.core.credentials_async import AsyncTokenCredential
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from langchain_azure_ai._user_agent import get_user_agent
from langchain_azure_ai.agents.hosting import FoundryCheckpointSaver, HostingFeature
from langchain_azure_ai.agents.hosting._foundry_checkpoint_saver import (
    DEFAULT_STORE_NAME_PREFIX,
)


class _GraphState(TypedDict):
    value: int


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
        self.get_item_calls: list[str] = []
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
        self.get_item_calls.append(key)
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
async def test_list_before_filters_by_checkpoint_id_not_storage_order() -> None:
    store = _FakeStateStore()

    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver."
        "FoundryStateStore.get_or_create",
        new=AsyncMock(return_value=store),
    ):
        saver = FoundryCheckpointSaver(_credential())
        # Lexicographic order of these ids is a-before < m-newer < z-older, which
        # intentionally disagrees with write/storage order so before-filtering must
        # use checkpoint ids rather than Foundry creation cursors.
        older_config = await saver.aput(
            _config(),
            _checkpoint("z-older", "older"),
            cast(CheckpointMetadata, {"source": "input", "step": -1}),
            {},
        )
        before_config = await saver.aput(
            older_config,
            _checkpoint("a-before", "before"),
            cast(CheckpointMetadata, {"source": "loop", "step": 0}),
            {},
        )
        await saver.aput(
            before_config,
            _checkpoint("m-newer", "newer"),
            cast(CheckpointMetadata, {"source": "loop", "step": 1}),
            {},
        )

        checkpoints = [
            item
            async for item in saver.alist(
                _config(),
                before=_config("m-newer"),
            )
        ]

    # Only ids strictly less than m-newer (a-before) belong before that point.
    assert [item.config["configurable"]["checkpoint_id"] for item in checkpoints] == [
        "a-before"
    ]


@pytest.mark.asyncio
async def test_latest_uses_checkpoint_id_when_storage_order_disagrees() -> None:
    """Regression for #1057: same-second storage order must not hide the true latest."""
    store = _FakeStateStore()
    # UUID6-shaped ids: lexicographically older < newer, like LangGraph's uuid6.
    older_id = "1f1b63e7-4a17-644a-bffe-764ddea781a4"
    newer_id = "1f1b63e7-4a17-67b7-8002-6c64d1d576fe"

    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver."
        "FoundryStateStore.get_or_create",
        new=AsyncMock(return_value=store),
    ):
        saver = FoundryCheckpointSaver(_credential())
        older_config = await saver.aput(
            _config(),
            _checkpoint(older_id, "older"),
            cast(CheckpointMetadata, {"source": "input", "step": -1}),
            {},
        )
        newer_config = await saver.aput(
            older_config,
            _checkpoint(newer_id, "newer"),
            cast(CheckpointMetadata, {"source": "loop", "step": 0}),
            {},
        )
        # Reverse Foundry insertion order to simulate same-second (created_at, id)
        # reordering that would make list_keys(order="desc", limit=1) return older.
        store.order.reverse()

        latest = await saver.aget_tuple(_config())
        exact = await saver.aget_tuple(newer_config)
        history = [item async for item in saver.alist(_config())]

    assert latest is not None
    assert latest.config == newer_config
    assert latest.checkpoint["channel_values"] == {"messages": ["newer"]}
    assert exact is not None
    assert exact.config == newer_config
    assert [item.config for item in history] == [newer_config, older_config]


@pytest.mark.asyncio
async def test_alist_limit_fetches_only_selected_checkpoint_values() -> None:
    """Regression for review on #1070: sort keys before reading bodies."""
    store = _FakeStateStore()
    ids = [f"ckpt-{i:02d}" for i in range(5)]

    with patch(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver."
        "FoundryStateStore.get_or_create",
        new=AsyncMock(return_value=store),
    ):
        saver = FoundryCheckpointSaver(_credential())
        for index, checkpoint_id in enumerate(ids):
            await saver.aput(
                _config(),
                _checkpoint(checkpoint_id, f"v{index}"),
                cast(CheckpointMetadata, {"source": "loop", "step": index}),
                {},
            )
        store.get_item_calls.clear()

        history = [
            item async for item in saver.alist(_config(), limit=2)
        ]

    assert [item.config["configurable"]["checkpoint_id"] for item in history] == [
        "ckpt-04",
        "ckpt-03",
    ]
    # Keys are listed for all five checkpoints, but bodies are fetched only for
    # the two that survive the limit (pending-write lookups use write keys).
    checkpoint_gets = [
        key for key in store.get_item_calls if "/writes/" not in key
    ]
    assert checkpoint_gets == ["/ckpt-04", "/ckpt-03"]


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
