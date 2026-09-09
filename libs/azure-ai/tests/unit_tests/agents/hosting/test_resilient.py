# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for Responses resilience storage."""

import asyncio
from typing import Any, cast

import pytest
from azure.ai.agentserver.responses import (
    ResponseContext,
    ResponseEventStream,
    ResponseObject,
)
from langchain_core.runnables import RunnableConfig

from langchain_azure_ai.agents.hosting._responses import (
    CONVERSATION_CHAIN_STORE_PREFIX,
    CONVERSATION_CHECKPOINT_KEY,
    METADATA_LANGGRAPH_CHECKPOINT_ID,
    METADATA_LANGGRAPH_THREAD_ID,
    CheckpointRef,
    FoundryConversationChainStore,
    HostingRunnableConfig,
    TaskStorageManager,
    conversation_chain_store,
)


def test_hosting_runnable_config_reads_checkpoint_ref() -> None:
    config = cast(
        RunnableConfig,
        {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_id": "checkpoint-1",
            }
        },
    )

    assert HostingRunnableConfig(config).checkpoint_ref == CheckpointRef(
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
    )


def test_task_storage_manager_reads_checkpoint_ref_from_seeded_stream() -> None:
    response = cast(
        ResponseObject,
        {
            "metadata": {
                "_internal_metadata": {
                    METADATA_LANGGRAPH_THREAD_ID: "thread-1",
                    METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-1",
                }
            }
        },
    )
    stream = ResponseEventStream(response_id="response-1", response=response)

    assert TaskStorageManager.from_stream(stream).checkpoint_ref == CheckpointRef(
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
    )


async def test_stored_checkpoint_requires_thread_id(
    foundry_state_stores: dict[str, dict[str, object]],
) -> None:
    foundry_state_stores[f"{CONVERSATION_CHAIN_STORE_PREFIX}/chain-1"] = {
        CONVERSATION_CHECKPOINT_KEY: {"checkpoint_id": "checkpoint-1"}
    }

    store = FoundryConversationChainStore()
    assert (
        CheckpointRef.from_dict(await store.get("chain-1", CONVERSATION_CHECKPOINT_KEY))
        is None
    )


def test_checkpoint_ref_readers_ignore_invalid_values() -> None:
    config = cast(
        RunnableConfig,
        {"configurable": {"thread_id": "", "checkpoint_id": 1}},
    )
    response = cast(
        ResponseObject,
        {
            "metadata": {
                "_internal_metadata": {
                    METADATA_LANGGRAPH_THREAD_ID: "",
                    METADATA_LANGGRAPH_CHECKPOINT_ID: 1,
                }
            }
        },
    )
    stream = ResponseEventStream(response_id="response-1", response=response)

    assert HostingRunnableConfig(config).checkpoint_ref is None
    assert TaskStorageManager.from_stream(stream).checkpoint_ref is None


async def test_foundry_conversation_chain_store_preserves_512_characters(
    foundry_state_stores: dict[str, dict[str, object]],
) -> None:
    store = FoundryConversationChainStore()
    data = {"key": "x" * 509}

    await store.set("chain-1", "custom-key", data)

    assert await store.get("chain-1", "custom-key") == data
    assert foundry_state_stores[f"{CONVERSATION_CHAIN_STORE_PREFIX}/chain-1"] == {
        "custom-key": data
    }


async def test_foundry_conversation_chain_store_rejects_non_string_dictionary(
    foundry_state_stores: dict[str, dict[str, object]],
) -> None:
    foundry_state_stores[f"{CONVERSATION_CHAIN_STORE_PREFIX}/chain-1"] = {
        "invalid-key": "not-a-dictionary"
    }

    with pytest.raises(TypeError, match="is not a string dictionary"):
        await FoundryConversationChainStore().get("chain-1", "invalid-key")

    foundry_state_stores[f"{CONVERSATION_CHAIN_STORE_PREFIX}/chain-1"] = {
        "invalid-key": {"value": 1}
    }
    with pytest.raises(TypeError, match="is not a string dictionary"):
        await FoundryConversationChainStore().get("chain-1", "invalid-key")


async def test_foundry_conversation_chain_store_forwards_all_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object, object, dict[str, object]]] = []
    get_or_create = conversation_chain_store.FoundryStateStore.get_or_create

    async def capture_options(
        name: str,
        credential: Any = None,
        endpoint: Any = None,
        **kwargs: Any,
    ) -> object:
        calls.append((name, credential, endpoint, kwargs))
        return await get_or_create(name, credential, endpoint, **kwargs)

    monkeypatch.setattr(
        conversation_chain_store.FoundryStateStore,
        "get_or_create",
        capture_options,
    )
    credential = cast(Any, object())
    store = FoundryConversationChainStore(
        credential,
        "https://example.services.ai.azure.com/api/projects/test",
        user_isolation=True,
        item_ttl_seconds=600,
        description="Custom description",
        tags={"environment": "test"},
        user_id="user-1",
        api_version="v2",
        retry_total=3,
    )

    await store.set("chain-1", "key", {"value": "1"})
    await store.get("chain-1", "key")

    expected = (
        f"{CONVERSATION_CHAIN_STORE_PREFIX}/chain-1",
        credential,
        "https://example.services.ai.azure.com/api/projects/test",
        {
            "user_isolation": True,
            "item_ttl_seconds": 600,
            "description": "Custom description",
            "tags": {"environment": "test"},
            "user_id": "user-1",
            "api_version": "v2",
            "retry_total": 3,
        },
    )
    assert calls == [expected, expected]


def test_hosting_runnable_config_returns_pinned_config_copy() -> None:
    config = cast(
        RunnableConfig,
        {
            "tags": ["existing"],
            "configurable": {
                "thread_id": "thread-old",
                "checkpoint_id": "checkpoint-old",
                "response_context": "context",
            },
        },
    )

    updated = (
        HostingRunnableConfig(config)
        .with_checkpoint_ref(CheckpointRef("thread-new", "checkpoint-new"))
        .runnable_config
    )

    assert updated == {
        "tags": ["existing"],
        "configurable": {
            "thread_id": "thread-new",
            "checkpoint_id": "checkpoint-new",
            "checkpoint_ns": "",
            "response_context": "context",
        },
    }
    assert config["configurable"]["checkpoint_id"] == "checkpoint-old"


def test_hosting_runnable_config_creates_thread_without_checkpoint_ref() -> None:
    response_context = cast(ResponseContext, object())

    hosting_config = HostingRunnableConfig.create(
        "thread-1",
        response_context,
    )

    assert hosting_config.checkpoint_ref is None
    assert hosting_config.runnable_config["configurable"] == {
        "thread_id": "thread-1",
        "response_context": response_context,
    }


def test_hosting_runnable_config_wraps_all_hosting_data() -> None:
    response_context = cast(ResponseContext, object())
    cancellation_signal = asyncio.Event()

    hosting_config = HostingRunnableConfig.create_from_checkpoint(
        CheckpointRef("thread-1", "checkpoint-1"),
        response_context,
    ).with_cancellation_signal(cancellation_signal)

    assert hosting_config.checkpoint_ref == CheckpointRef(
        "thread-1",
        "checkpoint-1",
    )
    assert hosting_config.runnable_config["configurable"] == {
        "thread_id": "thread-1",
        "checkpoint_id": "checkpoint-1",
        "checkpoint_ns": "",
        "response_context": response_context,
        "response_cancellation_signal": cancellation_signal,
    }
