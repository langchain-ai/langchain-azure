# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for Responses resilience storage managers."""

import asyncio
from typing import cast

from azure.ai.agentserver.responses import (
    ResponseContext,
    ResponseEventStream,
    ResponseObject,
)
from langchain_core.runnables import RunnableConfig

from langchain_azure_ai.agents.hosting._responses import (
    CONVERSATION_STATE_CHECKPOINT_REFERENCE_KEY,
    CONVERSATION_STATE_STORE_PREFIX,
    METADATA_LANGGRAPH_CHECKPOINT_ID,
    METADATA_LANGGRAPH_THREAD_ID,
    CheckpointRef,
    ConversationChainStorageManager,
    HostingRunnableConfig,
    TaskStorageManager,
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


async def test_conversation_storage_manager_requires_stored_thread(
    foundry_state_stores: dict[str, dict[str, object]],
) -> None:
    foundry_state_stores[f"{CONVERSATION_STATE_STORE_PREFIX}/chain-1"] = {
        CONVERSATION_STATE_CHECKPOINT_REFERENCE_KEY: {"checkpoint_id": "checkpoint-1"}
    }

    assert await ConversationChainStorageManager("chain-1").get_checkpoint_ref() is None


async def test_checkpoint_ref_readers_ignore_invalid_values(
    foundry_state_stores: dict[str, dict[str, object]],
) -> None:
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
    foundry_state_stores[f"{CONVERSATION_STATE_STORE_PREFIX}/chain-1"] = {
        CONVERSATION_STATE_CHECKPOINT_REFERENCE_KEY: {
            "thread_id": "",
            "checkpoint_id": 1,
        }
    }

    assert HostingRunnableConfig(config).checkpoint_ref is None
    assert TaskStorageManager.from_stream(stream).checkpoint_ref is None
    assert await ConversationChainStorageManager("chain-1").get_checkpoint_ref() is None


async def test_conversation_storage_manager_round_trips_checkpoint(
    foundry_state_stores: dict[str, dict[str, object]],
) -> None:
    manager = ConversationChainStorageManager("chain-1")
    checkpoint_ref = CheckpointRef("thread-1", "checkpoint-1")

    await manager.persist_checkpoint_ref(checkpoint_ref)

    assert await manager.get_checkpoint_ref() == checkpoint_ref
    assert foundry_state_stores[f"{CONVERSATION_STATE_STORE_PREFIX}/chain-1"] == {
        CONVERSATION_STATE_CHECKPOINT_REFERENCE_KEY: {
            "thread_id": "thread-1",
            "checkpoint_id": "checkpoint-1",
        }
    }


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
