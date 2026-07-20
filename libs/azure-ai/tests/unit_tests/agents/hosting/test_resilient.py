# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for Responses resilience storage managers."""

import asyncio
from types import SimpleNamespace
from typing import cast

from azure.ai.agentserver.responses import (
    ConversationChainMetadataNamespace,
    ResponseContext,
    ResponseObject,
)
from langchain_core.runnables import RunnableConfig

from langchain_azure_ai.agents.hosting._responses import (
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


def test_task_storage_manager_reads_checkpoint_ref_from_response() -> None:
    response = cast(
        ResponseObject,
        SimpleNamespace(
            internal_metadata={
                METADATA_LANGGRAPH_THREAD_ID: "thread-1",
                METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-1",
            }
        ),
    )

    assert TaskStorageManager.from_response(response).checkpoint_ref == CheckpointRef(
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
    )


def test_conversation_storage_manager_requires_stored_thread() -> None:
    namespace = cast(
        ConversationChainMetadataNamespace,
        lambda _: {"checkpoint_id": "checkpoint-1"},
    )

    assert ConversationChainStorageManager(namespace).checkpoint_ref is None


def test_checkpoint_ref_readers_ignore_invalid_values() -> None:
    config = cast(
        RunnableConfig,
        {"configurable": {"thread_id": "", "checkpoint_id": 1}},
    )
    response = cast(
        ResponseObject,
        SimpleNamespace(
            internal_metadata={
                METADATA_LANGGRAPH_THREAD_ID: "",
                METADATA_LANGGRAPH_CHECKPOINT_ID: 1,
            }
        ),
    )
    namespace = cast(
        ConversationChainMetadataNamespace,
        lambda _: {"thread_id": "", "checkpoint_id": 1},
    )

    assert HostingRunnableConfig(config).checkpoint_ref is None
    assert TaskStorageManager.from_response(response).checkpoint_ref is None
    assert ConversationChainStorageManager(namespace).checkpoint_ref is None


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
