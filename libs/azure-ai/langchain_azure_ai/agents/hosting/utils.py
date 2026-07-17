# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities for hosting LangGraph applications with the Responses API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from azure.ai.agentserver.responses import (
    ResponseContext,
    ResponseEventStream,
    ResponseObject,
)
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

METADATA_LANGGRAPH_CHECKPOINT_ID = "langgraph_checkpoint_id"
METADATA_LANGGRAPH_THREAD_ID = "langgraph_thread_id"


@dataclass(frozen=True)
class CheckpointInfo:
    """LangGraph thread and optional exact committed Responses checkpoint."""

    thread_id: str
    checkpoint_id: str | None = None


def store_langgraph_checkpoint_ref(
    stream: ResponseEventStream,
    config: RunnableConfig,
) -> None:
    """Copy the LangGraph thread and checkpoint references to the response."""
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        stream.internal_metadata[METADATA_LANGGRAPH_THREAD_ID] = thread_id

    checkpoint_id = configurable.get("checkpoint_id")
    if isinstance(checkpoint_id, str) and checkpoint_id:
        stream.internal_metadata[METADATA_LANGGRAPH_CHECKPOINT_ID] = checkpoint_id


def _response_conversation_id(response: ResponseObject) -> str | None:
    conversation = getattr(response, "conversation", None)
    if conversation is None and hasattr(response, "get"):
        conversation = response.get("conversation")
    if isinstance(conversation, dict):
        value = conversation.get("id")
    else:
        value = getattr(conversation, "id", None)
    return value if isinstance(value, str) and value else None


def _response_internal_metadata(response: ResponseObject | None) -> Any:
    if response is None:
        return None
    metadata = getattr(response, "internal_metadata", None)
    return metadata


def response_checkpoint_id(response: ResponseObject | None) -> str | None:
    """Return the LangGraph checkpoint stored on a Response, if any."""
    metadata = _response_internal_metadata(response)
    if metadata is None:
        return None
    value = metadata.get(METADATA_LANGGRAPH_CHECKPOINT_ID)
    return value if isinstance(value, str) and value else None


def _response_thread_id(response: ResponseObject) -> str | None:
    metadata = _response_internal_metadata(response)
    if metadata is None:
        return None
    value = metadata.get(METADATA_LANGGRAPH_THREAD_ID)
    return value if isinstance(value, str) and value else None


async def _checkpoint_info_from_parent_response(
    previous_response_id: str,
    context: ResponseContext,
) -> CheckpointInfo | None:
    provider = getattr(context, "_provider", None)
    if provider is None:
        return None

    try:
        response = await provider.get_response(
            previous_response_id,
            context=context.platform_context,
        )
    except Exception:
        logger.debug(
            "Failed to resolve parent response for thread mapping",
            exc_info=True,
        )
        return None

    thread_id = (
        _response_thread_id(response)
        or _response_conversation_id(response)
        or previous_response_id
    )
    return CheckpointInfo(thread_id, response_checkpoint_id(response))


async def resolve_checkpoint_info(
    conversation_id: str | None,
    previous_response_id: str | None,
    context: ResponseContext,
) -> CheckpointInfo:
    """Resolve the LangGraph thread and exact parent checkpoint."""
    if context.is_recovery and context.persisted_response is not None:
        checkpoint_id = response_checkpoint_id(context.persisted_response)
        thread_id = _response_thread_id(context.persisted_response)
        if thread_id and checkpoint_id:
            return CheckpointInfo(thread_id, checkpoint_id)

    if isinstance(conversation_id, str) and conversation_id:
        return CheckpointInfo(conversation_id)

    if isinstance(previous_response_id, str) and previous_response_id:
        info = await _checkpoint_info_from_parent_response(
            previous_response_id,
            context,
        )
        return info or CheckpointInfo(previous_response_id)

    # first turn of a new conversation
    return CheckpointInfo(context.response_id)