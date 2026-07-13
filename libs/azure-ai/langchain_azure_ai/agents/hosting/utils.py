# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities for hosting LangGraph applications with the Responses API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from azure.ai.agentserver.responses import ResponseContext

logger = logging.getLogger(__name__)

METADATA_LANGGRAPH_CHECKPOINT_ID = "langgraph_checkpoint_id"


@dataclass(frozen=True)
class CheckpointInfo:
    """LangGraph thread and optional exact committed Responses checkpoint."""

    thread_id: str
    checkpoint_id: str | None = None


def _response_field(response: Any, name: str) -> str | None:
    value = getattr(response, name, None)
    if value is None and hasattr(response, "get"):
        value = response.get(name)
    return value if isinstance(value, str) and value else None


def _response_conversation_id(response: Any) -> str | None:
    conversation = getattr(response, "conversation", None)
    if conversation is None and hasattr(response, "get"):
        conversation = response.get("conversation")
    if isinstance(conversation, dict):
        value = conversation.get("id")
    else:
        value = getattr(conversation, "id", None)
    return value if isinstance(value, str) and value else None


def _response_checkpoint_id(response: Any) -> str | None:
    metadata = getattr(response, "internal_metadata", None)
    if metadata is None and hasattr(response, "get"):
        metadata = response.get("internal_metadata")
    if metadata is None:
        return None
    value = metadata.get(METADATA_LANGGRAPH_CHECKPOINT_ID)
    return value if isinstance(value, str) and value else None


async def _checkpoint_info_from_response_chain(
    previous_response_id: str,
    context: ResponseContext,
) -> CheckpointInfo | None:
    provider = getattr(context, "_provider", None)
    if provider is None:
        return None

    response_id: str | None = previous_response_id
    seen: set[str] = set()
    root_response_id = previous_response_id
    parent_checkpoint_id: str | None = None
    while response_id and response_id not in seen:
        seen.add(response_id)
        try:
            response = await provider.get_response(
                response_id,
                context=context.platform_context,
            )
        except Exception:
            logger.debug(
                "Failed to resolve response chain for thread mapping",
                exc_info=True,
            )
            return None

        if response_id == previous_response_id:
            parent_checkpoint_id = _response_checkpoint_id(response)

        conversation_id = _response_conversation_id(response)
        if conversation_id:
            return CheckpointInfo(conversation_id, parent_checkpoint_id)

        root_response_id = response_id
        response_id = _response_field(response, "previous_response_id")

    return CheckpointInfo(root_response_id, parent_checkpoint_id)


async def resolve_checkpoint_info(
    conversation_id: str | None,
    previous_response_id: str | None,
    context: ResponseContext,
) -> CheckpointInfo:
    """Resolve the LangGraph thread and exact parent checkpoint."""
    if context.is_recovery and context.persisted_response is not None:
        metadata = getattr(context.persisted_response, "internal_metadata", None)
        if metadata is not None:
            persisted_thread_id = metadata.get("langgraph_thread_id")
            if isinstance(persisted_thread_id, str) and persisted_thread_id:
                persisted_checkpoint_id = metadata.get(
                    METADATA_LANGGRAPH_CHECKPOINT_ID
                )
                return CheckpointInfo(
                    persisted_thread_id,
                    persisted_checkpoint_id
                    if isinstance(persisted_checkpoint_id, str)
                    and persisted_checkpoint_id
                    else None,
                )

    if isinstance(conversation_id, str) and conversation_id:
        return CheckpointInfo(conversation_id)

    if isinstance(previous_response_id, str) and previous_response_id:
        info = await _checkpoint_info_from_response_chain(
            previous_response_id,
            context,
        )
        return info or CheckpointInfo(previous_response_id)

    # first turn of a new conversation
    return CheckpointInfo(context.response_id)