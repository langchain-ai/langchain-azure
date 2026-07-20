# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Conversation-chain LangGraph checkpoint storage management."""

from __future__ import annotations

from azure.ai.agentserver.responses import ConversationChainMetadataNamespace

from .checkpoint_ref import CheckpointRef

CONVERSATION_METADATA_NAMESPACE = "langgraph"
CONVERSATION_METADATA_CHECKPOINT_ID = "checkpoint_id"
CONVERSATION_METADATA_THREAD_ID = "thread_id"


class ConversationChainStorageManager:
    """Manage LangGraph references shared by a linear response chain."""

    def __init__(
        self,
        conversation_chain_metadata: ConversationChainMetadataNamespace,
    ) -> None:
        self._metadata = conversation_chain_metadata(CONVERSATION_METADATA_NAMESPACE)

    @property
    def checkpoint_ref(self) -> CheckpointRef | None:
        """Return the latest reference stored for the response chain."""
        thread_id = self._metadata.get(CONVERSATION_METADATA_THREAD_ID)
        if not isinstance(thread_id, str) or not thread_id:
            return None
        checkpoint_id = self._metadata.get(CONVERSATION_METADATA_CHECKPOINT_ID)
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            return None
        return CheckpointRef(thread_id, checkpoint_id)

    async def persist_checkpoint_ref(
        self,
        checkpoint_ref: CheckpointRef,
    ) -> None:
        """Persist the latest LangGraph checkpoint for the next turn."""
        self._metadata[CONVERSATION_METADATA_THREAD_ID] = checkpoint_ref.thread_id
        self._metadata[CONVERSATION_METADATA_CHECKPOINT_ID] = (
            checkpoint_ref.checkpoint_id
        )
        await self._metadata.flush()
