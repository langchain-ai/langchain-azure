# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Conversation-chain LangGraph checkpoint storage management."""

from __future__ import annotations

from azure.ai.agentserver.core.storage import FoundryStateStore

from .checkpoint_ref import CheckpointRef

CONVERSATION_STATE_STORE_PREFIX = "langchain_azure_ai.agents.hosting/responses"
CONVERSATION_STATE_CHECKPOINT_REFERENCE_KEY = "langgraph_checkpoint"
_CHECKPOINT_ID = "checkpoint_id"
_THREAD_ID = "thread_id"


class ConversationChainStorageManager:
    """Manage LangGraph references shared by a linear response chain."""

    def __init__(self, conversation_chain_id: str) -> None:
        self._store_name = f"{CONVERSATION_STATE_STORE_PREFIX}/{conversation_chain_id}"

    async def get_checkpoint_ref(self) -> CheckpointRef | None:
        """Return the latest reference stored for the response chain."""
        store = await FoundryStateStore.get_or_create(
            self._store_name,
            description="LangGraph state for a LangChain Responses conversation",
        )
        async with store:
            item = await store.get_item(CONVERSATION_STATE_CHECKPOINT_REFERENCE_KEY)
        if item is None or not isinstance(item.value, dict):
            return None
        thread_id = item.value.get(_THREAD_ID)
        if not isinstance(thread_id, str) or not thread_id:
            return None
        checkpoint_id = item.value.get(_CHECKPOINT_ID)
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            return None
        return CheckpointRef(thread_id, checkpoint_id)

    async def persist_checkpoint_ref(
        self,
        checkpoint_ref: CheckpointRef,
    ) -> None:
        """Persist the latest LangGraph checkpoint for the next turn."""
        store = await FoundryStateStore.get_or_create(
            self._store_name,
            description="LangGraph state for a LangChain Responses conversation",
        )
        async with store:
            await store.set_item(
                CONVERSATION_STATE_CHECKPOINT_REFERENCE_KEY,
                {
                    _THREAD_ID: checkpoint_ref.thread_id,
                    _CHECKPOINT_ID: checkpoint_ref.checkpoint_id,
                },
            )
