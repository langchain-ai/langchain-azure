# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Internal Responses API hosting support."""

from .checkpoint_ref import (
    CONVERSATION_CHECKPOINT_KEY,
    CheckpointRef,
    PendingInterruptIndex,
    consume_pending_interrupt_ids,
    load_pending_interrupt_index,
    store_pending_interrupt_ids,
)
from .conversation_chain_store import (
    CONVERSATION_CHAIN_STORE_PREFIX,
    ConversationChainStoreProtocol,
    FoundryConversationChainStore,
)
from .hosting_runnable_config import HostingRunnableConfig
from .task_storage_manager import (
    METADATA_LANGGRAPH_CHECKPOINT_ID,
    METADATA_LANGGRAPH_THREAD_ID,
    TaskStorageManager,
)

__all__ = [
    "CONVERSATION_CHECKPOINT_KEY",
    "CONVERSATION_CHAIN_STORE_PREFIX",
    "METADATA_LANGGRAPH_CHECKPOINT_ID",
    "METADATA_LANGGRAPH_THREAD_ID",
    "CheckpointRef",
    "PendingInterruptIndex",
    "ConversationChainStoreProtocol",
    "FoundryConversationChainStore",
    "HostingRunnableConfig",
    "TaskStorageManager",
    "consume_pending_interrupt_ids",
    "load_pending_interrupt_index",
    "store_pending_interrupt_ids",
]
