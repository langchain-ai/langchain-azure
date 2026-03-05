"""Azure AI Foundry Memory retriever for incremental search with chat history context.

This module provides the AzureAIMemoryRetriever class, which is designed to work
closely with AzureAIMemoryChatMessageHistory for retrieving contextual memories
during multi-turn conversations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.retrievers import BaseRetriever
from pydantic import model_validator

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.utils.env import get_from_dict_or_env

# Avoid circular imports - use TYPE_CHECKING for type hints
if False:  # TYPE_CHECKING
    pass

logger = logging.getLogger(__name__)


def _get_attr_or_key(obj: Any, key: str, default_value: Any = None) -> Any:
    """Helper to access attribute or dict key.

    Handles both object attributes and dictionary keys, useful for working with
    objects that may be either attribute-based or dict-like.

    Args:
        obj: Object to access (can be an object with attributes or a dict)
        key: Attribute or key name to access
        default_value: Value to return if key/attribute is not found

    Returns:
        The value of the attribute/key, or default_value if not found.
    """
    if hasattr(obj, key):
        return getattr(obj, key, default_value)
    if isinstance(obj, dict):
        return obj.get(key, default_value)
    return default_value


def _map_message_to_foundry_item(message: Any) -> Any:
    """Map LangChain message to Azure Foundry response message item.

    Uses substring matching to handle message type variations like
    AIMessage, AIMessageChunk, HumanMessage, etc.

    Args:
        message: LangChain BaseMessage instance

    Returns:
        Azure ResponsesMessageItemParam with appropriate role

    Note:
        Mapping:
        - contains 'human' → user (HumanMessage)
        - contains 'ai' → assistant (AIMessage, AIMessageChunk)
        - contains 'tool' → assistant (ToolMessage - treated as assistant output)
        - contains 'system' → system (SystemMessage)
        - contains 'developer' → developer
        - unknown → user (fallback with debug logging)
    """
    from azure.ai.projects.models import (
        ResponsesAssistantMessageItemParam,
        ResponsesDeveloperMessageItemParam,
        ResponsesSystemMessageItemParam,
        ResponsesUserMessageItemParam,
    )

    msg_type = getattr(message, "type", "") or message.__class__.__name__
    msg_type = msg_type.lower()
    content = (
        message.content if isinstance(message.content, str) else str(message.content)
    )

    if "human" in msg_type:
        return ResponsesUserMessageItemParam(content=content)
    if "ai" in msg_type:
        return ResponsesAssistantMessageItemParam(content=content)
    if "tool" in msg_type:
        # Tool messages are treated as assistant output
        return ResponsesAssistantMessageItemParam(content=content)
    if "system" in msg_type:
        return ResponsesSystemMessageItemParam(content=content)
    if "developer" in msg_type:
        return ResponsesDeveloperMessageItemParam(content=content)

    # Fallback for unknown types
    logger.debug(
        f"Unmapped message type '{msg_type}' from "
        f"{message.__class__.__name__}, defaulting to user role"
    )
    return ResponsesUserMessageItemParam(content=content)


@experimental()
class AzureAIMemoryRetriever(BaseRetriever):
    """LangChain retriever that queries Foundry Memory with multi-turn context.

    **NOTE:** This retriever is designed for close coupling with
    AzureAIMemoryChatMessageHistory. When bound to a history instance via
    history_ref, it provides incremental search capabilities with multi-turn
    conversation context. Use standalone only for one-off queries without
    conversation context.

    This retriever queries Azure AI Foundry Memory, supporting both standalone
    retrieval and history-bound incremental search with previous_search_id.

    Args:
        store_name: Memory store name (required if not using history_ref)
        scope: Memory scope (e.g., user:{user_id}) (required if not using history_ref)
        session_id: Optional session identifier for this retriever
        k: Maximum number of memories to retrieve
        project_endpoint: Azure AI project endpoint. If not provided, reads from
            AZURE_AI_PROJECT_ENDPOINT environment variable.
        credential: Azure credential for authentication. If not provided,
            uses DefaultAzureCredential().
        history_ref: Optional reference to a AzureAIMemoryChatMessageHistory instance.
            When provided, the retriever inherits client settings from the history
            and enables incremental search with conversation context.

    Example:
        Standalone retriever (one-off search without context):
        >>> from azure.identity import DefaultAzureCredential
        >>>
        >>> # Option 1: With explicit endpoint and credential
        >>> retriever = AzureAIMemoryRetriever(
        ...     project_endpoint="https://myproject.api.azureml.ms",
        ...     credential=DefaultAzureCredential(),
        ...     store_name="my_store",
        ...     scope="user:123",
        ...     k=5
        ... )
        >>> docs = retriever.invoke("What are my coffee preferences?")
        >>>
        >>> # Option 2: Using environment variable AZURE_AI_PROJECT_ENDPOINT
        >>> retriever = AzureAIMemoryRetriever(
        ...     store_name="my_store",
        ...     scope="user:123",
        ...     k=5
        ... )
        >>> docs = retriever.invoke("What are my preferences?")

        History-bound retriever (with conversation context - recommended):
        >>> from langchain_azure_ai.chat_message_histories import (
        ...     AzureAIMemoryChatMessageHistory
        ... )
        >>> history = AzureAIMemoryChatMessageHistory(
        ...     project_endpoint="https://myproject.api.azureml.ms",
        ...     store_name="my_store",
        ...     scope="user:123",
        ...     session_id="session_001",
        ...     base_history_factory=lambda _: InMemoryChatMessageHistory(),
        ... )
        >>> retriever = history.get_retriever(k=5)
        >>> docs = retriever.invoke("Tell me more")
    """

    client: Optional[Any] = None
    """AIProjectClient instance."""
    store_name: Optional[str] = None
    """Memory store name."""
    scope: Optional[str] = None
    """Memory scope (e.g., user or tenant ID)."""
    session_id: Optional[str] = None
    """Optional session identifier for this retriever."""
    k: int = 5
    """Maximum number of memories to retrieve."""
    project_endpoint: Optional[str] = None
    """Azure AI project endpoint."""
    credential: Optional[TokenCredential] = None
    """Azure credential for authentication."""
    history_ref: Optional[Any] = None  # AzureAIMemoryChatMessageHistory
    """Optional reference to a AzureAIMemoryChatMessageHistory instance."""
    _previous_search_id: Optional[str] = None
    """Cached search_id from the prior incremental query (if any)."""

    @model_validator(mode="before")
    @classmethod
    def _derive_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Derive store/scope/session/client from history_ref if provided.

        Note:
            When history_ref is provided, its properties take precedence over
            explicitly provided values to ensure the retriever is tightly bound
            to the history.
        """
        history_ref = values.get("history_ref")

        store_name = values.get("store_name")
        scope_val = values.get("scope")
        session_val = values.get("session_id")

        if history_ref is not None:
            # History properties take precedence when retriever is bound to history
            provided_store = store_name
            provided_scope = scope_val

            store_name = history_ref.store_name or store_name
            scope_val = history_ref.scope or scope_val
            session_val = history_ref.session_id or session_val

            # Use the client from the history
            values["client"] = history_ref._client

            # Warn if explicitly provided values differ from history
            if provided_store and provided_store != store_name:
                logger.warning(
                    f"Retriever store_name '{provided_store}' differs from "
                    f"history store_name '{store_name}'. Using history value."
                )
            if provided_scope and provided_scope != scope_val:
                logger.warning(
                    f"Retriever scope '{provided_scope}' differs from "
                    f"history scope '{scope_val}'. Using history value."
                )
        else:
            # Standalone retriever - create client from endpoint/credential
            if not store_name or not scope_val:
                raise ValueError(
                    "Either provide history_ref or both "
                    "store_name and scope explicitly."
                )

            # Read project_endpoint from environment if not provided
            project_endpoint = get_from_dict_or_env(
                values,
                "project_endpoint",
                "AZURE_AI_PROJECT_ENDPOINT",
            )

            if not project_endpoint:
                raise ValueError(
                    "project_endpoint must be provided either as a parameter or via "
                    "the AZURE_AI_PROJECT_ENDPOINT environment variable."
                )

            # Use provided credential or default
            credential = values.get("credential")
            cred: TokenCredential = credential or DefaultAzureCredential()

            # Create AIProjectClient with user-agent for monitoring
            from azure.ai.projects import AIProjectClient

            values["client"] = AIProjectClient(
                endpoint=project_endpoint,
                credential=cred,
                user_agent="langchain-azure-ai",
            )

        values["store_name"] = store_name
        values["scope"] = scope_val
        values["session_id"] = session_val

        return values

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search Foundry Memory with history context and incremental refinement.

        Args:
            query: The search query.
            run_manager: Callback manager for retrieval.
            **kwargs: Additional keyword arguments

        Returns:
            List of Document objects with memory content and metadata
        """
        incremental_search = self.history_ref is not None

        from azure.ai.projects.models import (
            MemorySearchOptions,
            ResponsesUserMessageItemParam,
        )

        # Build contextual items from the last assistant turn onward.
        items = []
        if self.history_ref is not None:
            messages = self.history_ref.messages
            last_assistant_idx = None
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                # Look for AI messages (including AIMessageChunk)
                if isinstance(msg, (AIMessage, AIMessageChunk)):
                    last_assistant_idx = i
                    break
            start_idx = last_assistant_idx if last_assistant_idx is not None else 0
            for m in messages[start_idx:]:
                items.append(_map_message_to_foundry_item(m))
        items.append(ResponsesUserMessageItemParam(content=query))

        # Client should always be initialized by the validator
        assert self.client is not None, "Client must be initialized"

        # Use previous_search_id only for history-bound (incremental) retrieval
        result = self.client.memory_stores.search_memories(
            name=self.store_name,
            scope=self.scope,
            items=items,
            previous_search_id=self._previous_search_id if incremental_search else None,
            options=MemorySearchOptions(max_memories=self.k),
        )

        # Cache search_id only if in incremental mode
        if incremental_search:
            try:
                self._previous_search_id = result.search_id
            except Exception as e:
                logger.debug(
                    f"Could not cache search_id from memory search result: {e}",
                    exc_info=False,
                )
                # Reset on failure to start fresh with next search
                self._previous_search_id = None
        else:
            # Reset in non-incremental mode (each call is independent)
            self._previous_search_id = None

        docs: List[Document] = []

        try:
            # result.memories is a list[MemorySearchItem]; each has .memory_item
            memories: list[Any] = _get_attr_or_key(result, "memories", [])
            for entry in memories:
                mem_item = _get_attr_or_key(entry, "memory_item")
                if not mem_item:
                    continue
                content = _get_attr_or_key(mem_item, "content", "")
                kind = _get_attr_or_key(mem_item, "kind")
                mem_id = _get_attr_or_key(mem_item, "memory_id")
                scope = _get_attr_or_key(mem_item, "scope")
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "memory_id": mem_id,
                            "kind": kind,
                            "scope": scope,
                            "source": "azure_ai_memory",
                        },
                    )
                )
        except Exception as e:
            # Return what we can even if parsing is partial
            logger.warning(
                f"Error parsing memory search results: {e}",
                exc_info=False,
            )
        return docs
