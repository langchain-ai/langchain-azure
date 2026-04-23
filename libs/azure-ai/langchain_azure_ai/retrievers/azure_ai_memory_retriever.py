"""Azure AI Foundry Memory retriever for incremental search with chat history context.

This module provides the AzureAIMemoryRetriever class, which is designed to work
closely with AzureAIMemoryChatMessageHistory for retrieving contextual memories
during multi-turn conversations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from azure.ai.projects import AIProjectClient
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.retrievers import BaseRetriever
from openai.types.responses import EasyInputMessageParam
from pydantic import ConfigDict, PrivateAttr, model_validator

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.chat_history.azure_ai_memory import (
    AzureAIMemoryChatMessageHistory,
)
from langchain_azure_ai.utils.env import get_project_endpoint

logger = logging.getLogger(__name__)


def _map_message_to_foundry_item(message: BaseMessage) -> EasyInputMessageParam:
    """Map LangChain message to Azure Foundry response message item.

    Uses substring matching to handle message type variations like
    AIMessage, AIMessageChunk, HumanMessage, etc.

    Args:
        message: LangChain BaseMessage instance

    Returns:
        EasyInputMessageParam with appropriate role

    Note:
        Mapping:
        - contains 'human' → user (HumanMessage)
        - contains 'ai' → assistant (AIMessage, AIMessageChunk)
        - contains 'tool' → assistant (ToolMessage - treated as assistant output)
        - contains 'system' → system (SystemMessage)
        - contains 'developer' → developer
        - unknown → user (fallback with debug logging)
    """
    msg_type = getattr(message, "type", "") or message.__class__.__name__
    msg_type = msg_type.lower()
    content = (
        message.content if isinstance(message.content, str) else str(message.content)
    )

    if "human" in msg_type:
        return EasyInputMessageParam(content=content, role="user")
    if "ai" in msg_type:
        return EasyInputMessageParam(content=content, role="assistant")
    if "tool" in msg_type:
        # Tool messages are treated as assistant output
        return EasyInputMessageParam(content=content, role="assistant")
    if "system" in msg_type:
        return EasyInputMessageParam(content=content, role="system")
    if "developer" in msg_type:
        return EasyInputMessageParam(content=content, role="developer")

    # Fallback for unknown types
    logger.debug(
        f"Unmapped message type '{msg_type}' from "
        f"{message.__class__.__name__}, defaulting to user role"
    )
    return EasyInputMessageParam(content=content, role="user")


@experimental()
class AzureAIMemoryRetriever(BaseRetriever):
    """LangChain retriever that queries Foundry Memory with multi-turn context.

    This retriever is designed for close coupling with
    `AzureAIMemoryChatMessageHistory`. When bound to a history instance via
    `history_ref`, it provides incremental search capabilities with multi-turn
    conversation context. Use standalone mode only for one-off queries without
    conversation context.

    This retriever queries Azure AI Foundry Memory, supporting both standalone
    retrieval and history-bound incremental search with `previous_search_id`.

    **Examples**

    Standalone retriever (one-off search without context):

    ```python
    from azure.identity import DefaultAzureCredential

    retriever = AzureAIMemoryRetriever(
        project_endpoint="https://myproject.api.azureml.ms",
        credential=DefaultAzureCredential(),
        store_name="my_store",
        scope="user:123",
        k=5,
    )
    docs = retriever.invoke("What are my coffee preferences?")
    ```

    With endpoint from environment variable:

    ```python
    retriever = AzureAIMemoryRetriever(
        store_name="my_store",
        scope="user:123",
        k=5,
    )
    docs = retriever.invoke("What are my preferences?")
    ```

    History-bound retriever (recommended):

    ```python
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_azure_ai.chat_history import (
        AzureAIMemoryChatMessageHistory,
    )

    history = AzureAIMemoryChatMessageHistory(
        project_endpoint="https://myproject.api.azureml.ms",
        store_name="my_store",
        scope="user:123",
        base_history=InMemoryChatMessageHistory(),
    )
    retriever = history.get_retriever(k=5)
    docs = retriever.invoke("Tell me more")
    ```
    """

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
    history_ref: Optional[AzureAIMemoryChatMessageHistory] = None
    """Optional reference to a AzureAIMemoryChatMessageHistory instance."""
    _client: Optional[AIProjectClient] = PrivateAttr(default=None)
    """Cached AIProjectClient instance."""
    _previous_search_id: Optional[str] = PrivateAttr(default=None)
    """Cached search_id from the prior incremental query (if any)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _derive_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Derive store/scope/session from history_ref if provided.

        Note:
            When history_ref is provided, its properties take precedence over
            explicitly provided values to ensure the retriever is tightly bound
            to the history.
        """
        history_ref = values.get("history_ref")

        store_name = values.get("store_name")
        scope_val = values.get("scope")

        if history_ref is not None:
            # History properties take precedence when retriever is bound to history
            provided_store = store_name
            provided_scope = scope_val

            store_name = history_ref.store_name or store_name
            scope_val = history_ref.scope or scope_val

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
            project_endpoint = get_project_endpoint(values)

            if not project_endpoint:
                raise ValueError(
                    "project_endpoint must be provided either as a parameter or via "
                    "the AZURE_AI_PROJECT_ENDPOINT environment variable."
                )

            values["project_endpoint"] = project_endpoint

        values["store_name"] = store_name
        values["scope"] = scope_val

        return values

    def _initialize_client(self) -> AIProjectClient:
        """Initialize and cache an AIProjectClient for this retriever."""
        if self._client is not None:
            return self._client

        if self.history_ref is not None:
            self._client = self.history_ref._client
            return self._client

        if not self.project_endpoint:
            raise ValueError(
                "project_endpoint must be provided either as a parameter or via "
                "the AZURE_AI_PROJECT_ENDPOINT environment variable."
            )

        cred: TokenCredential = self.credential or DefaultAzureCredential()
        client = AIProjectClient(
            endpoint=self.project_endpoint,
            credential=cred,
            user_agent="langchain-azure-ai",
        )
        if not hasattr(client, "beta") or not hasattr(client.beta, "memory_stores"):
            raise ValueError(
                "AzureAIMemoryRetriever requires azure-ai-projects>=2.0.0b4. "
                "Install the v2 extra: pip install 'langchain-azure-ai[v2]'"
            )
        self._client = client
        return self._client

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

        from azure.ai.projects.models import MemorySearchOptions

        # Build contextual items from the last assistant turn onward.
        items: list[EasyInputMessageParam] = []
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
        items.append(EasyInputMessageParam(content=query, role="user"))

        client = self._initialize_client()

        # Use previous_search_id only for history-bound (incremental) retrieval
        result = client.beta.memory_stores.search_memories(
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
            for entry in result.memories:
                mem_item = entry.memory_item
                docs.append(
                    Document(
                        page_content=mem_item.content,
                        metadata={
                            "memory_id": mem_item.memory_id,
                            "kind": mem_item.kind,
                            "scope": mem_item.scope,
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
