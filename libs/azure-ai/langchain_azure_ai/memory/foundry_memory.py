"""Azure AI Foundry Memory integration with LangChain."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    overload,
)

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, model_validator

if TYPE_CHECKING:
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import ResponsesMessageItemParam

logger = logging.getLogger(__name__)

# Type variable for generic return type in _get_attr_or_key
T = TypeVar("T")


@overload
def _get_attr_or_key(obj: Any, key: str) -> Any | None:
    ...


@overload
def _get_attr_or_key(obj: Any, key: str, default_value: T) -> T:
    ...


def _get_attr_or_key(
    obj: Any, key: str, default_value: T | None = None
) -> Any | T | None:
    """Helper to access attribute or dict key.
    
    Handles both object attributes and dictionary keys, useful for working with
    objects that may be either attribute-based or dict-like.
    
    Args:
        obj: Object to access (can be an object with attributes or a dict)
        key: Attribute or key name to access
        default_value: Value to return if key/attribute is not found
        
    Returns:
        The value of the attribute/key, or default_value if not found.
        When no default_value is provided, returns Any | None.
        When default_value of type T is provided, returns T.
    """
    if hasattr(obj, key):
        return getattr(obj, key, default_value)
    if isinstance(obj, dict):
        return obj.get(key, default_value)
    return default_value


def _map_message_to_foundry_item(message: BaseMessage) -> "ResponsesMessageItemParam":
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
        message.content
        if isinstance(message.content, str)
        else str(message.content)
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


class FoundryMemoryChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that wraps a base history and forwards turns to memory.

    This class decorates any LangChain BaseChatMessageHistory, keeping the short-term
    thread in your chosen store while forwarding each turn to Foundry Memory via
    begin_update_memories for long-term extraction and consolidation.

    Args:
        client: AIProjectClient instance from azure-ai-projects SDK
        store_name: Memory store name in Azure AI Foundry
        scope: Memory scope (e.g., user:{user_id} or tenant:{org_id}) for
            long-term recall across sessions
        session_id: Ephemeral session ID for this chat thread
        base_history_factory: Function to create base history for a session
        update_delay: Optional delay before memory extraction
            (None for default ~300s, 0 for immediate)
        role_mapper: Optional custom function to map LangChain messages
            to Foundry items

    Example:
        >>> from azure.identity import DefaultAzureCredential
        >>> from azure.ai.projects import AIProjectClient
        >>> from langchain_core.chat_history import InMemoryChatMessageHistory
        >>>
        >>> client = AIProjectClient(
        ...     endpoint="https://resource.azure.com/...",
        ...     credential=DefaultAzureCredential()
        ... )
        >>>
        >>> def base_factory(session_id: str):
        ...     return InMemoryChatMessageHistory()
        >>>
        >>> history = FoundryMemoryChatMessageHistory(
        ...     client=client,
        ...     store_name="my_store",
        ...     scope="user:123",
        ...     session_id="session_001",
        ...     base_history_factory=base_factory,
        ... )
    """

    def __init__(
        self,
        client: AIProjectClient,
        store_name: str,
        scope: str,
        session_id: str,
        base_history_factory: Callable[[str], BaseChatMessageHistory],
        *,
        update_delay: Optional[int] = None,  # None => service default (≈300s)
        role_mapper: Optional[Callable[[BaseMessage], Any]] = None,
    ):
        """Initialize FoundryMemoryChatMessageHistory."""
        self._client = client
        self._store = store_name
        self._scope = scope
        self._session_id = session_id
        self._base = base_history_factory(session_id)
        self._update_delay = update_delay
        self._role_mapper = role_mapper
        self._previous_update_id: Optional[str] = None  # advanced incremental updates

    @property
    def messages(self) -> List[BaseMessage]:
        """Return the underlying thread messages (short-term transcript)."""
        return self._base.messages

    @messages.setter
    def messages(self, value: List[BaseMessage]) -> None:
        """Set the underlying thread messages."""
        self._base.messages = value

    @property
    def store_name(self) -> str:
        """Memory store name."""
        return self._store

    @property
    def scope(self) -> str:
        """Memory scope (e.g., user ID or tenant ID)."""
        return self._scope

    @property
    def session_id(self) -> str:
        """Ephemeral session ID for this chat thread."""
        return self._session_id

    def add_message(self, message: BaseMessage) -> None:
        """Persist in short-term transcript AND asynchronously update Foundry Memory.

        This method adds the message to the base history and then fires off
        an asynchronous update to Foundry Memory without blocking the chat flow.

        Args:
            message: The message to add
        """
        # 1) always keep the session transcript
        self._base.add_message(message)

        # 2) best-effort memory update (do not block)
        try:
            item = self._map_lc_message_to_foundry_item(message)
            self._client.memory_stores.begin_update_memories(
                name=self._store,
                scope=self._scope,
                items=[item],
                update_delay=self._update_delay,
                # previous_update_id=self._previous_update_id,  # optional
            )
            # non-blocking: do NOT poll; let the service extract after update_delay
        except Exception as e:
            # Intentionally swallow to avoid breaking chat flow; log for observability
            logger.warning(
                f"Failed to update Foundry Memory for message: {e}",
                exc_info=False,
            )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Convenience: add multiple messages (each forwarded to Foundry).

        Args:
            messages: Sequence of messages to add
        """
        for m in messages:
            self.add_message(m)

    def clear(self) -> None:
        """Clear the short-term transcript for this session (no Foundry deletion)."""
        self._base.clear()

    def get_retriever(self, *, k: int = 5, **filters: Any) -> "FoundryMemoryRetriever":
        """Create a retriever bound to this store/scope/session.

        History-bound retrievers always use incremental search.

        Args:
            k: Maximum number of memories to retrieve
            **filters: Additional filter parameters

        Returns:
            A FoundryMemoryRetriever instance bound to this history
        """
        return FoundryMemoryRetriever(
            client=self._client,
            history_ref=self,
            k=k,
            filters=filters or {},
        )

    # helper kept private; override via role_mapper if needed
    def _map_lc_message_to_foundry_item(
        self, message: BaseMessage
    ) -> "ResponsesMessageItemParam":
        """Map LangChain message to Foundry message item.

        Args:
            message: LangChain message to map

        Returns:
            Foundry message item parameter
        """
        if self._role_mapper:
            return self._role_mapper(message)

        return _map_message_to_foundry_item(message)


class FoundryMemoryRetriever(BaseRetriever):
    """LangChain retriever that queries Foundry Memory with multi-turn context.

    This retriever queries Azure AI Foundry Memory, supporting both standalone
    retrieval and history-bound incremental search with previous_search_id.

    Args:
        client: AIProjectClient instance from azure-ai-projects SDK
        store_name: Memory store name (required if not using history_ref)
        scope: Memory scope (e.g., user:{user_id}) (required if not using history_ref)
        session_id: Optional session identifier for this retriever
        k: Maximum number of memories to retrieve
        history_ref: Optional reference to a FoundryMemoryChatMessageHistory instance
        filters: Optional filter parameters forwarded to the search call

    Example:
        Standalone retriever:
        >>> from azure.identity import DefaultAzureCredential
        >>> from azure.ai.projects import AIProjectClient
        >>>
        >>> client = AIProjectClient(
        ...     endpoint="https://resource.azure.com/...",
        ...     credential=DefaultAzureCredential()
        ... )
        >>>
        >>> retriever = FoundryMemoryRetriever(
        ...     client=client,
        ...     store_name="my_store",
        ...     scope="user:123",
        ...     k=5
        ... )
        >>> docs = retriever.invoke("What are my preferences?")

        History-bound retriever (incremental):
        >>> history = FoundryMemoryChatMessageHistory(...)
        >>> retriever = history.get_retriever(k=5)
        >>> docs = retriever.invoke("Tell me more")
    """

    # Fields that will be set as attributes
    client: Any
    """AIProjectClient instance."""
    store_name: Optional[str] = None
    """Memory store name."""
    scope: Optional[str] = None
    """Memory scope (e.g., user or tenant ID)."""
    session_id: Optional[str] = None
    """Optional session identifier for this retriever."""
    k: int = 5
    """Maximum number of memories to retrieve."""
    history_ref: Optional[FoundryMemoryChatMessageHistory] = None
    """Optional reference to a FoundryMemoryChatMessageHistory instance."""
    filters: Dict[str, Any] = Field(default_factory=dict)
    """Optional filter parameters forwarded to the search call."""
    _previous_search_id: Optional[str] = None
    """Cached search_id from the prior incremental query (if any)."""

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
        session_val = values.get("session_id")

        if history_ref is not None:
            # History properties take precedence when retriever is bound to history
            provided_store = store_name
            provided_scope = scope_val
            
            store_name = history_ref.store_name or store_name
            scope_val = history_ref.scope or scope_val
            session_val = history_ref.session_id or session_val
            
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
            if not store_name or not scope_val:
                raise ValueError(
                    "Either provide history_ref or both "
                    "store_name and scope explicitly."
                )

        values["store_name"] = store_name
        values["scope"] = scope_val
        values["session_id"] = session_val

        if values.get("filters") is None:
            values["filters"] = {}

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
                role = getattr(msg, "type", "") or msg.__class__.__name__
                role = role.lower()
                # Only look for AI messages, not tool messages
                # Tool messages will be included if they come after the AI message
                if "ai" in role:
                    last_assistant_idx = i
                    break
            start_idx = last_assistant_idx if last_assistant_idx is not None else 0
            for m in messages[start_idx:]:
                items.append(_map_message_to_foundry_item(m))
        items.append(ResponsesUserMessageItemParam(content=query))

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
                            "source": "foundry_memory",
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
