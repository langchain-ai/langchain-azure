"""Azure AI Foundry Memory integration with LangChain."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
)

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from openai.types.responses import EasyInputMessageParam

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.utils.env import get_from_dict_or_env

if TYPE_CHECKING:
    from langchain_azure_ai.retrievers.azure_ai_memory_retriever import (
        AzureAIMemoryRetriever,
    )

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
class AzureAIMemoryChatMessageHistory(BaseChatMessageHistory):
    """History wrapper that updates Azure AI Foundry Memory per chat turn.

    This class decorates a LangChain `BaseChatMessageHistory`, preserving
    short-term transcript storage while forwarding each turn to Foundry Memory
    for long-term extraction and consolidation.

    **Setup:**

    We will need to set the required environment variables:

    ```bash
    export AZURE_AI_PROJECT_ENDPOINT="<YOUR_PROJECT_ENDPOINT>"
    export MEMORY_STORE_CHAT_MODEL_DEPLOYMENT_NAME="<YOUR_CHAT_MODEL_DEPLOYMENT>"
    export MEMORY_STORE_EMBEDDING_MODEL_DEPLOYMENT_NAME="<YOUR_EMBEDDING_DEPLOYMENT>"
    ```

    Before using this class, create the memory store explicitly in your Azure AI
    Foundry project. The store is not created automatically by this integration.

    ```python
    import os

    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import (
        MemoryStoreDefaultDefinition,
        MemoryStoreDefaultOptions,
    )
    from azure.core.exceptions import ResourceNotFoundError
    from azure.identity import DefaultAzureCredential

    client = AIProjectClient(
        endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )

    store_name = "my_store"
    try:
        client.beta.memory_stores.get(store_name)
    except ResourceNotFoundError:
        client.beta.memory_stores.create(
            name=store_name,
            description="Long-term memory store",
            definition=MemoryStoreDefaultDefinition(
                chat_model=os.environ[
                    "MEMORY_STORE_CHAT_MODEL_DEPLOYMENT_NAME"
                ],
                embedding_model=os.environ[
                    "MEMORY_STORE_EMBEDDING_MODEL_DEPLOYMENT_NAME"
                ],
                options=MemoryStoreDefaultOptions(
                    user_profile_enabled=True,
                    chat_summary_enabled=True,
                ),
            ),
        )
    ```

    **Examples**

    With explicit endpoint and credential:

    ```python
    from azure.identity import DefaultAzureCredential
    from langchain_core.chat_history import InMemoryChatMessageHistory

    history = AzureAIMemoryChatMessageHistory(
        project_endpoint="https://myproject.api.azureml.ms",
        credential=DefaultAzureCredential(),
        store_name="my_store",
        scope="user:123",
        base_history=InMemoryChatMessageHistory(),
    )
    ```

    With endpoint from environment variable:

    ```python
    history = AzureAIMemoryChatMessageHistory(
        store_name="my_store",
        scope="user:123",
        base_history=InMemoryChatMessageHistory(),
    )
    ```
    """

    def __init__(
        self,
        store_name: str,
        scope: str,
        base_history: BaseChatMessageHistory,
        *,
        project_endpoint: Optional[str] = None,
        credential: Optional[TokenCredential] = None,
        update_delay: Optional[int] = None,  # None => service default (≈300s)
        role_mapper: Optional[Callable[[BaseMessage], EasyInputMessageParam]] = None,
    ):
        """Initialize history-backed integration with Azure AI Foundry Memory.

        Args:
            store_name: Memory store name in Azure AI Foundry.
            scope: Memory scope (for example, `user:{user_id}` or
                `tenant:{org_id}`) used for long-term recall across sessions.
            base_history: Underlying short-term history instance to wrap.
            project_endpoint: Azure AI project endpoint. If not provided,
                reads from `AZURE_AI_PROJECT_ENDPOINT`.
            credential: Azure credential for authentication. If not provided,
                uses `DefaultAzureCredential()`.
            update_delay: Optional delay before memory extraction. Use `None`
                for service default (about 300s) or `0` for immediate updates.
            role_mapper: Optional custom function to map LangChain messages
                to Foundry message items.
        """
        # Read project_endpoint from environment if not provided
        self._project_endpoint = get_from_dict_or_env(
            {"project_endpoint": project_endpoint},
            "project_endpoint",
            "AZURE_AI_PROJECT_ENDPOINT",
        )

        if not self._project_endpoint:
            raise ValueError(
                "project_endpoint must be provided either as a parameter or via "
                "the AZURE_AI_PROJECT_ENDPOINT environment variable."
            )

        # Use provided credential or default
        cred: TokenCredential = credential or DefaultAzureCredential()

        # Create AIProjectClient with user-agent for monitoring.
        # Requires azure-ai-projects>=2.0.0b4 for beta.memory_stores support.
        from azure.ai.projects import AIProjectClient

        client = AIProjectClient(
            endpoint=self._project_endpoint,
            credential=cred,
            user_agent="langchain-azure-ai",
        )
        if not hasattr(client, "beta") or not hasattr(client.beta, "memory_stores"):
            raise ImportError(
                "AzureAIMemoryChatMessageHistory requires azure-ai-projects>=2.0.0b4. "
                "Install the v2 extra: pip install 'langchain-azure-ai[v2]'"
            )
        self._client = client

        self._store = store_name
        self._scope = scope
        self._base = base_history
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
            self._client.beta.memory_stores.begin_update_memories(  # type: ignore[attr-defined]
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

    def get_retriever(self, *, k: int = 5) -> AzureAIMemoryRetriever:
        """Create a retriever bound to this store/scope/history.

        History-bound retrievers always use incremental search with multi-turn
        conversation context for better contextual memory retrieval.

        Args:
            k: Maximum number of memories to retrieve

        Returns:
            An AzureAIMemoryRetriever instance bound to this history
            for incremental search
        """
        from langchain_azure_ai.retrievers.azure_ai_memory_retriever import (
            AzureAIMemoryRetriever,
        )

        return AzureAIMemoryRetriever(
            history_ref=self,
            k=k,
        )

    # helper kept private; override via role_mapper if needed
    def _map_lc_message_to_foundry_item(
        self, message: BaseMessage
    ) -> EasyInputMessageParam:
        """Map LangChain message to Foundry message item.

        Args:
            message: LangChain message to map

        Returns:
            Foundry message item parameter
        """
        if self._role_mapper:
            return self._role_mapper(message)

        return _map_message_to_foundry_item(message)
