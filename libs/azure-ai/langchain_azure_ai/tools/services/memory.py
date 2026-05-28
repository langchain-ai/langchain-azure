"""Tools for saving and retrieving memories from Azure AI Foundry Memory stores.

These tools allow agents to store and retrieve memories scoped to a user or
agent using the Azure AI Foundry Memory API.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, Optional

from azure.ai.projects import AIProjectClient
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import BaseModel, ConfigDict, PrivateAttr, SkipValidation, model_validator

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.utils.env import get_project_endpoint

logger = logging.getLogger(__name__)


class RetrieveMemoryInput(BaseModel):
    """Input schema for the AzureAIMemoryRetrieveTool."""

    query: str
    """The natural-language query used to search for relevant memories."""


class SaveMemoryInput(BaseModel):
    """Input schema for the AzureAIMemorySaveTool."""

    content: str
    """The text content to save as a memory."""


@experimental()
class AzureAIMemoryRetrieveTool(BaseTool):
    """Tool that retrieves relevant memories from an Azure AI Foundry Memory store.

    The ``scope`` parameter determines which set of memories to query.  Use a
    user-scoped value (e.g. ``"user:{user_id}"``) to recall per-user facts, or
    an agent-scoped value (e.g. ``"agent:{agent_id}"``) for agent-level
    knowledge.

    **Setup:**

    Set the required environment variable before use:

    ```bash
    export AZURE_AI_PROJECT_ENDPOINT="<YOUR_PROJECT_ENDPOINT>"
    ```

    **Examples:**

    User-scoped retrieval:

    ```python
    from langchain_azure_ai.tools import AzureAIMemoryRetrieveTool

    tool = AzureAIMemoryRetrieveTool(
        store_name="my_store",
        scope="user:alice",
        k=5,
    )
    result = tool.invoke({"query": "What are my coffee preferences?"})
    ```

    Agent-scoped retrieval:

    ```python
    tool = AzureAIMemoryRetrieveTool(
        store_name="my_store",
        scope="agent:support-bot",
        k=3,
    )
    result = tool.invoke({"query": "Recent resolved tickets"})
    ```
    """

    name: str = "azure_ai_memory_retrieve"
    """The name of the tool."""

    description: str = (
        "Retrieves relevant memories from Azure AI Foundry Memory based on a "
        "natural-language query. Use this tool to recall previously stored facts, "
        "preferences, or context for a given user or agent scope."
    )
    """The description of the tool."""

    args_schema: Annotated[Optional[ArgsSchema], SkipValidation()] = RetrieveMemoryInput
    """The input args schema for the tool."""

    store_name: str
    """Name of the Azure AI Foundry Memory store to search."""

    scope: str
    """Memory scope that isolates memories, e.g. ``'user:{user_id}'`` or
    ``'agent:{agent_id}'``."""

    k: int = 5
    """Maximum number of memories to return."""

    project_endpoint: Optional[str] = None
    """Azure AI Foundry project endpoint.  Falls back to the
    ``AZURE_AI_PROJECT_ENDPOINT`` environment variable when not set."""

    credential: Optional[TokenCredential] = None
    """Azure credential used to authenticate against the project endpoint.
    Defaults to :class:`~azure.identity.DefaultAzureCredential` when not
    provided."""

    _client: Any = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _resolve_endpoint(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve ``project_endpoint`` from the environment when not provided."""
        endpoint = get_project_endpoint(values, nullable=True)
        if not endpoint:
            raise ValueError(
                "project_endpoint must be provided either as a parameter or via "
                "the AZURE_AI_PROJECT_ENDPOINT environment variable."
            )
        values["project_endpoint"] = endpoint
        return values

    @model_validator(mode="after")
    def _initialize_client(self) -> AzureAIMemoryRetrieveTool:
        """Initialize the Azure AI Project client."""
        cred: TokenCredential = self.credential or DefaultAzureCredential()
        client = AIProjectClient(
            endpoint=self.project_endpoint,
            credential=cred,
            user_agent="langchain-azure-ai",
        )
        if not hasattr(client, "beta") or not hasattr(client.beta, "memory_stores"):
            raise ImportError(
                "AzureAIMemoryRetrieveTool requires azure-ai-projects>=2.0.0b4. "
                "Install the v2 extra: pip install 'langchain-azure-ai[v2]'"
            )
        self._client = client
        return self

    def _run(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Retrieve memories relevant to *query* from the configured store.

        Args:
            query: Natural-language search query.
            run_manager: Optional callback manager.

        Returns:
            A formatted string listing the retrieved memories, or a message
            indicating that no memories were found.
        """
        from azure.ai.projects.models import MemorySearchOptions

        result = self._client.beta.memory_stores.search_memories(
            name=self.store_name,
            scope=self.scope,
            items=query,
            options=MemorySearchOptions(max_memories=self.k),
        )

        memories = []
        try:
            for entry in result.memories:
                mem_item = entry.memory_item
                memories.append(mem_item.content)
        except Exception as exc:
            logger.warning("Error parsing memory search results: %s", exc)

        if not memories:
            return "No relevant memories found."

        return "\n".join(f"- {m}" for m in memories)


@experimental()
class AzureAIMemorySaveTool(BaseTool):
    """Tool that saves a memory to an Azure AI Foundry Memory store.

    The ``scope`` parameter determines which memory namespace the content is
    saved to.  Use a user-scoped value (e.g. ``"user:{user_id}"``) to persist
    per-user facts, or an agent-scoped value (e.g. ``"agent:{agent_id}"``) for
    agent-level knowledge.

    The tool uses the Azure AI Foundry Memory extraction pipeline: the service
    processes the provided content and stores the derived memory items.  Set
    ``update_delay=0`` (the default) for immediate processing.

    **Setup:**

    Set the required environment variable before use:

    ```bash
    export AZURE_AI_PROJECT_ENDPOINT="<YOUR_PROJECT_ENDPOINT>"
    ```

    **Examples:**

    User-scoped save:

    ```python
    from langchain_azure_ai.tools import AzureAIMemorySaveTool

    tool = AzureAIMemorySaveTool(
        store_name="my_store",
        scope="user:alice",
    )
    result = tool.invoke({"content": "Alice prefers dark roast coffee."})
    ```

    Agent-scoped save:

    ```python
    tool = AzureAIMemorySaveTool(
        store_name="my_store",
        scope="agent:support-bot",
    )
    result = tool.invoke({"content": "Ticket #4821 was resolved on 2025-05-01."})
    ```
    """

    name: str = "azure_ai_memory_save"
    """The name of the tool."""

    description: str = (
        "Saves a memory to Azure AI Foundry Memory. Use this tool to persist "
        "facts, preferences, or context that should be recalled in future "
        "conversations for a given user or agent scope."
    )
    """The description of the tool."""

    args_schema: Annotated[Optional[ArgsSchema], SkipValidation()] = SaveMemoryInput
    """The input args schema for the tool."""

    store_name: str
    """Name of the Azure AI Foundry Memory store to update."""

    scope: str
    """Memory scope that isolates memories, e.g. ``'user:{user_id}'`` or
    ``'agent:{agent_id}'``."""

    update_delay: int = 0
    """Seconds to wait before processing the memory update.  Defaults to ``0``
    for immediate processing.  Set to a larger value to batch multiple updates
    within the delay window."""

    project_endpoint: Optional[str] = None
    """Azure AI Foundry project endpoint.  Falls back to the
    ``AZURE_AI_PROJECT_ENDPOINT`` environment variable when not set."""

    credential: Optional[TokenCredential] = None
    """Azure credential used to authenticate against the project endpoint.
    Defaults to :class:`~azure.identity.DefaultAzureCredential` when not
    provided."""

    _client: Any = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _resolve_endpoint(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve ``project_endpoint`` from the environment when not provided."""
        endpoint = get_project_endpoint(values, nullable=True)
        if not endpoint:
            raise ValueError(
                "project_endpoint must be provided either as a parameter or via "
                "the AZURE_AI_PROJECT_ENDPOINT environment variable."
            )
        values["project_endpoint"] = endpoint
        return values

    @model_validator(mode="after")
    def _initialize_client(self) -> AzureAIMemorySaveTool:
        """Initialize the Azure AI Project client."""
        cred: TokenCredential = self.credential or DefaultAzureCredential()
        client = AIProjectClient(
            endpoint=self.project_endpoint,
            credential=cred,
            user_agent="langchain-azure-ai",
        )
        if not hasattr(client, "beta") or not hasattr(client.beta, "memory_stores"):
            raise ImportError(
                "AzureAIMemorySaveTool requires azure-ai-projects>=2.0.0b4. "
                "Install the v2 extra: pip install 'langchain-azure-ai[v2]'"
            )
        self._client = client
        return self

    def _run(
        self,
        content: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Save *content* as a memory in the configured store.

        Args:
            content: The text to save as a memory.
            run_manager: Optional callback manager.

        Returns:
            A confirmation message on success, or an error description on
            failure.
        """
        try:
            self._client.beta.memory_stores.begin_update_memories(
                name=self.store_name,
                scope=self.scope,
                items=content,
                update_delay=self.update_delay,
            )
        except Exception as exc:
            logger.warning("Failed to save memory: %s", exc)
            return f"Failed to save memory: {exc}"

        return "Memory saved successfully."
