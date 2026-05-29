"""Azure AI Memory middleware for LangChain/LangGraph agents."""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Sequence

from azure.ai.projects import AIProjectClient
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from langchain.agents.middleware import AgentMiddleware, AgentState, Runtime
from langchain.tools import BaseTool
from langchain_core.messages import BaseMessage
from openai.types.responses import EasyInputMessageParam

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.utils.env import get_from_env

logger = logging.getLogger(__name__)


def _map_message_to_memory_item(
    message: BaseMessage,
) -> Optional[EasyInputMessageParam]:
    """Map a LangChain message to an Azure AI Memory item when relevant."""
    msg_type = getattr(message, "type", "") or message.__class__.__name__
    msg_type = msg_type.lower()
    role = getattr(message, "role", None)
    target_role: Literal["user", "assistant"]

    if role == "user" or "human" in msg_type:
        target_role = "user"
    elif role == "assistant" or "ai" in msg_type:
        target_role = "assistant"
    else:
        return None

    content = (
        message.content if isinstance(message.content, str) else str(message.content)
    )
    return EasyInputMessageParam(content=content, role=target_role)


@experimental()
class AzureAIMemoryMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Middleware that periodically extracts memories from turns into Azure AI Memory.

    This middleware collects messages from the agent's state after each turn and
    sends them to Azure AI Memory in batches. It supports both user and assistant
    messages, configurable by the `roles` parameter. Use the memory store
    configuration to control which memories are stored and how. To retrieve those
    memories back into your agent, use the :class:`AzureAIMemoryRetrieverTool`
    in your agent.

    Example usage:

    ```python
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model
    from langchain_azure_ai.agents.middleware import AzureAIMemoryMiddleware

    memory_middleware = AzureAIMemoryMiddleware(
        store_name="my-memory-store",
        scope="my-scope",
        update_every_n_turns=1,
        roles=["user", "assistant"],
        project_endpoint="https://resource.services.ai.azure.com/api/projects/my-project",
        credential=DefaultAzureCredential(),
    )
    tools = [memory_middleware.get_retriever_tool()]

    agent = create_agent(
        model=init_chat_model("azure_ai:gpt-5"),
        tools=tools,
        middleware=[memory_middleware],
    )
    ```

    This class doesn't create the memory store itself, so make sure to create the
    specified memory store before using the middleware.
    """

    name: str = "azure_ai_memory"
    tools: list = []

    def __init__(
        self,
        store_name: str,
        scope: str,
        *,
        messages_key: str = "messages",
        update_every_n_turns: int = 1,
        roles: Sequence[Literal["user", "assistant"]] = ("user",),
        project_endpoint: Optional[str] = None,
        credential: Optional[TokenCredential] = None,
        update_delay: Optional[int] = 0,
    ) -> None:
        """Initialize middleware for periodic memory updates.

        Args:
            store_name: Azure AI Memory store name.
            scope: Memory scope identifier.
            update_every_n_turns: Number of agent turns to buffer before flushing.
            roles: Message roles to remember. Allowed values are ``"user"`` and
                ``"assistant"``. Defaults to ``("user",)``. Pass
                ``["user", "assistant"]`` to remember both roles.
            project_endpoint: Azure AI Foundry project endpoint.
            credential: Token credential used to authenticate to Azure services.
            update_delay: Optional update delay for memory store updates.
            messages_key: Key in the agent state where the message list is
                stored. Defaults to "messages".
        """
        if update_every_n_turns < 1:
            raise ValueError("update_every_n_turns must be >= 1.")
        if not all(role in {"user", "assistant"} for role in roles):
            raise ValueError(
                f"roles must only contain 'user' and/or 'assistant', got: {roles!r}"
            )

        resolved_project_endpoint = project_endpoint or get_from_env(
            "project_endpoint",
            ["AZURE_AI_PROJECT_ENDPOINT", "FOUNDRY_PROJECT_ENDPOINT"],
        )
        cred: TokenCredential = credential or DefaultAzureCredential()
        client = AIProjectClient(
            endpoint=resolved_project_endpoint,
            credential=cred,
            user_agent="langchain-azure-ai",
        )
        if not hasattr(client, "beta") or not hasattr(client.beta, "memory_stores"):
            raise ValueError(
                "AzureAIMemoryMiddleware requires azure-ai-projects>=2.0.0b4. "
            )

        self._client = client
        self.project_endpoint = resolved_project_endpoint
        self._store_name = store_name
        self._scope = scope
        self._update_every_n_turns = update_every_n_turns
        self._update_delay = update_delay
        self._pending_turns = 0
        self._pending_items: list[EasyInputMessageParam] = []
        self._processed_message_count = 0
        self._previous_update_id: Optional[str] = None
        self._roles = set(roles)
        self._messages_key = messages_key

    def get_retriever_tool(self, **kwargs: Any) -> BaseTool:
        """Get a tool that can be used to retrieve memories from Azure AI Memory.

        This tool is pre-configured to retrieve memories from the same store/scope as
        the middleware, so you don't need to specify those parameters again. You can
        also configure additional parameters for the retriever tool via ``kwargs``. For
        example, you can specify the number of memories to retrieve with parameter
        ``k``.

        """
        from langchain_azure_ai.tools import AzureAIMemoryRetrieverTool

        if "store_name" in kwargs:
            raise ValueError(
                "store_name is already set by the middleware and cannot be overridden."
            )

        scope = kwargs.pop("scope", self._scope)

        return AzureAIMemoryRetrieverTool(
            project_endpoint=self.project_endpoint,
            store_name=self._store_name,
            scope=scope,
            **kwargs,
        )

    def _collect_new_items(self, state: AgentState[Any]) -> list[EasyInputMessageParam]:
        """Collect new user/assistant messages since the previous middleware run."""
        messages = state.get(self._messages_key)
        if not isinstance(messages, list):
            return []

        if self._processed_message_count > len(messages):
            self._processed_message_count = 0

        new_messages = messages[self._processed_message_count :]
        self._processed_message_count = len(messages)

        items: list[EasyInputMessageParam] = []
        for message in new_messages:
            if not isinstance(message, BaseMessage):
                continue
            item = _map_message_to_memory_item(message)
            if item is not None and item["role"] in self._roles:
                items.append(item)
        return items

    def _flush_pending_updates(self) -> None:
        """Send accumulated message delta to Azure AI Memory."""
        if not self._pending_items:
            return

        try:
            poller = self._client.beta.memory_stores.begin_update_memories(
                name=self._store_name,
                scope=self._scope,
                items=self._pending_items,
                previous_update_id=self._previous_update_id,
                update_delay=self._update_delay,
            )
            self._previous_update_id = getattr(poller, "update_id", None)
            self._pending_items = []
            self._pending_turns = 0
        except Exception as ex:
            logger.warning(
                "Failed to update Azure AI Memory from middleware: %s",
                ex,
                exc_info=False,
            )

    def after_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Queue memory updates from the latest turn and flush periodically."""
        new_items = self._collect_new_items(state)
        if not new_items:
            return None

        self._pending_items.extend(new_items)
        self._pending_turns += 1
        if self._pending_turns >= self._update_every_n_turns:
            self._flush_pending_updates()
        return None

    async def aafter_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async hook equivalent of :meth:`after_agent`."""
        return self.after_agent(state, runtime)
