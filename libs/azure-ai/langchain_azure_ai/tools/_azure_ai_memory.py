"""Tools for Azure AI Foundry Memory."""

from __future__ import annotations

from typing import Annotated, Optional

from azure.core.credentials import TokenCredential
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SkipValidation,
    model_validator,
)

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.retrievers.azure_ai_memory_retriever import (
    AzureAIMemoryRetriever,
)


class MemoryRetrieverInput(BaseModel):
    """Input schema for AzureAIMemoryRetrieverTool."""

    query: str = Field(description="User query to search related long-term memories.")


@experimental()
class AzureAIMemoryRetrieverTool(BaseTool):
    """Tool that retrieves relevant memories from Azure AI Foundry Memory.

    This tool is designed to be used in conjunction with the
    :class:`AzureAIMemoryMiddleware` to enable agents to store and retrieve
    long-term memories in Azure AI Foundry.

    **Example usage:**

    ```python
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model
    from langchain_azure_ai.agents.middleware import AzureAIMemoryMiddleware
    from langchain_azure_ai.tools import AzureAIMemoryRetrieverTool

    memory_middleware = AzureAIMemoryMiddleware(
        project_endpoint="https://resource...",
        store_name="agent-memories",
        scope="general",
        roles=["user"],
    )

    retriever_tool = memory_middleware.get_retriever_tool(k=1)

    agent = create_agent(
        system_prompt=(
            "You are a helpful assistant that can remember information over time."
            "Use the provided tool to retrieve relevant memories when needed."
        ),
        tools=[retriever_tool],
        model=init_chat_model("azure_ai:gpt-4o"),
        middleware=[memory_middleware],
    )
    ```
    """

    name: str = "azure_ai_memory_retriever"

    description: str = (
        "Searches Azure AI Memory and returns relevant long-term user memories for "
        "the provided query."
    )

    args_schema: Annotated[Optional[ArgsSchema], SkipValidation()] = (
        MemoryRetrieverInput
    )

    store_name: str
    """Name of the store where the memories are saved."""

    scope: str
    """Scope from which to retrieve memories."""

    k: int = 5
    """Number of relevant memories to retrieve."""

    project_endpoint: Optional[str] = None
    """Azure AI Foundry project endpoint URL."""

    credential: Optional[TokenCredential] = None
    """Azure AI Foundry credential."""

    _retriever: AzureAIMemoryRetriever = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _initialize_retriever(self) -> AzureAIMemoryRetrieverTool:
        """Initialize the underlying AzureAIMemoryRetriever instance."""
        self._retriever = AzureAIMemoryRetriever(
            project_endpoint=self.project_endpoint,
            credential=self.credential,
            store_name=self.store_name,
            scope=self.scope,
            k=self.k,
        )
        return self

    def _run(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Search and return matching memories."""
        del run_manager
        docs = self._retriever.invoke(query)
        if not docs:
            return "No relevant memories found."

        lines = []
        for doc in docs:
            memory_scope = doc.metadata.get("scope")
            prefix = f"[{memory_scope}] " if memory_scope else ""
            lines.append(f"- {prefix}{doc.page_content}")
        return "\n".join(lines)
