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
    """Tool that retrieves relevant memories from Azure AI Foundry Memory."""

    name: str = "azure_ai_memory_retriever"
    description: str = (
        "Searches Azure AI Memory and returns relevant long-term user memories for "
        "the provided query."
    )
    args_schema: Annotated[Optional[ArgsSchema], SkipValidation()] = (
        MemoryRetrieverInput
    )

    store_name: str
    scope: str
    k: int = 5
    project_endpoint: Optional[str] = None
    credential: Optional[TokenCredential] = None

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
            memory_id = doc.metadata.get("memory_id")
            prefix = f"[{memory_id}] " if memory_id else ""
            lines.append(f"- {prefix}{doc.page_content}")
        return "\n".join(lines)
