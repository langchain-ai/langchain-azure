"""Azure AI Foundry Agent Service Tools."""

from typing import Any, List

from azure.ai.agents.models import (
    FunctionToolDefinition,
    Tool,
    ToolResources,
)
from pydantic import BaseModel, ConfigDict


class AgentServiceBaseTool(BaseModel):
    """A tool that interacts with Azure AI Foundry Agent Service.

    Use this class to wrap tools from Azure AI Foundry for use with
    DeclarativeChatAgentNode.

    Example:
        ... code-block:: python
            from langchain_azure_ai.agents.prebuilt.tools import AgentServiceBaseTool
            from azure.ai.agents.models import CodeInterpreterTool

            code_interpreter_tool = AgentServiceBaseTool(tool=CodeInterpreterTool())

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool: Tool
    """The tool definition from Azure AI Foundry."""


class _OpenAIFunctionTool(Tool[FunctionToolDefinition]):
    """A tool that wraps OpenAI function definitions."""

    def __init__(self, definitions: List[FunctionToolDefinition]):
        """Initialize the OpenAIFunctionTool with function definitions.

        Args:
        definitions: A list of function definitions to be used by the tool.
        """
        self._definitions = definitions

    @property
    def definitions(self) -> List[FunctionToolDefinition]:
        """Get the function definitions.

        Returns:
            A list of function definitions.
        """
        return self._definitions

    @property
    def resources(self) -> ToolResources:
        """Get the tool resources for the agent.

        Returns:
            The tool resources.
        """
        return ToolResources()

    def execute(self, tool_call: Any) -> Any:
        """Execute the tool with the provided tool call.

        :param Any tool_call: The tool call to execute.
        :return: The output of the tool operations.
        """
        pass
