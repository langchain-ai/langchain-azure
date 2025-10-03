"""Azure AI Foundry Agent Service Tools."""

from azure.ai.agents.models import Tool
from pydantic import BaseModel, ConfigDict


class AgentServiceBaseTool(BaseModel):
    """A tool that interacts with Azure AI Foundry Agent Service.

    Use this class to wrap tools from Azure AI Foundry for use with
    DeclarativeChatAgentNode.

    Example:
        ... code-block:: python
            from langchain_azure_ai.tools.agent_service import AgentServiceBaseTool
            from azure.ai.agents.models import CodeInterpreterTool

            code_interpreter_tool = AgentServiceBaseTool(tool=CodeInterpreterTool())

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool: Tool
    """The tool definition from Azure AI Foundry."""
