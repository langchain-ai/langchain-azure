"""Azure AI Foundry Agent Service Tools for V2 (azure-ai-projects >= 2.0)."""


from azure.ai.projects.models import Tool
from pydantic import BaseModel, ConfigDict


class AgentServiceBaseToolV2(BaseModel):
    """A tool that interacts with Azure AI Foundry Agent Service V2.

    Use this class to wrap tools from Azure AI Foundry for use with
    PromptBasedAgentNodeV2.

    Example:
    ```python
    from langchain_azure_ai.agents.prebuilt.tools_v2 import AgentServiceBaseToolV2
    from azure.ai.projects.models import CodeInterpreterTool

    code_interpreter_tool = AgentServiceBaseToolV2(tool=CodeInterpreterTool())
    ```

    If your tool requires further configuration, you may need to use the
    Azure AI Foundry SDK directly to create and configure the tool.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool: Tool
    """The tool definition from Azure AI Foundry V2."""
