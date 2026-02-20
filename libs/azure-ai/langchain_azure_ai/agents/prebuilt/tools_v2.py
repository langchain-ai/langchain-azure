"""Azure AI Foundry Agent Service Tools for V2 (azure-ai-projects >= 2.0)."""

from typing import Any, List, Optional

from azure.ai.projects.models import (
    Tool,
)
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


def _get_v2_tool_definitions(
    tools: List[Any],
) -> List[Tool]:
    """Convert a list of tools to V2 Tool definitions for the agent.

    Separates tools into:
    - AgentServiceBaseToolV2 tools (native V2 tools like CodeInterpreterTool)
    - BaseTool / callable tools (converted to FunctionTool definitions)

    Args:
        tools: A list of tools to convert.

    Returns:
        A list of V2 Tool definitions.
    """
    from azure.ai.projects.models import FunctionTool as V2FunctionTool

    from langchain_core.tools import BaseTool
    from langchain_core.utils.function_calling import convert_to_openai_function

    tool_definitions: List[Tool] = []

    for tool in tools:
        if isinstance(tool, AgentServiceBaseToolV2):
            tool_definitions.append(tool.tool)
        elif isinstance(tool, BaseTool):
            function_def = convert_to_openai_function(tool)
            tool_definitions.append(
                V2FunctionTool(
                    name=function_def["name"],
                    description=function_def.get("description", ""),
                    parameters=function_def.get("parameters", {}),
                )
            )
        elif callable(tool):
            function_def = convert_to_openai_function(tool)
            tool_definitions.append(
                V2FunctionTool(
                    name=function_def["name"],
                    description=function_def.get("description", ""),
                    parameters=function_def.get("parameters", {}),
                )
            )
        else:
            raise ValueError(
                "Each tool must be an AgentServiceBaseToolV2, BaseTool, or a "
                f"callable. Got {type(tool)}"
            )

    return tool_definitions
