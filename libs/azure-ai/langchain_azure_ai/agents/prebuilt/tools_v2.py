"""Azure AI Foundry Agent Service Tools for V2 (azure-ai-projects >= 2.0)."""


from typing import Dict, Optional

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

    Some tools require extra HTTP headers when calling the Responses API.
    For example, ``ImageGenTool`` requires an
    ``x-ms-oai-image-generation-deployment`` header:

    ```python
    from azure.ai.projects.models import ImageGenTool

    image_tool = AgentServiceBaseToolV2(
        tool=ImageGenTool(model="gpt-image-1", quality="low", size="1024x1024"),
        extra_headers={
            "x-ms-oai-image-generation-deployment": "gpt-image-1",
        },
    )
    ```

    All ``extra_headers`` from every tool are merged and sent with each
    ``responses.create()`` call made by the agent node.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool: Tool
    """The tool definition from Azure AI Foundry V2."""

    extra_headers: Optional[Dict[str, str]] = None
    """Optional extra HTTP headers required by this tool.

    These headers are merged across all tools and passed to every
    ``openai_client.responses.create()`` call.  For example,
    ``ImageGenTool`` needs
    ``{"x-ms-oai-image-generation-deployment": "<deployment-name>"}``.
    """
