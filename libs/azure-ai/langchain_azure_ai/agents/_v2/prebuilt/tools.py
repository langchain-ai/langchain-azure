"""Azure AI Foundry Agent Service Tools for V2 (azure-ai-projects >= 2.0)."""

from typing import Dict, Literal, Optional

from azure.ai.projects.models import (
    CodeInterpreterContainerAuto,
    ImageGenToolInputImageMask,
    Tool,
)
from azure.ai.projects.models import CodeInterpreterTool as V2CodeInterpreterTool
from azure.ai.projects.models import ImageGenTool as V2ImageGenTool
from azure.ai.projects.models import MCPTool as V2MCPTool
from pydantic import BaseModel, ConfigDict


class AgentServiceBaseTool(BaseModel):
    """A tool that interacts with Azure AI Foundry Agent Service V2.

    Use this class to wrap tools from Azure AI Foundry for use with
    PromptBasedAgentNodeV2.

    Example:
    ```python
    from langchain_azure_ai.agents.prebuilt.tools_v2 import AgentServiceBaseToolV2
    from azure.ai.projects.models import CodeInterpreterTool

    code_interpreter_tool = AgentServiceBaseTool(tool=CodeInterpreterTool())
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

    requires_approval: bool = False
    """Whether this tool requires human approval before execution.

    When ``True``, the agent graph will include an MCP approval node
    that pauses execution via ``interrupt()`` so the user can approve
    or deny the tool call before it proceeds.
    """


class ImageGenTool(AgentServiceBaseTool):
    """A wrapper around the Foundry ImageGenTool for use in AgentServiceBaseToolV2.

    This class exists to provide a consistent import path for users who want
    to use the ImageGenTool with AgentServiceBaseToolV2, without needing to
    import from azure.ai.projects.models directly.
    """

    def __init__(
        self,
        model: Literal["gpt-image-1"] = "gpt-image-1",
        model_deployment: str = "gpt-image-1",
        quality: Literal["low", "medium", "high", "auto"] | None = None,
        size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"] | None = None,
        output_format: Literal["png", "webp", "jpeg"] | None = None,
        output_compression: int | None = None,
        moderation: Literal["auto", "low"] | None = None,
        background: Literal["transparent", "opaque", "auto"] | None = None,
        input_image_mask: ImageGenToolInputImageMask | None = None,
        partial_images: int | None = None,
    ):
        """Initialize the ImageGenTool with the given parameters."""
        super().__init__(
            tool=V2ImageGenTool(
                model=model,
                quality=quality,
                size=size,
                output_format=output_format,
                output_compression=output_compression,
                moderation=moderation,
                background=background,
                input_image_mask=input_image_mask,
                partial_images=partial_images,
            ),
            extra_headers={"x-ms-oai-image-generation-deployment": model_deployment},
        )


class CodeInterpreterTool(AgentServiceBaseTool):
    """A wrapper around the Foundry CodeInterpreterTool.

    This class exists to provide a consistent import path for users who want
    to use the CodeInterpreterTool with AgentServiceBaseToolV2, without needing
    to import from azure.ai.projects.models directly.
    """

    def __init__(
        self,
    ) -> None:
        """Initialize the CodeInterpreterTool with the given parameters."""
        super().__init__(
            tool=V2CodeInterpreterTool(container=CodeInterpreterContainerAuto())
        )


class MCPTool(AgentServiceBaseTool):
    """A wrapper around the Foundry MCPTool for use in AgentServiceBaseToolV2.

    This class exists to provide a consistent import path for users who want
    to use the MCPTool with AgentServiceBaseToolV2, without needing
    to import from azure.ai.projects.models directly.
    """

    def __init__(
        self,
        server_label: str,
        server_url: str,
        headers: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
        require_approval: Literal["always", "never"] | None = None,
        project_connection_id: str | None = None,
    ):
        """Initialize the MCPTool."""
        super().__init__(
            tool=V2MCPTool(
                server_label=server_label,
                server_url=server_url,
                headers=headers,
                allowed_tools=allowed_tools,
                require_approval=require_approval,
                project_connection_id=project_connection_id,
            ),
            requires_approval=require_approval not in (None, "never"),
        )
