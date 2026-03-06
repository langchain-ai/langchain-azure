"""Azure AI Foundry Agent Service Tools for V2 (azure-ai-projects >= 2.0)."""

from typing import Dict, Literal, Optional

from azure.ai.projects.models import (
    CodeInterpreterContainerAuto,
    ImageGenToolInputImageMask,
    MemorySearchOptions,
    MemorySearchPreviewTool,
    Tool,
)
from azure.ai.projects.models import CodeInterpreterTool as V2CodeInterpreterTool
from azure.ai.projects.models import ImageGenTool as V2ImageGenTool
from azure.ai.projects.models import MCPTool as V2MCPTool
from pydantic import BaseModel, ConfigDict

from langchain_azure_ai._api.base import experimental


class AgentServiceBaseTool(BaseModel):
    """A tool that interacts with Azure AI Foundry Agent Service V2.

    Use this class to wrap tools from Azure AI Foundry for use with
    `PromptBasedAgentNode`.

    Example:
    ```python
    from langchain_azure_ai.agents.v2.prebuilt.tools import AgentServiceBaseTool
    from azure.ai.projects.models import CodeInterpreterTool

    code_interpreter_tool = AgentServiceBaseTool(tool=CodeInterpreterTool())
    ```

    Some tools require extra HTTP headers when calling the Responses API.
    For example, ``ImageGenTool`` requires an
    ``x-ms-oai-image-generation-deployment`` header:

    ```python
    from azure.ai.projects.models import ImageGenTool

    image_tool = AgentServiceBaseTool(
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
    """A wrapper around the Foundry ImageGenTool for use in AgentServiceBaseTool.

    This class exists to provide a consistent import path for users who want
    to use the ImageGenTool with AgentServiceBaseTool, without needing to
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
        """Initialize the ImageGenTool with the given parameters.

        Args:
            model: The image generation model to use.  Only "gpt-image-1"
                is currently supported.
            model_deployment: The name of the model deployment to use for this tool.
                This is required and must match the deployment name of the model in
                Foundry. It is sent as an extra header with each tool call.
            quality: The quality of the generated image.  Higher quality images take
                longer to generate and may consume more credits.
            size: The size of the generated image.  Larger images take longer to
                generate and may consume more credits.
            output_format: The format of the generated image.
            output_compression: The compression level for the output image, from 0-100.
                Higher compression levels result in smaller file sizes but lower image
                quality.
            moderation: The moderation level to apply to the generated image.  Higher
                moderation levels may result in safer images but could also lead to
                more false positives.
            background: The background type for the generated image.  "transparent" is
                only supported for PNG output format.
            input_image_mask: An optional mask to apply to the input image, for
                inpainting tasks.  The mask should be a binary image where white pixels
                indicate the area to be modified and black pixels indicate the area to
                be preserved.
            partial_images: The number of partial images to return in the response. If
                provided, the tool will return intermediate images as they are
                generated, which can be used to provide feedback or stop the generation
                early.
        """
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
    to use the CodeInterpreterTool with AgentServiceBaseTool, without needing
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
    """A wrapper around the Foundry MCPTool for use in AgentServiceBaseTool.

    This class exists to provide a consistent import path for users who want
    to use the MCPTool with AgentServiceBaseTool, without needing
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
        """Initialize the MCPTool.

        Args:
            server_label: A human-friendly label for the MCP server, shown in the UI.
            server_url: The URL of the MCP server to connect to.
            headers: Optional HTTP headers to include in requests to the MCP server.
            allowed_tools: Optional list of tool names that this MCP can approve. If
                not provided, the MCP will be able to approve calls to any tool.
            require_approval: Whether to require approval for each tool before
                calling it.
            project_connection_id: Optional ID of the project connection to use for
                this MCPTool. Connections are used to retrieve credentials to
                authenticate to the tool.
        """
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


@experimental()
class MemorySearchTool(AgentServiceBaseTool):
    """A wrapper around the Foundry MemorySearchPreviewTool for use in agents.

    This class exists to provide a consistent import path for users who want
    to use the MemorySearchPreviewTool with AgentServiceBaseTool, without needing
    to import from azure.ai.projects.models directly.
    """

    def __init__(
        self,
        memory_store_name: str,
        scope: str,
        search_options: MemorySearchOptions | None = None,
        update_delay: int | None = None,
    ):
        """Initialize the MemorySearchPreviewTool.

        Args:
            memory_store_name: The name of the Azure AI memory store to search.
            scope: The scope within the memory store to search.
            search_options: The options for the memory search.
            update_delay: If provided, the tool will update the search results in
                Foundry every ``update_delay`` seconds while the agent is running.

        """
        super().__init__(
            tool=MemorySearchPreviewTool(
                memory_store_name=memory_store_name,
                scope=scope,
                search_options=search_options,
                update_delay=update_delay,
            )
        )
