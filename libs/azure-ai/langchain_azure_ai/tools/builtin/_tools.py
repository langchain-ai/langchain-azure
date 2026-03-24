"""Built-in server-side tools for OpenAI models deployed in Azure AI Foundry."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from openai.types.responses.file_search_tool_param import (
    Filters as FileSearchFilters,
)
from openai.types.responses.file_search_tool_param import (
    RankingOptions,
)
from openai.types.responses.response_input_item_param import (
    McpApprovalResponse,
)
from openai.types.responses.tool_param import (
    CodeInterpreter,
    CodeInterpreterContainerCodeInterpreterToolAuto,
    FileSearchToolParam,
    ImageGeneration,
    ImageGenerationInputImageMask,
    Mcp,
    McpAllowedTools,
    McpRequireApproval,
    WebSearchToolParam,
)
from openai.types.responses.web_search_tool_param import (
    Filters as WebSearchFilters,
)
from openai.types.responses.web_search_tool_param import (
    UserLocation,
)

from langchain_azure_ai._api.base import experimental

# Re-export SDK types that users commonly need when constructing tools,
# so they can be imported from this package without reaching into openai internals.
__all__ = [
    "BuiltinTool",
    "CodeInterpreterTool",
    "FileSearchTool",
    "FileSearchFilters",
    "ImageGenerationInputImageMask",
    "ImageGenerationTool",
    "McpAllowedTools",
    "McpRequireApproval",
    "McpTool",
    "RankingOptions",
    "UserLocation",
    "WebSearchFilters",
    "WebSearchTool",
    "McpApprovalResponse",
]


class BuiltinTool(dict):  # type: ignore[type-arg]
    """Base class for server-side built-in tools.

    Inherits from :class:`dict` so instances can be passed directly to
    ``model.bind_tools()`` without additional conversion.  Subclasses build
    the underlying :mod:`openai.types.responses` TypedDict and pass it to
    ``super().__init__(**sdk_typed_dict)`` so the dict payload always matches
    the OpenAI SDK schema.

    Some tools require extra HTTP headers to be sent with every inference API
    request.  Subclasses that need this should set ``self._request_headers``
    (a plain :class:`dict`) in their ``__init__``.  The
    :attr:`request_headers` property exposes these headers to ``bind_tools``
    implementations in the model classes, which merge and forward them to the
    underlying API client.

    Example – defining a custom built-in tool::

        class MyTool(BuiltinTool):
            def __init__(self, option: str = "default") -> None:
                super().__init__(type="my_tool", option=option)
    """

    @property
    def request_headers(self) -> Dict[str, str]:
        """Extra HTTP request headers required when this tool is active.

        These headers are injected into every inference API call made after
        ``model.bind_tools([this_tool, ...])``.  The default implementation
        returns an empty dict; subclasses set ``self._request_headers``
        in their ``__init__`` to declare requirements.

        Returns:
            A mapping of header name to header value.
        """
        return getattr(self, "_request_headers", {})


# ---------------------------------------------------------------------------
# Code Interpreter
# ---------------------------------------------------------------------------


@experimental()
class CodeInterpreterTool(BuiltinTool):
    """A tool that runs Python code server-side to help generate a response.

    The model can write and execute Python code within a sandboxed container
    and include the results in its response.

    Wraps :class:`openai.types.responses.tool_param.CodeInterpreter`.

    Example::

        from langchain.chat_models import init_chat_model
        from langchain_azure_ai.tools.builtin import CodeInterpreterTool
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
        model = init_chat_model(model="azure_ai:gpt-4.1", credential=credential)
        tool = CodeInterpreterTool()
        model_with_code = model.bind_tools([tool])
        response = model_with_code.invoke("Plot a sine wave using Python")

    Args:
        file_ids: Optional list of uploaded file IDs to make available inside
            the container.
        memory_limit: Memory limit for the container.  Accepted values are
            ``"1g"``, ``"4g"``, ``"16g"``, and ``"64g"``.
        network_policy: Network access policy for the
            container.

            .. seealso::
                :class:`~openai.types.responses.
                tool_param.CodeInterpreterContainerCodeInterpreterToolAutoNetworkPolicy`
    """

    def __init__(
        self,
        *,
        file_ids: Optional[List[str]] = None,
        memory_limit: Optional[str] = None,
        network_policy: Optional[Dict] = None,  # type: ignore[type-arg]
    ) -> None:
        container = CodeInterpreterContainerCodeInterpreterToolAuto(type="auto")
        if file_ids is not None:
            container["file_ids"] = file_ids
        if memory_limit is not None:
            container["memory_limit"] = memory_limit  # type: ignore[typeddict-item]
        if network_policy is not None:
            container["network_policy"] = network_policy  # type: ignore[typeddict-item]
        super().__init__(
            **CodeInterpreter(type="code_interpreter", container=container)
        )


# ---------------------------------------------------------------------------
# Web Search
# ---------------------------------------------------------------------------


@experimental()
class WebSearchTool(BuiltinTool):
    """A tool that searches the internet for sources related to the prompt.

    Wraps :class:`openai.types.responses.WebSearchToolParam`.

    Example::

        from langchain_azure_ai.tools.builtin import WebSearchTool

        tool = WebSearchTool(search_context_size="high")
        model_with_search = model.bind_tools([tool])

    Args:
        search_context_size: High-level guidance for the amount of context
            window space to use for the search.  One of ``"low"``,
            ``"medium"`` (default), or ``"high"``.
        user_location: Approximate location of the user.  Use
            :class:`~openai.types.responses.web_search_tool_param.UserLocation`
            or a plain :class:`dict` with optional keys ``city``, ``country``
            (ISO-3166 two-letter code), ``region``, ``timezone`` (IANA), and
            ``type="approximate"``.
        filters: Search filters.  Use
            :class:`~openai.types.responses.web_search_tool_param.Filters`
            or a plain :class:`dict` with an optional ``allowed_domains``
            list.
    """

    def __init__(
        self,
        *,
        search_context_size: Optional[str] = None,
        user_location: Optional[UserLocation] = None,
        filters: Optional[WebSearchFilters] = None,
    ) -> None:
        payload = WebSearchToolParam(type="web_search")
        if search_context_size is not None:
            payload["search_context_size"] = search_context_size  # type: ignore[typeddict-item]
        if user_location is not None:
            payload["user_location"] = user_location
        if filters is not None:
            payload["filters"] = filters
        super().__init__(**payload)


# ---------------------------------------------------------------------------
# File Search
# ---------------------------------------------------------------------------


@experimental()
class FileSearchTool(BuiltinTool):
    """A tool that searches for relevant content from uploaded vector stores.

    Wraps :class:`openai.types.responses.FileSearchToolParam`.

    Example::

        from langchain_azure_ai.tools.builtin import FileSearchTool

        tool = FileSearchTool(vector_store_ids=["vs_abc123"])
        model_with_search = model.bind_tools([tool])

    Args:
        vector_store_ids: IDs of the vector stores to search.  At least one
            ID must be provided.
        max_num_results: Maximum number of results to return (1–50).
        filters: Optional metadata filter to narrow results.  Use
            :data:`~openai.types.responses.file_search_tool_param.Filters`
            (a :class:`~openai.types.shared_params.ComparisonFilter` or
            :class:`~openai.types.shared_params.CompoundFilter`).
        ranking_options: Ranking options.  Use
            :class:`~openai.types.responses.file_search_tool_param.RankingOptions`
            or a plain :class:`dict` with optional keys ``ranker`` and
            ``score_threshold``.
    """

    def __init__(
        self,
        vector_store_ids: List[str],
        *,
        max_num_results: Optional[int] = None,
        filters: Optional[FileSearchFilters] = None,
        ranking_options: Optional[RankingOptions] = None,
    ) -> None:
        payload = FileSearchToolParam(
            type="file_search", vector_store_ids=vector_store_ids
        )
        if max_num_results is not None:
            payload["max_num_results"] = max_num_results
        if filters is not None:
            payload["filters"] = filters
        if ranking_options is not None:
            payload["ranking_options"] = ranking_options
        super().__init__(**payload)


# ---------------------------------------------------------------------------
# Image Generation
# ---------------------------------------------------------------------------


@experimental()
class ImageGenerationTool(BuiltinTool):
    """A tool that generates or edits images using GPT image models.

    Wraps :class:`openai.types.responses.tool_param.ImageGeneration`.

    When ``model_deployment`` is specified the tool automatically injects
    an ``x-ms-oai-image-generation-deployment`` HTTP header into every
    inference API call made via :meth:`model.bind_tools`.  This tells the
    Azure AI Foundry endpoint which model deployment to use for image
    generation.

    Example::

        from langchain_azure_ai.tools.builtin import ImageGenerationTool

        tool = ImageGenerationTool(
            quality="high",
            size="1024x1024",
            model_deployment="my-gpt-image-1-deployment",
        )
        model_with_img = model.bind_tools([tool])

    Args:
        model_deployment: Deployment name of the image generation model in
            Azure AI Foundry.  When set, the
            ``x-ms-oai-image-generation-deployment`` HTTP request header is
            injected automatically via :attr:`request_headers`.
        model: Image generation model to use (e.g. ``"gpt-image-1"``).
        action: Whether to generate a new image or edit an existing one.
            One of ``"generate"``, ``"edit"``, or ``"auto"`` (default).
        background: Background type.  One of ``"transparent"``,
            ``"opaque"``, or ``"auto"`` (default).
        input_fidelity: How closely the output should match style and
            facial features of input images.  One of ``"high"`` or
            ``"low"``.
        input_image_mask: Mask for inpainting.  Use
            :class:`~openai.types.responses.tool_param.ImageGenerationInputImageMask`
            or a plain :class:`dict` with optional ``image_url`` / ``file_id``
            keys.
        moderation: Moderation level.  One of ``"auto"`` (default) or
            ``"low"``.
        output_compression: Compression level (0–100, default 100).
        output_format: Output format.  One of ``"png"`` (default),
            ``"webp"``, or ``"jpeg"``.
        partial_images: Number of partial images to stream (0–3).
        quality: Image quality.  One of ``"low"``, ``"medium"``,
            ``"high"``, or ``"auto"`` (default).
        size: Image size.  One of ``"1024x1024"``, ``"1024x1536"``,
            ``"1536x1024"``, or ``"auto"`` (default).
    """

    def __init__(
        self,
        *,
        model: Optional[
            Literal["gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5"]
        ] = None,
        model_deployment: Optional[str] = None,
        action: Optional[Literal["generate", "edit", "auto"]] = None,
        background: Optional[Literal["transparent", "opaque", "auto"]] = None,
        input_fidelity: Optional[str] = None,
        input_image_mask: Optional[ImageGenerationInputImageMask] = None,
        moderation: Optional[Literal["auto", "low"]] = None,
        output_compression: Optional[int] = None,
        output_format: Optional[Literal["png", "webp", "jpeg"]] = None,
        partial_images: Optional[int] = None,
        quality: Optional[Literal["low", "medium", "high", "auto"]] = None,
        size: Optional[Literal["1024x1024", "1024x1536", "1536x1024", "auto"]] = None,
    ) -> None:
        payload = ImageGeneration(type="image_generation")
        if model is not None:
            payload["model"] = model  # type: ignore[typeddict-unknown-key]
        if action is not None:
            payload["action"] = action  # type: ignore[typeddict-item]
        if background is not None:
            payload["background"] = background  # type: ignore[typeddict-item]
        if input_fidelity is not None:
            payload["input_fidelity"] = input_fidelity  # type: ignore[typeddict-item]
        if input_image_mask is not None:
            payload["input_image_mask"] = (  # type: ignore[typeddict-unknown-key]
                input_image_mask
            )
        if moderation is not None:
            payload["moderation"] = moderation  # type: ignore[typeddict-item]
        if output_compression is not None:
            payload["output_compression"] = (  # type: ignore[typeddict-unknown-key]
                output_compression
            )
        if output_format is not None:
            payload["output_format"] = output_format  # type: ignore[typeddict-item]
        if partial_images is not None:
            payload["partial_images"] = (  # type: ignore[typeddict-unknown-key]
                partial_images
            )
        if quality is not None:
            payload["quality"] = quality  # type: ignore[typeddict-item]
        if size is not None:
            payload["size"] = size  # type: ignore[typeddict-item]
        super().__init__(**payload)
        # Store as instance attribute (not in the dict payload).
        self._request_headers: Dict[str, str] = {}
        deployment = model_deployment if model_deployment is not None else model
        if deployment is not None:
            self._request_headers["x-ms-oai-image-generation-deployment"] = deployment


# ---------------------------------------------------------------------------
# MCP (Model Context Protocol)
# ---------------------------------------------------------------------------


@experimental()
class McpTool(BuiltinTool):
    """A tool that gives the model access to an external MCP server.

    Allows the model to call tools exposed by a remote Model Context
    Protocol (MCP) server within a single conversational turn.

    Wraps :class:`openai.types.responses.tool_param.Mcp`.

    Example::

        from langchain_azure_ai.tools.builtin import McpTool

        tool = McpTool(
            server_label="my_server",
            server_url="https://my-mcp-server.example.com",
        )
        model_with_mcp = model.bind_tools([tool])

    Args:
        server_label: A label for this MCP server, used to identify it in
            tool calls.
        server_url: The URL for the MCP server.  Either ``server_url`` or
            ``connector_id`` must be provided.
        connector_id: Identifier for a built-in service connector (e.g.
            ``"connector_gmail"``).  Either ``server_url`` or
            ``connector_id`` must be provided.
        allowed_tools: List of tool names, or a
            :class:`~openai.types.responses.tool_param.McpAllowedToolsMcpToolFilter`
            dict, that the model is allowed to call on this server.
        headers: Optional HTTP headers to send with every request to the
            MCP server (e.g. for authentication).
        require_approval: Whether tool calls require human
            approval before execution.  Use
            :data:`~openai.types.responses.tool_param.McpRequireApproval`
            (``"always"``, ``"never"``, or an approval-filter dict).
        server_description: Optional description of the MCP server.
        authorization: OAuth access token for the MCP server.
    """

    def __init__(
        self,
        server_label: str,
        *,
        server_url: Optional[str] = None,
        connector_id: Optional[str] = None,
        allowed_tools: Optional[McpAllowedTools] = None,
        headers: Optional[Dict[str, str]] = None,
        require_approval: Optional[McpRequireApproval] = None,
        server_description: Optional[str] = None,
        authorization: Optional[str] = None,
    ) -> None:
        payload = Mcp(type="mcp", server_label=server_label)
        if server_url is not None:
            payload["server_url"] = server_url
        if connector_id is not None:
            payload["connector_id"] = connector_id  # type: ignore[typeddict-item]
        if allowed_tools is not None:
            payload["allowed_tools"] = allowed_tools
        if headers is not None:
            payload["headers"] = headers
        if require_approval is not None:
            payload["require_approval"] = require_approval
        if server_description is not None:
            payload["server_description"] = server_description
        if authorization is not None:
            payload["authorization"] = authorization
        super().__init__(**payload)
