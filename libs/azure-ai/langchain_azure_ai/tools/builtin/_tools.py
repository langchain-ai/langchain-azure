"""Built-in server-side tools for OpenAI models deployed in Azure AI Foundry.

These tools represent server-side capabilities that models can invoke within a
single conversational turn (e.g. web search, code execution, image generation).
Pass instances directly to ``model.bind_tools()``.

Example usage::

    from langchain_azure_ai.tools.builtin import CodeInterpreterTool, WebSearchTool

    model_with_tools = model.bind_tools([
        CodeInterpreterTool(),
        WebSearchTool(search_context_size="high"),
    ])

    response = model_with_tools.invoke("Use Python to plot a random graph")
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional


class BuiltinTool(dict):  # type: ignore[type-arg]
    """Base class for server-side built-in tools.

    Inherits from :class:`dict` so instances can be passed directly to
    ``model.bind_tools()`` without additional conversion.  Subclasses set
    their payload by calling ``super().__init__(**fields)`` with the required
    and optional tool fields, omitting ``None`` values so only explicitly
    provided parameters are forwarded to the API.

    Example – defining a custom built-in tool::

        class MyTool(BuiltinTool):
            def __init__(self, option: str = "default") -> None:
                super().__init__(type="my_tool", option=option)
    """


# ---------------------------------------------------------------------------
# Code Interpreter
# ---------------------------------------------------------------------------


class CodeInterpreterTool(BuiltinTool):
    """A tool that runs Python code server-side to help generate a response.

    The model can write and execute Python code within a sandboxed container
    and include the results in its response.

    Example::

        from langchain_azure_ai.tools.builtin import CodeInterpreterTool

        tool = CodeInterpreterTool()
        model_with_code = model.bind_tools([tool])
        response = model_with_code.invoke("Plot a sine wave using Python")

    Args:
        file_ids: Optional list of uploaded file IDs to make available inside
            the container.
        memory_limit: Memory limit for the container.  One of ``"1g"``,
            ``"4g"``, ``"16g"``, or ``"64g"``.
        network_policy: Network access policy for the container.
    """

    def __init__(
        self,
        *,
        file_ids: Optional[List[str]] = None,
        memory_limit: Optional[Literal["1g", "4g", "16g", "64g"]] = None,
        network_policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        container: Dict[str, Any] = {"type": "auto"}
        if file_ids is not None:
            container["file_ids"] = file_ids
        if memory_limit is not None:
            container["memory_limit"] = memory_limit
        if network_policy is not None:
            container["network_policy"] = network_policy
        super().__init__(type="code_interpreter", container=container)


# ---------------------------------------------------------------------------
# Web Search
# ---------------------------------------------------------------------------


class WebSearchTool(BuiltinTool):
    """A tool that searches the internet for sources related to the prompt.

    Example::

        from langchain_azure_ai.tools.builtin import WebSearchTool

        tool = WebSearchTool(search_context_size="high")
        model_with_search = model.bind_tools([tool])

    Args:
        search_context_size: High-level guidance for the amount of context
            window space to use for the search.  One of ``"low"``,
            ``"medium"`` (default), or ``"high"``.
        user_location: Approximate location of the user.  A dict with
            optional keys ``city``, ``country`` (ISO-3166 two-letter code),
            ``region``, ``timezone`` (IANA), and required
            ``type="approximate"``.
        filters: Search filters.  A dict with an optional
            ``allowed_domains`` list.
    """

    def __init__(
        self,
        *,
        search_context_size: Optional[Literal["low", "medium", "high"]] = None,
        user_location: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        data: Dict[str, Any] = {"type": "web_search"}
        if search_context_size is not None:
            data["search_context_size"] = search_context_size
        if user_location is not None:
            data["user_location"] = user_location
        if filters is not None:
            data["filters"] = filters
        super().__init__(**data)


# ---------------------------------------------------------------------------
# File Search
# ---------------------------------------------------------------------------


class FileSearchTool(BuiltinTool):
    """A tool that searches for relevant content from uploaded vector stores.

    Example::

        from langchain_azure_ai.tools.builtin import FileSearchTool

        tool = FileSearchTool(vector_store_ids=["vs_abc123"])
        model_with_search = model.bind_tools([tool])

    Args:
        vector_store_ids: IDs of the vector stores to search.  At least one
            ID must be provided.
        max_num_results: Maximum number of results to return (1–50).
        filters: Optional metadata filter to narrow results.
        ranking_options: Ranking options dict with optional keys ``ranker``
            and ``score_threshold``.
    """

    def __init__(
        self,
        vector_store_ids: List[str],
        *,
        max_num_results: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        ranking_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        data: Dict[str, Any] = {
            "type": "file_search",
            "vector_store_ids": vector_store_ids,
        }
        if max_num_results is not None:
            data["max_num_results"] = max_num_results
        if filters is not None:
            data["filters"] = filters
        if ranking_options is not None:
            data["ranking_options"] = ranking_options
        super().__init__(**data)


# ---------------------------------------------------------------------------
# Image Generation
# ---------------------------------------------------------------------------


class ImageGenerationTool(BuiltinTool):
    """A tool that generates or edits images using GPT image models.

    Example::

        from langchain_azure_ai.tools.builtin import ImageGenerationTool

        tool = ImageGenerationTool(quality="high", size="1024x1024")
        model_with_img = model.bind_tools([tool])

    Args:
        model: Image generation model to use.
        action: Whether to generate a new image or edit an existing one.
            One of ``"generate"``, ``"edit"``, or ``"auto"`` (default).
        background: Background type.  One of ``"transparent"``,
            ``"opaque"``, or ``"auto"`` (default).
        input_fidelity: How closely the output should match style and
            facial features of input images.  One of ``"high"`` or
            ``"low"``.
        input_image_mask: Mask for inpainting, as a dict with optional
            ``image_url`` or ``file_id`` keys.
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
        action: Optional[Literal["generate", "edit", "auto"]] = None,
        background: Optional[Literal["transparent", "opaque", "auto"]] = None,
        input_fidelity: Optional[Literal["high", "low"]] = None,
        input_image_mask: Optional[Dict[str, Any]] = None,
        moderation: Optional[Literal["auto", "low"]] = None,
        output_compression: Optional[int] = None,
        output_format: Optional[Literal["png", "webp", "jpeg"]] = None,
        partial_images: Optional[int] = None,
        quality: Optional[Literal["low", "medium", "high", "auto"]] = None,
        size: Optional[
            Literal["1024x1024", "1024x1536", "1536x1024", "auto"]
        ] = None,
    ) -> None:
        data: Dict[str, Any] = {"type": "image_generation"}
        if model is not None:
            data["model"] = model
        if action is not None:
            data["action"] = action
        if background is not None:
            data["background"] = background
        if input_fidelity is not None:
            data["input_fidelity"] = input_fidelity
        if input_image_mask is not None:
            data["input_image_mask"] = input_image_mask
        if moderation is not None:
            data["moderation"] = moderation
        if output_compression is not None:
            data["output_compression"] = output_compression
        if output_format is not None:
            data["output_format"] = output_format
        if partial_images is not None:
            data["partial_images"] = partial_images
        if quality is not None:
            data["quality"] = quality
        if size is not None:
            data["size"] = size
        super().__init__(**data)


# ---------------------------------------------------------------------------
# Computer Use
# ---------------------------------------------------------------------------


class ComputerUseTool(BuiltinTool):
    """A tool that gives the model access to a virtual computer interface.

    Allows the model to interact with a desktop environment (clicking,
    typing, taking screenshots) as part of its response.

    Example::

        from langchain_azure_ai.tools.builtin import ComputerUseTool

        tool = ComputerUseTool()
        model_with_computer = model.bind_tools([tool])
    """

    def __init__(self) -> None:
        super().__init__(type="computer_use_preview")


# ---------------------------------------------------------------------------
# MCP (Model Context Protocol)
# ---------------------------------------------------------------------------


class McpTool(BuiltinTool):
    """A tool that gives the model access to an external MCP server.

    Allows the model to call tools exposed by a remote Model Context
    Protocol (MCP) server within a single conversational turn.

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
        allowed_tools: List of tool names (or a filter dict) that the model
            is allowed to call on this server.
        headers: Optional HTTP headers to send with every request to the
            MCP server (e.g. for authentication).
        require_approval: Whether tool calls require human approval before
            execution.  One of ``"always"``, ``"never"``, or a filter dict.
        server_description: Optional description of the MCP server.
        authorization: OAuth access token for the MCP server.
    """

    def __init__(
        self,
        server_label: str,
        *,
        server_url: Optional[str] = None,
        connector_id: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None,
        require_approval: Optional[Literal["always", "never"]] = None,
        server_description: Optional[str] = None,
        authorization: Optional[str] = None,
    ) -> None:
        data: Dict[str, Any] = {"type": "mcp", "server_label": server_label}
        if server_url is not None:
            data["server_url"] = server_url
        if connector_id is not None:
            data["connector_id"] = connector_id
        if allowed_tools is not None:
            data["allowed_tools"] = allowed_tools
        if headers is not None:
            data["headers"] = headers
        if require_approval is not None:
            data["require_approval"] = require_approval
        if server_description is not None:
            data["server_description"] = server_description
        if authorization is not None:
            data["authorization"] = authorization
        super().__init__(**data)
