"""Declarative chat agent node for Azure AI Foundry agents V2.

This module implements the V2 agent node using the ``azure-ai-projects >= 2.0``
library.  The main paradigm shift from V1 is:

* Agents are created with ``project_client.agents.create_version()`` using a
  ``PromptAgentDefinition``.
* Agent invocation uses the OpenAI *Responses* API via
  ``openai_client.responses.create()`` with a *conversation* context, rather
  than the Threads / Runs model of V1.
* Function-tool calls are represented as ``ResponseFunctionToolCall``
  items in the response output, and results are sent back as
  ``FunctionCallOutput`` items in the next request.
"""

import base64
import binascii
import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Union

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AgentVersionDetails,
    CodeInterpreterTool,
    PromptAgentDefinition,
    Tool,
)
from azure.core.exceptions import HttpResponseError
from langchain.agents import AgentState
from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
    is_data_content_block,
)
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs.chat_result import ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph._internal._runnable import RunnableCallable
from langgraph.prebuilt.chat_agent_executor import StateSchema
from langgraph.store.base import BaseStore
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseInputImageContent,
    ResponseInputTextContent,
)
from openai.types.responses.response_input_item_param import (
    FunctionCallOutput,
    McpApprovalResponse,
)
from openai.types.responses.response_output_item import (
    McpApprovalRequest as McpApprovalRequestOutputItem,
)
from pydantic import Field

from langchain_azure_ai.agents._v2.prebuilt.tools import (
    AgentServiceBaseTool,
)
from langchain_azure_ai.utils.utils import get_mime_from_path

logger = logging.getLogger(__package__)

MCP_APPROVAL_REQUEST_TOOL_NAME = "mcp_approval_request"
"""Synthetic tool name used for MCP approval request tool calls."""


# ---------------------------------------------------------------------------
# Per-invocation state managed by the graph checkpointer
# ---------------------------------------------------------------------------


class AgentServiceAgentState(AgentState):
    """Extended ``AgentState`` that carries per-invocation agent context.

    By storing conversation IDs and pending-call type in the graph state
    (rather than on the node instance), the node becomes thread-safe:
    concurrent invocations each operate on their own copy of the state,
    and the graph's checkpointer can persist / restore it across
    interrupts.

    Fields
    ------
    azure_ai_agents_conversation_id : str | None
        The Responses-API conversation ID.  Created lazily on the first
        ``HumanMessage`` and reused for subsequent turns.
    azure_ai_agents_previous_response_id : str | None
        The ID of the most recent ``Response`` object, used to chain
        tool-call outputs within a single turn.
    azure_ai_agents_pending_type : str | None
        Indicates whether the last response left unresolved calls:
        ``"function_call"``, ``"mcp_approval"``, or ``None``.
    """

    azure_ai_agents_conversation_id: Optional[str]
    azure_ai_agents_previous_response_id: Optional[str]
    azure_ai_agents_pending_type: Optional[str]


def _get_agent_state(
    state: StateSchema,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Read agent context fields from the graph state.

    Returns ``(conversation_id, previous_response_id, pending_type)``.
    When the state schema does not carry these fields (e.g. the user
    supplied a plain ``AgentState``), the values default to ``None``.
    """
    if isinstance(state, dict):
        return (
            state.get("azure_ai_agents_conversation_id"),
            state.get("azure_ai_agents_previous_response_id"),
            state.get("azure_ai_agents_pending_type"),
        )
    return (
        getattr(state, "azure_ai_agents_conversation_id", None),
        getattr(state, "azure_ai_agents_previous_response_id", None),
        getattr(state, "azure_ai_agents_pending_type", None),
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _function_call_to_ai_message(
    func_call: ResponseFunctionToolCall,
) -> AIMessage:
    """Convert a V2 ``ResponseFunctionToolCall`` to a LangChain ``AIMessage``.

    Args:
        func_call: The function call item from the response output.

    Returns:
        An ``AIMessage`` with the corresponding ``tool_calls``.
    """
    tool_calls: List[ToolCall] = [
        ToolCall(
            id=func_call.call_id,
            name=func_call.name,
            args=json.loads(func_call.arguments),
        )
    ]
    return AIMessage(content="", tool_calls=tool_calls)


def _mcp_approval_to_ai_message(
    approval_request: McpApprovalRequestOutputItem,
) -> AIMessage:
    """Convert a V2 ``McpApprovalRequestOutputItem`` to a LangChain ``AIMessage``.

    MCP approval requests are surfaced as tool calls so they can flow through
    the standard LangGraph REACT tool-call loop.  The synthetic tool name is
    ``mcp_approval_request`` and the arguments carry the original request
    metadata so a downstream handler (or human-in-the-loop) can decide
    whether to approve.

    Args:
        approval_request: The MCP approval request item from the response
            output.

    Returns:
        An ``AIMessage`` whose ``tool_calls`` list contains one entry
        representing the approval request.
    """
    tool_calls: List[ToolCall] = [
        ToolCall(
            id=approval_request.id,
            name=MCP_APPROVAL_REQUEST_TOOL_NAME,
            args={
                "server_label": approval_request.server_label,
                "name": approval_request.name,
                "arguments": approval_request.arguments,
            },
        )
    ]
    return AIMessage(content="", tool_calls=tool_calls)


def _tool_message_to_output(
    tool_message: ToolMessage,
) -> FunctionCallOutput:
    """Convert a LangChain ``ToolMessage`` to a V2 ``FunctionCallOutput`` item."""
    if tool_message.tool_call_id is None:
        raise ValueError("ToolMessage must have a tool_call_id to submit as output.")
    output_value = (
        tool_message.content
        if isinstance(tool_message.content, str)
        else json.dumps(tool_message.content)
    )
    return FunctionCallOutput(
        call_id=tool_message.tool_call_id,
        output=output_value,
        type="function_call_output",
    )


def _get_v2_tool_definitions(
    tools: List[Any],
) -> List[Tool]:
    """Convert a list of tools to V2 Tool definitions for the agent.

    Separates tools into:
    - AgentServiceBaseTool tools (native V2 tools like CodeInterpreterTool)
    - BaseTool / callable tools (converted to FunctionTool definitions)

    Args:
        tools: A list of tools to convert.

    Returns:
        A list of V2 Tool definitions.
    """
    from azure.ai.projects.models import FunctionTool as V2FunctionTool

    tool_definitions: List[Tool] = []

    for tool in tools:
        if isinstance(tool, AgentServiceBaseTool):
            tool_definitions.append(tool.tool)
        elif isinstance(tool, BaseTool):
            function_def = convert_to_openai_function(tool)
            tool_definitions.append(
                V2FunctionTool(
                    name=function_def["name"],
                    description=function_def.get("description", ""),
                    parameters=function_def.get("parameters", {}),
                    strict=False,
                )
            )
        elif callable(tool):
            function_def = convert_to_openai_function(tool)
            tool_definitions.append(
                V2FunctionTool(
                    name=function_def["name"],
                    description=function_def.get("description", ""),
                    parameters=function_def.get("parameters", {}),
                    strict=False,
                )
            )
        else:
            raise ValueError(
                "Each tool must be an AgentServiceBaseToolV2, BaseTool, or a "
                f"callable. Got {type(tool)}"
            )

    return tool_definitions


def _approval_message_to_output(
    tool_message: ToolMessage,
) -> McpApprovalResponse:
    """Convert a ``ToolMessage`` for an MCP approval into a ``McpApprovalResponse``.

    The ``ToolMessage.content`` is interpreted as a JSON object (or plain
    string) that carries the approval decision.  Accepted shapes:

    * ``{"approve": true}`` / ``{"approve": false, "reason": "..."}``
    * ``"true"`` / ``"false"`` (shorthand – treated as approve/deny)

    Args:
        tool_message: The tool message whose ``tool_call_id`` matches the
            original ``McpApprovalRequestOutputItem.id``.

    Returns:
        A ``McpApprovalResponse`` ready to be sent back to the Responses API.
    """
    content = tool_message.content
    approve = True
    reason: Optional[str] = None

    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            parsed = content

        if isinstance(parsed, dict):
            approve = bool(parsed.get("approve", True))
            reason = parsed.get("reason")
        else:
            # Plain string: "true"/"false" (case-insensitive)
            approve = str(parsed).lower() not in ("false", "0", "no", "deny")
    elif isinstance(content, dict):
        approve = bool(content.get("approve", True))
        reason = content.get("reason")
    elif isinstance(content, list):
        # E.g. [{"type": "text", "text": "true"}]
        text_parts = [
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        ]
        combined = " ".join(text_parts).strip().lower()
        approve = combined not in ("false", "0", "no", "deny")

    if tool_message.tool_call_id is None:
        raise ValueError(
            "ToolMessage must have a tool_call_id to submit as approval response."
        )
    if reason is not None:
        return McpApprovalResponse(
            approval_request_id=tool_message.tool_call_id,
            approve=approve,
            type="mcp_approval_response",
            reason=reason,
        )
    return McpApprovalResponse(
        approval_request_id=tool_message.tool_call_id,
        approve=approve,
        type="mcp_approval_response",
    )


def _get_input_from_state(state: StateSchema) -> BaseMessage:
    """Extract the latest message from the state.

    Args:
        state: The current state, expected to have a ``messages`` key.

    Returns:
        The latest message.
    """
    messages = (
        state.get("messages", None)
        if isinstance(state, dict)
        else getattr(state, "messages", None)
    )
    if messages is None:
        raise ValueError(
            f"Expected input to call_model to have 'messages' key, but got {state}"
        )
    return messages[-1]


def _content_from_human_message(
    message: HumanMessage,
) -> Union[str, List[Union[ResponseInputTextContent, ResponseInputImageContent]]]:
    """Convert a ``HumanMessage`` to content suitable for the V2 API.

    Args:
        message: The human message to convert.

    Returns:
        Either a plain string or a list of V2 content blocks.
    """
    if isinstance(message.content, str):
        return message.content
    elif isinstance(message.content, list):
        content: List[Union[ResponseInputTextContent, ResponseInputImageContent]] = []
        for block in message.content:
            if isinstance(block, str):
                content.append(ResponseInputTextContent(type="input_text", text=block))
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    content.append(
                        ResponseInputTextContent(
                            type="input_text", text=block.get("text", "")
                        )
                    )
                elif block_type == "image_url":
                    content.append(
                        ResponseInputImageContent(
                            type="input_image",
                            image_url=block["image_url"]["url"],
                        )
                    )
                elif block_type == "image":
                    if block.get("source_type") == "base64":
                        content.append(
                            ResponseInputImageContent(
                                type="input_image",
                                image_url=(
                                    f"data:{block['mime_type']};base64,{block['data']}"
                                ),
                            )
                        )
                    elif block.get("source_type") == "url":
                        content.append(
                            ResponseInputImageContent(
                                type="input_image",
                                image_url=block["url"],
                            )
                        )
                    else:
                        raise ValueError(
                            "Only 'base64' and 'url' source types are supported "
                            "for image blocks."
                        )
                elif block_type == "file":
                    # File blocks that carry image data are sent inline
                    # so the model can see the content.  Non-image file
                    # blocks (CSV, PDF, etc.) are NOT inlined because
                    # the V2 API rejects non-image MIME types inside
                    # ``ResponseInputImageContent``.  Those files are still
                    # uploaded to a container by
                    # ``_upload_file_blocks_to_container`` and will be
                    # available to the agent via the code interpreter.
                    b64_data = block.get("base64") or block.get("data")
                    mime = block.get("mime_type", "application/octet-stream")
                    if b64_data and mime.startswith("image/"):
                        content.append(
                            ResponseInputImageContent(
                                type="input_image",
                                image_url=f"data:{mime};base64,{b64_data}",
                            )
                        )
                    elif not b64_data:
                        logger.warning(
                            "Skipping file block without base64/data payload "
                            "(mime_type=%s)",
                            mime,
                        )
                        continue
                    else:
                        # Non-image file – skip inline; it will be
                        # uploaded to a container instead.
                        logger.info(
                            "Skipping inline representation for non-image "
                            "file block (mime_type=%s); file will be "
                            "uploaded to a container.",
                            mime,
                        )
                        continue
                else:
                    raise ValueError(
                        f"Unsupported block type {block_type} in HumanMessage content."
                    )
            else:
                raise ValueError("Unexpected block type in HumanMessage content.")
        return content
    else:
        raise ValueError("HumanMessage content must be either a string or a list.")


def _upload_file_blocks_to_container(
    message: HumanMessage,
    openai_client: Any,
) -> tuple["HumanMessage", Optional[str]]:
    """Upload binary file blocks to a new container and return its ID.

    This follows the V2 pattern: create a container, upload each file block
    to it, and return the container ID so it can be passed to the agent via
    ``structured_inputs``.

    Args:
        message: The HumanMessage to inspect.
        openai_client: The OpenAI client obtained from
            ``project_client.get_openai_client()``.

    Returns:
        A tuple of (updated_message, container_id) where updated_message
        has the file blocks removed and container_id is the ID of the newly
        created container.  If the message has no eligible file blocks the
        original message and ``None`` are returned.
    """
    if isinstance(message.content, str):
        return message, None

    file_blocks: List[dict] = []
    remaining_content: List[Any] = []

    for block in message.content:
        if (
            isinstance(block, dict)
            and is_data_content_block(block)
            and block.get("type") == "file"
            and block.get("base64")
        ):
            file_blocks.append(block)
        else:
            remaining_content.append(block)

    if not file_blocks:
        return message, None

    # Create a bespoke container for this request.
    container = openai_client.containers.create(
        name=f"ci_{uuid.uuid4().hex[:12]}",
    )
    container_id: str = container.id
    logger.info("Created container: %s", container_id)

    # Upload each file block to the container.
    for block in file_blocks:
        try:
            raw = base64.b64decode(block["base64"])
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                f"Failed to decode base64 data in file content block: {exc}"
            ) from exc
        mime_type: str = block.get("mime_type", "application/octet-stream")
        raw_ext = mime_type.split("/")[-1].split(";")[0].strip()
        ext = "".join(c for c in raw_ext if c.isalnum())[:16] or "bin"
        filename = f"upload_{uuid.uuid4().hex}.{ext}"
        try:
            openai_client.containers.files.create(
                container_id=container_id,
                file=(filename, raw),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to upload file block '{filename}' to container "
                f"{container_id!r} (mime_type={mime_type!r}): {exc}"
            ) from exc
        logger.info("Uploaded file block '%s' to container %s", filename, container_id)

    updated_message = message.model_copy(update={"content": remaining_content})
    return updated_message, container_id


# ---------------------------------------------------------------------------
# Internal chat-model wrapper (used for having the right traces generated)
# ---------------------------------------------------------------------------


class _PromptBasedAgentModelV2(BaseChatModel):
    """A LangChain chat model wrapper for Azure AI Foundry V2 agents.

    It interprets a ``Response`` object produced by the OpenAI Responses API
    and converts its output items into LangChain messages.
    """

    response: Any  # azure.ai.projects.models.Response
    """The V2 Response object."""

    openai_client: Any = None
    """Optional OpenAI client for downloading container files."""

    agent_name: str
    """The agent name (used to tag messages)."""

    model_name: str
    """The model deployment name."""

    pending_function_calls: List[ResponseFunctionToolCall] = Field(default_factory=list)
    """Function calls that need external resolution."""

    pending_mcp_approvals: List[McpApprovalRequestOutputItem] = Field(
        default_factory=list
    )
    """MCP approval requests that need a human decision."""

    @property
    def _llm_type(self) -> str:
        return "PromptBasedAgentModelV2"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        generations: List[ChatGeneration] = []

        response = self.response
        status = response.status if hasattr(response, "status") else None

        if status == "failed":
            error = getattr(response, "error", None)
            raise RuntimeError(f"Response failed with error: {error}")

        # Check for function calls in the output
        function_calls = [
            item
            for item in (response.output or [])
            if getattr(item, "type", None) == "function_call"
        ]

        # Check for MCP approval requests in the output
        mcp_approvals = [
            item
            for item in (response.output or [])
            if getattr(item, "type", None) == "mcp_approval_request"
        ]

        if function_calls:
            # There are pending function calls – return them as tool calls
            self.pending_function_calls = function_calls
            self.pending_mcp_approvals = []
            for fc in function_calls:
                generations.append(
                    ChatGeneration(
                        message=_function_call_to_ai_message(fc),
                        generation_info={},
                    )
                )
        elif mcp_approvals:
            # There are MCP approval requests – surface as tool calls
            self.pending_mcp_approvals = mcp_approvals
            self.pending_function_calls = []
            for ar in mcp_approvals:
                generations.append(
                    ChatGeneration(
                        message=_mcp_approval_to_ai_message(ar),
                        generation_info={},
                    )
                )
        else:
            # Completed response – extract text and any generated files.
            self.pending_function_calls = []
            self.pending_mcp_approvals = []

            # Collect content parts: text + files from response.
            content_parts: List[Union[str, Dict[str, Any]]] = []

            output_text = getattr(response, "output_text", None)
            if output_text:
                content_parts.append(output_text)

            content_parts.extend(self._download_code_interpreter_files(response))

            # Extract generated images from image-generation tool calls.
            content_parts.extend(self._extract_image_generation_results(response))

            if content_parts:
                # Use a plain string when there's only text.
                content: Any
                if len(content_parts) == 1 and isinstance(content_parts[0], str):
                    content = content_parts[0]
                else:
                    content = content_parts
                msg = AIMessage(content=content)
                msg.name = self.agent_name
                generations.append(ChatGeneration(message=msg, generation_info={}))

        llm_output: Dict[str, Any] = {"model": self.model_name}
        usage = getattr(response, "usage", None)
        if usage:
            llm_output["token_usage"] = getattr(usage, "total_tokens", None)
        return ChatResult(generations=generations, llm_output=llm_output)

    # -- helpers ----------------------------------------------------------

    def _download_code_interpreter_files(self, response: Any) -> List[Dict[str, Any]]:
        """Download files generated by code-interpreter calls.

        Discovers files via ``container_file_citation`` annotations
        embedded in ``ResponseOutputText`` content parts.  Each
        annotation provides the ``container_id``, ``file_id`` and
        ``filename`` directly, so files can be downloaded without
        listing the container.

        Images are returned as
        ``{"type": "image", "mime_type": …, "base64": …}`` blocks.
        Non-image files are returned as
        ``{"type": "file", "mime_type": …, "data": …, "filename": …}``
        blocks.

        Returns an empty list when no files are found or when the
        ``openai_client`` is not available.
        """
        if self.openai_client is None:
            return []

        blocks: List[Dict[str, Any]] = []
        downloaded_file_ids: Set[str] = set()

        for item in response.output or []:
            if getattr(item, "type", None) != "message":
                continue
            for content_part in getattr(item, "content", []) or []:
                for annotation in getattr(content_part, "annotations", []) or []:
                    if getattr(annotation, "type", None) != "container_file_citation":
                        continue

                    container_id = getattr(annotation, "container_id", None)
                    file_id = getattr(annotation, "file_id", None)
                    filename = getattr(annotation, "filename", None) or ""
                    if not container_id or not file_id:
                        continue

                    if file_id in downloaded_file_ids:
                        continue

                    block = self._download_container_file(
                        container_id, file_id, filename
                    )
                    if block is not None:
                        blocks.append(block)
                        downloaded_file_ids.add(file_id)

        return blocks

    def _download_container_file(
        self,
        container_id: str,
        file_id: str,
        filename: str,
    ) -> Optional[Dict[str, Any]]:
        """Download a single file from a container and return a content block.

        Returns ``None`` when the download fails.
        """
        try:
            binary_resp = self.openai_client.containers.files.content.retrieve(
                file_id=file_id,
                container_id=container_id,
            )
            raw = binary_resp.read()
            b64 = base64.b64encode(raw).decode("utf-8")
            mime = get_mime_from_path(filename)

            if mime.startswith("image/"):
                block: Dict[str, Any] = {
                    "type": "image",
                    "mime_type": mime,
                    "base64": b64,
                }
            else:
                block = {
                    "type": "file",
                    "mime_type": mime,
                    "data": b64,
                    "filename": filename,
                }

            logger.info(
                "Downloaded file %s (%s) from container %s",
                filename,
                mime,
                container_id,
            )
            return block
        except Exception:
            logger.warning(
                "Failed to download file %s (%s) from container %s",
                file_id,
                filename,
                container_id,
                exc_info=True,
            )
            return None

    def _extract_image_generation_results(self, response: Any) -> List[Dict[str, Any]]:
        """Extract generated images from ``IMAGE_GENERATION_CALL`` output items.

        The ImageGenTool produces output items whose ``type`` is
        ``image_generation_call`` and whose ``result`` attribute holds the
        base64-encoded image data.  This method decodes that data and
        returns it as ``{"type": "image", "mime_type": …, "base64": …}``
        content blocks, consistent with the code-interpreter file blocks.

        Returns an empty list when no image-generation items are found.
        """
        blocks: List[Dict[str, Any]] = []
        for item in response.output or []:
            item_type = getattr(item, "type", None)
            if item_type != "image_generation_call":
                continue

            result = getattr(item, "result", None)
            if not result:
                continue

            # The result is base64-encoded image data (PNG by default).
            blocks.append(
                {
                    "type": "image",
                    "mime_type": "image/png",
                    "base64": result,
                }
            )
            logger.info("Extracted generated image from image_generation_call")

        return blocks


# ---------------------------------------------------------------------------
# Proxy model class (for ModelRequest.model in middleware)
# ---------------------------------------------------------------------------


class _AzureAIFoundryModelProxy(BaseChatModel):
    """A stub ``BaseChatModel`` that represents an Azure AI Foundry agent.

    This proxy is used as the ``model`` field in :class:`ModelRequest` when
    constructing a request for ``wrap_model_call`` middleware.  The actual
    Azure Responses API call is performed by the ``handler`` callback passed
    to the middleware – this proxy object is **not** invoked directly.

    If middleware attempts to invoke this model directly (e.g. a fallback
    strategy), a clear :exc:`NotImplementedError` is raised to indicate
    that direct invocation is unsupported.
    """

    agent_name: str
    """The Azure AI Foundry agent name, used for identification only."""

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "AzureAIFoundryAgent"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"agent_name": self.agent_name}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError(
            "Direct invocation of _AzureAIFoundryModelProxy is not supported. "
            "The Azure AI Foundry agent call is performed by the middleware "
            "handler callback, not by the model proxy."
        )


# ---------------------------------------------------------------------------
# Public node class
# ---------------------------------------------------------------------------


class PromptBasedAgentNode(RunnableCallable):
    """A LangGraph node for Azure AI Foundry agents using V2 (Responses API).

    You can use this node to create complex graphs that involve interactions
    with Azure AI Foundry agents under the V2 protocol.

    Example:
    ```python
    from azure.identity import DefaultAzureCredential
    from langchain_azure_ai.agents.agent_service_v2 import AgentServiceFactoryV2

    factory = AgentServiceFactoryV2(
        project_endpoint=(
            "https://resource.services.ai.azure.com/api/projects/demo-project"
        ),
        credential=DefaultAzureCredential(),
    )

    coder = factory.create_prompt_agent_node(
        name="code-interpreter-agent",
        model="gpt-4.1",
        instructions="You are a helpful assistant that can run Python code.",
        tools=[func1, func2],
    )
    ```
    """

    name: str = "PromptAgentV2"

    _client: AIProjectClient
    """The AIProjectClient instance."""

    _agent: Optional[AgentVersionDetails] = None
    """The agent version details."""

    _agent_name: Optional[str] = None
    """The agent name."""

    _agent_version: Optional[str] = None
    """The agent version."""

    _uses_container_template: bool = False
    """Whether the agent definition uses a ``{{container_id}}`` template.

    When True, every request creates a bespoke container and passes its ID
    via ``structured_inputs`` so the code interpreter can access uploaded
    files at runtime.
    """

    _wrap_model_call_handler: Optional[Callable] = None
    """Composed sync ``wrap_model_call`` handler from middleware (may be None)."""

    _awrap_model_call_handler: Optional[Callable] = None
    """Composed async ``wrap_model_call`` handler from middleware (may be None)."""

    def __init__(
        self,
        client: AIProjectClient,
        model: str,
        instructions: str,
        name: str,
        description: Optional[str] = None,
        agent_name: Optional[str] = None,
        tools: Optional[
            Sequence[Union[AgentServiceBaseTool, BaseTool, Callable]]
        ] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tags: Optional[Sequence[str]] = None,
        trace: bool = True,
        wrap_model_call_handler: Optional[Callable] = None,
        awrap_model_call_handler: Optional[Callable] = None,
    ) -> None:
        """Initialize the V2 agent node.

        Args:
            client: The AIProjectClient instance.
            model: The model deployment name.
            instructions: System instructions for the agent.
            name: Display name for the agent.
            description: Optional human-readable description.
            agent_name: If provided, retrieves an existing agent by name
                instead of creating a new one.  When set, ``model`` and
                ``instructions`` are still required but a new version is
                *not* created; the latest existing version is used.
            tools: Tools the agent can use.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            tags: Optional tags for the runnable.
            trace: Whether to enable tracing.
            wrap_model_call_handler: Optional composed sync handler from
                ``wrap_model_call`` middleware. When provided, this wraps
                the call to the Azure Responses API, enabling retry logic,
                observability, and request/response modification.
            awrap_model_call_handler: Optional composed async handler from
                ``awrap_model_call`` middleware.  Used in async execution
                paths (``_afunc``).
        """
        if ":" in name:
            raise ValueError(
                f"Agent name must not contain ':': {name!r}.  "
                "Colons are reserved for the internal name:version identifier."
            )

        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=trace)

        self._client = client
        self._uses_container_template = False
        self._wrap_model_call_handler = wrap_model_call_handler
        self._awrap_model_call_handler = awrap_model_call_handler

        # Collect extra HTTP headers declared on AgentServiceBaseTool
        # wrappers.  These are merged across all tools and passed to
        # every ``responses.create()`` call.
        self._extra_headers: Dict[str, str] = {}
        if tools:
            for t in tools:
                if isinstance(t, AgentServiceBaseTool) and t.extra_headers:
                    self._extra_headers.update(t.extra_headers)

        if agent_name is not None:
            try:
                existing = self._client.agents.get(agent_name=agent_name).versions[
                    "latest"
                ]
                self._agent = existing
                self._agent_name = existing.name
                self._agent_version = existing.version
                logger.info(
                    "Using existing agent: %s (version=%s)",
                    self._agent_name,
                    self._agent_version,
                )
                return
            except HttpResponseError as e:
                raise ValueError(
                    f"Could not find agent with name {agent_name} in the "
                    "connected project."
                ) from e

        # Build the PromptAgentDefinition
        definition_params: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
        }
        if temperature is not None:
            definition_params["temperature"] = temperature
        if top_p is not None:
            definition_params["top_p"] = top_p

        if tools is not None:
            tool_defs = _get_v2_tool_definitions(list(tools))

            # If a CodeInterpreterTool is present without a pre-configured
            # container, template it with ``{{container_id}}`` so that a
            # bespoke container can be provided at request time via
            # ``structured_inputs``.
            for i, td in enumerate(tool_defs):
                is_ci = isinstance(td, CodeInterpreterTool) or (
                    isinstance(td, dict) and td.get("type") == "code_interpreter"
                )
                if not is_ci:
                    continue

                # Check whether the tool already has a concrete container
                # (a string ID).  Placeholder values like ``None`` or
                # ``CodeInterpreterToolAuto`` should still be templated.
                existing_container = (
                    td.get("container", None)
                    if isinstance(td, dict)
                    else getattr(td, "container", None)
                )
                if isinstance(existing_container, str):
                    continue

                # Replace with a templated version.
                tool_defs[i] = CodeInterpreterTool(container="{{container_id}}")
                self._uses_container_template = True
                break  # At most one code-interpreter tool per agent

            definition_params["tools"] = tool_defs

            if self._uses_container_template:
                definition_params["structured_inputs"] = {
                    "container_id": {
                        "description": (
                            "Pre-configured container ID for the code interpreter"
                        ),
                        "required": True,
                    }
                }

        definition = PromptAgentDefinition(**definition_params)

        agent_create_params: Dict[str, Any] = {
            "agent_name": name,
            "definition": definition,
        }
        if description is not None:
            agent_create_params["description"] = description

        self._agent = self._client.agents.create_version(**agent_create_params)

        self._agent_name = self._agent.name
        self._agent_version = self._agent.version
        logger.info(
            "Created agent version: %s (name=%s, version=%s)",
            self._agent.id,
            self._agent.name,
            self._agent.version,
        )

    @property
    def _agent_id(self) -> Optional[str]:
        """Return a stable identifier for this agent (name:version)."""
        if self._agent_name and self._agent_version:
            return f"{self._agent_name}:{self._agent_version}"
        return None

    def delete_agent_from_node(self) -> None:
        """Delete the agent version associated with this node."""
        if self._agent_name is not None and self._agent_version is not None:
            self._client.agents.delete_version(
                agent_name=self._agent_name,
                agent_version=self._agent_version,
            )
            logger.info(
                "Deleted agent %s version %s",
                self._agent_name,
                self._agent_version,
            )
            self._agent = None
            self._agent_name = None
            self._agent_version = None
        else:
            raise ValueError("The node does not have an associated agent to delete.")

    # -----------------------------------------------------------------------
    # Core execution logic
    # -----------------------------------------------------------------------

    def _call_azure_api(
        self,
        state: StateSchema,
        config: RunnableConfig,
        openai_client: Any,
        wrap_handler: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Execute the Azure Responses API call, optionally wrapped by middleware.

        This method encapsulates the core Azure API invocation.  When
        ``wrap_handler`` is provided (a composed ``wrap_model_call`` handler
        from middleware), the actual API call is performed inside it so that
        middleware can add retry logic, observability, or request/response
        modification.

        Args:
            state: The current graph state.
            config: The runnable config (for callbacks/metadata/tags).
            openai_client: The OpenAI client for the Responses API.
            wrap_handler: Optional composed sync ``wrap_model_call`` handler
                from middleware.

        Returns:
            A dict suitable for returning from ``_func`` with keys
            ``messages``, ``azure_ai_agents_conversation_id``,
            ``azure_ai_agents_previous_response_id``, and
            ``azure_ai_agents_pending_type``.
        """
        if self._agent is None or self._agent_name is None:
            raise RuntimeError(
                "The agent has not been initialized properly or has been deleted."
            )

        message = _get_input_from_state(state)
        conversation_id, previous_response_id, pending_type = _get_agent_state(state)

        agent_name = self._agent_name
        agent_def = self._agent.definition
        extra_headers = self._extra_headers
        uses_container_template = self._uses_container_template

        # ----------------------------------------------------------------
        # Mutable holder for state produced by the Azure call.  Using a
        # dict instead of nonlocal variables so the closure can be called
        # multiple times by retry middleware safely.
        # ----------------------------------------------------------------
        azure_state_out: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "previous_response_id": previous_response_id,
            "pending_type": pending_type,
        }

        def _execute_azure_call(request: ModelRequest) -> ModelResponse:
            """Inner handler: performs the actual Azure Responses API call.

            This function is passed as the ``handler`` argument to
            ``wrap_model_call`` middleware.  It may be invoked multiple times
            (e.g. by retry middleware), and updates ``azure_state_out`` on
            each successful call.
            """
            # Re-read conversation state from the request's state so that
            # middleware can modify it if needed.
            req_state = request.state if request.state is not None else state
            req_conv_id, req_prev_resp_id, req_pending = _get_agent_state(req_state)

            # Use the most recent message from the request's message list.
            req_message = request.messages[-1] if request.messages else message
            # effective_message tracks any post-processing (e.g. file block
            # removal for container uploads) so the agent model is invoked
            # with the correct message content.
            effective_message = req_message

            if isinstance(req_message, ToolMessage):
                logger.info(
                    "Submitting tool message (tool_call_id=%s)",
                    req_message.tool_call_id,
                )

                if req_pending == "mcp_approval":
                    logger.info("Submitting MCP approval response")
                    input_items: List[McpApprovalResponse] = [
                        _approval_message_to_output(req_message)
                    ]

                    response_params: Dict[str, Any] = {
                        "input": input_items,
                        "extra_body": {
                            "agent_reference": {
                                "name": agent_name,
                                "type": "agent_reference",
                            }
                        },
                    }

                    if req_conv_id:
                        response_params["conversation"] = req_conv_id
                    elif req_prev_resp_id:
                        response_params["previous_response_id"] = req_prev_resp_id

                    if extra_headers:
                        response_params["extra_headers"] = extra_headers

                    response = openai_client.responses.create(**response_params)

                elif req_pending == "function_call":
                    input_items_fc: List[FunctionCallOutput] = [
                        _tool_message_to_output(req_message)
                    ]

                    response_params = {
                        "input": input_items_fc,
                        "extra_body": {
                            "agent_reference": {
                                "name": agent_name,
                                "type": "agent_reference",
                            }
                        },
                    }

                    if req_conv_id:
                        response_params["conversation"] = req_conv_id
                    elif req_prev_resp_id:
                        response_params["previous_response_id"] = req_prev_resp_id

                    if extra_headers:
                        response_params["extra_headers"] = extra_headers

                    response = openai_client.responses.create(**response_params)

                else:
                    raise RuntimeError(
                        "No pending function calls or MCP approval requests "
                        "to submit tool outputs to."
                    )

            elif isinstance(req_message, HumanMessage):
                logger.info("Submitting human message: %s", req_message.content)

                req_prev_resp_id = None

                req_msg_for_container = req_message
                new_conv_id = req_conv_id
                container_id: Optional[str] = None
                if uses_container_template:
                    req_msg_for_container, container_id = (
                        _upload_file_blocks_to_container(req_message, openai_client)
                    )
                    if container_id:
                        logger.info(
                            "Created container %s with uploaded files",
                            container_id,
                        )
                # Use the container-processed message for content extraction and
                # model invocation (file blocks have been removed from it).
                effective_message = req_msg_for_container

                content = _content_from_human_message(req_msg_for_container)

                if new_conv_id is None:
                    conversation = openai_client.conversations.create()
                    new_conv_id = conversation.id
                    logger.info("Created conversation: %s", new_conv_id)

                response_input: Any
                if isinstance(content, list):
                    response_input = [{"role": "user", "content": content}]
                else:
                    response_input = content

                extra_body: Dict[str, Any] = {
                    "agent_reference": {
                        "name": agent_name,
                        "type": "agent_reference",
                    }
                }

                if container_id is not None:
                    extra_body["structured_inputs"] = {
                        "container_id": container_id,
                    }

                response_params = {
                    "conversation": new_conv_id,
                    "input": response_input,
                    "extra_body": extra_body,
                }

                if extra_headers:
                    response_params["extra_headers"] = extra_headers

                response = openai_client.responses.create(**response_params)
                req_conv_id = new_conv_id

            else:
                raise RuntimeError(f"Unsupported message type: {type(req_message)}")

            agent_model = _PromptBasedAgentModelV2(
                response=response,
                openai_client=openai_client,
                agent_name=agent_name,
                model_name=agent_def.get("model", "unknown")
                if hasattr(agent_def, "get")
                else getattr(agent_def, "model", "unknown"),
                callbacks=config.get("callbacks", None),
                metadata=config.get("metadata", None),
                tags=config.get("tags", None),
            )

            responses = agent_model.invoke([effective_message])

            if agent_model.pending_function_calls:
                out_pending = "function_call"
            elif agent_model.pending_mcp_approvals:
                out_pending = "mcp_approval"
            else:
                out_pending = None

            # Update shared state holder for callers that inspect it after
            # wrap_model_call returns.
            azure_state_out["conversation_id"] = req_conv_id
            azure_state_out["previous_response_id"] = response.id
            azure_state_out["pending_type"] = out_pending

            return ModelResponse(result=responses)

        # ----------------------------------------------------------------
        # Build ModelRequest for middleware
        # ----------------------------------------------------------------
        messages = (
            state.get("messages", [])
            if isinstance(state, dict)
            else getattr(state, "messages", [])
        )
        proxy_model = _AzureAIFoundryModelProxy(agent_name=agent_name)
        request = ModelRequest(
            model=proxy_model,
            messages=list(messages),
            state=state,
        )

        # ----------------------------------------------------------------
        # Execute (with or without wrap_model_call middleware)
        # ----------------------------------------------------------------
        if wrap_handler is not None:
            result = wrap_handler(request, _execute_azure_call)
            # Normalize the result to ModelResponse
            if isinstance(result, ExtendedModelResponse):
                model_response: ModelResponse = result.model_response
            elif isinstance(result, AIMessage):
                model_response = ModelResponse(result=[result])
            elif isinstance(result, ModelResponse):
                model_response = result
            else:
                # _ComposedExtendedModelResponse (private internal type)
                model_response = getattr(result, "model_response", result)  # type: ignore[arg-type]
        else:
            model_response = _execute_azure_call(request)

        return {
            "messages": model_response.result,
            "azure_ai_agents_conversation_id": azure_state_out["conversation_id"],
            "azure_ai_agents_previous_response_id": azure_state_out[
                "previous_response_id"
            ],
            "azure_ai_agents_pending_type": azure_state_out["pending_type"],
        }

    def _func(
        self,
        state: StateSchema,
        config: RunnableConfig,
        *,
        store: Optional[BaseStore],
    ) -> StateSchema:
        if self._agent is None or self._agent_name is None:
            raise RuntimeError(
                "The agent has not been initialized properly or has been deleted."
            )

        logger.debug(
            "[_func] agent=%s, wrap_model_call=%s",
            self._agent_name,
            self._wrap_model_call_handler is not None,
        )

        openai_client = self._client.get_openai_client()
        try:
            return self._call_azure_api(  # type: ignore[return-value]
                state=state,
                config=config,
                openai_client=openai_client,
                wrap_handler=self._wrap_model_call_handler,
            )
        finally:
            openai_client.close()

    async def _afunc(
        self,
        state: StateSchema,
        config: RunnableConfig,
        *,
        store: Optional[BaseStore],
    ) -> StateSchema:
        import asyncio

        if self._agent is None or self._agent_name is None:
            raise RuntimeError(
                "The agent has not been initialized properly or has been deleted."
            )

        logger.debug(
            "[_afunc] agent=%s, awrap_model_call=%s",
            self._agent_name,
            self._awrap_model_call_handler is not None,
        )

        openai_client = self._client.get_openai_client()
        try:
            if self._awrap_model_call_handler is not None:
                # When async middleware is present we run the Azure API calls
                # in a thread pool and let the middleware run in the event loop.
                #
                # ``outer_azure_state`` is a shared mutable dict that is
                # populated inside the async handler so that after the
                # middleware returns we can build the correct graph-state
                # update.  If middleware short-circuits (never calls the
                # handler) we fall back to the current state values.
                conversation_id, previous_response_id, pending_type = (
                    _get_agent_state(state)
                )
                outer_azure_state: Dict[str, Any] = {
                    "azure_ai_agents_conversation_id": conversation_id,
                    "azure_ai_agents_previous_response_id": previous_response_id,
                    "azure_ai_agents_pending_type": pending_type,
                }

                async def _async_execute(req: ModelRequest) -> ModelResponse:
                    """Run the sync Azure call in a thread pool."""
                    req_state = req.state if req.state is not None else state

                    def _sync() -> Dict[str, Any]:
                        return self._call_azure_api(  # type: ignore[return-value]
                            state=req_state,
                            config=config,
                            openai_client=openai_client,
                            wrap_handler=None,
                        )

                    result_dict = await asyncio.to_thread(_sync)
                    # Capture azure state for use after middleware returns.
                    outer_azure_state.update(result_dict)
                    return ModelResponse(result=result_dict["messages"])

                messages = (
                    state.get("messages", [])
                    if isinstance(state, dict)
                    else getattr(state, "messages", [])
                )
                proxy_model = _AzureAIFoundryModelProxy(agent_name=self._agent_name)
                request = ModelRequest(
                    model=proxy_model,
                    messages=list(messages),
                    state=state,
                    runtime=None,
                )

                result = await self._awrap_model_call_handler(
                    request, _async_execute
                )

                if isinstance(result, ExtendedModelResponse):
                    model_response: ModelResponse = result.model_response
                elif isinstance(result, AIMessage):
                    model_response = ModelResponse(result=[result])
                elif isinstance(result, ModelResponse):
                    model_response = result
                else:
                    # _ComposedExtendedModelResponse (private internal type)
                    model_response = getattr(result, "model_response", result)  # type: ignore[arg-type]

                return {  # type: ignore[return-value]
                    "messages": model_response.result,
                    "azure_ai_agents_conversation_id": outer_azure_state.get(
                        "azure_ai_agents_conversation_id"
                    ),
                    "azure_ai_agents_previous_response_id": outer_azure_state.get(
                        "azure_ai_agents_previous_response_id"
                    ),
                    "azure_ai_agents_pending_type": outer_azure_state.get(
                        "azure_ai_agents_pending_type"
                    ),
                }

            # No async middleware - run sync call in a thread pool.
            def _sync_func() -> Dict[str, Any]:
                return self._call_azure_api(  # type: ignore[return-value]
                    state=state,
                    config=config,
                    openai_client=openai_client,
                    wrap_handler=self._wrap_model_call_handler,
                )

            return await asyncio.to_thread(_sync_func)  # type: ignore[return-value]
        finally:
            openai_client.close()
