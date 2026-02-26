"""Declarative chat agent node for Azure AI Foundry agents V2.

This module implements the V2 agent node using the ``azure-ai-projects >= 2.0``
library.  The main paradigm shift from V1 is:

* Agents are created with ``project_client.agents.create_version()`` using a
  ``PromptAgentDefinition``.
* Agent invocation uses the OpenAI *Responses* API via
  ``openai_client.responses.create()`` with a *conversation* context, rather
  than the Threads / Runs model of V1.
* Function-tool calls are represented as ``FunctionToolCallItemResource``
  items in the response output, and results are sent back as
  ``FunctionToolCallOutputItemParam`` items in the next request.
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
    FunctionToolCallItemResource,
    FunctionToolCallOutputItemParam,
    ItemType,
    MCPApprovalRequestItemResource,
    MCPApprovalResponseItemParam,
    PromptAgentDefinition,
)
from azure.core.exceptions import HttpResponseError
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
from langgraph._internal._runnable import RunnableCallable
from langgraph.prebuilt.chat_agent_executor import StateSchema
from langgraph.store.base import BaseStore
from pydantic import Field

from langchain_azure_ai.agents.prebuilt.tools_v2 import (
    AgentServiceBaseToolV2,
    _get_v2_tool_definitions,
)

logger = logging.getLogger(__package__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _function_call_to_ai_message(
    func_call: FunctionToolCallItemResource,
) -> AIMessage:
    """Convert a V2 ``FunctionToolCallItemResource`` to a LangChain ``AIMessage``.

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
    approval_request: MCPApprovalRequestItemResource,
) -> AIMessage:
    """Convert a V2 ``MCPApprovalRequestItemResource`` to a LangChain ``AIMessage``.

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
            name="mcp_approval_request",
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
) -> FunctionToolCallOutputItemParam:
    """Convert a LangChain ``ToolMessage`` to a V2 function-call output item."""
    return FunctionToolCallOutputItemParam(
        call_id=tool_message.tool_call_id,  # type: ignore[arg-type]
        output=tool_message.content
        if isinstance(tool_message.content, str)
        else json.dumps(tool_message.content),
    )


def _approval_message_to_output(
    tool_message: ToolMessage,
) -> MCPApprovalResponseItemParam:
    """Convert a ``ToolMessage`` for an MCP approval into an approval response.

    The ``ToolMessage.content`` is interpreted as a JSON object (or plain
    string) that carries the approval decision.  Accepted shapes:

    * ``{"approve": true}`` / ``{"approve": false, "reason": "..."}``
    * ``"true"`` / ``"false"`` (shorthand – treated as approve/deny)

    Args:
        tool_message: The tool message whose ``tool_call_id`` matches the
            original ``MCPApprovalRequestItemResource.id``.

    Returns:
        An ``MCPApprovalResponseItemParam`` ready to be sent back to the
        Responses API.
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

    params: Dict[str, Any] = {
        "approval_request_id": tool_message.tool_call_id,
        "approve": approve,
    }
    if reason is not None:
        params["reason"] = reason
    return MCPApprovalResponseItemParam(**params)


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
) -> Union[str, List[Any]]:
    """Convert a ``HumanMessage`` to content suitable for the V2 API.

    Args:
        message: The human message to convert.

    Returns:
        Either a plain string or a list of V2 ``ItemContent`` blocks.
    """
    from azure.ai.projects.models import (
        ItemContentInputImage,
        ItemContentInputText,
    )

    if isinstance(message.content, str):
        return message.content
    elif isinstance(message.content, list):
        content: List[Any] = []
        for block in message.content:
            if isinstance(block, str):
                content.append(ItemContentInputText(text=block))
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    content.append(ItemContentInputText(text=block.get("text", "")))
                elif block_type == "image_url":
                    content.append(
                        ItemContentInputImage(
                            image_url=block["image_url"]["url"],
                        )
                    )
                elif block_type == "image":
                    if block.get("source_type") == "base64":
                        content.append(
                            ItemContentInputImage(
                                image_url=(
                                    f"data:{block['mime_type']};base64,"
                                    f"{block['data']}"
                                ),
                            )
                        )
                    elif block.get("source_type") == "url":
                        content.append(
                            ItemContentInputImage(
                                image_url=block["url"],
                            )
                        )
                    else:
                        raise ValueError(
                            "Only 'base64' and 'url' source types are supported "
                            "for image blocks."
                        )
                elif block_type == "file":
                    # File blocks that carry image data should be sent
                    # inline so the model can see the content and decide
                    # which tool to invoke with the payload.  Non-image
                    # file blocks are also inlined as images when a data
                    # URI can be constructed (the model may still be able
                    # to interpret them), otherwise they are skipped with
                    # a warning. They may still get uploaded to a container
                    b64_data = block.get("base64") or block.get("data")
                    mime = block.get("mime_type", "application/octet-stream")
                    if b64_data:
                        content.append(
                            ItemContentInputImage(
                                image_url=f"data:{mime};base64,{b64_data}",
                            )
                        )
                    else:
                        logger.warning(
                            "Skipping file block without base64/data payload "
                            "(mime_type=%s)",
                            mime,
                        )
                        continue
                else:
                    raise ValueError(
                        f"Unsupported block type {block_type} in HumanMessage "
                        "content."
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

    pending_function_calls: List[FunctionToolCallItemResource] = Field(
        default_factory=list
    )
    """Function calls that need external resolution."""

    pending_mcp_approvals: List[MCPApprovalRequestItemResource] = Field(
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
            if getattr(item, "type", None) == ItemType.FUNCTION_CALL
        ]

        # Check for MCP approval requests in the output
        mcp_approvals = [
            item
            for item in (response.output or [])
            if getattr(item, "type", None) == ItemType.MCP_APPROVAL_REQUEST
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

            if not content_parts:
                # Fallback: iterate over output items for message-type items
                for item in response.output or []:
                    if getattr(item, "type", None) == ItemType.MESSAGE:
                        for content_part in getattr(item, "content", []):
                            text = getattr(content_part, "text", None)
                            if text:
                                content_parts.append(text)

            # Download files referenced in the response.
            content_parts.extend(self._download_code_interpreter_files(response))

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

        Uses two complementary strategies to discover files:

        1. **Annotations** (preferred) – ``container_file_citation``
           annotations embedded in ``ResponseOutputText`` content parts
           provide the ``container_id``, ``file_id`` and ``filename``
           directly, so files can be downloaded without listing the
           container.
        2. **OutputImage fallback** – for ``CODE_INTERPRETER_CALL`` items
           whose ``outputs`` contain ``OutputImage`` entries that were *not*
           already covered by an annotation, the method lists the container
           files and matches by basename.

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

        # -----------------------------------------------------------------
        # Strategy 1: annotations from message output items
        # -----------------------------------------------------------------
        for item in response.output or []:
            if getattr(item, "type", None) != ItemType.MESSAGE:
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

        # -----------------------------------------------------------------
        # Strategy 2: fallback for OutputImage entries without annotations
        # -----------------------------------------------------------------
        for item in response.output or []:
            if getattr(item, "type", None) != ItemType.CODE_INTERPRETER_CALL:
                continue

            container_id = getattr(item, "container_id", None)
            if not container_id:
                continue

            image_outputs = [
                o
                for o in (getattr(item, "outputs", []) or [])
                if getattr(o, "type", None) == "image"
            ]
            if not image_outputs:
                continue

            # Check if all images were already handled by annotations.
            # We try to avoid listing the container when possible.
            unresolved = []
            for output in image_outputs:
                url = getattr(output, "url", "") or ""
                basename = url.rsplit("/", 1)[-1] if "/" in url else url
                # A more precise check: see if the filename was covered.
                covered = False
                for blk in blocks:
                    blk_name = blk.get("filename", "")
                    if blk_name and blk_name == basename:
                        covered = True
                        break
                    # Images don't have a filename key; match via mime_type
                    # basename heuristic.
                    if blk.get("type") == "image" and basename:
                        blk_mime = blk.get("mime_type", "")
                        ext = (
                            basename.rsplit(".", 1)[-1].lower()
                            if "." in basename
                            else ""
                        )
                        if ext and blk_mime.endswith(ext):
                            covered = True
                            break
                if not covered:
                    unresolved.append((basename, url))

            if not unresolved:
                continue

            # List files in the container to resolve unmatched images.
            try:
                container_files = list(
                    self.openai_client.containers.files.list(
                        container_id=container_id,
                    )
                )
            except Exception:
                logger.warning(
                    "Failed to list files in container %s",
                    container_id,
                    exc_info=True,
                )
                continue

            files_by_name: Dict[str, Any] = {}
            for cf in container_files:
                path = getattr(cf, "path", "") or ""
                name = path.rsplit("/", 1)[-1] if "/" in path else path
                if name:
                    files_by_name[name] = cf

            for basename, url in unresolved:
                matched = files_by_name.get(basename)
                if matched is None:
                    for fname, cf in files_by_name.items():
                        if basename.endswith(fname) or fname.endswith(basename):
                            matched = cf
                            break

                if matched is not None:
                    file_id = getattr(matched, "id", None)
                    if not file_id or file_id in downloaded_file_ids:
                        continue
                    block = self._download_container_file(
                        container_id, file_id, basename
                    )
                    if block is not None:
                        blocks.append(block)
                        downloaded_file_ids.add(file_id)
                elif url:
                    blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": url},
                        }
                    )

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
            mime = _mime_from_path(filename)

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


def _mime_from_path(path: str) -> str:
    """Infer a MIME type from a file path or URL.

    Falls back to ``application/octet-stream`` for unrecognised extensions.
    """
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    _MIME_MAP = {
        # Images
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "svg": "image/svg+xml",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        # Documents
        "pdf": "application/pdf",
        "doc": "application/msword",
        "docx": (
            "application/vnd.openxmlformats-officedocument" ".wordprocessingml.document"
        ),
        # Spreadsheets
        "csv": "text/csv",
        "xls": "application/vnd.ms-excel",
        "xlsx": (
            "application/vnd.openxmlformats-officedocument" ".spreadsheetml.sheet"
        ),
        # Text / code
        "txt": "text/plain",
        "json": "application/json",
        "xml": "application/xml",
        "html": "text/html",
        "htm": "text/html",
        "md": "text/markdown",
        "py": "text/x-python",
        "log": "text/plain",
        # Archives
        "zip": "application/zip",
        "tar": "application/x-tar",
        "gz": "application/gzip",
    }
    return _MIME_MAP.get(ext, "application/octet-stream")


# ---------------------------------------------------------------------------
# Public node class
# ---------------------------------------------------------------------------


class PromptBasedAgentNodeV2(RunnableCallable):
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

    _previous_response_id: Optional[str] = None
    """The ID of the previous response, for chaining responses."""

    _pending_function_calls: List[FunctionToolCallItemResource] = []
    """Pending function calls from the last response."""

    _pending_mcp_approvals: List[MCPApprovalRequestItemResource] = []
    """Pending MCP approval requests from the last response."""

    _uses_container_template: bool = False
    """Whether the agent definition uses a ``{{container_id}}`` template.

    When True, every request creates a bespoke container and passes its ID
    via ``structured_inputs`` so the code interpreter can access uploaded
    files at runtime.
    """

    def __init__(
        self,
        client: AIProjectClient,
        model: str,
        instructions: str,
        name: str,
        description: Optional[str] = None,
        agent_name: Optional[str] = None,
        tools: Optional[
            Sequence[Union[AgentServiceBaseToolV2, BaseTool, Callable]]
        ] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tags: Optional[Sequence[str]] = None,
        trace: bool = True,
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
        """
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=trace)

        self._client = client
        self._conversation_id: Optional[str] = None
        self._previous_response_id: Optional[str] = None
        self._pending_function_calls = []
        self._pending_mcp_approvals = []
        self._uses_container_template = False

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
            for i, t in enumerate(tool_defs):
                is_ci = isinstance(t, CodeInterpreterTool) or (
                    isinstance(t, dict) and t.get("type") == "code_interpreter"
                )
                if not is_ci:
                    continue

                # Check whether the tool already has a concrete container
                # (a string ID).  Placeholder values like ``None`` or
                # ``CodeInterpreterToolAuto`` should still be templated.
                existing_container = (
                    t.get("container", None)
                    if isinstance(t, dict)
                    else getattr(t, "container", None)
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
                            "Pre-configured container ID for the code " "interpreter"
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

    @staticmethod
    def _responses_create_with_retry(
        openai_client: Any,
        response_params: Dict[str, Any],
        max_retries: int = 3,
    ) -> Any:
        """Call ``openai_client.responses.create`` with retry on transient 403.

        Azure AI Foundry can transiently return HTTP 403 right after an agent
        is created or when server-side resources are propagating.  The OpenAI
        SDK does not retry 403, so we handle it here.
        """
        from openai import PermissionDeniedError

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return openai_client.responses.create(**response_params)
            except PermissionDeniedError as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    logger.warning(
                        "responses.create returned 403 (attempt %d/%d), " "retrying …",
                        attempt + 1,
                        max_retries,
                    )
                    continue
        raise last_exc  # type: ignore[misc]

    def _func(
        self,
        state: StateSchema,
        config: RunnableConfig,
        *,
        store: Optional[BaseStore],
    ) -> StateSchema:
        if self._agent is None or self._agent_name is None:
            raise RuntimeError(
                "The agent has not been initialized properly or has been " "deleted."
            )

        message = _get_input_from_state(state)
        logger.debug(
            "[_func] message type=%s, agent=%s, prev_response_id=%s",
            type(message).__name__,
            self._agent_name,
            self._previous_response_id,
        )

        openai_client = self._client.get_openai_client()

        try:
            if isinstance(message, ToolMessage):
                logger.info(
                    "Submitting tool message (tool_call_id=%s)",
                    message.tool_call_id,
                )

                if self._pending_mcp_approvals:
                    # ---- MCP approval response path ----
                    logger.info("Submitting MCP approval response")
                    approval_output = _approval_message_to_output(message)
                    input_items: List[Any] = [
                        {
                            "type": "mcp_approval_response",
                            "approval_request_id": (
                                approval_output.approval_request_id
                            ),
                            "approve": approval_output.approve,
                            **(
                                {"reason": approval_output.reason}
                                if approval_output.reason
                                else {}
                            ),
                        }
                    ]

                    response_params: Dict[str, Any] = {
                        "input": input_items,
                        "extra_body": {
                            "agent_reference": {
                                "name": self._agent_name,
                                "type": "agent_reference",
                            }
                        },
                    }

                    if self._previous_response_id:
                        response_params["previous_response_id"] = (
                            self._previous_response_id
                        )

                    response = self._responses_create_with_retry(
                        openai_client, response_params
                    )

                elif self._pending_function_calls:
                    # ---- Function call output path ----
                    # Build function call output items
                    tool_outputs = [_tool_message_to_output(message)]

                    input_items = [
                        {
                            "type": "function_call_output",
                            "call_id": to.call_id,
                            "output": to.output,
                        }
                        for to in tool_outputs
                    ]

                    response_params = {
                        "input": input_items,
                        "extra_body": {
                            "agent_reference": {
                                "name": self._agent_name,
                                "type": "agent_reference",
                            }
                        },
                    }

                    if self._previous_response_id:
                        response_params["previous_response_id"] = (
                            self._previous_response_id
                        )

                    response = self._responses_create_with_retry(
                        openai_client, response_params
                    )

                else:
                    raise RuntimeError(
                        "No pending function calls or MCP approval requests "
                        "to submit tool outputs to."
                    )

            elif isinstance(message, HumanMessage):
                logger.info("Submitting human message: %s", message.content)

                # If the agent uses the container template, extract file
                # blocks, create a bespoke container, upload files to it,
                # and resolve the template via ``structured_inputs``.
                container_id: Optional[str] = None
                if self._uses_container_template:
                    message, container_id = _upload_file_blocks_to_container(
                        message, openai_client
                    )
                    if container_id:
                        logger.info(
                            "Created container %s with uploaded files",
                            container_id,
                        )

                content = _content_from_human_message(message)

                # Create conversation if needed (V2: empty conversation)
                if self._conversation_id is None:
                    conversation = openai_client.conversations.create()
                    self._conversation_id = conversation.id
                    logger.info("Created conversation: %s", self._conversation_id)

                # In V2, the user message is passed as the ``input``
                # parameter to ``responses.create``.
                response_input: Any
                if isinstance(content, list):
                    response_input = [{"role": "user", "content": content}]
                else:
                    response_input = content

                extra_body: Dict[str, Any] = {
                    "agent_reference": {
                        "name": self._agent_name,
                        "type": "agent_reference",
                    }
                }

                # Resolve the ``{{container_id}}`` template variable via
                # ``structured_inputs`` when a container was created.
                if container_id is not None:
                    extra_body["structured_inputs"] = {
                        "container_id": container_id,
                    }

                response_params: Dict[str, Any] = {
                    "conversation": self._conversation_id,
                    "input": response_input,
                    "extra_body": extra_body,
                }

                if self._previous_response_id:
                    response_params["previous_response_id"] = self._previous_response_id

                response = openai_client.responses.create(**response_params)
            else:
                raise RuntimeError(f"Unsupported message type: {type(message)}")

            self._previous_response_id = response.id

            agent_model = _PromptBasedAgentModelV2(
                response=response,
                openai_client=openai_client,
                agent_name=self._agent_name,
                model_name=self._agent.definition.get("model", "unknown")
                if hasattr(self._agent.definition, "get")
                else getattr(self._agent.definition, "model", "unknown"),
                callbacks=config.get("callbacks", None),
                metadata=config.get("metadata", None),
                tags=config.get("tags", None),
            )

            responses = agent_model.invoke([message])
            self._pending_function_calls = agent_model.pending_function_calls
            self._pending_mcp_approvals = agent_model.pending_mcp_approvals

            return {"messages": responses}  # type: ignore[return-value]
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

        def _sync_func() -> StateSchema:
            return self._func(state, config, store=store)  # type: ignore[return-value]

        return await asyncio.to_thread(_sync_func)
