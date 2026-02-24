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

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AgentVersionDetails,
    FunctionToolCallItemResource,
    FunctionToolCallOutputItemParam,
    ItemType,
    MCPApprovalRequestItemResource,
    MCPApprovalResponseItemParam,
    PromptAgentDefinition,
)
from azure.core.exceptions import HttpResponseError
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel, ChatResult
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph._internal._runnable import RunnableCallable
from langgraph.prebuilt.chat_agent_executor import StateSchema
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore

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


def _get_thread_input_from_state(state: StateSchema) -> BaseMessage:
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


# ---------------------------------------------------------------------------
# Internal chat-model wrapper (mirrors the V1 _PromptBasedAgentModel)
# ---------------------------------------------------------------------------


class _PromptBasedAgentModelV2(BaseChatModel):
    """A LangChain chat model wrapper for Azure AI Foundry V2 agents.

    It interprets a ``Response`` object produced by the OpenAI Responses API
    and converts its output items into LangChain messages.
    """

    response: Any  # azure.ai.projects.models.Response
    """The V2 Response object."""

    agent_name: str
    """The agent name (used to tag messages)."""

    model_name: str
    """The model deployment name."""

    pending_function_calls: List[FunctionToolCallItemResource] = []
    """Function calls that need external resolution."""

    pending_mcp_approvals: List[MCPApprovalRequestItemResource] = []
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
            # Completed response – extract text output
            self.pending_function_calls = []
            self.pending_mcp_approvals = []
            output_text = getattr(response, "output_text", None)
            if output_text:
                msg = AIMessage(content=output_text)
                msg.name = self.agent_name
                generations.append(ChatGeneration(message=msg, generation_info={}))
            else:
                # Fallback: iterate over output items for message-type items
                for item in response.output or []:
                    if getattr(item, "type", None) == ItemType.MESSAGE:
                        for content_part in getattr(item, "content", []):
                            text = getattr(content_part, "text", None)
                            if text:
                                msg = AIMessage(content=text)
                                msg.name = self.agent_name
                                generations.append(
                                    ChatGeneration(message=msg, generation_info={})
                                )

        llm_output: Dict[str, Any] = {"model": self.model_name}
        usage = getattr(response, "usage", None)
        if usage:
            llm_output["token_usage"] = getattr(usage, "total_tokens", None)
        return ChatResult(generations=generations, llm_output=llm_output)


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

    def __init__(
        self,
        client: AIProjectClient,
        model: str,
        instructions: str,
        name: str,
        description: Optional[str] = None,
        agent_name: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[
            Union[
                Sequence[Union[AgentServiceBaseToolV2, BaseTool, Callable]],
                ToolNode,
            ]
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
                instead of creating a new one.
            response_format: Optional response format configuration.
            tools: Tools the agent can use.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            tags: Optional tags for the runnable.
            trace: Whether to enable tracing.
        """
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=trace)

        self._client = client
        self._pending_function_calls = []
        self._pending_mcp_approvals = []

        if agent_name is not None:
            try:
                existing = self._client.agents.get(agent_name=agent_name)
                self._agent_name = existing.name
                logger.info("Using existing agent: %s", self._agent_name)
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
            if isinstance(tools, ToolNode):
                raise ValueError(
                    "ToolNode is not supported directly as tools input for V2. "
                    "Use a list of tools instead."
                )
            definition_params["tools"] = _get_v2_tool_definitions(list(tools))

        definition = PromptAgentDefinition(**definition_params)

        agent_create_params: Dict[str, Any] = {
            "agent_name": name,
            "definition": definition,
        }
        if description is not None:
            agent_create_params["description"] = description

        self._agent = self._client.agents.create_version(
            **agent_create_params
        )

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
                        "responses.create returned 403 (attempt %d/%d), "
                        "retrying …",
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

        message = _get_thread_input_from_state(state)
        logger.debug(
            "[_func] message type=%s, agent=%s, prev_response_id=%s",
            type(message).__name__,
            self._agent_name,
            self._previous_response_id,
        )

        openai_client = self._client.get_openai_client()

        try:
            if isinstance(message, ToolMessage):
                logger.info("Submitting tool message with ID %s", message.id)

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
                content = _content_from_human_message(message)

                response_params = {
                    "input": content,
                    "extra_body": {
                        "agent_reference": {
                            "name": self._agent_name,
                            "type": "agent_reference",
                        }
                    },
                }

                if self._previous_response_id is not None:
                    response_params["previous_response_id"] = (
                        self._previous_response_id
                    )

                response = self._responses_create_with_retry(
                    openai_client, response_params
                )
            else:
                raise RuntimeError(f"Unsupported message type: {type(message)}")

            self._previous_response_id = response.id

            agent_model = _PromptBasedAgentModelV2(
                response=response,
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
