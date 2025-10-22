"""Declarative chat agent node for Azure AI Foundry agents."""

import base64
import json
import logging
import tempfile
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

from azure.ai.agents.models import (
    Agent,
    FunctionDefinition,
    FunctionTool,
    FunctionToolDefinition,
    ListSortOrder,
    MessageImageUrlParam,
    MessageInputContentBlock,
    MessageInputImageUrlBlock,
    MessageInputTextBlock,
    RequiredFunctionToolCall,
    StructuredToolOutput,
    SubmitToolOutputsAction,
    ThreadMessage,
    Tool,
    ToolDefinition,
    ToolOutput,
    ToolResources,
    ToolSet,
)
from azure.ai.projects import AIProjectClient
from azure.core.exceptions import HttpResponseError
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
)
from langgraph._internal._runnable import RunnableCallable
from langgraph.prebuilt.chat_agent_executor import StateSchema
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore

from langchain_azure_ai.agents.prebuilt.tools import (
    AgentServiceBaseTool,
    _OpenAIFunctionTool,
)

logger = logging.getLogger(__package__)


def _required_tool_calls_to_message(
    required_tool_call: RequiredFunctionToolCall,
) -> AIMessage:
    """Convert a RequiredFunctionToolCall to an AIMessage with tool calls.

    Args:
        required_tool_call: The RequiredFunctionToolCall to convert.

    Returns:
        An AIMessage containing the tool calls.
    """
    tool_calls: List[ToolCall] = []
    tool_calls.append(
        ToolCall(
            id=required_tool_call.id,
            name=required_tool_call.function.name,
            args=json.loads(required_tool_call.function.arguments),
        )
    )
    return AIMessage(content="", tool_calls=tool_calls)


def _tool_message_to_output(tool_message: ToolMessage) -> StructuredToolOutput:
    """Convert a ToolMessage to a ToolOutput."""
    # TODO: Add support to artifacts

    return ToolOutput(
        tool_call_id=tool_message.tool_call_id,
        output=tool_message.content,  # type: ignore[arg-type]
    )


def _get_tool_resources(
    tools: Union[
        Sequence[Union[AgentServiceBaseTool, BaseTool, Callable]],
        ToolNode,
    ],
) -> Union[ToolResources, None]:
    """Get the tool resources for a list of tools.

    Args:
        tools: A list of tools to get resources for.

    Returns:
        The tool resources.
    """
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, AgentServiceBaseTool):
                if tool.tool.resources is not None:
                    return tool.tool.resources
            else:
                continue
    return None


def _get_tool_definitions(
    tools: Union[
        Sequence[Union[AgentServiceBaseTool, BaseTool, Callable]],
        ToolNode,
    ],
) -> List[ToolDefinition]:
    """Convert a list of tools to a ToolSet for the agent.

    Args:
        tools: A list of tools, which can be BaseTool instances, callables, or
            tool definitions.

    Returns:
    A ToolSet containing the converted tools.
    """
    toolset = ToolSet()
    function_tools: set[Callable] = set()
    openai_tools: list[FunctionToolDefinition] = []

    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, AgentServiceBaseTool):
                logger.debug(f"Adding AgentService tool: {tool.tool}")
                toolset.add(tool.tool)
            elif isinstance(tool, BaseTool):
                function_def = convert_to_openai_function(tool)
                logger.debug(f"Adding OpenAI function tool: {function_def['name']}")
                openai_tools.append(
                    FunctionToolDefinition(
                        function=FunctionDefinition(
                            name=function_def["name"],
                            description=function_def["description"],
                            parameters=function_def["parameters"],
                        )
                    )
                )
            elif callable(tool):
                logger.debug(f"Adding callable function tool: {tool.__name__}")
                function_tools.add(tool)
            else:
                if isinstance(tool, Tool):
                    raise ValueError(
                        "Passing raw Tool definitions from package azure-ai-agents "
                        "is not supported. Wrap the tool in "
                        "langchain_azure_ai.agents.prebuilt.tools.AgentServiceBaseTool"
                        " and pass `tool=<your_tool>`."
                    )
                else:
                    raise ValueError(
                        "Each tool must be an AgentServiceBaseTool, BaseTool, or a "
                        f"callable. Got {type(tool)}"
                    )
    elif isinstance(tools, ToolNode):
        raise ValueError(
            "ToolNode is not supported as a tool input. Use a list of " "tools instead."
        )
    else:
        raise ValueError("tools must be a list or a ToolNode.")

    if len(function_tools) > 0:
        toolset.add(FunctionTool(function_tools))
    if len(openai_tools) > 0:
        toolset.add(_OpenAIFunctionTool(openai_tools))

    return toolset.definitions


def _get_thread_input_from_state(state: StateSchema) -> BaseMessage:
    """Extract the latest message from the state.

    Args:
        state: The current state, expected to have a 'messages' key.

    Returns:
        The latest message from the state's messages.
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
) -> Union[str, List[Union[MessageInputContentBlock]]]:
    """Convert a HumanMessage content to a list of blocks.

    Args:
        message: The HumanMessage to convert.

    Returns:
        A list of MessageInputTextBlock or MessageInputImageFileBlock.
    """
    content: List[Union[MessageInputContentBlock]] = []
    if isinstance(message.content, str):
        return message.content
    elif isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, str):
                content.append(MessageInputTextBlock(text=block))
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    content.append(MessageInputTextBlock(text=block.get("text", "")))
                elif block_type == "image_url":
                    content.append(
                        MessageInputImageUrlBlock(
                            image_url=MessageImageUrlParam(
                                url=block["image_url"]["url"], detail="high"
                            )
                        ),
                    )
                elif block_type == "image":
                    if block.get("source_type") == "base64":
                        content.append(
                            MessageInputImageUrlBlock(
                                image_url=MessageImageUrlParam(
                                    url=f"data:{block['mime_type']};base64,{block['data']}",
                                    detail="high",
                                )
                            ),
                        )
                    elif block_type == "url":
                        content.append(
                            MessageInputImageUrlBlock(
                                image_url=MessageImageUrlParam(
                                    url=block["url"], detail="high"
                                )
                            ),
                        )
                    else:
                        raise ValueError(
                            "Only 'base64' and 'url' source types are supported for "
                            "image blocks."
                        )
                else:
                    raise ValueError(
                        f"Unsupported block type {block_type} in HumanMessage "
                        "content. Only 'image' type is supported as dict."
                    )
            else:
                raise ValueError("Unexpected block type in HumanMessage content.")
    else:
        raise ValueError(
            "HumanMessage content must be either a string or a list of strings and/or"
            " dicts."
        )
    return content


class PromptBasedAgentNode(RunnableCallable):
    """A LangGraph node that represents a prompt-based agent in Azure AI Foundry.

    You can use this node to create complex graphs that involve interactions with
    an Azure AI Foundry agent.

    You can also use `langchain_azure_ai.agents.AgentServiceFactory` to create
    instances of this node.

    Example:
    ```python
    from azure.identity import DefaultAzureCredential
    from langchain_azure_ai.agents import AgentServiceFactory

    factory = AgentServiceFactory(
        project_endpoint=(
            "https://resource.services.ai.azure.com/api/projects/demo-project",
        ),
        credential=DefaultAzureCredential()
    )

    coder = factory.create_prompt_agent_node(
        name="code-interpreter-agent",
        model="gpt-4.1",
        instructions="You are a helpful assistant that can run Python code.",
        tools=[func1, func2],
    )
    ```
    """

    name: str = "PromptAgent"

    _client: AIProjectClient
    """The AIProjectClient instance to use."""

    _agent: Optional[Agent] = None
    """The agent instance to use."""

    _agent_name: Optional[str] = None
    """The name of the agent to create or use."""

    _agent_id: Optional[str] = None
    """The ID of the agent to use. If not provided, a new agent will be created."""

    _thread_id: Optional[str] = None
    """The ID of the conversation thread to use. If not provided, a new thread will be
    created."""

    _pending_run_id: Optional[str] = None
    """The ID of the pending run, if any."""

    _polling_interval: int = 1
    """The interval (in seconds) to poll for updates on the agent's status."""

    def __init__(
        self,
        client: AIProjectClient,
        model: str,
        instructions: str,
        name: str,
        description: Optional[str] = None,
        agent_id: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[
            Union[
                Sequence[Union[AgentServiceBaseTool, BaseTool, Callable]],
                ToolNode,
            ]
        ] = None,
        tool_resources: Optional[Any] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        polling_interval: int = 1,
        tags: Optional[Sequence[str]] = None,
        trace: bool = True,
    ) -> None:
        """Initialize the DeclarativeChatAgentNode.

        Args:
            client: The AIProjectClient instance to use.
            model: The model to use for the agent.
            instructions: The prompt instructions to use for the agent.
            name: The name of the agent.
            agent_id: The ID of an existing agent to use. If not provided, a new
                agent will be created.
            response_format: The response format to use for the agent.
            description: An optional description for the agent.
            tools: A list of tools to use with the agent. Each tool can be a
            dictionary defining the tool.
            tool_resources: Optional tool resources to use with the agent.
            temperature: The temperature to use for the agent.
            top_p: The top_p value to use for the agent.
            tags: Optional tags to associate with the agent.
            polling_interval: The interval (in seconds) to poll for updates on the
                agent's status. Defaults to 1 second.
            trace: Whether to enable tracing for the node. Defaults to True.
        """
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=trace)

        self._client = client
        self._polling_interval = polling_interval

        if agent_id is not None:
            try:
                self._agent = self._client.agents.get_agent(agent_id=agent_id)
                self._agent_id = self._agent.id
                self._agent_name = self._agent.name
            except HttpResponseError as e:
                raise ValueError(
                    f"Could not find agent with ID {agent_id} in the "
                    "connected project. Do not pass agent_id when "
                    "creating a new agent."
                ) from e

        agent_params: Dict[str, Any] = {
            "model": model,
            "name": name,
            "instructions": instructions,
        }

        # Add optional parameters
        if description:
            agent_params["description"] = description
        if tool_resources:
            agent_params["tool_resources"] = tool_resources
        if tags:
            agent_params["metadata"] = tags
        if temperature is not None:
            agent_params["temperature"] = temperature
        if top_p is not None:
            agent_params["top_p"] = top_p
        if response_format is not None:
            agent_params["response_format"] = response_format

        if tools is not None:
            agent_params["tools"] = _get_tool_definitions(tools)
            tool_resources = _get_tool_resources(tools)
            if tool_resources is not None:
                agent_params["tool_resources"] = tool_resources

        self._agent = client.agents.create_agent(**agent_params)
        self._agent_id = self._agent.id
        self._agent_name = name
        logger.info(f"Created agent with name: {self._agent.name} ({self._agent.id})")

    def _to_langchain_message(self, msg: ThreadMessage) -> AIMessage:
        """Convert an Azure AI Foundry message to a LangChain message.

        Args:
            msg: The message from Azure AI Foundry.

        Returns:
            The corresponding LangChain message, or None if the message type is
            unsupported.
        """
        contents: List[Union[str, Dict[Any, Any]]] = []
        file_paths: Dict[str, str] = {}
        if msg.text_messages:
            for text in msg.text_messages:
                contents.append(text.text.value)
        if msg.file_path_annotations:
            for ann in msg.file_path_annotations:
                logger.info(
                    f"Found file path annotation: {ann.type} with text {ann.text}"
                )
                if ann.type == "file_path":
                    file_paths[ann.file_path.file_id] = ann.text.split("/")[-1]
        if msg.image_contents:
            for img in msg.image_contents:
                file_id = img.image_file.file_id
                file_name = file_paths.get(file_id, f"{file_id}.png")
                with tempfile.TemporaryDirectory() as target_dir:
                    logger.info(f"Downloading image file {file_id} as {file_name}")
                    self._client.agents.files.save(
                        file_id=file_id,
                        file_name=file_name,
                        target_dir=target_dir,
                    )
                    with open(f"{target_dir}/{file_name}", "rb") as f:
                        content = f.read()
                        contents.append(
                            {
                                "type": "image",
                                "mime_type": "image/png",
                                "base64": base64.b64encode(content).decode("utf-8"),
                            }
                        )

        if len(contents) == 1:
            return AIMessage(content=contents[0])  # type: ignore[arg-type]
        return AIMessage(content=contents)  # type: ignore[arg-type]

    def delete_agent_from_node(self) -> None:
        """Delete an agent associated with a DeclarativeChatAgentNode node."""
        if self._agent_id is not None:
            self._client.agents.delete_agent(self._agent_id)
            logger.info(f"Deleted agent with ID: {self._agent_id}")

            self._agent_id = None
            self._agent = None
        else:
            raise ValueError(
                "The node does not have an associated agent ID to eliminate"
            )

    def _func(
        self,
        state: StateSchema,
        config: RunnableConfig,
        *,
        store: Optional[BaseStore],
    ) -> StateSchema:
        if self._agent is None or self._agent_id is None:
            raise RuntimeError(
                "The agent has not been initialized properly "
                "its associated agent in Azure AI Foundry "
                "has been deleted."
            )

        if self._thread_id is None:
            thread = self._client.agents.threads.create()
            self._thread_id = thread.id
            logger.info(f"Created new thread with ID: {self._thread_id}")

        assert self._thread_id is not None

        message = _get_thread_input_from_state(state)

        if isinstance(message, ToolMessage):
            logger.info(f"Submitting tool message with ID {message.id}")
            if self._pending_run_id:
                run = self._client.agents.runs.get(
                    thread_id=self._thread_id, run_id=self._pending_run_id
                )
                if run.status == "requires_action" and isinstance(
                    run.required_action, SubmitToolOutputsAction
                ):
                    tool_outputs = [_tool_message_to_output(message)]
                    self._client.agents.runs.submit_tool_outputs(
                        thread_id=self._thread_id,
                        run_id=self._pending_run_id,
                        tool_outputs=tool_outputs,
                    )
                else:
                    raise RuntimeError(
                        f"Run {self._pending_run_id} is not in a state to accept "
                        "tool outputs."
                    )
            else:
                raise RuntimeError(
                    "No pending run to submit tool outputs to. Got ToolMessage "
                    "without a pending run."
                )
        elif isinstance(message, HumanMessage):
            logger.info(f"Submitting human message {message.content}")
            self._client.agents.messages.create(
                thread_id=self._thread_id,
                role="user",
                content=_content_from_human_message(message),  # type: ignore[arg-type]
            )
        else:
            raise RuntimeError(f"Unsupported message type: {type(message)}")

        if self._pending_run_id is None:
            logger.info("Creating and processing new run...")
            run = self._client.agents.runs.create(
                thread_id=self._thread_id,
                agent_id=self._agent_id,
            )
        else:
            logger.info(f"Getting existing run {self._pending_run_id}...")
            run = self._client.agents.runs.get(
                thread_id=self._thread_id, run_id=self._pending_run_id
            )

        while run.status in ["queued", "in_progress"]:
            time.sleep(self._polling_interval)
            run = self._client.agents.runs.get(thread_id=self._thread_id, run_id=run.id)

        responses: List[BaseMessage] = []

        if run.status == "requires_action" and isinstance(
            run.required_action, SubmitToolOutputsAction
        ):
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            for tool_call in tool_calls:
                if isinstance(tool_call, RequiredFunctionToolCall):
                    responses.append(_required_tool_calls_to_message(tool_call))
                else:
                    raise ValueError(
                        f"Unsupported tool call type: {type(tool_call)} in run "
                        f"{run.id}."
                    )
            self._pending_run_id = run.id
        elif run.status == "failed":
            raise RuntimeError(f"Run {run.id} failed with error: {run.last_error}")
        elif run.status == "completed":
            response = self._client.agents.messages.list(
                thread_id=self._thread_id,
                run_id=run.id,
                order=ListSortOrder.ASCENDING,
            )
            for msg in response:
                new_message = self._to_langchain_message(msg)
                new_message.name = self._agent_name
                if new_message:
                    responses.append(new_message)

            self._pending_run_id = None

        return {"messages": responses}  # type: ignore[return-value]

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
