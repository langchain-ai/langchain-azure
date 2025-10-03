"""Declarative chat agent node for Azure AI Foundry agents."""

import json
import logging
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
    MessageInputTextBlock,
    RequiredFunctionToolCall,
    StructuredToolOutput,
    SubmitToolOutputsAction,
    Tool,
    ToolOutput,
    ToolResources,
    ToolSet,
)
from azure.ai.projects import AIProjectClient
from azure.core.exceptions import HttpResponseError
from langchain.tools import BaseTool
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
)
from langgraph._internal._runnable import RunnableCallable
from langgraph.graph import MessagesState
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore

from langchain_azure_ai.tools.agent_service import AgentServiceBaseTool

logger = logging.getLogger(__package__)


class OpenAIFunctionTool(Tool[FunctionToolDefinition]):
    """A tool that wraps OpenAI function definitions."""

    def __init__(self, definitions: List[FunctionToolDefinition]):
        """Initialize the OpenAIFunctionTool with function definitions.

        Args:
        definitions: A list of function definitions to be used by the tool.
        """
        self._definitions = definitions

    @property
    def definitions(self) -> List[FunctionToolDefinition]:
        """Get the function definitions.

        Returns:
            A list of function definitions.
        """
        return self._definitions

    @property
    def resources(self) -> ToolResources:
        """Get the tool resources for the agent.

        Returns:
            The tool resources.
        """
        return ToolResources()

    def execute(self, tool_call: Any) -> Any:
        """Execute the tool with the provided tool call.

        :param Any tool_call: The tool call to execute.
        :return: The output of the tool operations.
        """
        pass


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
) -> ToolSet:
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
                toolset.add(tool.tool)
            elif isinstance(tool, BaseTool):
                function_def = convert_to_openai_function(tool)
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
                function_tools.add(tool)
            else:
                if isinstance(tool, Tool):
                    raise ValueError(
                        "Passing raw Tool definitions from package azure-ai-agents "
                        "is not supported. Wrap the tool in "
                        "`langchain_azure_ai.tools.agent_service.AgentServiceBaseTool` "
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
        toolset.add(OpenAIFunctionTool(openai_tools))

    return toolset


class DeclarativeChatAgentNode(RunnableCallable):
    """A LangGraph node that represents a declarative chat agent in Azure AI Foundry.

    You can use this node to create complex graphs that involve interactions with
    an Azure AI Foundry agent.

    You can also use `langchain_azure_ai.agents.AgentServiceFactory` to create
    instances of this node.

    Example:
        .. code-block:: python
            from langchain_azure_ai.agents import AgentServiceFactory
            from langchain_azure_ai.tools.agent_service import AgentServiceBaseTool
            from azure.identity import DefaultAzureCredential
            from langchain_core.messages import HumanMessage

            factory = AgentServiceFactory(
                project_endpoint=(
                    "https://resource.services.ai.azure.com/api/projects/demo-project",
                ),
                credential=DefaultAzureCredential()
            )

            coder = factory.create_declarative_chat_node(
                name="code-interpreter-agent",
                model="gpt-4.1",
                instructions="You are a helpful assistant that can run Python code.",
                tools=[AgentServiceBaseTool(tool=CodeInterpreterTool())],
            )
    """

    name: str = "DeclarativeChatAgent"

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

    def __init__(
        self,
        client: AIProjectClient,
        model: str,
        instructions: str,
        name: str,
        agent_id: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tools: Optional[
            Union[
                Sequence[Union[AgentServiceBaseTool, BaseTool, Callable]],
                ToolNode,
            ]
        ] = None,
        tool_resources: Optional[Any] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tags: Sequence[str] | None = None,
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
            trace: Whether to enable tracing for the node. Defaults to True.
        """
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=trace)

        self._client = client

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
            agent_params["toolset"] = _get_tool_definitions(tools)
            tool_resources = _get_tool_resources(tools)
            if tool_resources is not None:
                agent_params["tool_resources"] = tool_resources

        self._agent = client.agents.create_agent(**agent_params)
        self._agent_id = self._agent.id
        self._agent_name = name
        logger.info(f"Created agent with name: {self._agent.name} ({self._agent.id})")

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
        input: MessagesState,
        config: RunnableConfig,
        *,
        store: Optional[BaseStore],
    ) -> Any:
        if self._agent is None or self._agent_id is None:
            raise ValueError(
                "The agent has not been initialized properly "
                "its associated agent in Azure AI Foundry "
                "has been deleted."
            )

        if self._thread_id is None:
            thread = self._client.agents.threads.create()
            self._thread_id = thread.id
            logger.info(f"Created new thread with ID: {self._thread_id}")

        assert self._thread_id is not None

        message = input["messages"][-1]

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
                    raise ValueError(
                        f"Run {self._pending_run_id} is not in a state to accept "
                        "tool outputs."
                    )
            else:
                raise ValueError(
                    "No pending run to submit tool outputs to. Got ToolMessage "
                    "without a pending run."
                )
        elif isinstance(message, HumanMessage):
            logger.info(f"Submitting human message {message.content}")
            if isinstance(message.content, str):
                self._client.agents.messages.create(
                    thread_id=self._thread_id, role="user", content=message.content
                )
            elif isinstance(message.content, dict):
                raise ValueError(
                    "Message content as dict is not supported yet. "
                    "Please submit as string."
                )
            elif isinstance(message.content, list):
                self._client.agents.messages.create(
                    thread_id=self._thread_id,
                    role="user",
                    content=[MessageInputTextBlock(block) for block in message.content],  # type: ignore[arg-type]
                )
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

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
            time.sleep(1)
            run = self._client.agents.runs.get(thread_id=self._thread_id, run_id=run.id)

        if run.status == "requires_action" and isinstance(
            run.required_action, SubmitToolOutputsAction
        ):
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            for tool_call in tool_calls:
                if isinstance(tool_call, RequiredFunctionToolCall):
                    input["messages"].append(_required_tool_calls_to_message(tool_call))
                else:
                    raise ValueError(
                        f"Unsupported tool call type: {type(tool_call)} in run "
                        f"{run.id}."
                    )
            self._pending_run_id = run.id
        else:
            response = self._client.agents.messages.list(
                thread_id=self._thread_id,
                limit=1,
                order=ListSortOrder.DESCENDING,
            )
            for msg in response:
                # TODO: handle other types of content
                if msg.text_messages:
                    last_text = msg.text_messages[0]
                    input["messages"].append(AIMessage(content=last_text.text.value))
                    break
            self._pending_run_id = None

    async def _afunc(
        self,
        input: MessagesState,
        config: RunnableConfig,
        *,
        store: Optional[BaseStore],
    ) -> Any:
        import asyncio

        def _sync_func() -> Any:
            return self._func(input, config, store=store)

        return await asyncio.to_thread(_sync_func)
