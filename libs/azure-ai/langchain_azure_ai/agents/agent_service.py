"""Factory to create and manage agents in Azure AI Foundry."""

import logging
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Sequence,
    Set,
    Type,
    Union,
)

from azure.ai.projects import AIProjectClient
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.chat_agent_executor import Prompt
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from pydantic import BaseModel, ConfigDict

from langchain_azure_ai.agents.prebuilt.declarative import DeclarativeChatAgentNode
from langchain_azure_ai.agents.prebuilt.tools import AgentServiceBaseTool

logger = logging.getLogger(__package__)


def external_tools_condition(
    state: MessagesState,
) -> Literal["tools", "__end__"]:
    """Determine the next node based on whether the AI message contains tool calls."""
    ai_message = state["messages"][-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


class AgentServiceFactory(BaseModel):
    """Factory to create and manage declarative chat agents in Azure AI Foundry.

    Examples:
        To create a simple echo agent:

        .. code-block:: python
            from langchain_azure_ai.agents import AgentServiceFactory
            from langchain_core.messages import HumanMessage
            from azure.identity import DefaultAzureCredential

            factory = AgentServiceFactory(
                project_endpoint=(
                    "https://resource.services.ai.azure.com/api/projects/demo-project",
                ),
                credential=DefaultAzureCredential()
            )

            agent = factory.create_declarative_chat_agent(
                name="my-echo-agent",
                model="gpt-4.1",
                instructions="You are a helpful AI assistant that always replies back
                             "saying the opposite of what the user says.",
            )

            messages = [HumanMessage(content="I'm a genius and I love programming!")]
            state = agent.invoke({"messages": messages})

            for m in state['messages']:
                m.pretty_print()

        Agents can also be created with tools. For example, to create an agent that
        can perform arithmetic using a calculator tool:

        .. code-block:: python
            # add, multiply, divide are simple functions defined elsewhere
            # those functions are documented and with proper type hints

            tools = [add, multiply, divide]

            agent = factory.create_declarative_chat_agent(
                name="math-agent",
                model="gpt-4.1",
                instructions="You are a helpful assistant tasked with performing "
                             "arithmetic on a set of inputs.",
                tools=tools,
            )

        You can also use the built-in tools in the Agent Service. Those tools only
        work with agents created in Azure AI Foundry. For example, to create an agent
        that can use Code Interpreter.

        .. code-block:: python
            from langchain_azure_ai.tools.agent_service import CodeInterpreterTool

            document_parser_agent = factory.create_declarative_chat_agent(
                name="code-interpreter-agent",
                model="gpt-4.1",
                instructions="You are a helpful assistant that can run complex "
                             "mathematical functions precisely via tools.",
                tools=[CodeInterpreterTool()],
            )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    project_endpoint: Optional[str] = None
    """The project endpoint associated with the AI project. If this is specified,
    then the `endpoint` parameter becomes optional and `credential` has to be of type
    `TokenCredential`."""

    credential: Optional[TokenCredential] = None
    """The API key or credential to use to connect to the service. If using a project 
    endpoint, this must be of type `TokenCredential` since only Microsoft EntraID is 
    supported."""

    api_version: Optional[str] = None
    """The API version to use with Azure. If None, the 
    default version is used."""

    client_kwargs: Dict[str, Any] = {}
    """Additional keyword arguments to pass to the client."""

    def _initialize_client(self) -> AIProjectClient:
        """Initialize the AIProjectClient."""
        credential: TokenCredential
        if self.credential is None:
            credential = DefaultAzureCredential()
        else:
            credential = self.credential

        if self.project_endpoint is None:
            raise ValueError(
                "The `project_endpoint` parameter must be specified to create the "
                "AIProjectClient."
            )

        return AIProjectClient(
            endpoint=self.project_endpoint,
            credential=credential,
            user_agent="langchain-azure-ai",
            **self.client_kwargs,
        )

    def delete_agent(self, agent: CompiledStateGraph) -> None:
        """Delete an agent created with create_declarative_chat_agent.

        Args:
            agent: The CompiledStateGraph representing the agent to delete.

        Raises:
            ValueError: If the agent ID cannot be found in the graph metadata.
        """
        client = self._initialize_client()

        agent_ids = self.get_declarative_agents_id_from_graph(agent)
        for agent_id in agent_ids:
            client.agents.delete_agent(agent_id)
            logger.info(f"Deleted agent with ID: {agent_id}")
        else:
            logger.warning("No agent ID found in the graph metadata.")

    def get_declarative_agents_id_from_graph(
        self, graph: CompiledStateGraph
    ) -> Set[str]:
        """Get the agents associated with a state graph."""
        agent_ids = set()
        for node in graph.nodes.values():
            if node.metadata and "agent_id" in node.metadata:
                agent_ids.add(node.metadata.get("agent_id"))
        return agent_ids  # type: ignore[return-value]

    def create_declarative_chat_node(
        self,
        name: str,
        model: str,
        description: Optional[str] = None,
        tools: Optional[
            Union[
                Sequence[Union[AgentServiceBaseTool, BaseTool, Callable]],
                ToolNode,
            ]
        ] = None,
        instructions: Optional[Prompt] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        trace: bool = False,
    ) -> DeclarativeChatAgentNode:
        """Create a declarative chat agent node in Azure AI Foundry.

        Args:
            name: The name of the agent.
            model: The model to use for the agent.
            description: An optional description of the agent.
            tools: The tools to use with the agent. This can be a list of BaseTools
                callables, or tool definitions, or a ToolNode.
            instructions: The prompt instructions to use for the agent.
            temperature: The temperature to use for the agent.
            top_p: The top_p to use for the agent.
            response_format: The response format to use for the agent.
            trace: Whether to enable tracing.

        Returns:
            A DeclarativeChatAgentNode representing the agent.
        """
        logger.info("Validating parameters...")
        if not isinstance(instructions, str):
            raise ValueError("Only string instructions are supported momentarily.")

        logger.info("Initializing AIProjectClient")
        client = self._initialize_client()

        return DeclarativeChatAgentNode(
            client=client,
            name=name,
            description=description,
            model=model,
            instructions=instructions,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
            tools=tools,
            trace=trace,
        )

    def create_declarative_chat_agent(
        self,
        model: str,
        name: str,
        description: Optional[str] = None,
        tools: Optional[
            Union[
                Sequence[Union[AgentServiceBaseTool, BaseTool, Callable]],
                ToolNode,
            ]
        ] = None,
        instructions: Optional[Prompt] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        context_schema: Optional[Type[Any]] = None,
        checkpointer: Optional[Checkpointer] = None,
        store: Optional[BaseStore] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        trace: bool = False,
        debug: bool = False,
    ) -> CompiledStateGraph:
        """Create a declarative chat agent in Azure AI Foundry.

        Args:
            name: The name of the agent.
            description: An optional description of the agent.
            model: The model to use for the agent.
            tools: The tools to use with the agent. This can be a list of BaseTools,
                callables, or tool definitions, or a ToolNode.
            instructions: The prompt instructions to use for the agent.
            temperature: The temperature to use for the agent.
            top_p: The top_p to use for the agent.
            response_format: The response format to use for the agent.
            context_schema: The schema for the context to pass to the agent.
            checkpointer: The checkpointer to use for the agent.
            store: The store to use for the agent.
            interrupt_before: A list of node names to interrupt before.
            interrupt_after: A list of node names to interrupt after.
            trace: Whether to enable tracing.
            debug: Whether to enable debug mode.

        Returns:
            A CompiledStateGraph representing the agent workflow.
        """
        logger.info(f"Creating agent with name: {name}")

        builder = StateGraph(MessagesState, context_schema=context_schema)

        logger.info("Adding DeclarativeChatAgentNode")
        declarative_node = self.create_declarative_chat_node(
            name=name,
            description=description,
            model=model,
            tools=tools,
            instructions=instructions,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
            trace=trace,
        )
        builder.add_node(
            "foundryAgent",
            declarative_node,
            metadata={"agent_id": declarative_node._agent_id},
        )
        logger.info("DeclarativeChatAgentNode added")

        builder.add_edge(START, "foundryAgent")

        if tools is not None:
            if isinstance(tools, ToolNode):
                logger.info("Adding 1 ToolNode")
                builder.add_node("tools", tools)
            else:
                # Removing AgentServiceBaseTool from the tools passed to ToolNode
                # as these are automatically handled in the DeclarativeChatAgentNode
                filtered_tools = [
                    t for t in tools if not isinstance(t, AgentServiceBaseTool)
                ]
                if len(filtered_tools) > 0:
                    logger.info("Creating ToolNode with tools")
                    builder.add_node("tools", ToolNode(filtered_tools))
                else:
                    logger.info(
                        "All tools are AgentServiceBaseTool, skipping ToolNode creation"
                    )

        if "tools" in builder.nodes.keys():
            logger.info("Adding conditional edges")
            builder.add_conditional_edges(
                "foundryAgent",
                external_tools_condition,
            )
            logger.info("Conditional edges added")

            builder.add_edge("tools", "foundryAgent")
        else:
            logger.info("No tools found, adding edge to END")
            builder.add_edge("foundryAgent", END)

        logger.info("Compiling state graph")
        graph = builder.compile(
            name=name,
            checkpointer=checkpointer,
            store=store,
            interrupt_after=interrupt_after,
            interrupt_before=interrupt_before,
            debug=debug,
        )

        logger.info("State graph compiled")
        return graph
