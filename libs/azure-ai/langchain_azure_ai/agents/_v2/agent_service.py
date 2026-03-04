"""Factory to create and manage agents in Azure AI Foundry (V2).

This module provides ``AgentServiceFactory`` which uses the
``azure-ai-projects >= 2.0`` library (Responses / Conversations API).
"""

import itertools
import json
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
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
from langchain.agents.factory import (
    _add_middleware_edge,
    _chain_async_model_call_handlers,
    _chain_async_tool_call_wrappers,
    _chain_model_call_handlers,
    _chain_tool_call_wrappers,
    _get_can_jump_to,
    _resolve_schema,
)
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.utils import pre_init
from langgraph._internal._runnable import RunnableCallable
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.chat_agent_executor import (
    Prompt,
    StateSchemaType,
)
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, interrupt
from pydantic import BaseModel, ConfigDict

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.agents._v2.prebuilt.declarative import (
    MCP_APPROVAL_REQUEST_TOOL_NAME,
    AgentServiceAgentState,
    PromptBasedAgentNode,
)
from langchain_azure_ai.agents._v2.prebuilt.tools import AgentServiceBaseTool
from langchain_azure_ai.callbacks.tracers.inference_tracing import (
    AzureAIOpenTelemetryTracer,
)
from langchain_azure_ai.utils.env import get_from_dict_or_env

logger = logging.getLogger(__package__)


@experimental()
def external_tools_condition(
    state: MessagesState,
) -> Literal["tools", "__end__"]:
    """Determine the next node based on whether the AI message has tool calls."""
    ai_message = state["messages"][-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


def _mcp_approval_node(state: MessagesState) -> Dict[str, list]:
    r"""Pause execution for human approval of MCP tool calls.

    When the foundry agent returns an MCP approval request (surfaced as a
    tool call named ``mcp_approval_request``), this node interrupts graph
    execution and waits for the user to provide an approval decision.

    The interrupt payload is a list of approval-request dicts::

        [
            {
                "id": "approval_req_abc",
                "server_label": "api-specs",
                "tool_name": "read_file",
                "arguments": "{\\"path\\": \\"/README.md\\"}"
            }
        ]

    Resume the graph with ``Command(resume=...)`` where the value is one
    of:

    * ``True`` / ``False`` – approve or deny all pending requests.
    * ``{"approve": True}`` or ``{"approve": False, "reason": "..."}``
    * A plain string ``"true"`` / ``"false"``.

    Returns:
        A dict with ``messages`` containing ``ToolMessage`` instances for
        each approval request, ready for the agent to continue.
    """
    ai_message = state["messages"][-1]
    approval_requests = [
        tc
        for tc in getattr(ai_message, "tool_calls", []) or []
        if tc.get("name") == MCP_APPROVAL_REQUEST_TOOL_NAME
    ]

    if not approval_requests:
        return {"messages": []}

    # Surface approval details via interrupt – graph pauses here.
    interrupt_payload = [
        {
            "id": tc["id"],
            "server_label": tc["args"].get("server_label"),
            "tool_name": tc["args"].get("name"),
            "arguments": tc["args"].get("arguments"),
        }
        for tc in approval_requests
    ]
    decision = interrupt(interrupt_payload)

    # Convert the human decision into a content string.
    if isinstance(decision, bool):
        content = json.dumps({"approve": decision})
    elif isinstance(decision, dict):
        content = json.dumps(decision)
    elif isinstance(decision, str):
        content = decision
    else:
        content = json.dumps({"approve": bool(decision)})

    tool_messages = [
        ToolMessage(content=content, tool_call_id=tc["id"]) for tc in approval_requests
    ]
    return {"messages": tool_messages}


def _make_agent_routing_condition(
    has_tools_node: bool,
    has_mcp_approval_node: bool,
    model_destination: str = "foundryAgent",
    end_destination: str = END,
) -> Callable[[MessagesState], str]:
    """Build a routing function based on which downstream nodes exist.

    The returned callable inspects the last AI message and routes to:

    * ``"mcp_approval"`` – when tool calls include MCP approval requests
      and the graph has an approval node.
    * ``"tools"`` – when regular tool calls are present and the graph has
      a tools node.
    * ``end_destination`` – otherwise (defaults to ``END``).
    """

    def condition(state: MessagesState) -> str:
        ai_message = state["messages"][-1]
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if not tool_calls:
            return end_destination

        if has_mcp_approval_node and any(
            tc.get("name") == MCP_APPROVAL_REQUEST_TOOL_NAME for tc in tool_calls
        ):
            return "mcp_approval"

        if has_tools_node:
            return "tools"

        return end_destination

    return condition


@experimental()
class AgentServiceFactory(BaseModel):
    """Factory to create and manage prompt-based agents in Azure AI Foundry V2.

    Uses the ``azure-ai-projects >= 2.0`` library which relies on the
    OpenAI *Responses* and *Conversations* API instead of the older
    Threads / Runs model.

    To create a simple agent:

    ```python
    from langchain_azure_ai.agents.agent_service_v2 import AgentServiceFactory
    from langchain_core.messages import HumanMessage
    from azure.identity import DefaultAzureCredential

    factory = AgentServiceFactory(
        project_endpoint=(
            "https://resource.services.ai.azure.com/api/projects/demo-project"
        ),
        credential=DefaultAzureCredential(),
    )

    agent = factory.create_prompt_agent(
        name="my-echo-agent",
        model="gpt-4.1",
        instructions="You are a helpful AI assistant that always replies back "
                     "saying the opposite of what the user says.",
    )

    messages = [HumanMessage(content="I'm a genius and I love programming!")]
    state = agent.invoke({"messages": messages})

    for m in state['messages']:
        m.pretty_print()
    ```

    !!! note
        You can also create ``AgentServiceFactory`` without passing any
        parameters if you have set the ``AZURE_AI_PROJECT_ENDPOINT``
        environment variable and are using ``DefaultAzureCredential``
        for authentication.

    Agents can also be created with tools:

    ```python
    tools = [add, multiply, divide]

    agent = factory.create_prompt_agent(
        name="math-agent",
        model="gpt-4.1",
        instructions="You are a helpful assistant tasked with performing "
                     "arithmetic on a set of inputs.",
        tools=tools,
    )
    ```

    You can also use the built-in tools from the V2 Agent Service:

    ```python
    from azure.ai.projects.models import CodeInterpreterTool, CodeInterpreterToolAuto
    from langchain_azure_ai.agents.prebuilt.tools_v2 import (
        AgentServiceBaseTool,
    )

    agent = factory.create_prompt_agent(
        name="code-interpreter-agent",
        model="gpt-4.1",
        instructions="You are a helpful assistant that can run complex "
                     "mathematical functions precisely via tools.",
        tools=[AgentServiceBaseTool(
            tool=CodeInterpreterTool(CodeInterpreterToolAuto())
        )],
    )
    ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    project_endpoint: Optional[str] = None
    """The project endpoint associated with the AI project."""

    credential: Optional[TokenCredential] = None
    """The credential to use. Must be of type ``TokenCredential``."""

    api_version: Optional[str] = None
    """The API version to use. If None, the default is used."""

    client_kwargs: Dict[str, Any] = {}
    """Additional keyword arguments for the client."""

    @pre_init
    def validate_environment(cls, values: Dict) -> Any:
        """Validate required environment values."""
        values["project_endpoint"] = get_from_dict_or_env(
            values,
            "project_endpoint",
            "AZURE_AI_PROJECT_ENDPOINT",
        )

        if values["api_version"]:
            values["client_kwargs"]["api_version"] = values["api_version"]

        values["client_kwargs"]["user_agent"] = "langchain-azure-ai"

        return values

    def _initialize_client(self) -> AIProjectClient:
        """Initialize the AIProjectClient."""
        credential: TokenCredential
        if self.credential is None:
            credential = DefaultAzureCredential()
        else:
            credential = self.credential

        if self.project_endpoint is None:
            raise ValueError(
                "The `project_endpoint` parameter must be specified to create "
                "the AIProjectClient."
            )

        return AIProjectClient(
            endpoint=self.project_endpoint,
            credential=credential,
            **self.client_kwargs,
        )

    def delete_agent(
        self, agent: Union[CompiledStateGraph, PromptBasedAgentNode]
    ) -> None:
        """Delete an agent created with ``create_prompt_agent``.

        Args:
            agent: The compiled graph or node to delete.
        """
        if isinstance(agent, PromptBasedAgentNode):
            agent.delete_agent_from_node()
        else:
            if not isinstance(agent, CompiledStateGraph):
                raise ValueError(
                    "The agent must be a CompiledStateGraph or "
                    "PromptBasedAgentNode instance."
                )
            client = self._initialize_client()
            agent_ids = self.get_agents_id_from_graph(agent)
            if not agent_ids:
                logger.warning("[WARNING] No agent ID found in the graph metadata.")
            else:
                for agent_id in agent_ids:
                    # agent_id is "name:version"
                    parts = agent_id.split(":", 1)
                    if len(parts) == 2:
                        client.agents.delete_version(
                            agent_name=parts[0],
                            agent_version=parts[1],
                        )
                        logger.info("Deleted agent %s version %s", parts[0], parts[1])
                    else:
                        logger.warning("Unexpected agent ID format: %s", agent_id)

    def get_agents_id_from_graph(self, graph: CompiledStateGraph) -> Set[str]:
        """Get agent IDs (``name:version``) from a compiled state graph."""
        agent_ids: Set[str] = set()
        for node in graph.nodes.values():
            if node.metadata and "agent_id" in node.metadata:
                agent_id = node.metadata.get("agent_id")
                if isinstance(agent_id, str) and agent_id:
                    agent_ids.add(agent_id)
        return agent_ids

    def create_prompt_agent_node(
        self,
        name: str,
        model: str,
        description: Optional[str] = None,
        tools: Optional[
            Sequence[Union[AgentServiceBaseTool, BaseTool, Callable]]
        ] = None,
        instructions: Optional[Prompt] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        trace: bool = False,
        wrap_model_call_handler: Optional[Callable] = None,
        awrap_model_call_handler: Optional[Callable] = None,
    ) -> PromptBasedAgentNode:
        """Create a prompt-based agent node using V2.

        Args:
            name: The name of the agent.
            model: The model deployment name.
            description: Optional description.
            tools: Tools for the agent.
            instructions: System prompt instructions.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            trace: Whether to enable tracing.
            wrap_model_call_handler: Optional composed sync handler from
                ``wrap_model_call`` middleware.  When provided, this wraps
                the call to the Azure Responses API inside the node.
            awrap_model_call_handler: Optional composed async handler from
                ``awrap_model_call`` middleware.

        Returns:
            A ``PromptBasedAgentNode`` instance.
        """
        logger.info("Validating parameters...")
        if not isinstance(instructions, str):
            raise ValueError("Only string instructions are supported at this time.")

        logger.info("Initializing AIProjectClient")
        client = self._initialize_client()

        return PromptBasedAgentNode(
            client=client,
            name=name,
            description=description,
            model=model,
            instructions=instructions,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            trace=trace,
            wrap_model_call_handler=wrap_model_call_handler,
            awrap_model_call_handler=awrap_model_call_handler,
        )

    def create_prompt_agent(
        self,
        model: str,
        name: str,
        description: Optional[str] = None,
        tools: Optional[
            Sequence[Union[AgentServiceBaseTool, BaseTool, Callable]]
        ] = None,
        instructions: Optional[Prompt] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        middleware: Sequence[AgentMiddleware] = (),
        state_schema: Optional[StateSchemaType] = None,
        context_schema: Optional[Type[Any]] = None,
        checkpointer: Optional[Checkpointer] = None,
        store: Optional[BaseStore] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        trace: bool = False,
        debug: bool = False,
    ) -> CompiledStateGraph:
        """Create a prompt-based agent using V2.

        Args:
            model: The model deployment name.
            name: The name of the agent.
            description: Optional description.
            tools: Tools for the agent.
            instructions: System prompt instructions.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            middleware: A sequence of :class:`~langchain.agents.middleware.AgentMiddleware`
                instances to apply to the agent.

                Middleware can intercept and modify agent behaviour at various
                stages of the agent loop:

                * ``before_agent`` / ``after_agent`` – run once at the start/
                  end of the whole execution.
                * ``before_model`` / ``after_model`` – run before/after each
                  call to the Azure AI Foundry agent.
                * ``wrap_model_call`` / ``awrap_model_call`` – wrap the
                  Azure Responses API call itself, enabling retry logic,
                  observability, or request/response modification.
                * ``wrap_tool_call`` / ``awrap_tool_call`` – wrap
                  client-side tool execution in the ``ToolNode``.

                Additional tools registered via ``middleware.tools`` are
                added to the client-side ``ToolNode``.

                Middleware state schemas are merged with ``state_schema``
                automatically.

                Duplicate middleware instances (same object identity) are
                not allowed.
            state_schema: State schema. Defaults to ``AgentServiceAgentState``.
            context_schema: Context schema.
            checkpointer: Checkpointer to use.
            store: Store to use.
            interrupt_before: Nodes to interrupt before.
            interrupt_after: Nodes to interrupt after.
            trace: Whether to enable tracing.
            debug: Whether to enable debug mode.

        Returns:
            A compiled ``StateGraph`` representing the agent workflow.

        Raises:
            AssertionError: If duplicate middleware instances are provided.
        """
        logger.info("Creating V2 agent with name: %s", name)

        # ------------------------------------------------------------------
        # Validate middleware
        # ------------------------------------------------------------------
        if len({id(m) for m in middleware}) != len(list(middleware)):
            msg = "Please remove duplicate middleware instances."
            raise AssertionError(msg)

        # ------------------------------------------------------------------
        # Collect middleware hooks
        # ------------------------------------------------------------------

        # Middleware tools (additional tools added by middleware)
        middleware_tools = [t for m in middleware for t in getattr(m, "tools", [])]

        # Collect wrap_tool_call / awrap_tool_call handlers
        middleware_w_wrap_tool_call = [
            m
            for m in middleware
            if m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
        ]
        wrap_tool_call_wrapper = None
        if middleware_w_wrap_tool_call:
            wrappers = [m.wrap_tool_call for m in middleware_w_wrap_tool_call]
            wrap_tool_call_wrapper = _chain_tool_call_wrappers(wrappers)

        middleware_w_awrap_tool_call = [
            m
            for m in middleware
            if m.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
        ]
        awrap_tool_call_wrapper = None
        if middleware_w_awrap_tool_call:
            async_wrappers = [m.awrap_tool_call for m in middleware_w_awrap_tool_call]
            awrap_tool_call_wrapper = _chain_async_tool_call_wrappers(async_wrappers)

        # Collect wrap_model_call / awrap_model_call handlers
        middleware_w_wrap_model_call = [
            m
            for m in middleware
            if m.__class__.wrap_model_call is not AgentMiddleware.wrap_model_call
        ]
        wrap_model_call_handler = None
        if middleware_w_wrap_model_call:
            sync_handlers = [m.wrap_model_call for m in middleware_w_wrap_model_call]
            wrap_model_call_handler = _chain_model_call_handlers(sync_handlers)

        middleware_w_awrap_model_call = [
            m
            for m in middleware
            if m.__class__.awrap_model_call is not AgentMiddleware.awrap_model_call
        ]
        awrap_model_call_handler = None
        if middleware_w_awrap_model_call:
            async_handlers = [m.awrap_model_call for m in middleware_w_awrap_model_call]
            awrap_model_call_handler = _chain_async_model_call_handlers(async_handlers)

        # Collect lifecycle hook middleware
        middleware_w_before_agent = [
            m
            for m in middleware
            if m.__class__.before_agent is not AgentMiddleware.before_agent
            or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
        ]
        middleware_w_before_model = [
            m
            for m in middleware
            if m.__class__.before_model is not AgentMiddleware.before_model
            or m.__class__.abefore_model is not AgentMiddleware.abefore_model
        ]
        middleware_w_after_model = [
            m
            for m in middleware
            if m.__class__.after_model is not AgentMiddleware.after_model
            or m.__class__.aafter_model is not AgentMiddleware.aafter_model
        ]
        middleware_w_after_agent = [
            m
            for m in middleware
            if m.__class__.after_agent is not AgentMiddleware.after_agent
            or m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
        ]

        # ------------------------------------------------------------------
        # State schema: merge base + middleware state schemas
        # ------------------------------------------------------------------
        base_state = state_schema if state_schema is not None else AgentServiceAgentState
        state_schemas: Set[type] = {m.state_schema for m in middleware}
        state_schemas.add(base_state)
        resolved_state_schema = _resolve_schema(state_schemas, "StateSchema", None)
        input_schema = resolved_state_schema

        # ------------------------------------------------------------------
        # Build the graph
        # ------------------------------------------------------------------
        builder = StateGraph(resolved_state_schema, context_schema=context_schema)

        # ------------------------------------------------------------------
        # Create the PromptBasedAgentNode (foundryAgent)
        # ------------------------------------------------------------------
        logger.info("Adding PromptBasedAgentNode")
        prompt_node = self.create_prompt_agent_node(
            name=name,
            description=description,
            model=model,
            tools=tools,
            instructions=instructions,
            temperature=temperature,
            top_p=top_p,
            trace=trace,
            wrap_model_call_handler=wrap_model_call_handler,
            awrap_model_call_handler=awrap_model_call_handler,
        )
        builder.add_node(
            "foundryAgent",
            prompt_node,
            input_schema=input_schema,
            metadata={"agent_id": prompt_node._agent_id},
        )
        logger.info("PromptBasedAgentNode added")

        # ------------------------------------------------------------------
        # Tools node (client-side tools + middleware tools)
        # ------------------------------------------------------------------
        has_tools_node = False
        has_mcp_approval_node = False

        # Separate client-side tools from Azure-native tools
        filtered_tools: list = list(middleware_tools)
        service_tools: list = []
        if tools is not None:
            filtered_tools += [t for t in tools if not isinstance(t, AgentServiceBaseTool)]
            service_tools = [t for t in tools if isinstance(t, AgentServiceBaseTool)]

        if filtered_tools or wrap_tool_call_wrapper or awrap_tool_call_wrapper:
            has_tools_node = True
            logger.info("Creating ToolNode with tools")
            builder.add_node(
                "tools",
                ToolNode(
                    filtered_tools,
                    wrap_tool_call=wrap_tool_call_wrapper,
                    awrap_tool_call=awrap_tool_call_wrapper,
                ),
            )
        else:
            logger.info(
                "All tools are AgentServiceBaseTool or no tools, skipping ToolNode"
            )

        if service_tools and any(t.requires_approval for t in service_tools):
            has_mcp_approval_node = True
            logger.info("Creating MCP approval node")
            builder.add_node("mcp_approval", _mcp_approval_node)

        # ------------------------------------------------------------------
        # Middleware lifecycle nodes
        # ------------------------------------------------------------------
        for m in middleware:
            if (
                m.__class__.before_agent is not AgentMiddleware.before_agent
                or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
            ):
                sync_before_agent = (
                    m.before_agent
                    if m.__class__.before_agent is not AgentMiddleware.before_agent
                    else None
                )
                async_before_agent = (
                    m.abefore_agent
                    if m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
                    else None
                )
                before_agent_node = RunnableCallable(
                    sync_before_agent, async_before_agent, trace=False
                )
                builder.add_node(
                    f"{m.name}.before_agent",
                    before_agent_node,
                    input_schema=resolved_state_schema,
                )

            if (
                m.__class__.before_model is not AgentMiddleware.before_model
                or m.__class__.abefore_model is not AgentMiddleware.abefore_model
            ):
                sync_before = (
                    m.before_model
                    if m.__class__.before_model is not AgentMiddleware.before_model
                    else None
                )
                async_before = (
                    m.abefore_model
                    if m.__class__.abefore_model is not AgentMiddleware.abefore_model
                    else None
                )
                before_node = RunnableCallable(sync_before, async_before, trace=False)
                builder.add_node(
                    f"{m.name}.before_model",
                    before_node,
                    input_schema=resolved_state_schema,
                )

            if (
                m.__class__.after_model is not AgentMiddleware.after_model
                or m.__class__.aafter_model is not AgentMiddleware.aafter_model
            ):
                sync_after = (
                    m.after_model
                    if m.__class__.after_model is not AgentMiddleware.after_model
                    else None
                )
                async_after = (
                    m.aafter_model
                    if m.__class__.aafter_model is not AgentMiddleware.aafter_model
                    else None
                )
                after_node = RunnableCallable(sync_after, async_after, trace=False)
                builder.add_node(
                    f"{m.name}.after_model",
                    after_node,
                    input_schema=resolved_state_schema,
                )

            if (
                m.__class__.after_agent is not AgentMiddleware.after_agent
                or m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
            ):
                sync_after_agent = (
                    m.after_agent
                    if m.__class__.after_agent is not AgentMiddleware.after_agent
                    else None
                )
                async_after_agent = (
                    m.aafter_agent
                    if m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
                    else None
                )
                after_agent_node = RunnableCallable(
                    sync_after_agent, async_after_agent, trace=False
                )
                builder.add_node(
                    f"{m.name}.after_agent",
                    after_agent_node,
                    input_schema=resolved_state_schema,
                )

        # ------------------------------------------------------------------
        # Determine entry / loop-entry / loop-exit / exit nodes
        # ------------------------------------------------------------------
        # entry_node: first node to execute (runs once at start)
        if middleware_w_before_agent:
            entry_node = f"{middleware_w_before_agent[0].name}.before_agent"
        elif middleware_w_before_model:
            entry_node = f"{middleware_w_before_model[0].name}.before_model"
        else:
            entry_node = "foundryAgent"

        # loop_entry_node: start of each agent iteration (skips before_agent)
        if middleware_w_before_model:
            loop_entry_node = f"{middleware_w_before_model[0].name}.before_model"
        else:
            loop_entry_node = "foundryAgent"

        # loop_exit_node: end of each agent iteration (before routing)
        if middleware_w_after_model:
            loop_exit_node = f"{middleware_w_after_model[0].name}.after_model"
        else:
            loop_exit_node = "foundryAgent"

        # exit_node: last node before END (runs once at end)
        if middleware_w_after_agent:
            exit_node = f"{middleware_w_after_agent[-1].name}.after_agent"
        else:
            exit_node = END

        # ------------------------------------------------------------------
        # Edges
        # ------------------------------------------------------------------
        builder.add_edge(START, entry_node)

        if has_tools_node or has_mcp_approval_node:
            routing_fn = _make_agent_routing_condition(
                has_tools_node=has_tools_node,
                has_mcp_approval_node=has_mcp_approval_node,
                model_destination=loop_entry_node,
                end_destination=exit_node,
            )
            # The path_map must include all possible destinations from
            # the routing function.  When after_model middleware can use
            # ``jump_to`` to re-enter the loop, ``loop_entry_node`` is
            # also included.
            path_map: Dict[Hashable, str] = {exit_node: exit_node}
            if has_tools_node:
                path_map["tools"] = "tools"
            if has_mcp_approval_node:
                path_map["mcp_approval"] = "mcp_approval"
            # Include loop_entry_node when middleware nodes sit between
            # foundryAgent and the exit so that after_model middleware can
            # signal a re-entry (e.g. via jump_to in the state).
            if loop_exit_node != "foundryAgent" and loop_entry_node not in path_map:
                path_map[loop_entry_node] = loop_entry_node

            logger.info("Adding conditional edges from loop_exit_node")
            builder.add_conditional_edges(
                loop_exit_node,
                routing_fn,
                path_map,
            )
            logger.info("Conditional edges added")

            if has_tools_node:
                builder.add_edge("tools", loop_entry_node)
            if has_mcp_approval_node:
                builder.add_edge("mcp_approval", loop_entry_node)

        elif loop_exit_node == "foundryAgent":
            # No tools, no after_model nodes – simple direct edge.
            logger.info("No tools found, adding direct edge to exit_node")
            if exit_node == END:
                builder.add_edge("foundryAgent", END)
            else:
                builder.add_edge("foundryAgent", exit_node)
        else:
            # No tools but after_model nodes exist.  Use _add_middleware_edge
            # so that middleware can optionally jump back to the model.
            logger.info(
                "No tools, after_model middleware: connecting %s → %s",
                loop_exit_node,
                exit_node,
            )
            _add_middleware_edge(
                builder,
                name=loop_exit_node,
                default_destination=exit_node,
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(
                    middleware_w_after_model[0], "after_model"
                ),
            )

        # Connect before_agent middleware chain
        if middleware_w_before_agent:
            for m1, m2 in itertools.pairwise(middleware_w_before_agent):
                _add_middleware_edge(
                    builder,
                    name=f"{m1.name}.before_agent",
                    default_destination=f"{m2.name}.before_agent",
                    model_destination=loop_entry_node,
                    end_destination=exit_node,
                    can_jump_to=_get_can_jump_to(m1, "before_agent"),
                )
            _add_middleware_edge(
                builder,
                name=f"{middleware_w_before_agent[-1].name}.before_agent",
                default_destination=loop_entry_node,
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(middleware_w_before_agent[-1], "before_agent"),
            )

        # Connect before_model middleware chain
        if middleware_w_before_model:
            for m1, m2 in itertools.pairwise(middleware_w_before_model):
                _add_middleware_edge(
                    builder,
                    name=f"{m1.name}.before_model",
                    default_destination=f"{m2.name}.before_model",
                    model_destination=loop_entry_node,
                    end_destination=exit_node,
                    can_jump_to=_get_can_jump_to(m1, "before_model"),
                )
            _add_middleware_edge(
                builder,
                name=f"{middleware_w_before_model[-1].name}.before_model",
                default_destination="foundryAgent",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(middleware_w_before_model[-1], "before_model"),
            )

        # Connect after_model middleware chain
        # after_model runs in reverse order from foundryAgent outward.
        # The first after_model (index 0) is ``loop_exit_node`` whose edge
        # has already been added above (via conditional edges or
        # _add_middleware_edge).  The remaining nodes (indices > 0) are
        # chained back-to-front here.
        if middleware_w_after_model:
            builder.add_edge(
                "foundryAgent",
                f"{middleware_w_after_model[-1].name}.after_model",
            )
            for idx in range(len(middleware_w_after_model) - 1, 0, -1):
                m1 = middleware_w_after_model[idx]
                m2 = middleware_w_after_model[idx - 1]
                _add_middleware_edge(
                    builder,
                    name=f"{m1.name}.after_model",
                    default_destination=f"{m2.name}.after_model",
                    model_destination=loop_entry_node,
                    end_destination=exit_node,
                    can_jump_to=_get_can_jump_to(m1, "after_model"),
                )

        # Connect after_agent middleware chain
        if middleware_w_after_agent:
            for idx in range(len(middleware_w_after_agent) - 1, 0, -1):
                m1 = middleware_w_after_agent[idx]
                m2 = middleware_w_after_agent[idx - 1]
                _add_middleware_edge(
                    builder,
                    name=f"{m1.name}.after_agent",
                    default_destination=f"{m2.name}.after_agent",
                    model_destination=loop_entry_node,
                    end_destination=exit_node,
                    can_jump_to=_get_can_jump_to(m1, "after_agent"),
                )
            _add_middleware_edge(
                builder,
                name=f"{middleware_w_after_agent[0].name}.after_agent",
                default_destination=END,
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(middleware_w_after_agent[0], "after_agent"),
            )

        # ------------------------------------------------------------------
        # Compile
        # ------------------------------------------------------------------
        logger.info("Compiling state graph")
        graph = builder.compile(
            name=name,
            checkpointer=checkpointer,
            store=store,
            interrupt_after=interrupt_after,
            interrupt_before=interrupt_before,
            debug=debug,
        )

        if trace:
            logger.info("Configuring `AzureAIOpenTelemetry` tracer")
            try:
                tracer = AzureAIOpenTelemetryTracer(
                    enable_content_recording=True,
                    project_endpoint=self.project_endpoint,
                    credential=self.credential,
                    name=name,
                )
            except AttributeError as ex:
                raise ImportError(
                    "Failed to create OpenTelemetry tracer from the project "
                    "endpoint. Check the inner exception for details."
                ) from ex
            graph = graph.with_config({"callbacks": [tracer]})

        logger.info("State graph compiled")
        return graph
