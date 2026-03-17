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
    get_type_hints,
)

from azure.ai.projects import AIProjectClient
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from langchain.agents.middleware.types import AgentMiddleware
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
from typing_extensions import TypedDict

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


def _resolve_state_schema(
    state_schemas: "set[type]",
    schema_name: str,
) -> "type":
    """Merge multiple TypedDict schemas into a single TypedDict.

    Collects all field annotations from all provided schemas and produces a
    new ``TypedDict`` under ``schema_name``.  Later schemas override earlier
    ones when there are duplicate field names.

    Args:
        state_schemas: A set of TypedDict (or dataclass-like) types whose
            fields should be merged.
        schema_name: The ``__name__`` to give to the resulting TypedDict.

    Returns:
        A new TypedDict type that contains all fields from all schemas.
    """
    all_annotations: Dict[str, Any] = {}
    for schema in state_schemas:
        hints = get_type_hints(schema, include_extras=True)
        all_annotations.update(hints)
    return TypedDict(schema_name, all_annotations)  # type: ignore[operator]


def _add_middleware_edge(
    graph: StateGraph,
    *,
    name: str,
    default_destination: str,
) -> None:
    """Add a simple (unconditional) edge from a middleware node to its successor.

    Unlike the full LangChain implementation we do **not** support ``jump_to``
    from middleware nodes – those would require ``before_model`` / ``after_model``
    semantics which are not meaningful for the foundry-agent service call.

    Args:
        graph: The graph to add the edge to.
        name: The source middleware node name.
        default_destination: Target node for normal flow.
    """
    graph.add_edge(name, default_destination)


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
    end_destination: str = "__end__",
) -> Callable[[MessagesState], str]:
    """Build a routing function based on which downstream nodes exist.

    The returned callable inspects the last AI message and routes to:

    * ``"mcp_approval"`` – when tool calls include MCP approval requests
      and the graph has an approval node.
    * ``"tools"`` – when regular tool calls are present and the graph has
      a tools node.
    * ``end_destination`` – otherwise.

    Args:
        has_tools_node: Whether a ``"tools"`` node exists in the graph.
        has_mcp_approval_node: Whether an ``"mcp_approval"`` node exists.
        end_destination: The node name (or ``"__end__"``) to route to when
            there are no tool calls pending.  Defaults to ``"__end__"``
            for backward compatibility; pass the name of the first
            ``after_agent`` middleware node when one exists.
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


class AgentServiceFactory(BaseModel):
    """Factory to create and manage prompt-based agents in Microsoft Foundry V2.

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
        state_schema: Optional[StateSchemaType] = None,
        context_schema: Optional[Type[Any]] = None,
        checkpointer: Optional[Checkpointer] = None,
        store: Optional[BaseStore] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        trace: bool = False,
        debug: bool = False,
        middleware: Sequence[AgentMiddleware] = (),
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
            state_schema: State schema. Defaults to ``AgentServiceAgentState``.
            context_schema: Context schema.
            checkpointer: Checkpointer to use.
            store: Store to use.
            interrupt_before: Nodes to interrupt before.
            interrupt_after: Nodes to interrupt after.
            trace: Whether to enable tracing.
            debug: Whether to enable debug mode.
            middleware: A sequence of
                :class:`~langchain.agents.middleware.types.AgentMiddleware` instances
                to apply to the agent.

                Middleware can intercept and modify agent behavior at various stages.

                .. note::
                    The following middleware hooks are supported for the foundry
                    agent service:

                    * ``before_agent`` / ``abefore_agent`` – runs **once** before
                      the agent graph execution starts.
                    * ``after_agent`` / ``aafter_agent`` – runs **once** after
                      the agent graph execution completes.
                    * ``wrap_tool_call`` / ``awrap_tool_call`` – intercepts
                      **client-side** tool execution (i.e. tools that are executed
                      locally by LangGraph, not inside the Azure AI agent service).

                    The ``before_model``, ``after_model``, and ``wrap_model_call``
                    hooks are **not** supported because the Azure AI agent service
                    encapsulates the model interaction; the node acts as a proxy
                    rather than directly invoking an LLM.

        Returns:
            A compiled ``StateGraph`` representing the agent workflow.
        """
        logger.info("Creating V2 agent with name: %s", name)

        # ------------------------------------------------------------------ #
        # 1. Classify middleware by which hooks they implement
        # ------------------------------------------------------------------ #
        middleware_w_before_agent = [
            m
            for m in middleware
            if m.__class__.before_agent is not AgentMiddleware.before_agent
            or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
        ]
        middleware_w_after_agent = [
            m
            for m in middleware
            if m.__class__.after_agent is not AgentMiddleware.after_agent
            or m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
        ]

        # Collect middleware with wrap_tool_call / awrap_tool_call hooks.
        # Both lists intentionally include middleware that overrides EITHER the
        # sync or async variant.  If a middleware only implements one, calling
        # the other will raise ``NotImplementedError`` with a helpful message
        # directing users to the correct execution path.  This mirrors the
        # behaviour of LangChain's ``create_agent`` factory.
        middleware_w_wrap_tool_call = [
            m
            for m in middleware
            if m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
            or m.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
        ]
        middleware_w_awrap_tool_call = [
            m
            for m in middleware
            if m.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
            or m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
        ]

        # ------------------------------------------------------------------ #
        # 2. Build tool-call wrappers from middleware
        # ------------------------------------------------------------------ #
        wrap_tool_call_wrapper = None
        if middleware_w_wrap_tool_call:
            from langchain.agents.factory import _chain_tool_call_wrappers

            sync_wrappers = [m.wrap_tool_call for m in middleware_w_wrap_tool_call]
            wrap_tool_call_wrapper = _chain_tool_call_wrappers(sync_wrappers)

        awrap_tool_call_wrapper = None
        if middleware_w_awrap_tool_call:
            from langchain.agents.factory import _chain_async_tool_call_wrappers

            async_wrappers = [m.awrap_tool_call for m in middleware_w_awrap_tool_call]
            awrap_tool_call_wrapper = _chain_async_tool_call_wrappers(async_wrappers)

        # ------------------------------------------------------------------ #
        # 3. Resolve the state schema – merge middleware schemas into one
        # ------------------------------------------------------------------ #
        base_state = (
            state_schema if state_schema is not None else AgentServiceAgentState
        )
        state_schemas: set[type] = {m.state_schema for m in middleware}
        state_schemas.add(base_state)
        resolved_state_schema = _resolve_state_schema(state_schemas, "StateSchema")
        input_schema = resolved_state_schema

        # ------------------------------------------------------------------ #
        # 4. Build the graph
        # ------------------------------------------------------------------ #
        builder = StateGraph(resolved_state_schema, context_schema=context_schema)  # type: ignore[var-annotated]

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
        )
        builder.add_node(
            "foundryAgent",
            prompt_node,
            input_schema=input_schema,
            metadata={"agent_id": prompt_node._agent_id},
        )
        logger.info("PromptBasedAgentNode added")

        # ------------------------------------------------------------------ #
        # 5. Tool / MCP approval nodes
        # ------------------------------------------------------------------ #
        has_tools_node = False
        has_mcp_approval_node = False

        # Extra tools contributed by middleware
        middleware_tools = [t for m in middleware for t in getattr(m, "tools", [])]

        if tools is not None or middleware_tools:
            all_tools = list(tools) if tools is not None else []
            filtered_tools = [
                t for t in all_tools if not isinstance(t, AgentServiceBaseTool)
            ] + middleware_tools
            service_tools = [
                t for t in all_tools if isinstance(t, AgentServiceBaseTool)
            ]

            if (
                len(filtered_tools) > 0
                or wrap_tool_call_wrapper is not None
                or awrap_tool_call_wrapper is not None
            ):
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
                    "All tools are AgentServiceBaseTool, skipping ToolNode creation"
                )

            if any(t.requires_approval for t in service_tools):
                has_mcp_approval_node = True
                logger.info("Creating MCP approval node")
                builder.add_node("mcp_approval", _mcp_approval_node)

        # ------------------------------------------------------------------ #
        # 6. Add middleware nodes (before_agent / after_agent)
        # ------------------------------------------------------------------ #
        for m in middleware:
            if (
                m.__class__.before_agent is not AgentMiddleware.before_agent
                or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
            ):
                sync_fn = (
                    m.before_agent
                    if m.__class__.before_agent is not AgentMiddleware.before_agent
                    else None
                )
                async_fn = (
                    m.abefore_agent
                    if m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
                    else None
                )
                before_node = RunnableCallable(sync_fn, async_fn, trace=False)
                builder.add_node(
                    f"{m.name}.before_agent",
                    before_node,
                    input_schema=input_schema,
                )

            if (
                m.__class__.after_agent is not AgentMiddleware.after_agent
                or m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
            ):
                sync_fn = (
                    m.after_agent
                    if m.__class__.after_agent is not AgentMiddleware.after_agent
                    else None
                )
                async_fn = (
                    m.aafter_agent
                    if m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
                    else None
                )
                after_node = RunnableCallable(sync_fn, async_fn, trace=False)
                builder.add_node(
                    f"{m.name}.after_agent",
                    after_node,
                    input_schema=input_schema,
                )

        # ------------------------------------------------------------------ #
        # 7. Wire edges
        # ------------------------------------------------------------------ #
        # Determine the entry node: before_agent (first) -> foundryAgent
        if middleware_w_before_agent:
            entry_node = f"{middleware_w_before_agent[0].name}.before_agent"
        else:
            entry_node = "foundryAgent"

        # Determine the exit node: after_agent (last) -> END
        if middleware_w_after_agent:
            exit_node = f"{middleware_w_after_agent[-1].name}.after_agent"
        else:
            exit_node = END

        builder.add_edge(START, entry_node)

        # Chain before_agent nodes together
        if middleware_w_before_agent:
            for m1, m2 in itertools.pairwise(middleware_w_before_agent):
                _add_middleware_edge(
                    builder,
                    name=f"{m1.name}.before_agent",
                    default_destination=f"{m2.name}.before_agent",
                )
            # Last before_agent -> foundryAgent
            _add_middleware_edge(
                builder,
                name=f"{middleware_w_before_agent[-1].name}.before_agent",
                default_destination="foundryAgent",
            )

        # Routing from foundryAgent -> tools / mcp_approval / exit_node
        if has_tools_node or has_mcp_approval_node:
            routing_fn = _make_agent_routing_condition(
                has_tools_node=has_tools_node,
                has_mcp_approval_node=has_mcp_approval_node,
                end_destination=exit_node,
            )
            # Build path_map so LangGraph knows the valid destinations.
            # The condition function returns either exit_node, "tools", or
            # "mcp_approval".  When exit_node is END ("__end__") the key and
            # value are both "__end__", which is the LangGraph sentinel string.
            path_map: Dict[Hashable, str] = {exit_node: exit_node}
            if has_tools_node:
                path_map["tools"] = "tools"
            if has_mcp_approval_node:
                path_map["mcp_approval"] = "mcp_approval"

            logger.info("Adding conditional edges")
            builder.add_conditional_edges(
                "foundryAgent",
                routing_fn,
                path_map,
            )
            logger.info("Conditional edges added")

            if has_tools_node:
                builder.add_edge("tools", "foundryAgent")
            if has_mcp_approval_node:
                builder.add_edge("mcp_approval", "foundryAgent")
        else:
            logger.info("No tools found, adding edge to exit node")
            builder.add_edge("foundryAgent", exit_node)

        # Chain after_agent nodes together (last middleware runs first in chain)
        if middleware_w_after_agent:
            for idx in range(len(middleware_w_after_agent) - 1, 0, -1):
                m1 = middleware_w_after_agent[idx]
                m2 = middleware_w_after_agent[idx - 1]
                _add_middleware_edge(
                    builder,
                    name=f"{m1.name}.after_agent",
                    default_destination=f"{m2.name}.after_agent",
                )
            # First after_agent -> END
            _add_middleware_edge(
                builder,
                name=f"{middleware_w_after_agent[0].name}.after_agent",
                default_destination=END,
            )

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
                    agent_id=name,
                )
            except AttributeError as ex:
                raise ImportError(
                    "Failed to create OpenTelemetry tracer from the project "
                    "endpoint. Check the inner exception for details."
                ) from ex
            graph = graph.with_config({"callbacks": [tracer]})

        logger.info("State graph compiled")
        return graph
