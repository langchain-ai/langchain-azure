"""Graph-agnostic middleware utilities for LangGraph StateGraphs."""

import itertools
from typing import Any, Dict, Optional, Sequence, Set, get_type_hints

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict


def _resolve_state_schema(
    state_schemas: Set[type],
    schema_name: str,
) -> type:
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


def apply_middleware(
    builder: StateGraph,
    middleware: Sequence[Any],
    *,
    agent_node: str,
    input_schema: Optional[type] = None,
) -> tuple[str, Any]:
    """Add AgentMiddleware before/after hooks to any LangGraph StateGraph.

    This function is graph-agnostic: it works with any ``StateGraph``, not only
    ``AgentServiceFactory`` graphs.  It adds ``before_agent`` and ``after_agent``
    middleware nodes to the builder and wires the inter-node edges correctly.

    Args:
        builder: The ``StateGraph`` builder to modify in-place.
        middleware: Sequence of
            :class:`~langchain.agents.middleware.types.AgentMiddleware` instances
            to apply.
        agent_node: Name of the main agent node already added to *builder*.
        input_schema: Optional input schema for the middleware nodes.  When
            provided, each node is added with ``input_schema=input_schema``.

    Returns:
        A tuple ``(entry_node, after_agent_entry)`` where:

        * ``entry_node`` – the node that the graph's ``START`` edge should
          connect to.  This is the first ``before_agent`` node, or *agent_node*
          itself when no middleware implements ``before_agent``.
        * ``after_agent_entry`` – the node that the agent's exit path should
          route to when the agent is done.  This is the last ``after_agent``
          node, or ``END`` when no middleware implements ``after_agent``.

    Note:
        This function does **not** add the ``START → entry_node`` edge or the
        ``agent_node → after_agent_entry`` edge.  The caller is responsible for
        both connections so that it can integrate the middleware entry/exit
        points into its own conditional routing logic.

    Example:
        .. code-block:: python

            from langchain_azure_ai.guardrails import (
                AzureContentSafetyMiddleware,
                apply_middleware,
            )
            from langgraph.graph import END, START, MessagesState, StateGraph

            safety = AzureContentSafetyMiddleware(
                endpoint="https://my-resource.cognitiveservices.azure.com/",
                action="block",
            )

            builder = StateGraph(MessagesState)
            builder.add_node("agent", my_agent_fn)

            entry, after = apply_middleware(builder, [safety], agent_node="agent")

            # Wire START and routing using the returned node names.
            builder.add_edge(START, entry)
            builder.add_conditional_edges(
                "agent",
                route_fn,
                {"tools": "tools", after: after},
            )
            builder.add_node("tools", tool_node)
            builder.add_edge("tools", "agent")

            graph = builder.compile()
    """
    from langchain.agents.middleware.types import AgentMiddleware
    from langgraph._internal._runnable import RunnableCallable

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

    node_kwargs: Dict[str, Any] = {}
    if input_schema is not None:
        node_kwargs["input_schema"] = input_schema

    # ------------------------------------------------------------------ #
    # Add before_agent nodes
    # ------------------------------------------------------------------ #
    for m in middleware_w_before_agent:
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
        builder.add_node(f"{m.name}.before_agent", before_node, **node_kwargs)

    # ------------------------------------------------------------------ #
    # Add after_agent nodes
    # ------------------------------------------------------------------ #
    for m in middleware_w_after_agent:
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
        builder.add_node(f"{m.name}.after_agent", after_node, **node_kwargs)

    # ------------------------------------------------------------------ #
    # Chain before_agent: before[0] → before[1] → … → agent_node
    # ------------------------------------------------------------------ #
    if middleware_w_before_agent:
        for m1, m2 in itertools.pairwise(middleware_w_before_agent):
            builder.add_edge(
                f"{m1.name}.before_agent",
                f"{m2.name}.before_agent",
            )
        builder.add_edge(
            f"{middleware_w_before_agent[-1].name}.before_agent",
            agent_node,
        )

    # ------------------------------------------------------------------ #
    # Chain after_agent (reverse): after[-1] → after[-2] → … → after[0] → END
    # ------------------------------------------------------------------ #
    if middleware_w_after_agent:
        for idx in range(len(middleware_w_after_agent) - 1, 0, -1):
            m1 = middleware_w_after_agent[idx]
            m2 = middleware_w_after_agent[idx - 1]
            builder.add_edge(
                f"{m1.name}.after_agent",
                f"{m2.name}.after_agent",
            )
        builder.add_edge(
            f"{middleware_w_after_agent[0].name}.after_agent",
            END,
        )

    # ------------------------------------------------------------------ #
    # Compute and return entry / after-agent-entry names
    # ------------------------------------------------------------------ #
    entry_node: str = (
        f"{middleware_w_before_agent[0].name}.before_agent"
        if middleware_w_before_agent
        else agent_node
    )
    after_agent_entry: Any = (
        f"{middleware_w_after_agent[-1].name}.after_agent"
        if middleware_w_after_agent
        else END
    )

    return entry_node, after_agent_entry
