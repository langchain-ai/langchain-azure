"""Internal middleware utilities for LangGraph StateGraphs."""

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


def _apply_middleware(
    builder: StateGraph,
    middleware: Sequence[Any],
    *,
    agent_node: str,
    input_schema: Optional[type] = None,
) -> tuple[str, Any]:
    """Wire before_agent/after_agent middleware nodes into a StateGraph.

    Internal helper used by AgentServiceFactory.create_prompt_agent.  Not
    part of the public API.
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


# Internal alias preserved for the existing import in agent_service.py.
# Do not use this name in new code.
apply_middleware = _apply_middleware
