"""Reusable evaluator-optimizer graph builders for LangGraph.

Provides helper functions to construct evaluator-optimizer subgraphs
that can be embedded as nodes in larger LangGraph workflows. The
pattern follows the LangGraph evaluator-optimizer documentation:
a generator creates output, an evaluator grades it, and a conditional
edge routes back to the generator or exits.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, cast

from typing_extensions import TypedDict

from langchain_azure_ai._api.base import experimental

LOGGER = logging.getLogger(__name__)
_ITERATION_TRACKER_KEY = "_eval_optimize_iteration_count"


def _coerce_iteration_limit(value: Any, *, default: int) -> int:
    """Return a positive integer iteration limit."""
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return default
    return limit if limit > 0 else default


def _get_state_value(container: Any, key: str, default: Any = None) -> Any:
    """Read a key or attribute from a mapping-like or object-like state value."""
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _create_internal_state_schema(state_schema: type) -> type[dict[str, Any]]:
    """Extend the public state schema with an internal iteration counter."""
    annotations = dict(getattr(state_schema, "__annotations__", {}))
    annotations[_ITERATION_TRACKER_KEY] = int
    typed_dict_factory = cast(Any, TypedDict)
    return typed_dict_factory(
        f"{state_schema.__name__}Internal",
        annotations,
        total=getattr(state_schema, "__total__", True),
    )


@experimental(message="Foundry evaluation integration is in preview and may change.")
def create_eval_optimize_subgraph(
    *,
    evaluate_fn: Callable[[Any], dict[str, Any]],
    refine_fn: Callable[[Any], dict[str, Any]],
    should_refine_fn: Callable[[Any], str],
    state_schema: type,
    max_iterations: int = 3,
    accepted_route: str = "accepted",
    refine_route: str = "refine",
) -> Any:
    """Build a compiled evaluator-optimizer subgraph.

    Creates a LangGraph ``StateGraph`` that implements the
    evaluate→refine loop pattern. The returned compiled graph can
    be used as a node in a parent graph.

    Args:
        evaluate_fn: Node function that evaluates the current draft
            and returns state updates including evaluation results.
        refine_fn: Node function that refines the draft based on
            evaluation feedback.
        should_refine_fn: Routing function that returns
            *accepted_route* or *refine_route* based on state.
        state_schema: The TypedDict class for the subgraph state.
        max_iterations: Safety limit on refinement loops.
        accepted_route: Route name when evaluation passes.
        refine_route: Route name when refinement is needed.

    Returns:
        A compiled LangGraph ``StateGraph`` ready to be invoked
        or added as a node in a parent graph.
    """
    from langgraph.graph import END, START, StateGraph

    internal_state_schema = _create_internal_state_schema(state_schema)
    builder: Any = StateGraph(
        internal_state_schema,
        input_schema=state_schema,
        output_schema=state_schema,
    )

    def evaluate_with_guard(state: Any) -> dict[str, Any]:
        """Invoke the evaluator and track loop count in graph state."""
        tracked_iteration = _coerce_iteration_limit(
            _get_state_value(state, _ITERATION_TRACKER_KEY, 0),
            default=0,
        )
        updates = dict(evaluate_fn(state))
        updates[_ITERATION_TRACKER_KEY] = tracked_iteration + 1
        return updates

    def should_refine_with_guard(state: Any) -> str:
        route = should_refine_fn(state)
        if route == accepted_route:
            return accepted_route

        iteration_limit = _coerce_iteration_limit(
            _get_state_value(state, "max_iterations", max_iterations),
            default=max_iterations,
        )
        tracked_iteration = _coerce_iteration_limit(
            _get_state_value(state, _ITERATION_TRACKER_KEY, 0),
            default=0,
        )
        if tracked_iteration >= iteration_limit:
            return accepted_route

        return route

    builder.add_node("evaluate", cast(Any, evaluate_with_guard))
    builder.add_node("refine", cast(Any, refine_fn))

    builder.add_edge(START, "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        should_refine_with_guard,
        {
            accepted_route: END,
            refine_route: "refine",
        },
    )
    builder.add_edge("refine", "evaluate")

    return builder.compile()


@experimental(message="Foundry evaluation integration is in preview and may change.")
def create_analyst_subgraph(
    *,
    research_fn: Callable[[Any], dict[str, Any]],
    write_fn: Callable[[Any], dict[str, Any]],
    eval_optimize_graph: Any,
    build_completed_fn: Callable[[Any], dict[str, Any]],
    state_schema: type,
    max_iterations: int = 3,
    state_to_eval_input: Callable[[dict[str, Any], int], dict[str, Any]] | None = None,
    eval_output_to_state: (
        Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None
    ) = None,
) -> Any:
    """Build a compiled analyst subgraph with embedded eval-optimize loop.

    Creates a LangGraph ``StateGraph`` for a specialist analyst:
    research → write → eval-optimize (subgraph) → build_completed.

    The eval-optimize loop is invoked as a subgraph with a different
    state schema, demonstrating the parent→child→grandchild pattern
    with state transformation.

    Args:
        research_fn: Node that gathers research data.
        write_fn: Node that writes the section draft.
        eval_optimize_graph: Compiled eval-optimize subgraph.
        build_completed_fn: Node that packages final output.
        state_schema: The analyst TypedDict state.
        max_iterations: Safety limit passed to the eval-optimize subgraph.
        state_to_eval_input: Optional callback that maps the analyst state to the
            eval-optimize subgraph input. When omitted, the helper uses the
            conventional analyst state keys ``section``, ``draft_content``,
            ``evaluation_feedback``, ``evaluation_result``, ``accepted``,
            ``iteration``/``iteration_count``, and ``max_iterations``.
            ``section`` may be either a mapping or an object with ``area`` and
            ``title`` attributes.
        eval_output_to_state: Optional callback that maps the eval-optimize
            output back into analyst state updates. When omitted, the helper
            returns ``draft_content``, ``evaluation_result``, and
            ``iteration_count`` updates.

    Returns:
        A compiled LangGraph ``StateGraph``.
    """
    from langgraph.graph import END, START, StateGraph

    def default_state_to_eval_input(
        state: dict[str, Any], iteration_limit: int
    ) -> dict[str, Any]:
        """Build the default eval-optimize input payload."""
        section = state.get("section")
        return {
            "section_area": _get_state_value(section, "area", "unknown"),
            "section_title": _get_state_value(section, "title", "untitled"),
            "draft_content": state.get("draft_content", ""),
            "evaluation_feedback": state.get("evaluation_feedback", ""),
            "evaluation_result": state.get("evaluation_result"),
            "accepted": bool(state.get("accepted", False)),
            "iteration": state.get("iteration_count", state.get("iteration", 0)),
            "max_iterations": _coerce_iteration_limit(
                state.get("max_iterations", iteration_limit),
                default=iteration_limit,
            ),
        }

    def default_eval_output_to_state(
        state: dict[str, Any], eval_output: dict[str, Any]
    ) -> dict[str, Any]:
        """Build the default analyst state update payload."""
        return {
            "draft_content": eval_output.get(
                "draft_content", state.get("draft_content", "")
            ),
            "evaluation_result": eval_output.get("evaluation_result"),
            "iteration_count": eval_output.get("iteration", 0),
        }

    state_to_eval_input_fn = state_to_eval_input or default_state_to_eval_input
    eval_output_to_state_fn = eval_output_to_state or default_eval_output_to_state

    def call_eval_optimize(state: dict[str, Any]) -> dict[str, Any]:
        """Transform analyst state → eval state, invoke, transform back."""
        eval_input = state_to_eval_input_fn(state, max_iterations)
        eval_output = eval_optimize_graph.invoke(eval_input)
        return eval_output_to_state_fn(state, eval_output)

    builder: Any = StateGraph(state_schema)

    builder.add_node("research", cast(Any, research_fn))
    builder.add_node("write", cast(Any, write_fn))
    builder.add_node("eval_optimize", call_eval_optimize)
    builder.add_node("build_completed", cast(Any, build_completed_fn))

    builder.add_edge(START, "research")
    builder.add_edge("research", "write")
    builder.add_edge("write", "eval_optimize")
    builder.add_edge("eval_optimize", "build_completed")
    builder.add_edge("build_completed", END)

    return builder.compile()
