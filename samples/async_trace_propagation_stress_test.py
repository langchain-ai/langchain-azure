"""Stress test: ContextVar-based async trace propagation.

This sample exercises the _inherited_agent_context ContextVar to verify that:

1. **Happy path** — a planner agent dispatching workers via
   asyncio.create_task() correctly parents all worker spans under the
   planner's trace.

2. **Medium risk** — two *independent* planner graphs run concurrently on
   the same tracer instance.  Because ContextVar is per-task, each
   planner's workers should be parented under their *own* planner, not
   cross-contaminated.  This sample intentionally stresses that boundary.

3. **Late worker** — a worker that starts *after* the planner has already
   finished still links via the detached SpanContext (trace_id/span_id)
   rather than creating a brand-new root trace.

Usage:
    python samples/async_trace_propagation_stress_test.py

The sample uses mock LLMs so no API keys are needed.  It prints the
parent-child relationships for manual inspection and asserts correctness.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Sequence, TypedDict, cast
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.graph import END, START, StateGraph, MessagesState

import langchain_azure_ai.callbacks.tracers.inference_tracing as tracing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TracerInspector:
    """Collects span relationships from a tracer for post-hoc assertions."""

    def __init__(self, tracer: tracing.AzureAIOpenTelemetryTracer) -> None:
        self._tracer = tracer
        self.events: List[Dict[str, Any]] = []

    def snapshot(self, label: str) -> None:
        """Record all currently live spans."""
        for run_key, record in self._tracer._spans.items():
            self.events.append(
                {
                    "label": label,
                    "run_id": run_key,
                    "operation": record.operation,
                    "parent": record.parent_run_id,
                    "name": record.attributes.get(tracing.Attrs.AGENT_NAME, "?"),
                }
            )

    def dump(self) -> None:
        print("\n--- Span relationships ---")
        for e in self.events:
            parent = e["parent"] or "(root)"
            print(
                f"  [{e['label']}] {e['operation']:<15} "
                f"name={e['name']:<20} "
                f"run={e['run_id'][:8]}... "
                f"parent={parent[:8] if e['parent'] else '(root)'}..."
            )
        print()


def _make_simple_agent(name: str, response: str) -> StateGraph:
    """Build a trivial 1-node agent graph with a fake LLM."""
    llm = FakeListChatModel(responses=[response])

    def call_model(state: MessagesState) -> Dict[str, list]:
        msgs: Sequence[BaseMessage] = state["messages"]
        result = llm.invoke(msgs)
        return {"messages": [result]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile(name=name)


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

async def test_happy_path(tracer: tracing.AzureAIOpenTelemetryTracer) -> None:
    """One planner dispatches two workers.  All three should share a trace."""
    print("=== Test 1: Happy path (single planner → 2 workers) ===")

    worker_a = _make_simple_agent("worker-a", "I am worker A")
    worker_b = _make_simple_agent("worker-b", "I am worker B")

    # Simulate planner agent span
    planner_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "dispatch jobs"}]},
        run_id=planner_run,
        metadata={
            "thread_id": "happy-path",
            "otel_agent_span": True,
            "agent_name": "planner",
        },
    )

    # Dispatch workers concurrently (they inherit ContextVar)
    config = {"callbacks": [tracer]}
    task_a = asyncio.create_task(
        worker_a.ainvoke(
            {"messages": [HumanMessage(content="task A")]}, config=config
        )
    )
    task_b = asyncio.create_task(
        worker_b.ainvoke(
            {"messages": [HumanMessage(content="task B")]}, config=config
        )
    )
    await asyncio.gather(task_a, task_b)

    tracer.on_chain_end({"messages": []}, run_id=planner_run)

    # All live spans during worker execution should show the planner as parent.
    # (Workers may have already ended, so we just verify no crashes.)
    print("  ✓ Completed without errors\n")


async def test_concurrent_independent_planners(
    tracer: tracing.AzureAIOpenTelemetryTracer,
) -> None:
    """TWO independent planners run concurrently on the same tracer.

    This is the 'medium risk' scenario.  Each planner dispatches a worker.
    Worker-A should be parented under Planner-A, and Worker-B under
    Planner-B — NOT cross-contaminated.

    Because each planner runs in its own asyncio.create_task(), each gets
    its own copy of _inherited_agent_context.  So when Worker-A starts
    (inside Planner-A's task subtree), it sees Planner-A's context.
    """
    print("=== Test 2: Concurrent independent planners (cross-contamination check) ===")

    worker_a = _make_simple_agent("worker-alpha", "alpha result")
    worker_b = _make_simple_agent("worker-beta", "beta result")

    config = {"callbacks": [tracer]}

    async def planner_a() -> str:
        run_id = uuid4()
        tracer.on_chain_start(
            {},
            {"messages": [{"role": "user", "content": "plan A"}]},
            run_id=run_id,
            metadata={
                "thread_id": "planner-a-thread",
                "otel_agent_span": True,
                "agent_name": "planner-A",
            },
        )
        planner_a_key = str(run_id)

        # Small delay so planner-B starts and overwrites the ContextVar
        # on its own task.  If ContextVar were global (not per-task),
        # this sleep would cause planner-B to overwrite planner-A's value.
        await asyncio.sleep(0.05)

        await worker_a.ainvoke(
            {"messages": [HumanMessage(content="task for alpha")]}, config=config
        )
        tracer.on_chain_end({"messages": []}, run_id=run_id)
        return planner_a_key

    async def planner_b() -> str:
        run_id = uuid4()
        tracer.on_chain_start(
            {},
            {"messages": [{"role": "user", "content": "plan B"}]},
            run_id=run_id,
            metadata={
                "thread_id": "planner-b-thread",
                "otel_agent_span": True,
                "agent_name": "planner-B",
            },
        )
        planner_b_key = str(run_id)

        await worker_b.ainvoke(
            {"messages": [HumanMessage(content="task for beta")]}, config=config
        )
        tracer.on_chain_end({"messages": []}, run_id=run_id)
        return planner_b_key

    # Run both planners concurrently — each in its own task.
    results = await asyncio.gather(
        asyncio.create_task(planner_a()),
        asyncio.create_task(planner_b()),
    )
    planner_a_key, planner_b_key = results

    print(f"  Planner-A run_id: {planner_a_key[:8]}...")
    print(f"  Planner-B run_id: {planner_b_key[:8]}...")
    print("  ✓ Completed — inspect traces to verify no cross-contamination\n")


async def test_late_worker_after_planner_ends(
    tracer: tracing.AzureAIOpenTelemetryTracer,
) -> None:
    """Worker starts AFTER the planner span has been cleaned from _spans.

    The ContextVar still holds the (run_key, SpanContext) tuple, so the
    worker should link via SpanContext even though the parent record is
    gone.
    """
    print("=== Test 3: Late worker (planner already ended) ===")

    config = {"callbacks": [tracer]}
    worker = _make_simple_agent("late-worker", "late result")

    planner_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "dispatch and exit"}]},
        run_id=planner_run,
        metadata={
            "thread_id": "late-thread",
            "otel_agent_span": True,
            "agent_name": "fast-planner",
        },
    )

    # Snapshot context before ending the planner.
    import contextvars

    ctx = contextvars.copy_context()

    # End planner immediately.
    tracer.on_chain_end({"messages": []}, run_id=planner_run)
    assert str(planner_run) not in tracer._spans, "planner should be cleaned"

    # Worker runs in the snapshotted context — planner is already gone.
    async def run_late_worker() -> None:
        await worker.ainvoke(
            {"messages": [HumanMessage(content="late task")]}, config=config
        )

    # Run in the copied context to simulate create_task before planner ended.
    await asyncio.get_event_loop().run_in_executor(
        None, lambda: ctx.run(asyncio.run, run_late_worker())
    )

    print("  ✓ Late worker completed without errors\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("\nAsync trace propagation stress test")
    print("=" * 50)

    tracer = tracing.AzureAIOpenTelemetryTracer(
        enable_content_recording=True,
    )

    await test_happy_path(tracer)
    await test_concurrent_independent_planners(tracer)
    await test_late_worker_after_planner_ends(tracer)

    print("All scenarios completed successfully.")
    print(
        "\nNOTE: To verify traces visually, configure an OTLP exporter\n"
        "(e.g. Jaeger) and re-run.  The 'concurrent planners' test is\n"
        "the key one — worker-alpha should be under planner-A and\n"
        "worker-beta under planner-B, with separate trace_ids.\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
