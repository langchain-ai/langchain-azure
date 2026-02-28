"""Reproduce and verify the fix for issue #290: async agent trace fragmentation.

Architecture (from the issue):
  Planner agent dispatches multiple worker agents via asyncio.create_task().
  Workers run concurrently — each has LLM calls and tool use.
  WITHOUT the fix, each worker gets a separate trace_id (fragmentation).
  WITH the fix, all workers share the planner's trace_id (unified trace).

Usage:
    cd libs/azure-ai
    uv run python ../../samples/issue_290_async_trace_fragmentation.py
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict
from uuid import uuid4

import dotenv

# Load .env from the standard location (override with ENV_FILE env var)
dotenv.load_dotenv(os.environ.get("ENV_FILE", ".env"))

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from langchain_azure_ai.callbacks.tracers.inference_tracing import (
    AzureAIOpenTelemetryTracer,
)

# ---------------------------------------------------------------------------
# In-memory span exporter for reliable local validation
# ---------------------------------------------------------------------------
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


class MemoryExporter(SpanExporter):
    """Captures all exported spans in-memory for programmatic validation."""

    def __init__(self) -> None:
        self.spans: list[Any] = []

    def export(self, spans: Any) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# LLM + Tools
# ---------------------------------------------------------------------------

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    temperature=0,
    max_tokens=100,
)


@tool
def get_candidate_info(name: str) -> str:
    """Look up candidate information from the database."""
    time.sleep(0.5)
    return f"Candidate {name}: 5 years experience, Python/ML, available immediately"


@tool
def schedule_interview(candidate: str, date: str) -> str:
    """Schedule an interview for a candidate."""
    time.sleep(0.3)
    return f"Interview scheduled for {candidate} on {date}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email notification."""
    time.sleep(0.2)
    return f"Email sent to {to}: {subject}"


# ---------------------------------------------------------------------------
# Worker graph builders (matching issue #290 architecture)
# ---------------------------------------------------------------------------

candidate_tools = [get_candidate_info, send_email]
interview_tools = [schedule_interview]


def _make_candidate_worker(name: str) -> Any:
    """Worker CC agent — processes a candidate with tool calls."""
    llm_with_tools = llm.bind_tools(candidate_tools)

    def agent_node(state: MessagesState) -> Dict[str, list]:
        result = llm_with_tools.invoke(
            [HumanMessage(content=f"You are {name}. Process this candidate.")]
            + list(state["messages"])
        )
        return {"messages": [result]}

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(candidate_tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(name=name)


def _make_interview_worker(name: str) -> Any:
    """EITL CC agent — schedules interviews."""
    llm_with_tools = llm.bind_tools(interview_tools)

    def agent_node(state: MessagesState) -> Dict[str, list]:
        result = llm_with_tools.invoke(
            [HumanMessage(content=f"You are {name}. Schedule the interview.")]
            + list(state["messages"])
        )
        return {"messages": [result]}

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(interview_tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(name=name)


def _make_summary_agent(name: str) -> Any:
    """Simple LLM-only summary agent."""

    def call_model(state: MessagesState) -> Dict[str, list]:
        result = llm.invoke(
            [HumanMessage(content="Summarize the processing results in 1 sentence.")]
            + list(state["messages"])
        )
        return {"messages": [result]}

    g = StateGraph(MessagesState)
    g.add_node("agent", call_model)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    return g.compile(name=name)


# ---------------------------------------------------------------------------
# Planner: dispatch node (fire-and-forget via asyncio.create_task)
# ---------------------------------------------------------------------------


async def planner_dispatch(tracer: AzureAIOpenTelemetryTracer) -> list[str]:
    """Simulate the planner's dispatch node from issue #290.

    Creates a planner span, then fires off concurrent worker agents
    via asyncio.create_task() — exactly as described in the issue.
    """
    # Build workers
    worker_rose = _make_candidate_worker("worker-rose")
    worker_aurora = _make_candidate_worker("worker-aurora")
    eitl_worker = _make_interview_worker("eitl-review")
    summary_agent = _make_summary_agent("summary-llm")

    config = {"callbacks": [tracer]}

    # --- Planner span (manually created, as a real LangGraph planner would) ---
    planner_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "Process 2 candidates + review"}]},
        run_id=planner_run,
        metadata={
            "thread_id": "job-12345",
            "otel_agent_span": True,
            "agent_name": "recruitment-planner",
        },
    )

    # --- Fire-and-forget dispatch (the core of issue #290) ---
    tasks = [
        asyncio.create_task(
            worker_rose.ainvoke(
                {"messages": [HumanMessage(content="Process candidate Rose. Look up their info and send a confirmation email.")]},
                config=config,
            ),
            name="worker-rose",
        ),
        asyncio.create_task(
            worker_aurora.ainvoke(
                {"messages": [HumanMessage(content="Process candidate Aurora. Look up their info and send a status email.")]},
                config=config,
            ),
            name="worker-aurora",
        ),
        asyncio.create_task(
            eitl_worker.ainvoke(
                {"messages": [HumanMessage(content="Schedule an interview for candidate Rose on 2026-03-15.")]},
                config=config,
            ),
            name="eitl-review",
        ),
        asyncio.create_task(
            summary_agent.ainvoke(
                {"messages": [HumanMessage(content="All candidates processed successfully.")]},
                config=config,
            ),
            name="summary-llm",
        ),
    ]

    results = await asyncio.gather(*tasks)

    tracer.on_chain_end(
        {"messages": [{"role": "assistant", "content": "All tasks dispatched and completed."}]},
        run_id=planner_run,
    )

    return [r["messages"][-1].content[:80] for r in results]


# ---------------------------------------------------------------------------
# Trace tree printer
# ---------------------------------------------------------------------------


def print_trace_tree(spans: list[Any]) -> dict[str, list[str]]:
    """Print the trace tree and return {trace_id: [span_names]}."""
    by_id = {}
    children: dict[str, list] = {}
    traces: dict[str, list] = {}

    for s in spans:
        sid = format(s.context.span_id, "016x")
        tid = format(s.context.trace_id, "032x")
        by_id[sid] = s
        traces.setdefault(tid, []).append(s)
        pid = format(s.parent.span_id, "016x") if s.parent else None
        if pid and pid in [format(x.context.span_id, "016x") for x in spans]:
            children.setdefault(pid, []).append(s)

    trace_spans: dict[str, list[str]] = {}

    for tid, tspans in sorted(traces.items()):
        span_ids_in_trace = {format(s.context.span_id, "016x") for s in tspans}
        roots = [
            s
            for s in tspans
            if not s.parent
            or format(s.parent.span_id, "016x") not in span_ids_in_trace
        ]
        root_name = roots[0].name if roots else "?"
        print(f"\n  Trace {tid[:12]}  ({root_name})  [{len(tspans)} spans]")
        trace_spans[tid[:12]] = []

        def tree(span: Any, indent: int = 0) -> None:
            dur_ms = (span.end_time - span.start_time) / 1_000_000
            sid = format(span.context.span_id, "016x")[:8]
            prefix = "  " * indent + ("└── " if indent > 0 else "● ")
            print(f"    {prefix}{span.name}  {dur_ms:.0f}ms  [{sid}]")
            trace_spans[tid[:12]].append(span.name)
            for child in sorted(
                children.get(format(span.context.span_id, "016x"), []),
                key=lambda x: x.start_time,
            ):
                tree(child, indent + 1)

        for root in sorted(roots, key=lambda x: x.start_time):
            tree(root)

    return trace_spans


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_traces(spans: list[Any]) -> bool:
    """Validate that all spans share one trace_id and have correct parenting."""
    ok = True

    # 1) All spans should share the same trace_id
    trace_ids = {format(s.context.trace_id, "032x") for s in spans}
    if len(trace_ids) == 1:
        print("  ✅ All spans share a single trace_id")
    else:
        print(f"  ❌ FRAGMENTED: {len(trace_ids)} different trace_ids!")
        ok = False

    # 2) Find root (planner) and workers
    by_id = {format(s.context.span_id, "016x"): s for s in spans}
    roots = [s for s in spans if not s.parent or format(s.parent.span_id, "016x") not in by_id]
    if len(roots) == 1 and "recruitment-planner" in roots[0].name:
        print("  ✅ Single root span: recruitment-planner")
    else:
        print(f"  ❌ Expected 1 root (recruitment-planner), got {len(roots)}: {[r.name for r in roots]}")
        ok = False

    # 3) Workers should be children of planner
    if roots:
        root_sid = format(roots[0].context.span_id, "016x")
        workers = [
            s for s in spans
            if s.parent and format(s.parent.span_id, "016x") == root_sid
            and "invoke_agent" in s.name
        ]
        worker_names = sorted(s.name for s in workers)
        expected = sorted([
            "invoke_agent worker-rose",
            "invoke_agent worker-aurora",
            "invoke_agent eitl-review",
            "invoke_agent summary-llm",
        ])
        if worker_names == expected:
            print(f"  ✅ All 4 workers are direct children of planner")
        else:
            print(f"  ❌ Expected workers {expected}, got {worker_names}")
            ok = False

    # 4) Tool and chat spans should be under their worker, not planner
    tool_spans = [s for s in spans if "execute_tool" in s.name]
    chat_spans = [s for s in spans if "chat" in s.name]
    misparented = []
    for s in tool_spans + chat_spans:
        if s.parent:
            parent_sid = format(s.parent.span_id, "016x")
            parent_span = by_id.get(parent_sid)
            if parent_span and "planner" in parent_span.name:
                misparented.append(s.name)
    if not misparented:
        print(f"  ✅ All {len(tool_spans)} tool + {len(chat_spans)} chat spans nested under workers (not flat)")
    else:
        print(f"  ❌ Misparented spans (flat under planner): {misparented}")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 70)
    print("Issue #290 Reproduction: Async Agent Trace Fragmentation")
    print("=" * 70)

    # --- Set up in-memory exporter BEFORE configure_azure_monitor ---
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry import trace as otel_trace

    exporter = MemoryExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)

    # --- Optionally also send to App Insights ---
    conn_str = os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING")
    if conn_str:
        try:
            from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

            ai_exporter = AzureMonitorTraceExporter(connection_string=conn_str)
            provider.add_span_processor(SimpleSpanProcessor(ai_exporter))
            print("  → Also sending to App Insights\n")
        except ImportError:
            print("  → azure-monitor-opentelemetry-exporter not installed; local only\n")
    else:
        print("  → No APPLICATION_INSIGHTS_CONNECTION_STRING; local validation only\n")

    tracer = AzureAIOpenTelemetryTracer(enable_content_recording=True)

    # --- Run the planner dispatch ---
    print("Running planner → 4 concurrent workers (create_task)...\n")
    results = await planner_dispatch(tracer)

    for i, r in enumerate(results):
        print(f"  Worker {i+1}: {r}")

    # --- Print trace tree ---
    spans = exporter.spans
    print(f"\n{'='*70}")
    print(f"Trace Tree ({len(spans)} spans)")
    print("=" * 70)
    print_trace_tree(spans)

    # --- Validate ---
    print(f"\n{'='*70}")
    print("Validation")
    print("=" * 70)
    all_ok = validate_traces(spans)

    if all_ok:
        print("\n  🎉 All validations passed — issue #290 is FIXED!")
    else:
        print("\n  💥 Some validations FAILED — issue #290 is NOT fully fixed.")

    provider.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
