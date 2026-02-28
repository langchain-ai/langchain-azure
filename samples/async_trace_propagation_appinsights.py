"""Comprehensive trace propagation stress test → App Insights.

Covers all edge cases supported by the tracer compatibility branch:

1. Async dispatch  — planner → concurrent workers via asyncio.create_task
2. Long-running tools — tools with simulated I/O latency
3. GOTO nodes      — Command(goto=...) rerouting graph execution
4. Traceparent     — explicit W3C header propagation across boundaries
5. Concurrent planners — two independent graphs, no cross-contamination
6. Late worker     — worker starts after planner span has ended

Usage:
    cd libs/azure-ai
    uv run python ../../samples/async_trace_propagation_appinsights.py
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import time
from typing import Any, Annotated, Dict, Sequence, TypedDict
from uuid import uuid4

import dotenv

dotenv.load_dotenv(os.environ.get("ENV_FILE", ".env"))

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from langchain_azure_ai.callbacks.tracers.inference_tracing import (
    AzureAIOpenTelemetryTracer,
)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    temperature=0,
    max_tokens=100,
)

# ---------------------------------------------------------------------------
# Tools (with simulated latency)
# ---------------------------------------------------------------------------


@tool
def slow_database_lookup(query: str) -> str:
    """Simulate a slow database query that takes 2 seconds."""
    time.sleep(2)
    return f"DB result for '{query}': 42 rows found"


@tool
def slow_api_call(endpoint: str) -> str:
    """Simulate a slow external API call that takes 1.5 seconds."""
    time.sleep(1.5)
    return f"API response from '{endpoint}': status=200, data={{items: 7}}"


@tool
def fast_calculator(expression: str) -> str:
    """Evaluate a simple math expression."""
    try:
        return str(eval(expression))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


tools = [slow_database_lookup, slow_api_call, fast_calculator]
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def _make_simple_agent(name: str, system_prompt: str) -> Any:
    """1-node agent graph (no tools)."""

    def call_model(state: MessagesState) -> Dict[str, list]:
        result = llm.invoke(
            [HumanMessage(content=system_prompt)] + list(state["messages"])
        )
        return {"messages": [result]}

    g = StateGraph(MessagesState)
    g.add_node("agent", call_model)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    return g.compile(name=name)


def _make_tool_agent(name: str, system_prompt: str) -> Any:
    """Agent graph with tool calling (ReAct-style)."""

    def call_model(state: MessagesState) -> Dict[str, list]:
        result = llm_with_tools.invoke(
            [HumanMessage(content=system_prompt)] + list(state["messages"])
        )
        return {"messages": [result]}

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    g = StateGraph(MessagesState)
    g.add_node("agent", call_model)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(name=name)


class GotoState(TypedDict):
    messages: list[BaseMessage]
    step: str


def _make_goto_graph(name: str) -> Any:
    """Graph that uses Command(goto=...) to reroute execution."""

    def router(state: GotoState) -> Command:
        """Decide which specialist to call."""
        last_msg = state["messages"][-1].content if state["messages"] else ""
        if "math" in last_msg.lower():
            return Command(
                update={"step": "math"},
                goto="math_specialist",
            )
        else:
            return Command(
                update={"step": "general"},
                goto="general_specialist",
            )

    def math_specialist(state: GotoState) -> Dict[str, Any]:
        result = llm.invoke(
            [HumanMessage(content="You are a math expert. Answer concisely.")]
            + state["messages"]
        )
        return {"messages": [result], "step": "done"}

    def general_specialist(state: GotoState) -> Dict[str, Any]:
        result = llm.invoke(
            [HumanMessage(content="You are a general assistant. Answer concisely.")]
            + state["messages"]
        )
        return {"messages": [result], "step": "done"}

    g = StateGraph(GotoState)
    g.add_node("router", router)
    g.add_node("math_specialist", math_specialist)
    g.add_node("general_specialist", general_specialist)
    g.add_edge(START, "router")
    g.add_edge("math_specialist", END)
    g.add_edge("general_specialist", END)
    return g.compile(name=name)


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------


async def test_1_async_dispatch_with_tools(
    tracer: AzureAIOpenTelemetryTracer,
) -> None:
    """Planner dispatches two workers — one with long-running tool calls."""
    print("\n=== Test 1: Async dispatch + long-running tools ===")

    worker_simple = _make_simple_agent(
        "worker-simple", "Reply with exactly one sentence."
    )
    worker_tools = _make_tool_agent(
        "worker-tools",
        "You are a data analyst. Use the slow_database_lookup tool to query "
        "'sales_2024' and the fast_calculator tool to compute '1000 * 1.15'. "
        "Report both results.",
    )

    planner_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "dispatch workers with tools"}]},
        run_id=planner_run,
        metadata={
            "thread_id": "test1-tools",
            "otel_agent_span": True,
            "agent_name": "planner-with-tools",
        },
    )

    config = {"callbacks": [tracer]}
    t1 = asyncio.create_task(
        worker_simple.ainvoke(
            {"messages": [HumanMessage(content="Say hello")]}, config=config
        )
    )
    t2 = asyncio.create_task(
        worker_tools.ainvoke(
            {"messages": [HumanMessage(content="Run the analysis")]}, config=config
        )
    )
    r1, r2 = await asyncio.gather(t1, t2)

    tracer.on_chain_end({"messages": []}, run_id=planner_run)

    print(f"  Simple worker: {r1['messages'][-1].content[:60]}")
    print(f"  Tool worker:   {r2['messages'][-1].content[:60]}")
    print("  ✓ Expect: planner → 2 workers, tool worker has tool + LLM spans\n")


async def test_2_goto_routing(tracer: AzureAIOpenTelemetryTracer) -> None:
    """Graph with Command(goto=...) routing to specialists."""
    print("=== Test 2: GOTO node routing ===")

    goto_graph = _make_goto_graph("goto-router-agent")
    config = {"callbacks": [tracer]}

    # Route to math specialist
    r1 = await goto_graph.ainvoke(
        {
            "messages": [HumanMessage(content="What is the math formula for area of a circle?")],
            "step": "",
        },
        config=config,
    )
    print(f"  Math route: {r1['messages'][-1].content[:60]}")

    # Route to general specialist
    r2 = await goto_graph.ainvoke(
        {
            "messages": [HumanMessage(content="What is the capital of France?")],
            "step": "",
        },
        config=config,
    )
    print(f"  General route: {r2['messages'][-1].content[:60]}")
    print("  ✓ Expect: 2 traces, each with router → specialist spans + goto attr\n")


async def test_3_traceparent_propagation(
    tracer: AzureAIOpenTelemetryTracer,
) -> None:
    """Explicit traceparent header propagation across a simulated boundary."""
    print("=== Test 3: Explicit traceparent header propagation ===")

    worker = _make_simple_agent("traceparent-worker", "Reply in 1 sentence.")
    config = {"callbacks": [tracer]}

    # Start a "remote" parent span manually.
    remote_parent_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "remote dispatch"}]},
        run_id=remote_parent_run,
        metadata={
            "thread_id": "traceparent-test",
            "otel_agent_span": True,
            "agent_name": "remote-orchestrator",
        },
    )

    # Extract the traceparent from the span we just created.
    parent_record = tracer._spans.get(str(remote_parent_run))
    if parent_record:
        span_ctx = parent_record.span.get_span_context()
        traceparent = (
            f"00-{format(span_ctx.trace_id, '032x')}-"
            f"{format(span_ctx.span_id, '016x')}-01"
        )
        print(f"  Traceparent: {traceparent[:50]}...")

        # Worker receives the traceparent in metadata (simulating cross-process).
        r = await worker.ainvoke(
            {"messages": [HumanMessage(content="I was dispatched via traceparent")]},
            config={
                "callbacks": [tracer],
                "metadata": {"traceparent": traceparent},
            },
        )
        print(f"  Worker: {r['messages'][-1].content[:60]}")
    else:
        print("  ⚠ Could not extract parent span context")

    tracer.on_chain_end({"messages": []}, run_id=remote_parent_run)
    print("  ✓ Expect: worker parented under remote-orchestrator via traceparent\n")


async def test_4_concurrent_planners(
    tracer: AzureAIOpenTelemetryTracer,
) -> None:
    """Two independent planners — workers should NOT cross-contaminate."""
    print("=== Test 4: Concurrent planners (cross-contamination check) ===")

    worker_a = _make_tool_agent(
        "worker-A-analyst",
        "Use fast_calculator to compute '999 + 1'. Report the result.",
    )
    worker_b = _make_simple_agent(
        "worker-B-greeter", "Reply with a short greeting."
    )
    config = {"callbacks": [tracer]}

    async def planner_a() -> str:
        run_id = uuid4()
        tracer.on_chain_start(
            {},
            {"messages": [{"role": "user", "content": "plan A with tools"}]},
            run_id=run_id,
            metadata={
                "thread_id": "planner-a-concurrent",
                "otel_agent_span": True,
                "agent_name": "planner-A",
            },
        )
        await asyncio.sleep(0.1)  # Let B start
        r = await worker_a.ainvoke(
            {"messages": [HumanMessage(content="Compute the value")]},
            config=config,
        )
        tracer.on_chain_end({"messages": []}, run_id=run_id)
        return f"A: {r['messages'][-1].content[:40]}"

    async def planner_b() -> str:
        run_id = uuid4()
        tracer.on_chain_start(
            {},
            {"messages": [{"role": "user", "content": "plan B simple"}]},
            run_id=run_id,
            metadata={
                "thread_id": "planner-b-concurrent",
                "otel_agent_span": True,
                "agent_name": "planner-B",
            },
        )
        r = await worker_b.ainvoke(
            {"messages": [HumanMessage(content="Hi there")]},
            config=config,
        )
        tracer.on_chain_end({"messages": []}, run_id=run_id)
        return f"B: {r['messages'][-1].content[:40]}"

    results = await asyncio.gather(
        asyncio.create_task(planner_a()),
        asyncio.create_task(planner_b()),
    )
    print(f"  {results[0]}")
    print(f"  {results[1]}")
    print("  ✓ Expect: 2 separate traces, no cross-contamination\n")


async def test_5_late_worker(tracer: AzureAIOpenTelemetryTracer) -> None:
    """Worker starts after planner ends — links via detached SpanContext."""
    print("=== Test 5: Late worker (planner already ended) ===")

    worker = _make_simple_agent("late-worker", "Reply in 1 sentence.")
    config = {"callbacks": [tracer]}

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

    ctx = contextvars.copy_context()
    tracer.on_chain_end({"messages": []}, run_id=planner_run)
    assert str(planner_run) not in tracer._spans

    # Worker in snapshotted context — planner record is gone.
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: ctx.run(
            asyncio.run,
            worker.ainvoke(
                {"messages": [HumanMessage(content="Am I late?")]}, config=config
            ),
        ),
    )
    print(f"  Late worker: {result['messages'][-1].content[:60]}")
    print("  ✓ Expect: worker linked under fast-planner's trace_id\n")


# ---------------------------------------------------------------------------
# Trace verification helper
# ---------------------------------------------------------------------------

async def verify_traces(conn_str: str) -> None:
    """Query App Insights to show the traces we just sent."""
    print("=== Querying App Insights (may take a few seconds) ===\n")
    import subprocess
    import json

    # Extract the app ID from the connection string
    app_id = None
    for part in conn_str.split(";"):
        if part.startswith("ApplicationId="):
            app_id = part.split("=", 1)[1]

    if not app_id:
        print("  Could not extract ApplicationId — check traces manually.")
        return

    query = """
    union dependencies, requests
    | where timestamp > ago(5m)
    | where name startswith "invoke_agent" or name startswith "chat" or name startswith "execute_tool"
    | project timestamp, name, id=substring(id, 0, 12),
              parentId=substring(operation_ParentId, 0, 12),
              traceId=substring(operation_Id, 0, 12),
              duration
    | order by timestamp asc
    """

    try:
        result = subprocess.run(
            [
                "az", "monitor", "app-insights", "query",
                "--app", app_id,
                "--analytics-query", query,
                "-o", "json",
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            tables = data.get("tables", [])
            if tables and tables[0]["rows"]:
                cols = [c["name"] for c in tables[0]["columns"]]
                print(f"  {'Name':<40} {'TraceID':<14} {'SpanID':<14} {'ParentID':<14} {'Duration'}")
                print(f"  {'-'*100}")
                for row in tables[0]["rows"]:
                    r = dict(zip(cols, row))
                    print(
                        f"  {r.get('name','?'):<40} "
                        f"{r.get('traceId','?'):<14} "
                        f"{r.get('id','?'):<14} "
                        f"{r.get('parentId','?'):<14} "
                        f"{r.get('duration','?')}"
                    )
            else:
                print("  No traces found yet — wait 2-3 minutes and check the portal.")
        else:
            print(f"  Query failed: {result.stderr[:200]}")
    except Exception as e:
        print(f"  Could not query: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    conn_str = os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING")
    if not conn_str:
        print("ERROR: APPLICATION_INSIGHTS_CONNECTION_STRING not set")
        return

    print("\nComprehensive Trace Propagation Stress Test → App Insights")
    print("=" * 60)

    tracer = AzureAIOpenTelemetryTracer(
        enable_content_recording=True,
        connection_string=conn_str,
    )

    from opentelemetry import trace as otel_trace

    def flush():
        provider = otel_trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=10_000)

    await test_1_async_dispatch_with_tools(tracer)
    flush()
    await test_2_goto_routing(tracer)
    flush()
    await test_3_traceparent_propagation(tracer)
    flush()
    await test_4_concurrent_planners(tracer)
    flush()
    await test_5_late_worker(tracer)
    flush()

    print("All 5 scenarios completed.")
    print("\nShutting down TracerProvider to force all pending spans...")
    provider = otel_trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
    await asyncio.sleep(5)

    await verify_traces(conn_str)

    print("\nDone. Full traces should appear in App Insights within 2-5 minutes.")
    print("Look for: planner-with-tools, goto-router-agent, remote-orchestrator,")
    print("          planner-A, planner-B, fast-planner\n")


if __name__ == "__main__":
    asyncio.run(main())
