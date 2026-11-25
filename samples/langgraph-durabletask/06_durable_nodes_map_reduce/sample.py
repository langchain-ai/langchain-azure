# Copyright (c) Microsoft. All rights reserved.

"""Example 06: Durable Nodes (`@durable_node` & `@durable_node`) - Static Parallel Fan-Out & `Send()` Map-Reduce.

Demonstrates:
1. Static parallel fan-out (`START -> [branch_a, branch_b, branch_c] -> join`) using `@durable_node` and `@durable_node`
   so each branch executes concurrently as its own `langgraph.<name>.node` activity.
2. Dynamic `Send("research_worker", item)` map-reduce fan-out with per-branch retry isolation (`branch #2` /
   `security` fails once on attempt 1 and retries independently without re-running sibling branches) and
   `node_max_concurrency`.
"""

from __future__ import annotations

import operator
import os
import sys
import threading
from typing import Annotated, TypedDict

from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from langchain_azure_compute.graph import attach_durability, durable_node, sdk_options


class StaticParallelState(TypedDict):
    seed: int
    results: Annotated[list[str], operator.add]
    summary: str


def build_static_parallel_graph(branch_attempts: dict[str, int]):
    lock = threading.Lock()

    @durable_node
    def branch_a(state: StaticParallelState) -> dict:
        with lock:
            branch_attempts["branch_a"] = branch_attempts.get("branch_a", 0) + 1
        return {"results": [f"A={state['seed'] + 1}"]}

    @durable_node(retries=3)
    def branch_b(state: StaticParallelState) -> dict:
        with lock:
            branch_attempts["branch_b"] = branch_attempts.get("branch_b", 0) + 1
            attempt = branch_attempts["branch_b"]
        if attempt == 1:
            raise RuntimeError("Transient failure in branch_b (attempt 1)")
        return {"results": [f"B={state['seed'] + 2}"]}

    def branch_c(state: StaticParallelState) -> dict:
        with lock:
            branch_attempts["branch_c"] = branch_attempts.get("branch_c", 0) + 1
        return {"results": [f"C={state['seed'] + 3}"]}

    durable_node(branch_c, durable=True, retries=2)

    builder = StateGraph(StaticParallelState)
    builder.add_node("branch_a", branch_a)
    builder.add_node("branch_b", branch_b)
    builder.add_node("branch_c", branch_c)
    builder.add_node("join", lambda s: {"summary": " + ".join(sorted(s["results"]))})
    builder.add_edge(START, "branch_a")
    builder.add_edge(START, "branch_b")
    builder.add_edge(START, "branch_c")
    builder.add_edge(["branch_a", "branch_b", "branch_c"], "join")
    builder.add_edge("join", END)
    return builder.compile()


class ResearchState(TypedDict):
    topics: list[str]
    findings: Annotated[list[str], operator.add]
    synthesis: str


def build_map_reduce_graph(call_counts: dict[str, int]):
    lock = threading.Lock()

    def planner(state: ResearchState) -> dict:
        return {"findings": [f"PLANNED({len(state['topics'])} topics)"]}

    def dispatch_workers(state: ResearchState) -> list[Send]:
        return [Send("research_worker", {"topic": topic}) for topic in state["topics"]]

    @durable_node(retries=3)
    def research_worker(arg: dict) -> dict:
        topic = arg["topic"]
        with lock:
            call_counts[topic] = call_counts.get(topic, 0) + 1
            attempt = call_counts[topic]
        if topic == "security" and attempt == 1:
            raise RuntimeError("Simulated transient timeout in security worker (attempt 1)")
        return {"findings": [f"REPORT[{topic}]"]}

    def synthesize(state: ResearchState) -> dict:
        return {"synthesis": " | ".join(sorted(state["findings"]))}

    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner)
    builder.add_node("research_worker", research_worker)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", dispatch_workers, ["research_worker"])
    builder.add_edge("research_worker", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


def main() -> None:
    endpoint = (
        os.environ.get("DTS_ENDPOINT")
        or os.environ.get("DTS_EMULATOR")
        or os.environ.get("DURABLE_TASK_SCHEDULER_CONNECTION_STRING")
    )
    if not endpoint:
        import socket

        for _cand_port in (18080, 8080):
            try:
                with socket.create_connection(("127.0.0.1", _cand_port), timeout=0.3):
                    endpoint = f"127.0.0.1:{_cand_port}"
                    break
            except OSError:
                pass
    if not endpoint:
        print("Set DTS_EMULATOR=127.0.0.1:8080 (or DURABLE_TASK_SCHEDULER_CONNECTION_STRING) to run.")
        print("Start one with: docker run -d -p 8080:8080 -p 8082:8082 mcr.microsoft.com/dts/dts-emulator:latest")
        sys.exit(0)

    branch_attempts: dict[str, int] = {}
    call_counts: dict[str, int] = {}

    reg_static = attach_durability(build_static_parallel_graph(branch_attempts), name="static_parallel_nodes")
    reg_mr = attach_durability(
        build_map_reduce_graph(call_counts),
        name="research_map_reduce",
        node_max_concurrency=2,
    )

    worker = DurableTaskSchedulerWorker(**sdk_options(endpoint))
    sdk = DurableTaskSchedulerClient(**sdk_options(endpoint))
    reg_static.register(worker)
    reg_mr.register(worker)
    worker.start()
    try:
        # Part 1: Static parallel fan-out (START -> [branch_a, branch_b, branch_c] -> join)
        res_static = reg_static.client(sdk).start({"seed": 10, "results": [], "summary": ""}).wait_result(timeout=60)
        print("1. Static parallel summary:", res_static["summary"])
        print("   Static branch attempts (only branch_b retried):", branch_attempts)

        # Part 2: Dynamic Send('research_worker', item) map-reduce with node_max_concurrency=2
        res_mr = (
            reg_mr.client(sdk)
            .start(
                {"topics": ["architecture", "security", "performance"], "findings": [], "synthesis": ""},
                node_max_concurrency=2,
            )
            .wait_result(timeout=60)
        )
        print("2. Map-reduce synthesis:", res_mr["synthesis"])
        print("   Map-reduce attempt counts (only 'security' retried):", call_counts)
    finally:
        worker.stop()
        sdk.close()


if __name__ == "__main__":
    main()
