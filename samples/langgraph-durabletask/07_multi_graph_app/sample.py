# Copyright (c) Microsoft. All rights reserved.

"""Example 07: Multiple Durable Graphs in One Application (`triage`, `vacation`, `refund`).

Answers the question:
"What happens if I have multiple durable graphs I want to run in my app - how does that work?"

How it works:
1. Each graph is attached with its own unique `name` (`attach_durability(graph, name="triage")`, etc.),
   which namespaces its Durable Task entities:
     - `langgraph.triage.{run,step,tool,node}`
     - `langgraph.vacation.{run,step,tool,node}`
     - `langgraph.refund.{run,step,tool,node}`
2. All registrations call `.register(worker)` on the **same `DurableTaskSchedulerWorker`** and share a
   **single `DurableTaskSchedulerClient`** and a **single `SqliteSaver`** checkpointer (or run in Zero-Config mode).
3. Multiple graphs can run concurrently on the same worker fleet, AND one durable graph (`triage`) can
   delegate work to specialist durable graphs (`vacation` or `refund`) via a `@durable_tool` or `@durable_node`!
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Annotated, NotRequired, TypedDict

from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_azure_compute.graph import attach_durability, durable_node, durable_tool, sdk_options


# ---------------------------------------------------------------------------
# Graph 1: Vacation Specialist (`langgraph.vacation.*`)
# ---------------------------------------------------------------------------
class VacationState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    destination: str
    itinerary: NotRequired[str]


@durable_tool(retries=3)
@tool
def reserve_package(destination: str) -> str:
    """Book flight + hotel bundle for destination."""
    return f"PKG-{destination.upper()}-9001"


def build_vacation_graph(saver=None):
    def plan(state: VacationState) -> dict:
        dest = state["destination"]
        return {
            "messages": [
                AIMessage(
                    content=f"Booking package for {dest}",
                    tool_calls=[
                        {"name": "reserve_package", "args": {"destination": dest}, "id": "pkg-1", "type": "tool_call"}
                    ],
                )
            ]
        }

    def finalize(state: VacationState) -> dict:
        codes = [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
        return {"itinerary": f"Confirmed {state['destination']}: {', '.join(codes)}"}

    b = StateGraph(VacationState)
    b.add_node("plan", plan)
    b.add_node("tools", ToolNode([reserve_package]))
    b.add_node("finalize", finalize)
    b.add_edge(START, "plan")
    b.add_edge("plan", "tools")
    b.add_edge("tools", "finalize")
    b.add_edge("finalize", END)
    return b.compile(checkpointer=saver) if saver is not None else b.compile()


# ---------------------------------------------------------------------------
# Graph 2: Refund Specialist (`langgraph.refund.*`)
# ---------------------------------------------------------------------------
class RefundState(TypedDict):
    order_id: str
    amount: float
    refund_receipt: NotRequired[str]


def build_refund_graph(saver=None):
    @durable_node(retries=2)
    def issue_refund(state: RefundState) -> dict:
        return {"refund_receipt": f"REFUND-{state['order_id']}-${state['amount']:.2f}"}

    b = StateGraph(RefundState)
    b.add_node("issue_refund", issue_refund)
    b.add_edge(START, "issue_refund")
    b.add_edge("issue_refund", END)
    return b.compile(checkpointer=saver) if saver is not None else b.compile()


# ---------------------------------------------------------------------------
# Graph 3: Front-Door Triage Router (`langgraph.triage.*`)
# Delegates to either the Vacation or Refund durable graph!
# ---------------------------------------------------------------------------
class TriageState(TypedDict):
    request_id: str
    query: str
    category: NotRequired[str]
    resolution: NotRequired[str]


def build_triage_graph(vacation_client_fn, refund_client_fn, saver=None):
    def classify(state: TriageState) -> dict:
        q = state["query"].lower()
        cat = "refund" if "refund" in q else "vacation"
        return {"category": cat}

    @durable_node(retries=2)
    def dispatch_specialist(state: TriageState) -> dict:
        req_id = state["request_id"]
        if state["category"] == "refund":
            # Deterministically derive the child instance_id from request_id so retries are idempotent
            child_run = refund_client_fn().start(
                {"order_id": req_id, "amount": 199.00},
                instance_id=f"delegated-refund-{req_id}",
            )
            out = child_run.wait_result(timeout=30)
            return {"resolution": f"Routed to refund graph -> {out['refund_receipt']}"}

        child_run = vacation_client_fn().start(
            {"messages": [HumanMessage(content=state["query"])], "destination": "Lisbon"},
            instance_id=f"delegated-vacation-{req_id}",
        )
        out = child_run.wait_result(timeout=30)
        return {"resolution": f"Routed to vacation graph -> {out['itinerary']}"}

    b = StateGraph(TriageState)
    b.add_node("classify", classify)
    b.add_node("dispatch_specialist", dispatch_specialist)
    b.add_edge(START, "classify")
    b.add_edge("classify", "dispatch_specialist")
    b.add_edge("dispatch_specialist", END)
    return b.compile(checkpointer=saver) if saver is not None else b.compile()


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

    with tempfile.TemporaryDirectory() as tmp:
        conn = sqlite3.connect(Path(tmp) / "shared_app.db", check_same_thread=False)
        shared_saver = SqliteSaver(conn)

        sdk = DurableTaskSchedulerClient(**sdk_options(endpoint))
        worker = DurableTaskSchedulerWorker(**sdk_options(endpoint))

        # 1. Attach durability to all 3 graphs under distinct names sharing one SqliteSaver
        reg_vacation = attach_durability(build_vacation_graph(shared_saver), name="vacation")
        reg_refund = attach_durability(build_refund_graph(shared_saver), name="refund")

        vacation_client = reg_vacation.client(sdk)
        refund_client = reg_refund.client(sdk)

        reg_triage = attach_durability(
            build_triage_graph(lambda: vacation_client, lambda: refund_client, shared_saver),
            name="triage",
        )
        triage_client = reg_triage.client(sdk)

        # 2. Register all 3 durable graphs onto the SAME worker
        reg_vacation.register(worker)
        reg_refund.register(worker)
        reg_triage.register(worker)
        worker.start()

        try:
            # A. Start all 3 durable graphs concurrently on the same worker & checkpointer
            run_vac = vacation_client.start({"messages": [HumanMessage(content="Trip to Maui")], "destination": "Maui"})
            run_ref = refund_client.start({"order_id": "ORD-500", "amount": 85.50})
            run_tri = triage_client.start(
                {"request_id": f"REQ-77-{os.urandom(3).hex()}", "query": "Please refund my broken headphones"}
            )

            out_vac = run_vac.wait_result(timeout=60)
            out_ref = run_ref.wait_result(timeout=60)
            out_tri = run_tri.wait_result(timeout=60)

            print("1. Direct vacation graph (`langgraph.vacation.*`):", out_vac["itinerary"])
            print("2. Direct refund graph   (`langgraph.refund.*`)  :", out_ref["refund_receipt"])
            print("3. Delegating triage graph (`langgraph.triage.*` -> `langgraph.refund.*`):", out_tri["resolution"])
        finally:
            worker.stop()
            sdk.close()
            conn.close()


if __name__ == "__main__":
    main()
