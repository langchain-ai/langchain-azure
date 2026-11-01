# Copyright (c) Microsoft. All rights reserved.

"""Example 02: Vacation Booking (`plan_trip -> ToolNode([book_flight, book_hotel, book_car]) -> review -> confirm`).

Demonstrates:
1. Scenario 1 - Per-tool retry isolation during a `ToolNode` superstep: `book_flight`, `book_hotel`, and `book_car`
   fan out concurrently in a single batch. `book_car` injects a transient failure on attempt 1 and is retried by DTS without
   re-executing `book_flight` or `book_hotel` (preventing the double-booking that occurs when replaying an entire
   `ToolNode` superstep from a checkpointer alone).
2. Scenario 2 - Idempotent client retry via `client.start(..., instance_id=...)`: passing the same `instance_id`
   attaches to the existing/completed orchestration rather than scheduling a duplicate run.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Annotated, NotRequired, TypedDict

from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_azure_compute.graph import attach_durability, durable_tool, sdk_options


class VacationState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    destination: str
    confirmations: NotRequired[list[str]]
    confirmed: NotRequired[bool]


def build_vacation_graph(log_path: Path, *, fail_car_once: bool = True):
    seq = {"n": 0, "car_attempts": 0}
    log_lock = threading.Lock()

    def record_booking(kind: str, destination: str) -> str:
        with log_lock:
            seq["n"] += 1
            confirmation = f"{kind}-{1000 + seq['n']}"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{kind} BOOKED ({destination}): {confirmation}\n")
            return confirmation

    @durable_tool(retries=3)
    @tool
    def book_flight(destination: str) -> str:
        """Book a flight to the destination."""
        return record_booking("FLIGHT", destination)

    @durable_tool(retries=3)
    @tool
    def book_hotel(destination: str) -> str:
        """Book a hotel at the destination."""
        return record_booking("HOTEL", destination)

    @durable_tool(retries=3)
    @tool
    def book_car(destination: str) -> str:
        """Book a rental car at the destination (simulates a transient failure on attempt 1)."""
        with log_lock:
            seq["car_attempts"] += 1
            attempt = seq["car_attempts"]
        if fail_car_once and attempt == 1:
            raise RuntimeError("Transient rental car API outage on attempt 1")
        return record_booking("CAR", destination)

    def plan_trip(state: VacationState) -> dict:
        dest = state["destination"]
        return {
            "messages": [
                AIMessage(
                    content=f"Planning vacation to {dest}",
                    tool_calls=[
                        {"name": "book_flight", "args": {"destination": dest}, "id": "flight-1", "type": "tool_call"},
                        {"name": "book_hotel", "args": {"destination": dest}, "id": "hotel-1", "type": "tool_call"},
                        {"name": "book_car", "args": {"destination": dest}, "id": "car-1", "type": "tool_call"},
                    ],
                )
            ]
        }

    def review(state: VacationState) -> dict:
        confs = [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
        return {"confirmations": confs, "messages": [AIMessage(content="Reviewed: " + ", ".join(confs))]}

    def confirm(state: VacationState) -> dict:
        return {"confirmed": True, "messages": [AIMessage(content="Trip confirmed!")]}

    builder = StateGraph(VacationState)
    builder.add_node("plan_trip", plan_trip)
    builder.add_node("tools", ToolNode([book_flight, book_hotel, book_car]))
    builder.add_node("review", review)
    builder.add_node("confirm", confirm)
    builder.add_edge(START, "plan_trip")
    builder.add_edge("plan_trip", "tools")
    builder.add_edge("tools", "review")
    builder.add_edge("review", "confirm")
    builder.add_edge("confirm", END)
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

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "bookings.log"
        reg = attach_durability(build_vacation_graph(log_path, fail_car_once=True), name="vacation_planner")
        worker = DurableTaskSchedulerWorker(**sdk_options(endpoint))
        sdk = DurableTaskSchedulerClient(**sdk_options(endpoint))
        reg.register(worker)
        worker.start()
        try:
            client = reg.client(sdk)
            run_id = f"vacation-kyoto-{uuid.uuid4().hex[:6]}"
            inp = {"messages": [HumanMessage(content="Book trip to Kyoto")], "destination": "Kyoto"}

            print(f"\n=== [Run 1] Starting new vacation booking orchestration (instance_id={run_id!r}) ===", flush=True)
            run1 = client.start(inp, instance_id=run_id)
            result1 = run1.wait_result(timeout=60)
            print(f"=== [Run 1 Complete] Confirmations: {result1['confirmations']} ===\n", flush=True)

            # Retrying with the same instance_id attaches to the completed run without re-booking
            print(
                f"=== [Run 2: Idempotent Client Retry] Calling client.start() again with SAME instance_id={run_id!r} ===",
                flush=True,
            )
            print("  (Attaches to the already-completed orchestration -- no nodes or tools re-execute!)", flush=True)
            run2 = client.start(inp, instance_id=run_id)
            result2 = run2.wait_result(timeout=60)
            print(f"=== [Run 2 Complete] Confirmations: {result2['confirmations']} ===", flush=True)

            print("Confirmations:", result1["confirmations"])
            print("Idempotent retry matched:", result1["confirmations"] == result2["confirmations"])
            print("\nExternal booking ledger (exactly 3 bookings despite book_car failing once):")
            print(log_path.read_text(encoding="utf-8").strip())
        finally:
            worker.stop()
            sdk.close()


if __name__ == "__main__":
    main()
