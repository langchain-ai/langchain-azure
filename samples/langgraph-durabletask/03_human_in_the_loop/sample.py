# Copyright (c) Microsoft. All rights reserved.

"""Example 03: Human-in-the-Loop (`interrupt()`, parallel interrupts, `pending_interrupts()`, and `run.resume()`)."""

from __future__ import annotations

import operator
import os
import sys
from typing import Annotated, NotRequired, TypedDict

from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from langchain_azure_compute.graph import attach_durability, sdk_options


class ExpenseState(TypedDict):
    employee: str
    amount: float
    approved: NotRequired[bool]
    status: NotRequired[str]


def review_expense(state: ExpenseState) -> dict:
    decision = interrupt(
        {
            "question": "Approve expense reimbursement?",
            "employee": state["employee"],
            "amount": state["amount"],
        }
    )
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    return {
        "approved": approved,
        "status": f"Approved (${state['amount']:.2f})" if approved else "Rejected",
    }


def build_single_interrupt_graph():
    builder = StateGraph(ExpenseState)
    builder.add_node("review_expense", review_expense)
    builder.add_edge(START, "review_expense")
    builder.add_edge("review_expense", END)
    return builder.compile()


class ParallelReviewState(TypedDict):
    proposal: str
    decisions: Annotated[list[str], operator.add]
    final_status: NotRequired[str]


def finance_gate(state: ParallelReviewState) -> dict:
    note = interrupt({"gate": "finance", "proposal": state["proposal"]})
    return {"decisions": [f"finance:{note}"]}


def legal_gate(state: ParallelReviewState) -> dict:
    note = interrupt({"gate": "legal", "proposal": state["proposal"]})
    return {"decisions": [f"legal:{note}"]}


def build_parallel_interrupt_graph():
    builder = StateGraph(ParallelReviewState)
    builder.add_node("finance_gate", finance_gate)
    builder.add_node("legal_gate", legal_gate)
    builder.add_node("finalize", lambda s: {"final_status": " & ".join(sorted(s["decisions"]))})
    builder.add_edge(START, "finance_gate")
    builder.add_edge(START, "legal_gate")
    builder.add_edge(["finance_gate", "legal_gate"], "finalize")
    builder.add_edge("finalize", END)
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

    reg_single = attach_durability(build_single_interrupt_graph(), name="expense_approval")
    reg_parallel = attach_durability(build_parallel_interrupt_graph(), name="parallel_approval")

    worker = DurableTaskSchedulerWorker(**sdk_options(endpoint))
    sdk = DurableTaskSchedulerClient(**sdk_options(endpoint))
    reg_single.register(worker)
    reg_parallel.register(worker)
    worker.start()
    try:
        # 1. Single interrupt() + run.resume(receipt, value)
        client_single = reg_single.client(sdk)
        run1 = client_single.start({"employee": "Ada Lovelace", "amount": 420.00})
        receipt = run1.wait_pending(timeout=30)
        print("1. Single interrupt receipt:", receipt)
        run1.resume(receipt, {"approved": True})
        print("   Final state:", run1.wait_result(timeout=30))

        # 2. Parallel branch interrupts + run.pending_interrupts() + batch run.resume({...})
        client_parallel = reg_parallel.client(sdk)
        run2 = client_parallel.start({"proposal": "Enterprise Vendor Contract", "decisions": []})
        run2.wait_pending(timeout=30)
        receipts = run2.pending_interrupts()
        print("2. Parallel pending interrupts:", receipts)
        run2.resume({r: f"approved-by-{r.value['gate']}" for r in receipts})
        print("   Parallel final state:", run2.wait_result(timeout=30))
    finally:
        worker.stop()
        sdk.close()


if __name__ == "__main__":
    main()
