# Copyright (c) Microsoft. All rights reserved.

"""Azure Functions Python v2 app exposing a Durable LangGraph workflow with Human-in-the-Loop interrupts.

Workflow topology (`expense_approval`):
  `START -> submit_expense -> tools (@durable_tool policy_check) -> manager_review (interrupt) -> [process_payout | reject_expense] -> END`

HTTP Endpoints exposed by `create_function_app(...)`:
  - `POST /api/graphs/expense_approval/start`
  - `GET  /api/graphs/expense_approval/status/{instance_id}`
  - `POST /api/graphs/expense_approval/respond/{instance_id}`
  - `POST /api/graphs/expense_approval/respond/{instance_id}/{interrupt_id}`
  - `GET  /api/health`
"""

from __future__ import annotations

import argparse
import operator
import os
import time
from typing import Annotated, Any, NotRequired, TypedDict

from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command, interrupt

from langchain_azure_compute.graph import attach_durability, durable_tool, sdk_options
from langchain_azure_compute.graph.azure_functions import (
    FunctionAppHttpServer,
    LangGraphFunctionApp,
)

GRAPH_NAME = "expense_approval"


def _local_demo_identity(req: Any) -> str | None:
    """Opt-in, insecure identity stub for loopback demos only; never deploy enabled."""
    return "demo-user" if req.headers.get("x-demo-user") == "demo-user" else None


def _identity_options() -> dict[str, Any]:
    if os.environ.get("LOCAL_DEMO_IDENTITY") == "1":
        return {"identity_provider": _local_demo_identity}
    return {}


class ExpenseInput(TypedDict):
    expense_id: str
    employee: str
    amount: float
    category: NotRequired[str]


class ExpenseState(TypedDict):
    expense_id: str
    employee: str
    amount: float
    category: NotRequired[str]
    policy_status: NotRequired[str]
    approved: NotRequired[bool]
    reviewer: NotRequired[str]
    review_notes: NotRequired[str]
    final_status: NotRequired[str]
    payout_reference: NotRequired[str]
    worker_trace: Annotated[list[str], operator.add]
    messages: Annotated[list[AnyMessage], add_messages]


class ExpenseOutput(TypedDict):
    expense_id: str
    employee: str
    amount: float
    policy_status: str
    approved: bool
    reviewer: str
    review_notes: str
    final_status: str
    payout_reference: str
    worker_trace: list[str]


def build_expense_graph(worker_id: str = "func-host"):
    """Build the Expense Reimbursement LangGraph workflow in Zero-Config mode (`builder.compile()`)."""

    @durable_tool(retries=3)
    @tool
    def policy_check(
        expense_id: str,
        amount: float,
        category: str,
        runtime: ToolRuntime[Any, Any],
    ) -> Command:
        """Run corporate expense policy verification as an isolated DTS tool activity."""
        trace_entry = f"policy_check:worker={worker_id}:pid={os.getpid()}"
        policy_result = "REQUIRES_VP_APPROVAL" if amount >= 1000.0 else "WITHIN_STANDARD_LIMIT"
        return Command(
            update={
                "policy_status": policy_result,
                "worker_trace": [trace_entry],
                "messages": [
                    ToolMessage(
                        content=f"Policy check for {expense_id} (${amount:.2f}, {category}): {policy_result}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    def submit_expense(state: ExpenseState) -> dict[str, Any]:
        expense_id = state["expense_id"]
        amount = float(state["amount"])
        category = state.get("category") or "travel"
        trace_entry = f"submit_expense:worker={worker_id}:pid={os.getpid()}"
        return {
            "category": category,
            "policy_status": "PENDING",
            "approved": False,
            "reviewer": "",
            "review_notes": "",
            "final_status": "SUBMITTED",
            "payout_reference": "",
            "worker_trace": [trace_entry],
            "messages": [
                AIMessage(
                    content=f"Submitted expense {expense_id} for {state['employee']} (${amount:.2f})",
                    tool_calls=[
                        {
                            "name": "policy_check",
                            "args": {
                                "expense_id": expense_id,
                                "amount": amount,
                                "category": category,
                            },
                            "id": f"call-policy-{expense_id}",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
        }

    def manager_review(state: ExpenseState) -> dict[str, Any]:
        human_decision = interrupt(
            {
                "expense_id": state["expense_id"],
                "employee": state["employee"],
                "amount": state["amount"],
                "policy_status": state.get("policy_status", "UNKNOWN"),
                "prompt": "Approve or reject this expense?",
            }
        )
        if isinstance(human_decision, dict):
            approved = bool(human_decision.get("approved", False))
            reviewer = str(human_decision.get("reviewer") or "manager")
            notes = str(human_decision.get("notes") or "")
        else:
            approved = bool(human_decision)
            reviewer = "manager"
            notes = str(human_decision)

        trace_entry = f"manager_review:worker={worker_id}:pid={os.getpid()}"
        return {
            "approved": approved,
            "reviewer": reviewer,
            "review_notes": notes,
            "worker_trace": [trace_entry],
        }

    def route_after_review(state: ExpenseState) -> str:
        return "process_payout" if state.get("approved") else "reject_expense"

    def process_payout(state: ExpenseState) -> dict[str, Any]:
        trace_entry = f"process_payout:worker={worker_id}:pid={os.getpid()}"
        return {
            "final_status": "APPROVED_AND_PAID",
            "payout_reference": f"PAY-{state['expense_id']}-${state['amount']:.2f}",
            "worker_trace": [trace_entry],
        }

    def reject_expense(state: ExpenseState) -> dict[str, Any]:
        trace_entry = f"reject_expense:worker={worker_id}:pid={os.getpid()}"
        return {
            "final_status": "REJECTED",
            "payout_reference": "NONE",
            "worker_trace": [trace_entry],
        }

    builder = StateGraph(ExpenseState, input_schema=ExpenseInput, output_schema=ExpenseOutput)
    builder.add_node("submit_expense", submit_expense)
    builder.add_node("tools", ToolNode([policy_check]))
    builder.add_node("manager_review", manager_review)
    builder.add_node("process_payout", process_payout)
    builder.add_node("reject_expense", reject_expense)

    builder.add_edge(START, "submit_expense")
    builder.add_edge("submit_expense", "tools")
    builder.add_edge("tools", "manager_review")
    builder.add_conditional_edges(
        "manager_review",
        route_after_review,
        {"process_payout": "process_payout", "reject_expense": "reject_expense"},
    )
    builder.add_edge("process_payout", END)
    builder.add_edge("reject_expense", END)

    return builder.compile()


def build_expense_function_app(
    *,
    graph_name: str = GRAPH_NAME,
    worker_id: str = "func-host",
    dts_options: Any = None,
    start_worker: bool = True,
):
    """Build an `azure.functions.FunctionApp` for the expense approval workflow."""
    return LangGraphFunctionApp(
        graph=build_expense_graph(worker_id),
        name=graph_name,
        dts_options=dts_options,
        start_worker=start_worker,
        **_identity_options(),
    )


# Default Azure Functions v2 entry point discovered by `func start` (one-line, like MAF's AgentFunctionApp!)
_default_worker_id = os.environ.get("WORKER_ID", "func-host")
_start_embedded = os.environ.get("START_EMBEDDED_WORKER", "1").strip().lower() not in ("0", "false", "no")
app = LangGraphFunctionApp(
    graph=build_expense_graph(_default_worker_id),
    name=GRAPH_NAME,
    start_worker=_start_embedded,
    **_identity_options(),
)


def _run_standalone_worker(worker_id: str, graph_name: str) -> None:
    """Run a dedicated DTS worker process (`python function_app.py --worker worker-1`)."""
    worker_reg = attach_durability(build_expense_graph(worker_id), name=graph_name)
    worker = DurableTaskSchedulerWorker(**sdk_options())
    worker_reg.register(worker)
    worker.start()
    print(f"WORKER_READY worker={worker_id} pid={os.getpid()}", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Azure Functions HITL Expense Approval App / Worker")
    parser.add_argument("--worker", type=str, default=None, help="Run as a standalone DTS worker with this worker_id.")
    parser.add_argument("--name", type=str, default=GRAPH_NAME, help="Durable graph registration name.")
    parser.add_argument(
        "--port", type=int, default=7071, help="HTTP port when running the Function App server directly."
    )
    parser.add_argument(
        "--no-embedded-worker",
        action="store_true",
        help="Serve only HTTP Function routes without starting an embedded DTS worker in the HTTP process.",
    )
    args = parser.parse_args()

    if args.worker:
        _run_standalone_worker(args.worker, args.name)
        return

    func_app = build_expense_function_app(
        graph_name=args.name,
        worker_id=os.environ.get("WORKER_ID", "func-host"),
        start_worker=not args.no_embedded_worker,
    )
    with FunctionAppHttpServer(func_app, host="127.0.0.1", port=args.port) as server:
        print(f"FUNCTION_APP_READY url={server.base_url}/api pid={os.getpid()}", flush=True)
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
