# Copyright (c) Microsoft. All rights reserved.

"""Example 08: Graph Versioning & Live Migration (`version`, `previous_versions`, `migrations`, `run.upgrade()`).

Demonstrates:
1. Non-breaking downstream topology evolution in-place (adding a downstream node after an already-checkpointed
   step on the same `thread_id`).
2. Breaking topology evolution with `attach_durability(graph_v2, name="order", version="v2", previous_versions={"v1": graph_v1}, migrations={("v1", "v2"): ...}, upgrade_on_continue_as_new=True)`:
   - Automatic live migration from `"v1"` to `"v2"` across a `continue_as_new` boundary (`upgrade_on_continue_as_new=True`).
   - Explicit mid-run migration of a paused `"v1"` workflow at an `interrupt()` to `"v2"` via `run.upgrade("v2", ...)` / `run.resume(..., version="v2", as_node=...)`.
"""

from __future__ import annotations

import operator
import os
import sys
from typing import Annotated, TypedDict

from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from langchain_azure_compute.graph import Limits, attach_durability, sdk_options


class OrderState(TypedDict):
    total: int
    log: Annotated[list[str], operator.add]


def build_auto_upgrade_graphs():
    # v1: step1 (+1) -> step2 (*2)
    b1 = StateGraph(OrderState)
    b1.add_node("step1", lambda s: {"total": s["total"] + 1, "log": ["v1:step1"]})
    b1.add_node("step2", lambda s: {"total": s["total"] * 2, "log": ["v1:step2"]})
    b1.add_edge(START, "step1")
    b1.add_edge("step1", "step2")
    b1.add_edge("step2", END)

    # v2: step1 (+1) -> step2 (*10) -> step3_v2 (new downstream compliance audit node)
    b2 = StateGraph(OrderState)
    b2.add_node("step1", lambda s: {"total": s["total"] + 1, "log": ["v2:step1"]})
    b2.add_node("step2", lambda s: {"total": s["total"] * 10, "log": ["v2:step2"]})
    b2.add_node("step3_v2", lambda s: {"log": ["v2:step3_audit"]})
    b2.add_edge(START, "step1")
    b2.add_edge("step1", "step2")
    b2.add_edge("step2", "step3_v2")
    b2.add_edge("step3_v2", END)
    return b1.compile(), b2.compile()


def build_interrupt_upgrade_graphs():
    # v1: prep -> old_approval (interrupt) -> finish (*2)
    b1 = StateGraph(OrderState)
    b1.add_node("prep", lambda s: {"total": s["total"] + 1, "log": ["v1:prep"]})
    b1.add_node("old_approval", lambda s: {"log": [f"v1:approval:{interrupt('Approve order on v1?')}"]})
    b1.add_node("finish", lambda s: {"total": s["total"] * 2, "log": ["v1:finish"]})
    b1.add_edge(START, "prep")
    b1.add_edge("prep", "old_approval")
    b1.add_edge("old_approval", "finish")
    b1.add_edge("finish", END)

    # v2: renames old_approval -> new_approval and adds downstream audit_v2 node
    b2 = StateGraph(OrderState)
    b2.add_node("prep", lambda s: {"total": s["total"] + 1, "log": ["v2:prep"]})
    b2.add_node("new_approval", lambda s: {"log": [f"v2:approval:{interrupt('Approve order on v2?')}"]})
    b2.add_node("finish", lambda s: {"total": s["total"] * 10, "log": ["v2:finish"]})
    b2.add_node("audit_v2", lambda s: {"log": ["v2:audit"]})
    b2.add_edge(START, "prep")
    b2.add_edge("prep", "new_approval")
    b2.add_edge("new_approval", "finish")
    b2.add_edge("finish", "audit_v2")
    b2.add_edge("audit_v2", END)
    return b1.compile(), b2.compile()


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

    g_can_v1, g_can_v2 = build_auto_upgrade_graphs()
    reg_auto = attach_durability(
        g_can_v2,
        name="order_auto_upgrade",
        limits=Limits(steps_per_generation=1),
        version="v2",
        previous_versions={"v1": g_can_v1},
        migrations={
            ("v1", "v2"): lambda state: {"total": state["total"] + 100, "log": state["log"] + ["migrated:v1->v2"]}
        },
        upgrade_on_continue_as_new=True,
    )

    g_int_v1, g_int_v2 = build_interrupt_upgrade_graphs()
    reg_explicit = attach_durability(
        g_int_v2,
        name="order_interrupt_upgrade",
        version="v2",
        previous_versions={"v1": g_int_v1},
        migrations={
            ("v1", "v2"): lambda state, snap, resume_val: {
                "as_node": "new_approval",
                "values": {"log": state["log"] + [f"v2:upgraded_approval:{resume_val}"]},
            }
        },
    )

    worker = DurableTaskSchedulerWorker(**sdk_options(endpoint))
    sdk = DurableTaskSchedulerClient(**sdk_options(endpoint))
    reg_auto.register(worker)
    reg_explicit.register(worker)
    worker.start()
    try:
        # Part 1: Run starts on v1, executes step1 on v1, rolls over via continue_as_new to v2,
        # applies the ('v1', 'v2') state migration, and executes step2 + step3_v2 on v2!
        run_auto = reg_auto.client(sdk).start({"total": 5, "log": []}, version="v1")
        out_auto = run_auto.wait_result(timeout=60)
        print(f"1. Auto-upgrade on continue_as_new (final version={run_auto.version!r}):", out_auto)

        # Part 2: Run starts on v1 and pauses at old_approval; caller upgrades it live to v2 on resume!
        run_exp = reg_explicit.client(sdk).start({"total": 4, "log": []}, version="v1")
        receipt = run_exp.wait_pending(timeout=30)
        print("2. Paused on v1 interrupt:", receipt)
        run_exp.upgrade("v2", resume="approved-by-manager", receipt=receipt)
        out_exp = run_exp.wait_result(timeout=60)
        print(f"   Explicit upgrade on resume (final version={run_exp.version!r}):", out_exp)
    finally:
        worker.stop()
        sdk.close()


if __name__ == "__main__":
    main()
