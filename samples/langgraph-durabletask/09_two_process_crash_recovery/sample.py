# Copyright (c) Microsoft. All rights reserved.

"""Example 09: Two-Process Crash Recovery & Multi-Instance Handoff with Zero-Config `InMemorySaver`.

Answers a common production question:
  "If I have many concurrent workflow instances and use Zero-Config `InMemorySaver`
   (`builder.compile()` with no external database), what happens when a worker pod
   crashes mid-run and a brand-new pod continues? Will it work or raise an exception?"

How it works under the hood:
  In Zero-Config mode (`inline_checkpoints=True`), after every superstep and durable
  tool call, `langgraph-durabletask` exports the active thread's latest checkpoint slice
  into a compact `checkpoint_bundle` carried in the DTS orchestration state. When a
  new worker process (`Pod B`) picks up the next step, it automatically hydrates its own
  fresh `InMemorySaver` from that `checkpoint_bundle` before invoking LangGraph - with
  zero data loss and no external database required.

What this script does:
  1. Spawns Worker Process 1 (`pod-A`, OS PID 1) via `subprocess.Popen` with a
     Zero-Config `InMemorySaver` graph (`builder.compile()`).
  2. Starts 3 concurrent order workflows (`ORD-9001`, `ORD-9002`, `ORD-9003`) and waits
     until all 3 execute `prepare_order` + `@durable_tool` `reserve_inventory` on `pod-A`
     and pause at `interrupt()` inside `manager_approval`.
  3. Hard-kills Worker Process 1 (`proc_a.kill(); proc_a.wait()`) - destroying `pod-A`'s
     entire process memory and its `InMemorySaver`.
  4. While ZERO worker processes are alive, sends `run.resume("approved-by-vp")` for all
     3 orders (DTS durably queues the external events).
  5. Spawns Worker Process 2 (`pod-B`, OS PID 2) in a brand-new OS process with a
     completely empty `InMemorySaver`.
  6. Verifies all 3 workflows complete on `pod-B` (`pid_a != pid_b`) with 100% of their
     pre-crash state (`items_reserved=True`, `order_id`, and `pid_trace`) preserved.
"""

from __future__ import annotations

import argparse
import operator
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Annotated, Any, NotRequired, TypedDict

from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command, interrupt

from langchain_azure_compute.graph import attach_durability, durable_tool, sdk_options

GRAPH_NAME = "order_crash_recovery"


class OrderInput(TypedDict):
    order_id: str


class OrderState(TypedDict):
    order_id: str
    items_reserved: NotRequired[bool]
    approved_by: NotRequired[str]
    shipped: NotRequired[bool]
    pid_trace: Annotated[list[str], operator.add]
    messages: Annotated[list[AnyMessage], add_messages]


class OrderOutput(TypedDict):
    order_id: str
    items_reserved: bool
    approved_by: str
    shipped: bool
    pid_trace: list[str]


def build_order_graph(pod_name: str = "client"):
    """Build the order fulfillment graph in Zero-Config mode (`builder.compile()`, uses `InMemorySaver`)."""

    @durable_tool(retries=3)
    @tool
    def reserve_inventory(order_id: str, runtime: ToolRuntime[Any, Any]) -> Command:
        """Reserve inventory for an order as an isolated DTS tool activity."""
        step_entry = f"reserve_inventory:pid={os.getpid()}:pod={pod_name}"
        return Command(
            update={
                "items_reserved": True,
                "pid_trace": [step_entry],
                "messages": [
                    ToolMessage(
                        content=f"Inventory reserved for {order_id} on {pod_name} (pid={os.getpid()})",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    def prepare_order(state: OrderState) -> dict:
        order_id = state["order_id"]
        step_entry = f"prepare_order:pid={os.getpid()}:pod={pod_name}"
        return {
            "items_reserved": False,
            "approved_by": "",
            "shipped": False,
            "pid_trace": [step_entry],
            "messages": [
                AIMessage(
                    content=f"Preparing order {order_id}",
                    tool_calls=[
                        {
                            "name": "reserve_inventory",
                            "args": {"order_id": order_id},
                            "id": f"call-reserve-{order_id}",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
        }

    def manager_approval(state: OrderState) -> dict:
        decision = interrupt(
            {
                "order_id": state["order_id"],
                "items_reserved": state.get("items_reserved", False),
                "question": "Approve shipment?",
            }
        )
        step_entry = f"manager_approval:pid={os.getpid()}:pod={pod_name}"
        return {
            "approved_by": str(decision),
            "pid_trace": [step_entry],
        }

    def ship_order(state: OrderState) -> dict:
        if not state.get("items_reserved"):
            raise RuntimeError(f"Cannot ship {state['order_id']}: items_reserved state was lost!")
        if not state.get("approved_by"):
            raise RuntimeError(f"Cannot ship {state['order_id']}: approved_by state was lost!")
        step_entry = f"ship_order:pid={os.getpid()}:pod={pod_name}"
        return {
            "shipped": True,
            "pid_trace": [step_entry],
        }

    builder = StateGraph(OrderState, input_schema=OrderInput, output_schema=OrderOutput)
    builder.add_node("prepare_order", prepare_order)
    builder.add_node("tools", ToolNode([reserve_inventory]))
    builder.add_node("manager_approval", manager_approval)
    builder.add_node("ship_order", ship_order)

    builder.add_edge(START, "prepare_order")
    builder.add_edge("prepare_order", "tools")
    builder.add_edge("tools", "manager_approval")
    builder.add_edge("manager_approval", "ship_order")
    builder.add_edge("ship_order", END)

    # Zero-Config mode: no checkpointer passed -> uses ephemeral per-process InMemorySaver
    return builder.compile()


def _resolve_endpoint() -> str | None:
    return (
        os.environ.get("DTS_ENDPOINT")
        or os.environ.get("DTS_EMULATOR")
        or os.environ.get("DURABLE_TASK_SCHEDULER_CONNECTION_STRING")
    )


def _run_worker_process(pod_name: str, graph_name: str) -> None:
    """Entry point for child worker OS process (`--worker pod-A` or `--worker pod-B`)."""
    endpoint = _resolve_endpoint()
    if not endpoint:
        raise RuntimeError("DTS_EMULATOR, DTS_ENDPOINT, or DURABLE_TASK_SCHEDULER_CONNECTION_STRING is required")

    reg = attach_durability(build_order_graph(pod_name), name=graph_name)
    worker = DurableTaskSchedulerWorker(**sdk_options(endpoint))
    reg.register(worker)
    worker.start()
    print(f"WORKER_READY pod={pod_name} pid={os.getpid()}", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()


def _spawn_worker_subprocess(
    pod_name: str,
    graph_name: str = GRAPH_NAME,
    *,
    endpoint: str | None = None,
) -> tuple[subprocess.Popen[str], int]:
    """Spawn a separate OS process running a DTS worker and wait for its `WORKER_READY` banner."""
    child_env = os.environ.copy()
    src_dir = next(
        (str(p / "src") for p in Path(__file__).resolve().parents if (p / "src" / "langgraph_durabletask").is_dir()),
        str(Path(__file__).resolve().parent.parent / "src"),
    )
    existing_pp = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing_pp}" if existing_pp else src_dir
    if endpoint and not (
        child_env.get("DTS_ENDPOINT")
        or child_env.get("DTS_EMULATOR")
        or child_env.get("DURABLE_TASK_SCHEDULER_CONNECTION_STRING")
    ):
        child_env["DTS_ENDPOINT"] = endpoint

    proc = subprocess.Popen(
        [sys.executable, __file__, "--worker", pod_name, "--name", graph_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
    )
    assert proc.stdout is not None
    deadline = time.monotonic() + 30.0
    captured: list[str] = []
    worker_pid: int | None = None

    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
            continue
        stripped = line.strip()
        captured.append(stripped)
        if stripped.startswith("WORKER_READY "):
            for token in stripped.split():
                if token.startswith("pid="):
                    worker_pid = int(token.split("=", 1)[1])
            break

    if worker_pid is None:
        proc.kill()
        proc.wait()
        raise RuntimeError(f"Worker subprocess {pod_name!r} failed to start. Output:\n" + "\n".join(captured))

    # Drain remaining child stdout in a daemon thread so pipe buffers never block
    def _drain() -> None:
        assert proc.stdout is not None
        for _ in proc.stdout:
            pass

    threading.Thread(target=_drain, daemon=True).start()
    return proc, worker_pid


def run_two_process_crash_demo(
    endpoint: str,
    *,
    order_ids: tuple[str, ...] = ("ORD-9001", "ORD-9002", "ORD-9003"),
    graph_name: str = GRAPH_NAME,
) -> dict[str, Any]:
    """Run the two-process crash recovery demonstration and return a structured verification summary."""
    reg = attach_durability(build_order_graph("client"), name=graph_name)
    sdk = DurableTaskSchedulerClient(**sdk_options(endpoint))
    client = reg.client(sdk)

    proc_a: subprocess.Popen[str] | None = None
    proc_b: subprocess.Popen[str] | None = None
    try:
        # 1. Spawn Process 1 (Pod A)
        proc_a, pid_a = _spawn_worker_subprocess("pod-A", graph_name=graph_name, endpoint=endpoint)
        print(f"[1/6] Spawned Worker Process 1 (pod-A) -> OS PID={pid_a} (fresh InMemorySaver)", flush=True)

        # 2. Start concurrent order instances and wait until all pause at manager_approval on Pod A
        runs = {oid: client.start({"order_id": oid}) for oid in order_ids}
        receipts = {}
        for oid, run in runs.items():
            receipt = run.wait_pending(timeout=60)
            assert receipt is not None, f"Expected interrupt receipt for {oid}"
            receipts[oid] = receipt
        print(
            f"[2/6] All {len(order_ids)} concurrent instances ({', '.join(order_ids)}) executed "
            f"prepare_order + @durable_tool reserve_inventory on pod-A (PID={pid_a}) and paused at manager_approval:",
            flush=True,
        )
        for oid, receipt in receipts.items():
            print(f"      - {oid}: interrupt payload = {receipt.value}", flush=True)

        # 3. HARD KILL Process 1 (Pod A) - its RAM and InMemorySaver are completely wiped
        proc_a.kill()
        proc_a.wait(timeout=10)
        print(
            f"[3/6] HARD KILLED Worker Process 1 (pod-A, PID={pid_a}) -> InMemorySaver RAM destroyed!",
            flush=True,
        )
        proc_a = None

        # 4. While ZERO workers are alive, send human-in-the-loop resume events to DTS
        for oid, run in runs.items():
            run.resume("approved-by-vp")
        print(
            f"[4/6] Sent run.resume('approved-by-vp') for all {len(order_ids)} instances while ZERO workers are alive",
            flush=True,
        )

        # 5. Spawn Process 2 (Pod B) - brand-new OS process with an empty InMemorySaver
        proc_b, pid_b = _spawn_worker_subprocess("pod-B", graph_name=graph_name, endpoint=endpoint)
        print(
            f"[5/6] Spawned Worker Process 2 (pod-B) -> OS PID={pid_b} (brand-new empty InMemorySaver)",
            flush=True,
        )
        assert pid_a != pid_b, f"Expected two distinct OS processes, got pid_a={pid_a} == pid_b={pid_b}"

        # 6. Wait for all concurrent instances to hydrate their InMemorySaver from DTS and finish on Pod B
        results: dict[str, OrderOutput] = {}
        for oid, run in runs.items():
            results[oid] = run.wait_result(timeout=60)

        print(
            f"[6/6] All {len(order_ids)} instances hydrated InMemorySaver from DTS checkpoint_bundle and completed on pod-B!",
            flush=True,
        )
        for oid, res in results.items():
            print(
                f"      - {oid}: items_reserved={res['items_reserved']}, "
                f"approved_by={res['approved_by']!r}, shipped={res['shipped']}",
                flush=True,
            )
            print(f"        pid_trace = {res['pid_trace']}", flush=True)

            # Verify pre-crash steps ran on Pod A (PID 1) and post-crash steps ran on Pod B (PID 2)
            assert res["order_id"] == oid
            assert res["items_reserved"] is True
            assert res["approved_by"] == "approved-by-vp"
            assert res["shipped"] is True
            assert res["pid_trace"] == [
                f"prepare_order:pid={pid_a}:pod=pod-A",
                f"reserve_inventory:pid={pid_a}:pod=pod-A",
                f"manager_approval:pid={pid_b}:pod=pod-B",
                f"ship_order:pid={pid_b}:pod=pod-B",
            ]

        return {"pid_a": pid_a, "pid_b": pid_b, "results": results}
    finally:
        for proc in (proc_a, proc_b):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
        sdk.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-process InMemorySaver crash recovery demo.")
    parser.add_argument("--worker", type=str, default=None, help="Internal flag: run as named worker subprocess.")
    parser.add_argument("--name", type=str, default=GRAPH_NAME, help="Durable graph registration name.")
    args = parser.parse_args()

    if args.worker:
        _run_worker_process(args.worker, args.name)
        return

    endpoint = _resolve_endpoint()
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

    summary = run_two_process_crash_demo(endpoint, graph_name=args.name)
    print(
        f"\nVerified cross-process InMemorySaver handoff: "
        f"pod-A (PID {summary['pid_a']}) -> [CRASH] -> pod-B (PID {summary['pid_b']}) "
        f"across {len(summary['results'])} concurrent instances with 0 state loss."
    )


if __name__ == "__main__":
    main()
