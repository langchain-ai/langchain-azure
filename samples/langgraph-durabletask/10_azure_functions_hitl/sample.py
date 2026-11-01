# Copyright (c) Microsoft. All rights reserved.

"""Example 10: Azure Functions Human-in-the-Loop (`interrupt()`) with Two Separate Worker Processes.

Demonstrates an Azure Functions Python v2 app (`examples/10_azure_functions_hitl/function_app.py`)
hosting a durable LangGraph workflow (`expense_approval`) that pauses at `interrupt()`, exposes
an HTTP Function endpoint (`POST /api/graphs/expense_approval/respond/{instance_id}/{interrupt_id}`)
to set the human response, and resumes seamlessly across **two separate worker processes**:

  1. Starts the Azure Functions HTTP routes (`/api/graphs/expense_approval/*` and `/api/health`)
     on a local TCP port (`start_worker=False` on the HTTP host so all graph work runs on the
     dedicated worker subprocesses).
  2. Spawns Worker 1 (`worker-1`, OS `PID 1`) via `subprocess.Popen`.
  3. Sends `POST /api/graphs/expense_approval/start` with `EXP-501` (`$1850.00` for `Alice`).
  4. Polls `GET /api/graphs/expense_approval/status/{instanceId}` until `pendingInterrupts` is
     populated (`worker-1` executed `submit_expense` + `@durable_tool` `policy_check` and paused
     at `interrupt()` inside `manager_review`).
  5. Spawns Worker 2 (`worker-2`, OS `PID 2`) and **hard-kills Worker 1 (`proc_1.kill()`)** while
     the graph is paused waiting for the human response!
  6. Sends the human approval response via HTTP `POST /api/graphs/expense_approval/respond/{instanceId}/{interruptId}`
     (`{"approved": True, "reviewer": "vp-finance", "notes": "Approved for Q4 travel"}`).
  7. Polls `GET /api/graphs/expense_approval/status/{instanceId}` until `runtimeStatus == "COMPLETED"`
     and verifies that pre-interrupt steps ran on `worker-1 (PID 1)` and post-resume steps
     (`manager_review` and `process_payout`) ran on `worker-2 (PID 2)` (`PID 1 != PID 2`).
  8. Starts a second instance (`EXP-502`), pauses at `interrupt()`, and POSTs a rejection response
     (`{"approved": False, "reviewer": "vp-finance", "notes": "Missing receipt"}`) to
     `POST /api/graphs/expense_approval/respond/{instanceId}` to verify the rejection branch.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from langchain_azure_compute.graph import FunctionAppHttpServer, sdk_options

_HERE = Path(__file__).resolve().parent
FUNC_APP_PATH = _HERE / "function_app.py"


def _load_function_app_module() -> Any:
    spec = importlib.util.spec_from_file_location("azure_functions_hitl_app", FUNC_APP_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_endpoint() -> str | None:
    return (
        os.environ.get("DTS_ENDPOINT")
        or os.environ.get("DTS_EMULATOR")
        or os.environ.get("DURABLE_TASK_SCHEDULER_CONNECTION_STRING")
    )


def _http_json(method: str, url: str, body: Any = None) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return int(resp.status), json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        return int(exc.code), json.loads(raw) if raw else {"error": str(exc)}


def _spawn_worker_subprocess(
    worker_id: str,
    graph_name: str,
    *,
    endpoint: str,
) -> tuple[subprocess.Popen[str], int]:
    """Spawn a separate OS process running `function_app.py --worker <worker_id>`."""
    child_env = os.environ.copy()
    src_dir = next(
        (str(p / "src") for p in Path(__file__).resolve().parents if (p / "src" / "langgraph_durabletask").is_dir()),
        str(Path(__file__).resolve().parent.parent / "src"),
    )
    existing_pp = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing_pp}" if existing_pp else src_dir
    child_env["START_EMBEDDED_WORKER"] = "0"
    if not (
        child_env.get("DTS_ENDPOINT")
        or child_env.get("DTS_EMULATOR")
        or child_env.get("DURABLE_TASK_SCHEDULER_CONNECTION_STRING")
    ):
        if "endpoint=" in endpoint.lower():
            child_env["DURABLE_TASK_SCHEDULER_CONNECTION_STRING"] = endpoint
        else:
            child_env["DTS_ENDPOINT"] = endpoint

    proc = subprocess.Popen(
        [sys.executable, str(FUNC_APP_PATH), "--worker", worker_id, "--name", graph_name],
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
        raise RuntimeError(f"Worker subprocess {worker_id!r} failed to start. Output:\n" + "\n".join(captured))

    def _drain() -> None:
        assert proc.stdout is not None
        for _ in proc.stdout:
            pass

    threading.Thread(target=_drain, daemon=True).start()
    return proc, worker_pid


def _poll_until_interrupt(status_url: str, *, timeout: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    delay = 0.05
    while time.monotonic() < deadline:
        code, payload = _http_json("GET", status_url)
        assert code == 200, f"Unexpected status code {code}: {payload}"
        if payload.get("pendingInterrupts"):
            return payload
        if payload.get("runtimeStatus") in ("FAILED", "TERMINATED", "COMPLETED"):
            raise AssertionError(f"Run reached {payload.get('runtimeStatus')} before interrupt: {payload}")
        time.sleep(delay)
        delay = min(0.25, delay * 1.3)
    raise TimeoutError(f"Timed out waiting for pendingInterrupts at {status_url}")


def _poll_until_completed(status_url: str, *, timeout: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    delay = 0.05
    while time.monotonic() < deadline:
        code, payload = _http_json("GET", status_url)
        assert code == 200, f"Unexpected status code {code}: {payload}"
        st = payload.get("runtimeStatus")
        if st == "COMPLETED":
            return payload
        if st in ("FAILED", "TERMINATED"):
            raise AssertionError(f"Run failed with status {st}: {payload}")
        time.sleep(delay)
        delay = min(0.25, delay * 1.3)
    raise TimeoutError(f"Timed out waiting for COMPLETED at {status_url}")


def run_azure_functions_two_worker_demo(
    endpoint: str,
    *,
    graph_name: str = "expense_approval",
) -> dict[str, Any]:
    """Run the end-to-end Azure Functions HITL two-worker verification and return the summary."""
    func_mod = _load_function_app_module()
    dts_opts = sdk_options(endpoint)
    func_app = func_mod.build_expense_function_app(
        graph_name=graph_name,
        worker_id="http-client",
        dts_options=dts_opts,
        start_worker=False,
    )

    proc_1: subprocess.Popen[str] | None = None
    proc_2: subprocess.Popen[str] | None = None

    with FunctionAppHttpServer(func_app, host="127.0.0.1", port=0) as server:
        api_base = f"{server.base_url}/api"
        try:
            # Verify /api/health
            h_code, h_body = _http_json("GET", f"{api_base}/health")
            assert h_code == 200 and h_body["status"] == "healthy" and graph_name in h_body["graphs"]
            print(f"[1/7] Azure Functions HTTP host listening at {api_base} (health={h_body['status']})", flush=True)

            # Spawn Worker 1 (OS Process 1)
            proc_1, pid_1 = _spawn_worker_subprocess("worker-1", graph_name, endpoint=endpoint)
            print(f"[2/7] Spawned Worker 1 (worker-1) -> OS PID={pid_1}", flush=True)

            # Start Expense EXP-501 via POST /api/graphs/{graph_name}/start
            start_code, start_1 = _http_json(
                "POST",
                f"{api_base}/graphs/{graph_name}/start",
                {
                    "expense_id": "EXP-501",
                    "employee": "Alice",
                    "amount": 1850.0,
                    "category": "conference",
                },
            )
            assert start_code == 202, f"Expected 202 Accepted, got {start_code}: {start_1}"
            iid_1 = start_1["instanceId"]
            status_url_1 = start_1["statusQueryGetUri"]

            # Poll GET /api/graphs/{graph_name}/status/{instanceId} until paused at interrupt()
            paused_1 = _poll_until_interrupt(status_url_1, timeout=60.0)
            intr_1 = paused_1["pendingInterrupts"][0]
            intr_id_1 = intr_1["interruptId"]
            respond_url_1 = intr_1["respondUrl"]
            assert intr_1["value"] == {
                "expense_id": "EXP-501",
                "employee": "Alice",
                "amount": 1850.0,
                "policy_status": "REQUIRES_VP_APPROVAL",
                "prompt": "Approve or reject this expense?",
            }
            print(
                f"[3/7] EXP-501 ({iid_1}) paused at interrupt {intr_id_1[:8]} on worker-1 (PID={pid_1}): "
                f"payload={intr_1['value']}",
                flush=True,
            )

            # Spawn Worker 2 (OS Process 2) and hard-kill Worker 1 while the workflow is paused at interrupt()!
            proc_2, pid_2 = _spawn_worker_subprocess("worker-2", graph_name, endpoint=endpoint)
            assert pid_1 != pid_2, f"Expected distinct worker PIDs, got pid_1={pid_1} == pid_2={pid_2}"
            proc_1.kill()
            proc_1.wait(timeout=10)
            proc_1 = None
            print(
                f"[4/7] Spawned Worker 2 (worker-2, PID={pid_2}) and HARD-KILLED Worker 1 (PID={pid_1}) "
                f"while EXP-501 is waiting for human input!",
                flush=True,
            )

            # Send human approval response via POST /api/graphs/{graph_name}/respond/{instanceId}/{interruptId}
            resp_code_1, resp_body_1 = _http_json(
                "POST",
                respond_url_1,
                {
                    "approved": True,
                    "reviewer": "vp-finance",
                    "notes": "Approved for Q4 travel",
                },
            )
            assert resp_code_1 == 200 and resp_body_1["resumed"] is True
            print(
                f"[5/7] POSTed human approval to Azure Function {respond_url_1} -> {resp_body_1}",
                flush=True,
            )

            # Poll GET /api/graphs/{graph_name}/status/{instanceId} until COMPLETED on Worker 2
            done_1 = _poll_until_completed(status_url_1, timeout=60.0)
            out_1 = done_1["output"]
            print(
                f"[6/7] EXP-501 completed on Worker 2 (PID={pid_2})! "
                f"final_status={out_1['final_status']!r}, payout_reference={out_1['payout_reference']!r}",
                flush=True,
            )
            print(f"      worker_trace = {out_1['worker_trace']}", flush=True)

            assert out_1["expense_id"] == "EXP-501"
            assert out_1["policy_status"] == "REQUIRES_VP_APPROVAL"
            assert out_1["approved"] is True
            assert out_1["reviewer"] == "vp-finance"
            assert out_1["review_notes"] == "Approved for Q4 travel"
            assert out_1["final_status"] == "APPROVED_AND_PAID"
            assert out_1["payout_reference"] == "PAY-EXP-501-$1850.00"
            assert out_1["worker_trace"] == [
                f"submit_expense:worker=worker-1:pid={pid_1}",
                f"policy_check:worker=worker-1:pid={pid_1}",
                f"manager_review:worker=worker-2:pid={pid_2}",
                f"process_payout:worker=worker-2:pid={pid_2}",
            ]

            # Run a second instance (EXP-502) testing the rejection path via POST /respond/{instanceId}
            start_code_2, start_2 = _http_json(
                "POST",
                f"{api_base}/graphs/{graph_name}/start",
                {
                    "input": {
                        "expense_id": "EXP-502",
                        "employee": "Bob",
                        "amount": 320.0,
                        "category": "meals",
                    }
                },
            )
            assert start_code_2 == 202
            iid_2 = start_2["instanceId"]
            status_url_2 = start_2["statusQueryGetUri"]
            paused_2 = _poll_until_interrupt(status_url_2, timeout=60.0)
            assert paused_2["pendingInterrupts"][0]["value"]["policy_status"] == "WITHIN_STANDARD_LIMIT"

            resp_code_2, resp_body_2 = _http_json(
                "POST",
                f"{api_base}/graphs/{graph_name}/respond/{iid_2}",
                {
                    "approved": False,
                    "reviewer": "vp-finance",
                    "notes": "Missing itemized receipt",
                },
            )
            assert resp_code_2 == 200 and resp_body_2["resumed"] is True
            done_2 = _poll_until_completed(status_url_2, timeout=60.0)
            out_2 = done_2["output"]
            print(
                f"[7/7] EXP-502 rejection path verified via POST /respond/{iid_2}: "
                f"final_status={out_2['final_status']!r}, review_notes={out_2['review_notes']!r}",
                flush=True,
            )

            assert out_2["expense_id"] == "EXP-502"
            assert out_2["approved"] is False
            assert out_2["final_status"] == "REJECTED"
            assert out_2["payout_reference"] == "NONE"
            assert out_2["review_notes"] == "Missing itemized receipt"

            return {
                "pid_1": pid_1,
                "pid_2": pid_2,
                "approved_output": out_1,
                "rejected_output": out_2,
            }
        finally:
            for proc in (proc_1, proc_2):
                if proc is not None and proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)


def main() -> None:
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

    summary = run_azure_functions_two_worker_demo(endpoint)
    print(
        f"\nVerified Azure Functions HITL across 2 workers: "
        f"worker-1 (PID {summary['pid_1']}) -> [HTTP /respond resume] -> worker-2 (PID {summary['pid_2']})."
    )


if __name__ == "__main__":
    main()
