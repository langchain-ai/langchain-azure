# Copyright (c) Microsoft. All rights reserved.

"""Emulator integration tests and sample smoke tests for `langchain_azure_compute.graph`."""  # noqa: E501

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlsplit

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("durabletask.azuremanaged")

from durabletask.azuremanaged.client import DurableTaskSchedulerClient  # noqa: E402
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import interrupt  # noqa: E402

from langchain_azure_compute.graph import (  # noqa: E402
    attach_durability,
    sdk_options,
)


def _emulator_endpoint() -> str | None:
    """Return the endpoint, failing loudly if explicitly configured but unreachable."""
    ep = os.environ.get("DTS_ENDPOINT") or os.environ.get("DTS_EMULATOR")
    if not ep:
        return None
    if ep.strip().lower() in ("1", "true", "yes"):
        ep = "127.0.0.1:8080"
    parsed = urlsplit(ep if "://" in ep else f"//{ep}")
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 8080)
    if host is None:
        pytest.fail("Invalid explicitly configured DTS endpoint")
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return ep
    except OSError as exc:
        pytest.fail(f"Explicit DTS endpoint {ep!r} is unreachable: {exc}")


@pytest.mark.compile
def test_placeholder() -> None:
    """Used for compiling integration tests without running any real work."""


def test_emulator_durable_graph_and_hitl_roundtrip() -> None:
    """Run a durable StateGraph with human-in-the-loop interrupt() against a live DTS emulator."""  # noqa: E501
    ep = _emulator_endpoint()
    if ep is None:
        pytest.skip("DTS_EMULATOR / DTS_ENDPOINT is not configured or not reachable")

    class ApprovalState(TypedDict):
        item: str
        approved: bool

    def review_step(state: ApprovalState) -> dict[str, bool]:
        decision = bool(interrupt({"item": state["item"]}))
        return {"approved": decision}

    b = StateGraph(ApprovalState)
    b.add_node("review", review_step)
    b.add_edge(START, "review")
    b.add_edge("review", END)

    reg = attach_durability(b.compile(), name="it_hitl_graph")
    opts = sdk_options(ep)
    with DurableTaskSchedulerWorker(**opts) as worker:
        reg.register(worker)
        worker.start()
        sdk = DurableTaskSchedulerClient(**opts)
        try:
            client = reg.client(sdk)
            run = client.start({"item": "laptop", "approved": False})
            pending = run.wait_pending(timeout=15.0)
            assert pending is not None
            assert pending.value == {"item": "laptop"}
            run.resume(pending, True)
            result = run.wait_result(timeout=15.0)
            assert result == {"item": "laptop", "approved": True}
        finally:
            if hasattr(sdk, "close"):
                sdk.close()


@pytest.mark.parametrize(
    "sample_rel",
    [
        "01_hello_graph/sample.py",
        "02_vacation_booking/sample.py",
        "03_human_in_the_loop/sample.py",
        "04_react_agent_and_subgraphs/sample.py",
        "05_multi_turn_conversation/sample.py",
        "06_durable_nodes_map_reduce/sample.py",
        "07_multi_graph_app/sample.py",
        "08_graph_versioning_and_migration/sample.py",
        "09_two_process_crash_recovery/sample.py",
    ],
)
def test_emulator_sample_smoke(sample_rel: str) -> None:
    """Smoke-test runnable samples in samples/langgraph-durabletask/ when DTS_EMULATOR is live."""  # noqa: E501
    ep = _emulator_endpoint()
    if ep is None:
        pytest.skip("DTS_EMULATOR / DTS_ENDPOINT is not configured or not reachable")

    repo_root = Path(__file__).resolve().parents[5]
    sample_path = repo_root / "samples" / "langgraph-durabletask" / sample_rel
    if not sample_path.is_file():
        pytest.skip(f"Sample file {sample_path} not present")
    env = dict(os.environ)
    taskhub = sdk_options(ep)["taskhub"]
    for key in (
        "DURABLE_TASK_SCHEDULER_CONNECTION_STRING",
        "DTS_CONNECTION_STRING",
        "DTS_ENDPOINT",
        "DTS_EMULATOR",
        "DTS_TASKHUB",
        "TASKHUB_NAME",
    ):
        env.pop(key, None)
    env["DTS_ENDPOINT"] = ep
    env["DTS_TASKHUB"] = taskhub
    proc = subprocess.run(
        [sys.executable, str(sample_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, f"{sample_rel} failed:\n{proc.stdout}\n{proc.stderr}"
