# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Cross-process server lifecycle helpers for local resilience tests."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from pathlib import Path
from time import monotonic
from typing import BinaryIO

import httpx

PACKAGE_ROOT = Path(__file__).resolve().parents[6]
SERVER = Path(__file__).resolve().parents[1] / "server" / "main.py"


def free_port() -> int:
    """Reserve and return an available localhost port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def spawn_server(
    tmp_path: Path,
    port: int,
    lifetime: int,
) -> tuple[subprocess.Popen[bytes], BinaryIO]:
    """Start one server lifetime against shared durable local state."""

    state_root = tmp_path / "state"
    state_root.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "AGENTSERVER_STATE_ROOT": str(state_root / "agentserver"),
            "PORT": str(port),
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(PACKAGE_ROOT), env.get("PYTHONPATH")))
            ),
            "RESILIENCE_E2E_STATE_ROOT": str(state_root / "workflow"),
        }
    )
    log = (tmp_path / f"host-{lifetime}.log").open("wb")
    process = subprocess.Popen(
        [sys.executable, str(SERVER)],
        cwd=SERVER.parent,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return process, log


async def wait_ready(process: subprocess.Popen[bytes], port: int) -> None:
    """Wait until the server HTTP host is accepting requests."""

    deadline = monotonic() + 20.0
    url = f"http://127.0.0.1:{port}/health/live"
    while monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"Host exited during startup with {process.returncode}"
            )
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                response = await client.get(url)
            if response.status_code < 500:
                return
        except httpx.TransportError:
            pass
        await asyncio.sleep(0.1)
    raise TimeoutError("Host did not become ready")


async def wait_exit(process: subprocess.Popen[bytes]) -> int:
    """Wait for the server's intentional crash."""

    deadline = monotonic() + 30.0
    while monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            return return_code
        await asyncio.sleep(0.1)
    raise TimeoutError("Server did not reach the configured crash point")


def clear_stale_stream_locks(tmp_path: Path) -> None:
    """Remove local replay-stream locks left by the terminated process."""

    streams_path = tmp_path / "state" / "agentserver" / "streams"
    for lock_path in streams_path.glob("*.lock"):
        lock_path.unlink(missing_ok=True)


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    """Terminate a surviving server process."""

    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)