# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Real end-to-end fixtures for the hosting samples.

Each sample is launched **as documented** in its own module docstring:

    python sample_XX_*.py

The sample binds a Starlette server to ``127.0.0.1:$PORT`` using a real
Azure OpenAI deployment behind ``DefaultAzureCredential``. No code path
is mocked. The tests then talk to the running process over HTTP exactly
like the curl snippets in each sample's README.

Tests require:

* a working ``az login`` (or any other ``DefaultAzureCredential`` source),
* ``AZURE_AI_PROJECT_ENDPOINT`` set in the shell or in
  ``samples/hosting/.env``,
* optionally ``AZURE_AI_MODEL_DEPLOYMENT_NAME`` (defaults to ``gpt-4o``),
* for sample 07: ``FOUNDRY_AGENT_TOOLBOX_NAME``.

If those preconditions are not met, the affected tests are skipped, not
failed — keeping the sample suite usable in offline / no-credentials
development.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest
from dotenv import load_dotenv

SAMPLES_DIR = Path(__file__).resolve().parent.parent

# Load samples/hosting/.env into the test process so the same
# configuration the samples document also drives the tests.
load_dotenv(SAMPLES_DIR / ".env")


# ---------------------------------------------------------------------------
# Environment gating
# ---------------------------------------------------------------------------


def _require_env(*names: str) -> Optional[str]:
    """Return the name of the first missing env var, or ``None``."""
    for name in names:
        if not os.environ.get(name):
            return name
    return None


def requires_foundry_endpoint() -> None:
    """Skip the calling test if no Azure AI project endpoint is configured."""
    missing = _require_env("AZURE_AI_PROJECT_ENDPOINT")
    if missing:
        pytest.skip(
            f"{missing} is not set; skipping end-to-end sample test. "
            "Set it in samples/hosting/.env to enable."
        )


def requires_foundry_toolbox() -> None:
    """Skip the calling test if no Foundry toolbox name is configured."""
    requires_foundry_endpoint()
    missing = _require_env("FOUNDRY_AGENT_TOOLBOX_NAME")
    if missing:
        pytest.skip(
            f"{missing} is not set; skipping sample 07 (toolbox-dependent) test."
        )


# ---------------------------------------------------------------------------
# Subprocess management
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Return an OS-assigned free TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_health(
    base_url: str,
    *,
    proc: subprocess.Popen,
    log_path: Path,
    timeout: float = 90.0,
) -> None:
    """Poll the sample's ``/readiness`` endpoint until it responds 200."""
    deadline = time.monotonic() + timeout
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log_tail = log_path.read_text(
                encoding="utf-8", errors="replace"
            )
            raise RuntimeError(
                "Sample process exited before becoming healthy "
                f"(rc={proc.returncode}).\nLOG:\n{log_tail}"
            )
        try:
            resp = httpx.get(f"{base_url}/readiness", timeout=5.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_err = exc
        time.sleep(0.3)
    raise TimeoutError(
        f"Sample did not become healthy within {timeout}s "
        f"(last error: {last_err!r})."
    )


class SampleServer:
    """Handle for a running sample subprocess + its HTTP base URL."""

    def __init__(
        self,
        proc: subprocess.Popen,
        base_url: str,
        log_path: Path,
        log_handle: Any,
    ) -> None:
        self.proc = proc
        self.base_url = base_url
        self.log_path = log_path
        self._log_handle = log_handle

    @property
    def url(self) -> str:
        return self.base_url

    def terminate(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
        finally:
            try:
                self._log_handle.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass


def _start_sample(
    script: str,
    *,
    extra_env: Optional[dict[str, str]] = None,
    health_timeout: float = 90.0,
) -> SampleServer:
    """Launch a sample script and wait until it serves ``/readiness``.

    Subprocess stdout/stderr are redirected to a file on disk so the OS
    pipe buffer never fills up — long-running samples emit a lot of logs
    once the LLM round-trip starts, and a blocked pipe would otherwise
    stall the server mid-request.
    """
    port = _free_port()
    env = os.environ.copy()
    env["PORT"] = str(port)
    if extra_env:
        env.update(extra_env)
    env.setdefault("PYTHONUNBUFFERED", "1")

    log_fd, log_str = tempfile.mkstemp(
        prefix=f"hosting-sample-{Path(script).stem}-", suffix=".log"
    )
    log_path = Path(log_str)
    log_handle = os.fdopen(log_fd, "w", encoding="utf-8", errors="replace")

    cmd = [sys.executable, script]
    proc = subprocess.Popen(  # noqa: S603 - controlled command list
        cmd,
        cwd=str(SAMPLES_DIR),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(
            base_url, proc=proc, log_path=log_path, timeout=health_timeout
        )
    except Exception:
        # Make sure we don't leak a stuck process on startup failure.
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            log_handle.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
        raise

    return SampleServer(proc, base_url, log_path, log_handle)


@pytest.fixture
def start_sample() -> Iterator[callable]:
    """Pytest fixture yielding a factory for launching sample subprocesses.

    Usage::

        def test_something(start_sample):
            server = start_sample("sample_01_responses_basic.py")
            r = httpx.post(f"{server.url}/responses", json={...})
            ...

    Started servers are torn down automatically at test teardown.
    """
    started: list[SampleServer] = []

    def _factory(
        script: str,
        *,
        extra_env: Optional[dict[str, str]] = None,
        health_timeout: float = 90.0,
    ) -> SampleServer:
        server = _start_sample(
            script, extra_env=extra_env, health_timeout=health_timeout
        )
        started.append(server)
        return server

    try:
        yield _factory
    finally:
        for server in started:
            server.terminate()
