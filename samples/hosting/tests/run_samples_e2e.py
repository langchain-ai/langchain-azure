# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Manual end-to-end runner for every sample under ``samples/hosting/``.

This is **not** an automated test suite — it's a verbose script you run by
hand to sanity-check that every sample boots, talks to a real Azure AI
Foundry project, and produces the documented HTTP behavior. Each check
launches the sample exactly the way its docstring advertises
(``python sample_XX_*.py``) and drives the running process over HTTP the
same way the curl snippets do. ``DefaultAzureCredential`` → Foundry →
real Azure OpenAI deployment → LangGraph → host → HTTP. Nothing is mocked.

Prereqs:

* a working ``az login`` (or any other ``DefaultAzureCredential`` source),
* ``AZURE_AI_PROJECT_ENDPOINT`` set in the shell or in
  ``samples/hosting/.env``,
* optionally ``AZURE_AI_MODEL_DEPLOYMENT_NAME`` (defaults to ``gpt-4o``),
* for sample 07: ``FOUNDRY_AGENT_TOOLBOX_NAME``.

Usage (from ``samples/hosting/``)::

    python run_samples_e2e.py                 # run every check
    python run_samples_e2e.py 01 03           # only samples 01 and 03
    python run_samples_e2e.py --list          # list available checks
    python run_samples_e2e.py --fail-fast     # stop on first failure
    python run_samples_e2e.py --keep-logs     # don't delete sample logs

Each check prints the request it sent, the response it got back, the
assertions it ran, and a final PASS/FAIL line. A summary table is
printed at the very end and the exit code reflects whether any check
failed.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

SAMPLES_DIR = Path(__file__).resolve().parent
load_dotenv(SAMPLES_DIR / ".env")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _hr(char: str = "─", width: int = 78) -> str:
    return char * width


def _section(title: str) -> None:
    print()
    print(_hr("═"))
    print(f"  {title}")
    print(_hr("═"))


def _step(title: str) -> None:
    print()
    print(f"── {title} " + _hr()[len(title) + 4 :])


def _info(msg: str) -> None:
    print(f"  • {msg}")


def _kv(key: str, value: Any) -> None:
    print(f"    {key:<22} {value}")


def _dump_json(label: str, payload: Any, *, limit: int = 1200) -> None:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if len(rendered) > limit:
        rendered = rendered[:limit] + f"\n... [truncated, {len(rendered)} chars total]"
    print(f"    {label}:")
    print(textwrap.indent(rendered, "      "))


def _dump_text(label: str, body: str, *, limit: int = 1200) -> None:
    snippet = body if len(body) <= limit else body[:limit] + f"\n... [truncated, {len(body)} chars total]"
    print(f"    {label}:")
    print(textwrap.indent(snippet, "      "))


# ---------------------------------------------------------------------------
# Environment gating
# ---------------------------------------------------------------------------


class SkipCheck(Exception):
    """Raised by a check to indicate the precondition isn't met."""


def _require_env(*names: str) -> Optional[str]:
    for name in names:
        if not os.environ.get(name):
            return name
    return None


def requires_foundry_endpoint() -> None:
    missing = _require_env("AZURE_AI_PROJECT_ENDPOINT")
    if missing:
        raise SkipCheck(
            f"{missing} is not set. Put it in samples/hosting/.env to enable."
        )


def requires_foundry_toolbox() -> None:
    requires_foundry_endpoint()
    missing = _require_env("FOUNDRY_AGENT_TOOLBOX_NAME")
    if missing:
        raise SkipCheck(
            f"{missing} is not set; sample 07 needs a Foundry Toolbox."
        )


# ---------------------------------------------------------------------------
# Subprocess management
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class SampleServer:
    proc: subprocess.Popen
    base_url: str
    log_path: Path
    _log_handle: Any

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
            except Exception:
                pass


def _wait_for_health(
    base_url: str,
    *,
    proc: subprocess.Popen,
    log_path: Path,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"Sample process exited before becoming healthy (rc={proc.returncode}).\n"
                f"LOG:\n{log_tail}"
            )
        try:
            resp = httpx.get(f"{base_url}/readiness", timeout=5.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_err = exc
        time.sleep(0.3)
    raise TimeoutError(
        f"Sample did not become healthy within {timeout}s (last error: {last_err!r})."
    )


def start_sample(
    script: str,
    *,
    extra_env: Optional[dict[str, str]] = None,
    health_timeout: float = 90.0,
) -> SampleServer:
    """Launch a sample script and wait until it serves ``/readiness``."""
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

    _info(f"launching: python {script} (PORT={port})")
    _info(f"log file:  {log_path}")
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
        _wait_for_health(base_url, proc=proc, log_path=log_path, timeout=health_timeout)
    except Exception:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            log_handle.close()
        except Exception:
            pass
        raise
    _info(f"ready at:  {base_url}")
    return SampleServer(proc=proc, base_url=base_url, log_path=log_path, _log_handle=log_handle)


# ---------------------------------------------------------------------------
# HTTP / SSE helpers
# ---------------------------------------------------------------------------


def _post(
    server: SampleServer,
    path: str,
    *,
    json_body: dict[str, Any],
    stream: bool = False,
    timeout: float = 120.0,
) -> httpx.Response:
    url = f"{server.url}{path}"
    _info(f"POST {path}  (stream={stream})")
    _dump_json("request", json_body)
    if stream:
        with httpx.stream("POST", url, json=json_body, timeout=timeout) as resp:
            body = "".join(resp.iter_text())
        out = httpx.Response(
            status_code=resp.status_code,
            headers=resp.headers,
            content=body.encode("utf-8"),
            request=resp.request,
        )
        _info(f"status: {out.status_code}")
        _dump_text("sse body", body)
        return out
    resp = httpx.post(url, json=json_body, timeout=timeout)
    _info(f"status: {resp.status_code}")
    try:
        _dump_json("response", resp.json())
    except json.JSONDecodeError:
        _dump_text("response", resp.text)
    return resp


def _parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    current_type = ""
    for line in body.splitlines():
        if line.startswith("event:"):
            current_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data = line.split(":", 1)[1].strip()
            if not data:
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                payload = {"raw": data}
            events.append((current_type, payload))
    return events


def _response_text(payload: dict[str, Any]) -> str:
    return "".join(
        part.get("text", "")
        for item in payload.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
    )


def _output_types(payload: dict[str, Any]) -> list[str]:
    return [item.get("type") for item in payload.get("output", [])]


def _assert(cond: bool, message: str) -> None:
    """Assertion helper that prints a clear pass/fail line."""
    if cond:
        print(f"    ✓ {message}")
    else:
        print(f"    ✗ {message}")
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# Checks (one or more per sample). Each takes no args and may raise
# AssertionError on failure or SkipCheck on missing prereqs.
# ---------------------------------------------------------------------------


def check_01_non_streaming() -> None:
    """sample_01: Responses API returns an assistant message."""
    requires_foundry_endpoint()
    server = start_sample("sample_01_responses_basic.py")
    try:
        _step("POST /responses (non-streaming)")
        resp = _post(
            server,
            "/responses",
            json_body={"input": "Say hello in one short sentence.", "model": "gpt-4o"},
        )
        _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        payload = resp.json()
        _assert(payload["status"] == "completed", "status == completed")
        _assert("message" in _output_types(payload), "output contains a `message` item")
        _assert(bool(_response_text(payload).strip()), "assistant text is non-empty")
    finally:
        server.terminate()


def check_01_streaming() -> None:
    """sample_01: Responses API stream emits response.created + response.completed."""
    requires_foundry_endpoint()
    server = start_sample("sample_01_responses_basic.py")
    try:
        _step("POST /responses (stream=True)")
        resp = _post(
            server,
            "/responses",
            json_body={"input": "Say hi.", "model": "gpt-4o", "stream": True},
            stream=True,
        )
        _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        types = [t for t, _ in _parse_sse(resp.text)]
        _assert("response.created" in types, "stream contains response.created")
        _assert("response.completed" in types, "stream contains response.completed")
    finally:
        server.terminate()


def check_02_non_streaming() -> None:
    """sample_02: Responses API surfaces function_call + function_call_output."""
    requires_foundry_endpoint()
    server = start_sample("sample_02_responses_tools.py")
    try:
        _step("POST /responses (non-streaming, tool round-trip)")
        resp = _post(
            server,
            "/responses",
            json_body={"input": "What is the weather in Seattle?", "model": "gpt-4o"},
        )
        _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        payload = resp.json()
        types = _output_types(payload)
        _assert("function_call" in types, "output contains a `function_call` item")
        _assert(
            "function_call_output" in types,
            "output contains a `function_call_output` item",
        )
        _assert("message" in types, "output contains a `message` item")
    finally:
        server.terminate()


def check_02_streaming() -> None:
    """sample_02: streamed run emits at least one function_call output item."""
    requires_foundry_endpoint()
    server = start_sample("sample_02_responses_tools.py")
    try:
        _step("POST /responses (stream=True, tool round-trip)")
        resp = _post(
            server,
            "/responses",
            json_body={
                "input": "What is the weather in Tokyo?",
                "model": "gpt-4o",
                "stream": True,
            },
            stream=True,
        )
        _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        events = _parse_sse(resp.text)
        types = [t for t, _ in events]
        _assert("response.created" in types, "stream contains response.created")
        _assert("response.completed" in types, "stream contains response.completed")
        function_call_items = [
            payload
            for event_type, payload in events
            if event_type == "response.output_item.done"
            and payload.get("item", {}).get("type") == "function_call"
        ]
        _assert(bool(function_call_items), "stream surfaces ≥1 function_call output item")
    finally:
        server.terminate()


def check_03_multi_turn() -> None:
    """sample_03: Invocations API + MemorySaver remembers across turns."""
    requires_foundry_endpoint()
    server = start_sample("sample_03_invocations_basic.py")
    try:
        _step("POST /invocations (turn 1 — establish a fact)")
        r1 = _post(
            server,
            "/invocations",
            json_body={"message": "My name is Alice. Just say OK."},
        )
        _assert(r1.status_code == 200, f"turn 1 HTTP 200 (got {r1.status_code})")
        session_id = r1.headers.get("x-agent-session-id")
        _kv("x-agent-session-id", session_id)
        _assert(bool(session_id), "server returned x-agent-session-id")

        _step("POST /invocations (turn 2 — recall the fact)")
        r2 = _post(
            server,
            f"/invocations?agent_session_id={session_id}",
            json_body={"message": "What is my name?"},
        )
        _assert(r2.status_code == 200, f"turn 2 HTTP 200 (got {r2.status_code})")
        body = r2.json()
        _assert("response" in body, "response body has a `response` field")
        _assert("alice" in body["response"].lower(), "model recalls the name `alice`")
    finally:
        server.terminate()


def check_03_streaming() -> None:
    """sample_03: streaming emits token frames and a final `done` event."""
    requires_foundry_endpoint()
    server = start_sample("sample_03_invocations_basic.py")
    try:
        _step("POST /invocations (stream=True)")
        resp = _post(
            server,
            "/invocations",
            json_body={"message": "Count from 1 to 3.", "stream": True},
            stream=True,
        )
        _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        tokens: list[str] = []
        saw_done = False
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
                if not data or data == "{}":
                    continue
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "token" in payload:
                    tokens.append(payload["token"])
            elif line.startswith("event:") and line.split(":", 1)[1].strip() == "done":
                saw_done = True
        _kv("tokens received", len(tokens))
        _assert(bool(tokens), "stream produced ≥1 token frame")
        _assert(saw_done, "stream ended with an `event: done`")
    finally:
        server.terminate()


def check_04_final_text() -> None:
    """sample_04: Invocations API runs the tool server-side, returns final text."""
    requires_foundry_endpoint()
    server = start_sample("sample_04_invocations_tools.py")
    try:
        _step("POST /invocations (tool runs server-side)")
        resp = _post(
            server,
            "/invocations",
            json_body={"message": "What is the weather in Seattle?"},
        )
        _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        body = resp.json()
        _assert(set(body.keys()) == {"response"}, "body has exactly `response` key")
        _assert("seattle" in body["response"].lower(), "final text mentions seattle")
    finally:
        server.terminate()


def check_05_responses_tool_round_trip() -> None:
    """sample_05: dual host — /responses surfaces tool round-trip items."""
    requires_foundry_endpoint()
    server = start_sample("sample_05_workflow_all_in_one.py")
    try:
        _step("POST /responses (tool round-trip)")
        resp = _post(
            server,
            "/responses",
            json_body={"input": "What is the weather in Seattle?", "model": "gpt-4o"},
        )
        _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        payload = resp.json()
        types = _output_types(payload)
        _assert("function_call" in types, "output contains a `function_call` item")
        _assert("function_call_output" in types, "output contains `function_call_output`")
        _assert("message" in types, "output contains a `message` item")
    finally:
        server.terminate()


def check_05_invocations_returns_42() -> None:
    """sample_05: dual host — /invocations runs `add(17, 25)` and surfaces 42."""
    requires_foundry_endpoint()
    server = start_sample("sample_05_workflow_all_in_one.py")
    try:
        _step("POST /invocations (add tool → 42)")
        resp = _post(
            server,
            "/invocations",
            json_body={"message": "What is 17 plus 25?"},
        )
        _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        body = resp.json()
        _assert("response" in body, "body has a `response` field")
        _assert("42" in body["response"], "final text contains `42`")
    finally:
        server.terminate()


def check_06_pause_then_resume() -> None:
    """sample_06: HITL — interrupt sentinel + resume via function_call_output."""
    requires_foundry_endpoint()
    server = start_sample("sample_06_responses_hitl.py")
    try:
        conversation_id = f"e2e-hitl-{uuid.uuid4().hex[:8]}"
        _kv("conversation.id", conversation_id)

        _step("POST /responses (turn 1 — expect interrupt)")
        first = _post(
            server,
            "/responses",
            json_body={
                "input": "Ask me where I am, then look up the weather there.",
                "conversation": {"id": conversation_id},
            },
            timeout=180.0,
        )
        _assert(first.status_code == 200, f"turn 1 HTTP 200 (got {first.status_code})")
        first_payload = first.json()
        _assert(first_payload["status"] == "completed", "turn 1 status == completed")
        interrupts = [
            item
            for item in first_payload["output"]
            if item.get("type") == "function_call"
            and item.get("name") == "__hosted_agent_adapter_interrupt__"
        ]
        _assert(bool(interrupts), "turn 1 surfaces interrupt sentinel function_call")
        call_id = interrupts[0]["call_id"]
        _kv("interrupt call_id", call_id)
        _assert(bool(call_id), "interrupt call_id is non-empty")

        _step("POST /responses (turn 2 — resume with Seattle)")
        second = _post(
            server,
            "/responses",
            json_body={
                "conversation": {"id": conversation_id},
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps({"resume": "Seattle"}),
                    }
                ],
            },
            timeout=300.0,
        )
        _assert(second.status_code == 200, f"turn 2 HTTP 200 (got {second.status_code})")
        second_payload = second.json()
        _assert(second_payload["status"] == "completed", "turn 2 status == completed")
        new_interrupts = [
            it
            for it in second_payload["output"]
            if it.get("type") == "function_call"
            and it.get("name") == "__hosted_agent_adapter_interrupt__"
        ]
        _assert(not new_interrupts, "turn 2 does not re-interrupt")
        _assert(
            "seattle" in _response_text(second_payload).lower(),
            "turn 2 final text mentions seattle",
        )
    finally:
        server.terminate()


def check_07_toolbox() -> None:
    """sample_07: Foundry Toolbox-backed agent serves /responses."""
    requires_foundry_toolbox()
    server = start_sample("sample_07_responses_toolbox.py", health_timeout=180.0)
    try:
        _step("POST /responses (toolbox-loaded agent)")
        resp = _post(
            server,
            "/responses",
            json_body={
                "input": "What tools do you have available?",
                "model": "gpt-4o",
            },
            timeout=180.0,
        )
        _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        payload = resp.json()
        _assert(payload["status"] == "completed", "status == completed")
        _assert(bool(_response_text(payload).strip()), "assistant text is non-empty")
    finally:
        server.terminate()


def check_health_endpoint(script: str) -> Callable[[], None]:
    """Build a check that boots ``script`` and re-hits /readiness fast."""

    def _check() -> None:
        requires_foundry_endpoint()
        server = start_sample(script)
        try:
            _step("GET /readiness (steady state)")
            start = time.monotonic()
            resp = httpx.get(f"{server.url}/readiness", timeout=5.0)
            elapsed = time.monotonic() - start
            _kv("elapsed", f"{elapsed * 1000:.0f} ms")
            _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
            _assert(elapsed < 5.0, "/readiness responds in < 5s")
        finally:
            server.terminate()

    _check.__doc__ = f"{script}: /readiness responds 200 in steady state."
    return _check


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class Check:
    sample_id: str  # e.g. "01"
    name: str       # short identifier
    fn: Callable[[], None]
    description: str
    server: Optional[SampleServer] = field(default=None, repr=False)

    @property
    def full_name(self) -> str:
        return f"sample_{self.sample_id}::{self.name}"


def _build_registry() -> list[Check]:
    items: list[Check] = [
        Check("01", "non_streaming", check_01_non_streaming, check_01_non_streaming.__doc__ or ""),
        Check("01", "streaming", check_01_streaming, check_01_streaming.__doc__ or ""),
        Check("02", "non_streaming", check_02_non_streaming, check_02_non_streaming.__doc__ or ""),
        Check("02", "streaming", check_02_streaming, check_02_streaming.__doc__ or ""),
        Check("03", "multi_turn", check_03_multi_turn, check_03_multi_turn.__doc__ or ""),
        Check("03", "streaming", check_03_streaming, check_03_streaming.__doc__ or ""),
        Check("04", "final_text", check_04_final_text, check_04_final_text.__doc__ or ""),
        Check("05", "responses_tool_round_trip", check_05_responses_tool_round_trip, check_05_responses_tool_round_trip.__doc__ or ""),
        Check("05", "invocations_returns_42", check_05_invocations_returns_42, check_05_invocations_returns_42.__doc__ or ""),
        Check("06", "pause_then_resume", check_06_pause_then_resume, check_06_pause_then_resume.__doc__ or ""),
        Check("07", "toolbox", check_07_toolbox, check_07_toolbox.__doc__ or ""),
    ]

    health_samples = [
        ("01", "sample_01_responses_basic.py"),
        ("02", "sample_02_responses_tools.py"),
        ("03", "sample_03_invocations_basic.py"),
        ("04", "sample_04_invocations_tools.py"),
        ("05", "sample_05_workflow_all_in_one.py"),
        ("06", "sample_06_responses_hitl.py"),
    ]
    for sid, script in health_samples:
        items.append(
            Check(sid, "readiness", check_health_endpoint(script), f"{script}: /readiness 200 in <5s")
        )
    return items


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class Result:
    check: Check
    status: str            # "pass", "fail", "skip"
    elapsed: float
    detail: str = ""


def _run_check(check: Check) -> Result:
    _section(f"{check.full_name}  —  {check.description.strip()}")
    started = time.monotonic()
    try:
        check.fn()
    except SkipCheck as exc:
        elapsed = time.monotonic() - started
        print(f"\n  ⏭  SKIP  ({elapsed:.1f}s): {exc}")
        return Result(check, "skip", elapsed, str(exc))
    except AssertionError as exc:
        elapsed = time.monotonic() - started
        print(f"\n  ✗ FAIL  ({elapsed:.1f}s): {exc}")
        return Result(check, "fail", elapsed, str(exc))
    except Exception as exc:  # noqa: BLE001 - runner needs to catch all
        elapsed = time.monotonic() - started
        print(f"\n  ✗ ERROR ({elapsed:.1f}s): {exc.__class__.__name__}: {exc}")
        traceback.print_exc()
        return Result(check, "fail", elapsed, f"{exc.__class__.__name__}: {exc}")
    else:
        elapsed = time.monotonic() - started
        print(f"\n  ✓ PASS  ({elapsed:.1f}s)")
        return Result(check, "pass", elapsed)


def _filter(registry: list[Check], selectors: list[str]) -> list[Check]:
    if not selectors:
        return registry
    wanted_ids = {s.lstrip("0") or "0" for s in selectors}
    matched: list[Check] = []
    for chk in registry:
        if chk.sample_id.lstrip("0") in wanted_ids or chk.sample_id in selectors:
            matched.append(chk)
    return matched


def _print_summary(results: list[Result]) -> None:
    _section("Summary")
    width = max(len(r.check.full_name) for r in results) if results else 20
    for r in results:
        icon = {"pass": "✓", "fail": "✗", "skip": "⏭"}[r.status]
        print(f"  {icon} {r.status.upper():<5} {r.check.full_name:<{width}}  ({r.elapsed:.1f}s)")
        if r.detail and r.status != "pass":
            print(f"      {r.detail}")
    n_pass = sum(1 for r in results if r.status == "pass")
    n_fail = sum(1 for r in results if r.status == "fail")
    n_skip = sum(1 for r in results if r.status == "skip")
    print()
    print(f"  Totals: {n_pass} passed, {n_fail} failed, {n_skip} skipped ({len(results)} total)")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manual end-to-end runner for samples/hosting samples."
    )
    parser.add_argument(
        "samples",
        nargs="*",
        help="Sample IDs to run (e.g. 01 03). Default: every check.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available checks and exit."
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failing check.",
    )
    parser.add_argument(
        "--keep-logs",
        action="store_true",
        help="Do not delete sample log files on success.",
    )
    args = parser.parse_args(argv)

    registry = _build_registry()
    if args.list:
        print("Available checks:")
        for chk in registry:
            print(f"  {chk.full_name:<45} {chk.description.strip()}")
        return 0

    selected = _filter(registry, args.samples)
    if not selected:
        print(f"No checks match {args.samples!r}. Try --list.")
        return 2

    _section("Run plan")
    print(f"  Working dir : {SAMPLES_DIR}")
    print(f"  Python      : {sys.executable}")
    print(f"  Checks      : {len(selected)}")
    for chk in selected:
        print(f"    - {chk.full_name}")
    print(f"  AZURE_AI_PROJECT_ENDPOINT  set: {bool(os.environ.get('AZURE_AI_PROJECT_ENDPOINT'))}")
    print(f"  FOUNDRY_AGENT_TOOLBOX_NAME set: {bool(os.environ.get('FOUNDRY_AGENT_TOOLBOX_NAME'))}")

    results: list[Result] = []
    for chk in selected:
        result = _run_check(chk)
        results.append(result)
        if result.status == "fail" and args.fail_fast:
            print("\n  (--fail-fast set: stopping)")
            break

    _print_summary(results)
    return 0 if not any(r.status == "fail" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
