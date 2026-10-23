# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Manual end-to-end runner for every sample under
``samples/hosting/langgraph-hosted-agents/``.

This is **not** an automated test suite — it's a verbose script you run by
hand to sanity-check that every sample boots, talks to a real Azure AI
Foundry project, and produces the documented HTTP behavior. Each check
launches the sample exactly the way its README advertises
(``python <folder>/main.py``) and drives the running process over HTTP the
same way the curl snippets do. ``DefaultAzureCredential`` → Foundry →
real Azure OpenAI deployment → LangGraph → host → HTTP. Nothing is mocked.

Prereqs:

* a working ``az login`` (or any other ``DefaultAzureCredential`` source),
* ``FOUNDRY_PROJECT_ENDPOINT`` set in the shell or in
  ``samples/hosting/langgraph-hosted-agents/.env``,
* optionally ``AZURE_AI_MODEL_DEPLOYMENT_NAME`` (defaults to ``gpt-4o``),
* for the Foundry-Toolbox and local Deep Agents samples: ``TOOLBOX_NAME``;
* optionally ``DEEP_AGENTS_NAME`` to validate an already-deployed Deep Agent
  instead of launching its local host.

Usage (from ``samples/hosting/langgraph-hosted-agents/``)::

    python tests/run_samples_e2e.py                 # run every check
    python tests/run_samples_e2e.py 01              # every sample whose folder starts with 01
    python tests/run_samples_e2e.py responses/04    # only responses/04_foundry_toolbox
    python tests/run_samples_e2e.py .\responses\04_foundry_toolbox\
    python tests/run_samples_e2e.py invocations/01_basic
    python tests/run_samples_e2e.py --list          # list available checks
    python tests/run_samples_e2e.py --fail-fast     # stop on first failure
    python tests/run_samples_e2e.py --keep-logs     # don't delete sample logs

Each check prints the request it sent, the response it got back, the
assertions it ran, and a final PASS/FAIL line. A summary table is
printed at the very end and the exit code reflects whether any check
failed. A markdown report is always written to the current directory
for sharing/PR comments, including formatted API request/response payloads
for troubleshooting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

SAMPLES_DIR = Path(__file__).resolve().parent.parent
load_dotenv()


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
    global _CURRENT_STEP_TITLE

    _CURRENT_STEP_TITLE = title
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


def _sse_data_text(line: str) -> str:
    data = line.split(":", 1)[1]
    return data[1:] if data.startswith(" ") else data


def _iter_sse_events(body: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    current_type = "message"
    data_lines: list[str] = []

    for line in [*body.splitlines(), ""]:
        if line == "":
            if data_lines:
                events.append((current_type, "\n".join(data_lines)))
            current_type = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            current_type = line.split(":", 1)[1].strip() or "message"
        elif line.startswith("data:"):
            data_lines.append(_sse_data_text(line))

    return events


def _format_response_body(body: str, *, stream: bool) -> tuple[str, str]:
    if not body:
        return "text", ""

    if stream:
        events = _iter_sse_events(body)
        if not events:
            return "text", body
        lines: list[str] = []
        for idx, (event_type, data) in enumerate(events, start=1):
            lines.append(f"[{idx}] event: {event_type}")
            if data:
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    lines.append(textwrap.indent(data, "  data: "))
                else:
                    lines.append("  data:")
                    lines.append(textwrap.indent(_json_text(payload), "    "))
            lines.append("")
        return "text", "\n".join(lines).rstrip()

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return "text", body
    return "json", _json_text(payload)


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
    missing = _require_env("FOUNDRY_PROJECT_ENDPOINT")
    if missing:
        raise SkipCheck(
            f"{missing} is not set. Put it in "
            f"samples/hosting/langgraph-hosted-agents/.env to enable."
        )


def requires_foundry_toolbox() -> None:
    requires_foundry_endpoint()
    missing = _require_env("TOOLBOX_NAME")
    if missing:
        raise SkipCheck(
            f"{missing} is not set; the Foundry Toolbox sample needs it."
        )


def requires_github_pat() -> None:
    requires_foundry_endpoint()
    missing = _require_env("GITHUB_PAT")
    if missing:
        raise SkipCheck(
            f"{missing} is not set; the remote MCP sample needs a GitHub token."
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


@dataclass
class RawApiExchange:
    check_name: str
    step_title: str
    ordinal: int
    method: str
    url: str
    path: str
    stream: bool
    timeout: float
    request_json: dict[str, Any]
    response_status: Optional[int] = None
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body: str = ""
    error: str = ""


_CURRENT_CHECK_NAME = ""
_CURRENT_STEP_TITLE = ""
_RAW_API_EXCHANGES: list[RawApiExchange] = []


def _start_raw_api_exchange(
    *,
    method: str,
    url: str,
    path: str,
    stream: bool,
    timeout: float,
    request_json: dict[str, Any],
) -> RawApiExchange:
    exchange = RawApiExchange(
        check_name=_CURRENT_CHECK_NAME or "<unknown check>",
        step_title=_CURRENT_STEP_TITLE,
        ordinal=len(_RAW_API_EXCHANGES) + 1,
        method=method,
        url=url,
        path=path,
        stream=stream,
        timeout=timeout,
        request_json=request_json,
    )
    _RAW_API_EXCHANGES.append(exchange)
    return exchange


def _post(
    server: SampleServer,
    path: str,
    *,
    json_body: dict[str, Any],
    stream: bool = False,
    timeout: float = 120.0,
) -> httpx.Response:
    url = f"{server.url}{path}"
    exchange = _start_raw_api_exchange(
        method="POST",
        url=url,
        path=path,
        stream=stream,
        timeout=timeout,
        request_json=json_body,
    )
    _info(f"POST {path}  (stream={stream})")
    _dump_json("request", json_body)
    if stream:
        try:
            with httpx.stream("POST", url, json=json_body, timeout=timeout) as resp:
                body = "".join(resp.iter_text())
                exchange.response_status = resp.status_code
                exchange.response_headers = dict(resp.headers)
                exchange.response_body = body
        except Exception as exc:
            exchange.error = f"{exc.__class__.__name__}: {exc}"
            raise
        out = httpx.Response(
            status_code=resp.status_code,
            headers=resp.headers,
            content=body.encode("utf-8"),
            request=resp.request,
        )
        _info(f"status: {out.status_code}")
        _, formatted_body = _format_response_body(body, stream=True)
        _dump_text("sse events", formatted_body, limit=2400)
        return out
    try:
        resp = httpx.post(url, json=json_body, timeout=timeout)
        exchange.response_status = resp.status_code
        exchange.response_headers = dict(resp.headers)
        exchange.response_body = resp.text
    except Exception as exc:
        exchange.error = f"{exc.__class__.__name__}: {exc}"
        raise
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


def _tool_call_names(payload: dict[str, Any]) -> list[str]:
    return [
        item.get("name", "")
        for item in payload.get("output", [])
        if item.get("type") == "function_call"
    ]


def _loaded_tool_names(log_path: Path) -> list[str]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tool_names: list[str] = []
    collecting = False
    for line in lines:
        if (
            "tool(s) from Foundry toolbox" in line
            or "tool(s) from MCP server" in line
        ):
            collecting = True
            continue
        if not collecting:
            continue

        stripped = line.strip()
        if stripped.startswith("- "):
            tool_names.append(stripped[2:])
            continue
        if stripped:
            break
    return tool_names


def _toolbox_prompt(tool_names: list[str]) -> str:
    lower_names = {name.lower(): name for name in tool_names}
    if "code_interpreter" in lower_names:
        return (
            "You must use the code_interpreter tool to calculate 17 * 25. "
            "Do not do the arithmetic yourself. After the tool call, reply with "
            "the result in one short sentence."
        )
    if "web_search" in lower_names:
        return (
            "You must use the web_search tool to look up the current weather in "
            "Seattle. Do not answer from memory. After the tool call, summarize "
            "the result in one short sentence."
        )

    tool_name = tool_names[0]
    return (
        f"You must use the {tool_name} tool at least once before answering. "
        "Do not answer from memory. After the tool call, briefly say which tool "
        "you used and what it returned."
    )


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
    """responses/01_basic: Responses API returns an assistant message."""
    requires_foundry_endpoint()
    server = start_sample("responses/01_basic/main.py")
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
    """responses/01_basic: Responses API stream emits response.created + response.completed."""
    requires_foundry_endpoint()
    server = start_sample("responses/01_basic/main.py")
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
    """responses/02_tools: Responses API surfaces function_call + function_call_output."""
    requires_foundry_endpoint()
    server = start_sample("responses/02_tools/main.py")
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
    """responses/02_tools: streamed run emits at least one function_call output item."""
    requires_foundry_endpoint()
    server = start_sample("responses/02_tools/main.py")
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


def check_03_remote_mcp() -> None:
    """responses/03_mcp: remote MCP tools are loaded and invoked."""
    requires_github_pat()
    server = start_sample("responses/03_mcp/main.py", health_timeout=180.0)
    try:
        tool_names = _loaded_tool_names(server.log_path)
        _kv("MCP tools", ", ".join(tool_names) or "<none>")
        _assert(bool(tool_names), "startup log lists ≥1 remote MCP tool")

        _step("POST /responses (remote MCP tool round-trip)")
        resp = _post(
            server,
            "/responses",
            json_body={
                "input": (
                    "Use a GitHub MCP tool to search GitHub for public issues "
                    "mentioning langchain-azure. After the tool call, summarize "
                    "one result in one short sentence."
                ),
                "model": "gpt-4o",
            },
            timeout=180.0,
        )
        _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        payload = resp.json()
        _assert(payload["status"] == "completed", "status == completed")
        types = _output_types(payload)
        _assert("function_call" in types, "output contains a `function_call` item")
        _assert(
            "function_call_output" in types,
            "output contains a `function_call_output` item",
        )
        _assert(
            any(name in tool_names for name in _tool_call_names(payload)),
            "response invoked one of the loaded MCP tools",
        )
        _assert(bool(_response_text(payload).strip()), "assistant text is non-empty")
    finally:
        server.terminate()


def check_03_multi_turn() -> None:
    """invocations/01_basic: Invocations API + MemorySaver remembers across turns."""
    requires_foundry_endpoint()
    server = start_sample("invocations/01_basic/main.py")
    try:
        turn1_request = {"message": "My name is Alice. Just say OK."}
        turn2_request = {"message": "What is my name?"}

        _step("POST /invocations (turn 1 — establish a fact)")
        r1 = _post(
            server,
            "/invocations",
            json_body=turn1_request,
        )
        _assert(r1.status_code == 200, f"turn 1 HTTP 200 (got {r1.status_code})")
        session_id = r1.headers.get("x-agent-session-id")
        _kv("x-agent-session-id", session_id)
        _assert(bool(session_id), "server returned x-agent-session-id")

        _step("POST /invocations (turn 2 — recall the fact)")
        r2 = _post(
            server,
            f"/invocations?agent_session_id={session_id}",
            json_body=turn2_request,
        )
        _assert(r2.status_code == 200, f"turn 2 HTTP 200 (got {r2.status_code})")
        body = r2.json()
        _assert("response" in body, "response body has a `response` field")
        _assert("alice" in body["response"].lower(), "model recalls the name `alice`")

        check_name = _CURRENT_CHECK_NAME or "<unknown check>"
        transcript_exchanges = [
            exchange
            for exchange in _RAW_API_EXCHANGES
            if exchange.check_name == check_name
        ]
        _assert(
            len(transcript_exchanges) >= 2,
            "raw API transcript captured both multi-turn requests",
        )
        turn1_exchange, turn2_exchange = transcript_exchanges[-2:]
        _assert(
            turn1_exchange.path == "/invocations",
            "raw API transcript captured turn 1 path",
        )
        _assert(
            turn1_exchange.request_json == turn1_request,
            "raw API transcript captured turn 1 request body",
        )
        _assert(
            turn1_exchange.response_headers.get("x-agent-session-id") == session_id,
            "raw API transcript captured turn 1 session header",
        )
        _assert(
            json.loads(turn1_exchange.response_body) == r1.json(),
            "raw API transcript captured turn 1 response body",
        )
        _assert(
            turn2_exchange.path == f"/invocations?agent_session_id={session_id}",
            "raw API transcript captured turn 2 session query",
        )
        _assert(
            turn2_exchange.request_json == turn2_request,
            "raw API transcript captured turn 2 request body",
        )
        _assert(
            turn2_exchange.response_headers.get("x-agent-session-id") == session_id,
            "raw API transcript captured reused session header",
        )
        _assert(
            json.loads(turn2_exchange.response_body) == body,
            "raw API transcript captured turn 2 recalled response",
        )
    finally:
        server.terminate()


def check_03_streaming() -> None:
    """invocations/01_basic: streaming emits token frames and a final `done` event."""
    requires_foundry_endpoint()
    server = start_sample("invocations/01_basic/main.py")
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
    """invocations/02_tools: Invocations API runs the tool server-side, returns final text."""
    requires_foundry_endpoint()
    server = start_sample("invocations/02_tools/main.py")
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
    """responses/05_workflows: /responses surfaces tool round-trip items."""
    requires_foundry_endpoint()
    server = start_sample("responses/05_workflows/main.py")
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


def check_06_files() -> None:
    """responses/06_files: model reads the code from an inline PDF attachment."""
    requires_foundry_endpoint()
    request = json.loads(
        (SAMPLES_DIR / "responses/06_files/request.json").read_text(encoding="utf-8")
    )
    server = start_sample("responses/06_files/main.py")
    try:
        _step("POST /responses (read an inline PDF attachment)")
        resp = _post(server, "/responses", json_body=request, timeout=180.0)
        _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        payload = resp.json()
        _assert(payload["status"] == "completed", "status == completed")
        _assert(
            "ORCHID-4827" in _response_text(payload),
            "assistant reads the verification code from the PDF, not the prompt",
        )
    finally:
        server.terminate()


def check_07_observability() -> None:
    """responses/07_observability: traced Responses host returns assistant text."""
    requires_foundry_endpoint()
    server = start_sample(
        "responses/07_observability/main.py",
        extra_env={"AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED": "true"},
    )
    try:
        _step("POST /responses (observability sample response)")
        resp = _post(
            server,
            "/responses",
            json_body={
                "input": "Tell me a fun fact about distributed tracing.",
                "model": "gpt-4o",
            },
            timeout=180.0,
        )
        _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        payload = resp.json()
        _assert(payload["status"] == "completed", "status == completed")
        _assert("message" in _output_types(payload), "output contains a `message` item")
        _assert(bool(_response_text(payload).strip()), "assistant text is non-empty")
    finally:
        server.terminate()


def check_06_pause_then_resume() -> None:
    """responses/08_hitl: approval-style HITL — mcp_approval_request paired
    with function_call; primary resume via mcp_approval_response; rejection
    surfaces as response.failed; rich override still works via
    function_call_output.
    """
    requires_foundry_endpoint()
    server = start_sample("responses/08_hitl/main.py")
    try:
        # ------------------------------------------------------------------
        # Run A (primary path): approve via the OpenAI-standard
        # mcp_approval_response. This is the headline UX.
        # ------------------------------------------------------------------
        conversation_a = f"e2e-hitl-approve-{uuid.uuid4().hex[:8]}"
        _kv("conversation.id (Run A — approve)", conversation_a)

        _step("Run A turn 1 — POST /responses (expect paired approval items)")
        first = _post(
            server,
            "/responses",
            json_body={
                "input": "What is the weather in Seattle?",
                "conversation": {"id": conversation_a},
            },
            timeout=180.0,
        )
        _assert(first.status_code == 200, f"turn 1 HTTP 200 (got {first.status_code})")
        first_payload = first.json()
        _assert(first_payload["status"] == "completed", "turn 1 status == completed")
        approvals = [
            item
            for item in first_payload["output"]
            if item.get("type") == "mcp_approval_request"
            and item.get("name") == "__hosted_agent_adapter_interrupt__"
        ]
        interrupts = [
            item
            for item in first_payload["output"]
            if item.get("type") == "function_call"
            and item.get("name") == "__hosted_agent_adapter_interrupt__"
        ]
        _assert(
            bool(approvals),
            "turn 1 surfaces the OpenAI-standard mcp_approval_request item",
        )
        _assert(
            bool(interrupts),
            "turn 1 also surfaces the paired function_call (advanced channel)",
        )
        approval_id = approvals[0]["id"]
        call_id = interrupts[0]["call_id"]
        _kv("mcp_approval_request.id", approval_id)
        _kv("function_call.call_id", call_id)
        _assert(
            approval_id == call_id,
            "mcp_approval_request.id == function_call.call_id (same interrupt id)",
        )
        # Envelope shape: arguments JSON describes the proposed tool call.
        try:
            envelope = json.loads(approvals[0]["arguments"])
        except (TypeError, ValueError):
            envelope = None
        _assert(
            isinstance(envelope, dict)
            and envelope.get("interrupt_id") == approval_id
            and isinstance(envelope.get("value"), dict)
            and envelope["value"].get("tool") == "get_weather",
            "approval arguments envelope describes the proposed get_weather call",
        )

        _step("Run A turn 2 — approve via mcp_approval_response")
        second = _post(
            server,
            "/responses",
            json_body={
                "conversation": {"id": conversation_a},
                "input": [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": approval_id,
                        "approve": True,
                    }
                ],
            },
            timeout=300.0,
        )
        _assert(second.status_code == 200, f"turn 2 HTTP 200 (got {second.status_code})")
        second_payload = second.json()
        _assert(second_payload["status"] == "completed", "turn 2 status == completed")
        _assert(
            not [
                it
                for it in second_payload["output"]
                if it.get("type")
                in ("function_call", "mcp_approval_request")
                and it.get("name") == "__hosted_agent_adapter_interrupt__"
            ],
            "turn 2 does not re-interrupt after approve=true",
        )
        _assert(
            "seattle" in _response_text(second_payload).lower(),
            "turn 2 final text mentions seattle (tool ran with approved args)",
        )

        # ------------------------------------------------------------------
        # Run B: reject via mcp_approval_response — the host should
        # surface response.failed(code="interrupt_rejected").
        # ------------------------------------------------------------------
        conversation_b = f"e2e-hitl-reject-{uuid.uuid4().hex[:8]}"
        _kv("conversation.id (Run B — reject)", conversation_b)

        _step("Run B turn 1 — POST /responses (expect approval prompt)")
        firstb = _post(
            server,
            "/responses",
            json_body={
                "input": "What is the weather in Seattle?",
                "conversation": {"id": conversation_b},
            },
            timeout=180.0,
        )
        _assert(
            firstb.status_code == 200,
            f"Run B turn 1 HTTP 200 (got {firstb.status_code})",
        )
        approvals_b = [
            it
            for it in firstb.json()["output"]
            if it.get("type") == "mcp_approval_request"
        ]
        _assert(bool(approvals_b), "Run B turn 1 surfaces mcp_approval_request")
        approval_id_b = approvals_b[0]["id"]

        _step("Run B turn 2 — reject via mcp_approval_response")
        secondb = _post(
            server,
            "/responses",
            json_body={
                "conversation": {"id": conversation_b},
                "input": [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": approval_id_b,
                        "approve": False,
                        "reason": "automated test rejection",
                    }
                ],
            },
            timeout=120.0,
        )
        _assert(
            secondb.status_code == 200,
            f"Run B turn 2 HTTP 200 (got {secondb.status_code})",
        )
        secondb_payload = secondb.json()
        _assert(
            secondb_payload["status"] == "failed",
            f"Run B turn 2 status == failed (got {secondb_payload.get('status')!r})",
        )
        err = secondb_payload.get("error") or {}
        _assert(
            err.get("code") == "interrupt_rejected",
            f"error.code == interrupt_rejected (got {err.get('code')!r})",
        )
        _assert(
            approval_id_b in (err.get("message") or ""),
            "rejection message references the interrupt id",
        )

        # ------------------------------------------------------------------
        # Run C (advanced): override the proposed tool args via
        # function_call_output (richer channel for non-vanilla clients).
        # ------------------------------------------------------------------
        conversation_c = f"e2e-hitl-override-{uuid.uuid4().hex[:8]}"
        _kv("conversation.id (Run C — function_call_output override)", conversation_c)

        _step("Run C turn 1 — POST /responses (expect approval prompt)")
        firstc = _post(
            server,
            "/responses",
            json_body={
                "input": "What is the weather in Seattle?",
                "conversation": {"id": conversation_c},
            },
            timeout=180.0,
        )
        _assert(
            firstc.status_code == 200,
            f"Run C turn 1 HTTP 200 (got {firstc.status_code})",
        )
        interrupts_c = [
            it
            for it in firstc.json()["output"]
            if it.get("type") == "function_call"
            and it.get("name") == "__hosted_agent_adapter_interrupt__"
        ]
        _assert(bool(interrupts_c), "Run C turn 1 surfaces sentinel function_call")
        call_id_c = interrupts_c[0]["call_id"]

        _step("Run C turn 2 — override args via function_call_output")
        override_payload = {
            "resume": {
                "tool": "get_weather",
                "arguments": {"location": "Vancouver"},
            }
        }
        secondc = _post(
            server,
            "/responses",
            json_body={
                "conversation": {"id": conversation_c},
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call_id_c,
                        "output": json.dumps(override_payload),
                    }
                ],
            },
            timeout=300.0,
        )
        _assert(
            secondc.status_code == 200,
            f"Run C turn 2 HTTP 200 (got {secondc.status_code})",
        )
        secondc_payload = secondc.json()
        _assert(
            secondc_payload["status"] == "completed",
            "Run C turn 2 status == completed",
        )
        _assert(
            "vancouver" in _response_text(secondc_payload).lower(),
            "Run C final text reflects the overridden location (Vancouver)",
        )
    finally:
        server.terminate()


def check_07_toolbox() -> None:
    """responses/04_foundry_toolbox: Foundry Toolbox-backed agent serves /responses."""
    requires_foundry_toolbox()
    server = start_sample("responses/04_foundry_toolbox/main.py", health_timeout=180.0)
    try:
        tool_names = _loaded_tool_names(server.log_path)
        _kv("toolbox tools", ", ".join(tool_names) or "<none>")
        _assert(bool(tool_names), "startup log lists ≥1 toolbox tool")

        _step("POST /responses (toolbox-loaded agent, expect a real tool call)")
        resp = _post(
            server,
            "/responses",
            json_body={
                "input": _toolbox_prompt(tool_names),
                "model": "gpt-4o",
            },
            timeout=180.0,
        )
        _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        payload = resp.json()
        _assert(payload["status"] == "completed", "status == completed")
        types = _output_types(payload)
        _assert("function_call" in types, "output contains a `function_call` item")
        _assert(
            "function_call_output" in types,
            "output contains a `function_call_output` item",
        )
        called_tools = _tool_call_names(payload)
        _kv("called tools", ", ".join(called_tools) or "<none>")
        _assert(
            any(name in tool_names for name in called_tools),
            "response invoked one of the loaded toolbox tools",
        )
        _assert(bool(_response_text(payload).strip()), "assistant text is non-empty")
    finally:
        server.terminate()


def check_07_toolbox_streaming() -> None:
    """responses/04_foundry_toolbox: streamed run emits toolbox tool events."""
    requires_foundry_toolbox()
    server = start_sample("responses/04_foundry_toolbox/main.py", health_timeout=180.0)
    try:
        tool_names = _loaded_tool_names(server.log_path)
        _kv("toolbox tools", ", ".join(tool_names) or "<none>")
        _assert(bool(tool_names), "startup log lists ≥1 toolbox tool")

        _step("POST /responses (stream=True, toolbox-loaded agent)")
        resp = _post(
            server,
            "/responses",
            json_body={
                "input": _toolbox_prompt(tool_names),
                "model": "gpt-4o",
                "stream": True,
            },
            stream=True,
            timeout=180.0,
        )
        _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        events = _parse_sse(resp.text)
        event_types = [event_type for event_type, _ in events]
        _kv("SSE events", len(events))
        _assert(len(events) > 1, "stream contains multiple SSE events")
        _assert("response.created" in event_types, "stream contains response.created")
        _assert(
            "response.completed" in event_types,
            "stream contains response.completed",
        )

        function_call_items = [
            payload.get("item", {})
            for event_type, payload in events
            if event_type == "response.output_item.done"
            and payload.get("item", {}).get("type") == "function_call"
        ]
        completed_calls = [
            item
            for event_type, payload in events
            if event_type == "response.completed"
            for item in payload.get("response", {}).get("output", [])
            if item.get("type") == "function_call"
        ]
        called_tools = [
            item.get("name", "") for item in [*function_call_items, *completed_calls]
        ]
        _kv("called tools", ", ".join(called_tools) or "<none>")
        _assert(bool(function_call_items), "stream surfaces ≥1 function_call output item")
        _assert(
            any(name in tool_names for name in called_tools),
            "stream invoked one of the loaded toolbox tools",
        )
    finally:
        server.terminate()


def check_11_middleware(decision: str, *, stream: bool) -> None:
    """responses/11_middleware: native approval decisions over JSON or SSE."""
    requires_foundry_endpoint()
    spec = importlib.util.spec_from_file_location(
        "middleware_sample_reviewer", SAMPLES_DIR / "responses/11_middleware/client.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the middleware sample reviewer")
    reviewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reviewer)
    server = start_sample("responses/11_middleware/main.py")

    def post_response(input_items: Any, **options: Any) -> dict[str, Any]:
        response = _post(
            server,
            "/responses",
            json_body={"input": input_items, "stream": stream, **options},
            stream=stream,
        )
        _assert(response.status_code == 200, f"HTTP 200 (got {response.status_code})")
        if stream:
            events = _parse_sse(response.text)
            event_types = [event_type for event_type, _ in events]
            _assert(
                "response.created" in event_types, "stream contains response.created"
            )
            _assert(
                "response.completed" in event_types,
                "stream contains response.completed",
            )
            failures = set(event_types) & {
                "error", "response.failed", "response.incomplete"
            }
            _assert(not failures, "stream has no terminal errors")
            payload = next(
                event["response"]
                for event_type, event in reversed(events)
                if event_type == "response.completed"
            )
        else:
            payload = response.json()
        _assert(payload.get("status") == "completed", "response status == completed")
        return payload

    try:
        _step(f"{decision}: pause before tool execution (stream={stream})")
        paused = post_response(reviewer.DEFAULT_PROMPT)
        interrupts = [
            item
            for item in paused["output"]
            if item.get("type") == "function_call"
            and item.get("name") == reviewer.INTERRUPT_NAME
        ]
        _assert(len(interrupts) == 1, "response contains one native middleware review")
        actions = json.loads(interrupts[0]["arguments"])["value"]["action_requests"]
        _assert(
            len(actions) == 1 and actions[0]["name"] == "send_email",
            "review proposes one send_email action",
        )
        arguments = actions[0]["args"]
        _assert(
            arguments["recipient"] == "alice@example.com",
            "review has the requested recipient",
        )
        _assert(
            "function_call_output" not in _output_types(paused),
            "paused response has no tool result",
        )
        transport = "rich edit" if decision == "edit" else "standard MCP approval"
        _step(f"{decision}: resume through the {transport} channel")
        replies = reviewer.build_resume_items(paused, decision, "bob@example.com")
        expected_type = (
            "function_call_output" if decision == "edit" else "mcp_approval_response"
        )
        _assert(
            len(replies) == 1 and replies[0]["type"] == expected_type,
            f"reviewer uses {expected_type} without a fallback",
        )
        finished = post_response(
            replies,
            previous_response_id=paused["id"],
        )
        _assert(
            reviewer.INTERRUPT_NAME not in _tool_call_names(finished),
            "resume does not leave another pending review",
        )
        outputs = [
            str(item.get("output", ""))
            for item in finished["output"]
            if item.get("type") == "function_call_output"
        ]
        _assert(len(outputs) == 1, "response contains the native tool result")
        if decision == "reject":
            _assert(
                "Rejected by the reviewer" in outputs[0],
                "rejection returns native feedback",
            )
        else:
            recipient = (
                "bob@example.com" if decision == "edit" else arguments["recipient"]
            )
            receipt = (
                f"Simulated email to {recipient}: {arguments['subject']}\n"
                f"{arguments['body']}"
            )
            _assert(
                receipt in outputs[0],
                "tool receives the reviewed recipient, subject, and body",
            )
            if decision == "edit":
                calls = [
                    item
                    for item in finished["output"]
                    if item.get("type") == "function_call"
                    and item.get("name") == "send_email"
                ]
                _assert(
                    len(calls) == 1 and json.loads(calls[0]["arguments"]) == arguments,
                    "edit preserves the model-generated tool call",
                )
                _assert(
                    "human reviewer" in outputs[0]
                    and "Executed instead: send_email" in outputs[0],
                    "tool result explains the human edit",
                )
        _assert(
            bool(_response_text(finished).strip()),
            "assistant completes the review round",
        )
    finally:
        server.terminate()


def check_11_message_limit() -> None:
    """responses/11_middleware: custom hook stops at 50 accumulated messages."""
    requires_foundry_endpoint()
    server = start_sample("responses/11_middleware/main.py")
    try:
        _step("50 messages: custom hook ends the turn")
        response = _post(
            server,
            "/responses",
            json_body={
                "input": [{"role": "user", "content": "Check the message limit."}] * 50,
            },
        )
        _assert(response.status_code == 200, f"HTTP 200 (got {response.status_code})")
        payload = response.json()
        _assert(payload.get("status") == "completed", "response status == completed")
        _assert(
            _response_text(payload).strip() == "Conversation limit reached.",
            "custom hook returns the conversation limit message",
        )
        _assert(
            not set(_output_types(payload))
            & {"function_call", "function_call_output", "mcp_approval_request"},
            "limit response has no tool or approval output",
        )
    finally:
        server.terminate()


# ---------------------------------------------------------------------------
def _deep_agents_response(
    client: Any,
    *,
    stream: bool,
    session_ids: set[str] | None = None,
    blob_names: set[str] | None = None,
    **request: Any,
) -> dict[str, Any]:
    def remember_session(value: Any) -> None:
        if session_ids is not None and isinstance(value, str) and value:
            session_ids.add(value)

    request.setdefault("max_output_tokens", 12000)
    exchange = _start_raw_api_exchange(
        method="POST",
        url=f"{str(client.base_url).rstrip('/')}/responses",
        path="/responses",
        stream=stream,
        timeout=600,
        request_json={**request, "stream": stream},
    )

    def record_response(value: Any) -> None:
        exchange.url = str(value.request.url)
        exchange.request_json = json.loads(value.request.content)
        exchange.response_status = value.status_code
        exchange.response_headers = dict(value.headers)
        remember_session(value.headers.get("x-agent-session-id"))

    response = None
    try:
        if not stream:
            raw = client.responses.with_raw_response.create(stream=False, **request)
            record_response(raw.http_response)
            exchange.response_body = raw.http_response.text
            response = raw.parse().model_dump()
            remember_session(response.get("agent_session_id"))
            if blob_names is not None:
                _deep_agents_exports(response, blob_names)
        else:
            with client.responses.with_streaming_response.create(
                stream=True,
                **request,
            ) as raw:
                record_response(raw.http_response)
                lines = []
                try:
                    for line in raw.iter_lines():
                        lines.append(line)
                finally:
                    exchange.response_body = "\n".join(lines)
                    if blob_names is not None:
                        partial_output = []
                        for _, data in _iter_sse_events(exchange.response_body):
                            try:
                                event = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            snapshot = event.get("response")
                            if isinstance(snapshot, dict):
                                remember_session(snapshot.get("agent_session_id"))
                                _deep_agents_exports(snapshot, blob_names)
                            if event.get("type") == "response.output_item.done":
                                partial_output.append(event["item"])
                        _deep_agents_exports({"output": partial_output}, blob_names)
            event_types = []
            stream_errors = []
            for event_name, data in _iter_sse_events(exchange.response_body):
                if data == "[DONE]":
                    continue
                event = json.loads(data)
                event_type = event.get("type", event_name)
                _assert(
                    event_name in {"message", event_type},
                    "SSE event name matches its payload type",
                )
                event_types.append(event_type)
                snapshot = event.get("response")
                if isinstance(snapshot, dict):
                    remember_session(snapshot.get("agent_session_id"))
                    if blob_names is not None:
                        _deep_agents_exports(snapshot, blob_names)
                if event_type in {"error", "response.failed", "response.incomplete"}:
                    stream_errors.append(event)
                if event_type in {
                    "response.completed",
                    "response.failed",
                    "response.incomplete",
                }:
                    response = snapshot
            _assert(not stream_errors, f"SSE has no error events: {stream_errors}")
            _assert(
                event_types.count("response.created") == 1
                and event_types.count("response.completed") == 1
                and event_types[0] == "response.created"
                and event_types[-1] == "response.completed",
                "SSE has one ordered created/completed lifecycle",
            )
    except Exception as exc:
        failed_response = getattr(exc, "response", None)
        if failed_response is not None:
            record_response(failed_response)
            if not stream:
                exchange.response_body = failed_response.text
        exchange.error = f"{exc.__class__.__name__}: {exc}"
        raise
    if not isinstance(response, dict):
        raise AssertionError("Response is missing a terminal snapshot")
    remember_session(response.get("agent_session_id"))
    _dump_json("Deep Research response", response)
    _assert(
        response.get("status") == "completed",
        f"response completed: {response.get('status')}; "
        f"error={response.get('error')}; "
        f"incomplete_details={response.get('incomplete_details')}",
    )
    return response


def _deep_agents_file_outputs(
    response: dict[str, Any], file_path: str = "/final_report.md"
) -> list[str]:
    last_edit = max(
        (
            index
            for index, item in enumerate(response.get("output", []))
            if item.get("type") == "function_call"
            and item.get("name") in {"write_file", "edit_file"}
            and json.loads(item["arguments"]).get("file_path") == file_path
        ),
        default=-1,
    )
    read_ids = {
        item["call_id"]
        for item in response.get("output", [])[last_edit + 1 :]
        if item.get("type") == "function_call"
        and item.get("name") == "read_file"
        and json.loads(item["arguments"]).get("file_path") == file_path
    }
    return [
        str(item.get("output", ""))
        for item in response.get("output", [])
        if item.get("type") == "function_call_output"
        and item.get("call_id") in read_ids
    ]


def _deep_agents_report_matches(
    response: dict[str, Any], report: str, file_path: str = "/final_report.md"
) -> bool:
    expected = report.replace("\r\n", "\n").removesuffix("\n").split("\n")
    covered: set[int] = set()
    for output in _deep_agents_file_outputs(response, file_path):
        match = re.fullmatch(
            r"@@ lines (\d+)-(\d+)(?: of (\d+))?(?: \| [^\n]*?)? @@\n(.*)",
            output.replace("\r\n", "\n"),
            re.DOTALL,
        )
        if match is None:
            continue
        start, end = int(match[1]), int(match[2])
        if not 1 <= start <= end <= len(expected):
            return False
        if match[3] is not None and int(match[3]) != len(expected):
            return False
        lines = match[4].split("\n")
        if len(lines) == end - start + 2 and lines[-1] == "":
            lines.pop()
        if lines != expected[start - 1 : end]:
            return False
        covered.update(range(start - 1, end))
    return len(covered) == len(expected)


def _deep_agents_answer_matches(
    response: dict[str, Any], report: str, download_url: str | None = None
) -> bool:
    messages = [
        item for item in response.get("output", []) if item.get("type") == "message"
    ]
    if not messages:
        return False
    answer = "".join(part.get("text", "") for part in messages[-1].get("content", []))
    expected = report.replace("\r\n", "\n").rstrip("\n")
    if download_url is not None:
        expected += f"\n\nDownload: {download_url}"
    return answer.replace("\r\n", "\n").rstrip("\n") == expected


def _deep_agents_exports(
    response: dict[str, Any], blob_names: set[str]
) -> list[dict[str, Any]]:
    from urllib.parse import urlsplit

    call_ids = {
        item["call_id"]
        for item in response.get("output", [])
        if item.get("type") == "function_call" and item.get("name") == "publish_report"
    }
    exports = []
    for item in response.get("output", []):
        if (
            item.get("type") != "function_call_output"
            or item.get("call_id") not in call_ids
        ):
            continue
        try:
            exported = json.loads(item["output"])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(exported, dict) and "download_url" in exported:
            exports.append(exported)
    if not exports:
        return []
    account = urlsplit(os.environ["REPORT_STORAGE_ACCOUNT_URL"])
    container = os.environ["REPORT_STORAGE_CONTAINER"]
    for exported in exports:
        url = urlsplit(exported["download_url"])
        blob_name = url.path.removeprefix(f"/{container}/")
        match = re.fullmatch(
            r"[0-9a-f]{32}/([A-Za-z0-9][A-Za-z0-9_-]{0,63}\.md)", blob_name
        )
        _assert(
            url.scheme == "https"
            and url.netloc == account.netloc
            and url.path.startswith(f"/{container}/")
            and match is not None
            and exported.get("file_path") == f"/{match[1]}",
            "export targets a new blob in the configured account and container",
        )
        blob_names.add(blob_name)
    return exports


def _deep_agents_check_export(
    response: dict[str, Any],
    report: str,
    blob_names: set[str],
    report_name: str = "final_report.md",
) -> str:
    import hashlib
    from urllib.parse import parse_qs, urlsplit

    exports = _deep_agents_exports(response, blob_names)
    _assert(bool(exports), "publish_report returned a Blob export")
    exported = exports[-1]
    download_url = exported["download_url"]
    permissions = parse_qs(urlsplit(download_url).query)
    _assert(
        exported.get("file_path") == f"/{report_name}"
        and exported.get("sha256")
        == hashlib.sha256(report.encode("utf-8")).hexdigest(),
        "export digest matches the complete saved report",
    )
    _assert(
        permissions.get("sp") == ["r"]
        and permissions.get("spr") == ["https"]
        and permissions.get("sr") == ["b"],
        "download SAS is read-only, HTTPS-only, and scoped to one blob",
    )
    expires_at = datetime.fromisoformat(exported["expires_at"])
    _assert(
        0 < (expires_at - datetime.now(timezone.utc)).total_seconds() <= 3600,
        "download link expires within one hour",
    )
    _assert(
        _deep_agents_answer_matches(response, report, download_url),
        "final assistant answer contains the complete report and exact download URL",
    )
    downloaded = httpx.get(download_url, timeout=30)
    downloaded.raise_for_status()
    _assert(
        downloaded.headers.get("content-disposition")
        == f'attachment; filename="{report_name}"',
        "download uses the selected report filename",
    )
    _assert(
        downloaded.content == report.encode("utf-8"),
        "downloaded Blob bytes exactly match the saved and delivered report",
    )
    return download_url


def _deep_agents_revised_report(
    response: dict[str, Any], report: str, file_path: str = "/final_report.md"
) -> str:
    outputs = {
        item["call_id"]: str(item.get("output", ""))
        for item in response.get("output", [])
        if item.get("type") == "function_call_output"
    }
    revised = report
    for item in response.get("output", []):
        if item.get("type") != "function_call" or item.get("name") != "edit_file":
            continue
        arguments = json.loads(item["arguments"])
        if arguments.get("file_path") != file_path:
            continue
        result = outputs.get(item["call_id"], "")
        if not result or result.lower().startswith("error"):
            continue
        old = arguments["old_string"]
        _assert(bool(old) and old in revised, "revision edits existing saved content")
        revised = revised.replace(
            old, arguments["new_string"], -1 if arguments.get("replace_all") else 1
        )
    _assert(revised != report, "coordinator edited the existing report")
    return revised


def _deep_agents_delete_exports(storage: Any, container: str, names: set[str]) -> None:
    errors = []
    for name in sorted(names):
        try:
            storage.get_blob_client(container, name).delete_blob()
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise RuntimeError(f"Could not delete Deep Agents test exports: {errors}")


def _deep_agents_stop_sessions(
    project: Any, agent_name: str, session_ids: set[str]
) -> None:
    errors = []
    for session_id in sorted(session_ids):
        try:
            project.agents.stop_session(agent_name, session_id)
        except Exception as exc:
            errors.append(f"{session_id}: {exc}")
    if errors:
        raise RuntimeError(f"Could not stop Deep Agents test sessions: {errors}")


def _check_deep_agents(*, stream: bool) -> None:
    requires_foundry_endpoint()
    if not all(
        os.environ.get(name)
        for name in ("REPORT_STORAGE_ACCOUNT_URL", "REPORT_STORAGE_CONTAINER")
    ):
        raise SkipCheck(
            "Configure REPORT_STORAGE_ACCOUNT_URL and REPORT_STORAGE_CONTAINER."
        )
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential
    from openai import OpenAI

    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:
        raise SkipCheck(
            "Install responses/12_deep_agents/requirements.txt first."
        ) from exc

    with ExitStack() as resources:
        agent_name = os.environ.get("DEEP_AGENTS_NAME")
        client: Any
        session_ids: set[str] = set()
        credential = resources.enter_context(DefaultAzureCredential())
        if agent_name:
            project = resources.enter_context(
                AIProjectClient(
                    endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
                    credential=credential,
                )
            )
            resources.callback(
                _deep_agents_stop_sessions, project, agent_name, session_ids
            )
            client = resources.enter_context(
                project.get_openai_client(
                    agent_name=agent_name,
                    timeout=600,
                    max_retries=0,
                )
            )
        else:
            requires_foundry_toolbox()
            from importlib.util import find_spec

            missing = [
                name
                for name in ("deepagents", "langchain_mcp_adapters")
                if find_spec(name) is None
            ]
            if missing:
                raise SkipCheck(
                    f"Missing Deep Agents dependencies: {', '.join(missing)}. "
                    "Install responses/12_deep_agents/requirements.txt first."
                )
            state_root = resources.enter_context(
                tempfile.TemporaryDirectory(prefix="deep-agents-checkpoints-")
            )
            server = start_sample(
                "responses/12_deep_agents/main.py",
                extra_env={"AGENTSERVER_STATE_ROOT": state_root},
            )
            resources.callback(server.terminate)
            client = resources.enter_context(
                OpenAI(
                    base_url=server.base_url,
                    api_key="local",
                    timeout=600,
                    max_retries=0,
                )
            )
        storage = resources.enter_context(
            BlobServiceClient(os.environ["REPORT_STORAGE_ACCOUNT_URL"], credential)
        )
        blob_names: set[str] = set()
        resources.callback(
            _deep_agents_delete_exports,
            storage,
            os.environ["REPORT_STORAGE_CONTAINER"],
            blob_names,
        )
        marker = f"research-{uuid.uuid4().hex}"
        _step("Research with planning, delegation, and a saved report")
        response = _deep_agents_response(
            client,
            stream=stream,
            session_ids=session_ids,
            blob_names=blob_names,
            input=(
                "Research how LangGraph checkpoints support multi-turn agents. "
                "Plan with write_todos, save /research_request.md, delegate to one "
                "research-agent to use web_search and think_tool and return "
                "findings with source URLs, without writing the final report. "
                "The coordinator must write /report-v1.md itself, with a "
                "short cited report based on those findings. Read back the "
                "entire report, reading all remaining pages if needed, before "
                "publishing it with publish_report(report_name='report-v1.md') "
                "and returning its full contents "
                "followed by the exact Download paragraph. Limit this to one search. "
                f"Include this exact label in the report: {marker}"
            ),
        )
        calls = [
            item
            for item in response.get("output", [])
            if item.get("type") == "function_call"
        ]
        names = {item.get("name") for item in calls}
        _assert(
            {"write_todos", "task", "write_file", "read_file"} <= names,
            "coordinator planned, delegated, wrote, and read working files",
        )
        _assert(
            any(
                item.get("name") == "task"
                and json.loads(item["arguments"]).get("subagent_type")
                == "research-agent"
                for item in calls
            ),
            "coordinator used the configured research-agent",
        )
        reports = [
            arguments["content"]
            for item in calls
            if item.get("name") == "write_file"
            for arguments in [json.loads(item["arguments"])]
            if arguments.get("file_path") == "/report-v1.md"
        ]
        _assert(bool(reports), "coordinator wrote /report-v1.md")
        report = reports[-1]
        task_outputs = [
            item.get("output", "")
            for item in response.get("output", [])
            if item.get("type") == "function_call_output"
            and item.get("call_id")
            in {call.get("call_id") for call in calls if call.get("name") == "task"}
        ]
        _assert(
            any(
                isinstance(output, str) and "https://" in output
                for output in task_outputs
            ),
            "delegated researcher returned findings with source URLs",
        )
        _assert(
            marker in report and "https://" in report,
            "saved report contains its label and source URLs",
        )
        _assert(
            _deep_agents_report_matches(response, report, "/report-v1.md"),
            "read_file returned the complete report written during research",
        )
        original_url = _deep_agents_check_export(
            response, report, blob_names, "report-v1.md"
        )

        session_options: dict[str, Any] = {}
        if agent_name:
            session_id = response.get("agent_session_id")
            _assert(
                isinstance(session_id, str) and bool(session_id),
                "deployed response identifies its test session",
            )
            session_options["extra_body"] = {"agent_session_id": session_id}

        if not agent_name:
            _step("Restart the local host with the same checkpoint storage")
            server.terminate()
            _assert(
                any((Path(state_root) / "state_stores").glob("*.json")),
                "local checkpoint files exist before restart",
            )
            server = start_sample(
                "responses/12_deep_agents/main.py",
                extra_env={"AGENTSERVER_STATE_ROOT": state_root},
            )
            resources.callback(server.terminate)
            client = resources.enter_context(
                OpenAI(
                    base_url=server.base_url,
                    api_key="local",
                    timeout=600,
                    max_retries=0,
                )
            )

        _step("Read the report in the same conversation")
        continued = _deep_agents_response(
            client,
            stream=stream,
            session_ids=session_ids,
            blob_names=blob_names,
            previous_response_id=response["id"],
            input=(
                "Use read_file to return the entire existing /report-v1.md. "
                "Read all remaining pages if needed. Do not research, edit or write."
            ),
            **session_options,
        )
        if agent_name:
            _assert(
                continued.get("agent_session_id") == session_id,
                "continuation stayed in the original test session",
            )
        _assert(
            "read_file" in _tool_call_names(continued), "continuation read saved state"
        )
        _assert(
            not {"task", "web_search", "write_file", "edit_file", "publish_report"}
            & set(_tool_call_names(continued)),
            "continuation did not research or rewrite the report",
        )
        _assert(
            _deep_agents_report_matches(continued, report, "/report-v1.md"),
            "continued read_file returned the complete saved report",
        )
        _assert(
            _deep_agents_answer_matches(continued, report),
            "continued assistant answer contains the complete saved report",
        )

        _step("Revise and re-export the saved report without repeating research")
        revision_marker = f"revision-{uuid.uuid4().hex}"
        revision_section = (
            "## Recovery limitations\n\n"
            "Checkpoints restore saved state, not arbitrary in-flight work.\n\n"
            f"Validation label: {revision_marker}"
        )
        revised_response = _deep_agents_response(
            client,
            stream=stream,
            session_ids=session_ids,
            blob_names=blob_names,
            previous_response_id=continued["id"],
            input=(
                "Read /report-v1.md and use write_file to copy its exact content "
                "to /report-v2.md. Then use edit_file to add the following exact "
                "section to /report-v2.md. Do not change /report-v1.md. Keep the "
                "entire original report, citations and URLs "
                "unchanged; this revision needs no new research. The validation "
                "label is only an identifier, not a research finding. "
                "Read the entire updated file after editing, call "
                "publish_report(report_name='report-v2.md'), "
                "then return its full contents followed by the new Download paragraph."
                f"\n\n{revision_section}"
            ),
            **session_options,
        )
        if agent_name:
            _assert(
                revised_response.get("agent_session_id") == session_id,
                "revision stayed in the original test session",
            )
        copies = [
            arguments["content"]
            for item in revised_response.get("output", [])
            if item.get("type") == "function_call" and item.get("name") == "write_file"
            for arguments in [json.loads(item["arguments"])]
            if arguments.get("file_path") == "/report-v2.md"
        ]
        _assert(copies == [report], "revision starts from an exact named working copy")
        revised = _deep_agents_revised_report(revised_response, report, "/report-v2.md")
        revised_lines = iter(revised.splitlines())
        _assert(
            all(
                any(original == line for line in revised_lines)
                for original in report.splitlines()
            )
            and revision_section in revised,
            "revision preserves the original report and adds the requested section",
        )
        _assert(
            not {"task", "web_search"} & set(_tool_call_names(revised_response)),
            "revision did not repeat research",
        )
        _assert(
            not any(
                item.get("type") == "function_call"
                and item.get("name") in {"write_file", "edit_file"}
                and json.loads(item["arguments"]).get("file_path") == "/report-v1.md"
                for item in revised_response.get("output", [])
            ),
            "revision did not modify the original working file",
        )
        _assert(
            _deep_agents_report_matches(revised_response, revised, "/report-v2.md"),
            "read_file returned the complete revised saved report",
        )
        revised_url = _deep_agents_check_export(
            revised_response, revised, blob_names, "report-v2.md"
        )
        _assert(revised_url != original_url, "revision has a new download snapshot")
        original_download = httpx.get(original_url, timeout=30)
        original_download.raise_for_status()
        _assert(
            original_download.content == report.encode("utf-8"),
            "the original download still contains the original report",
        )

        _step("Read the revised report using the latest response ID")
        reread = _deep_agents_response(
            client,
            stream=stream,
            session_ids=session_ids,
            blob_names=blob_names,
            previous_response_id=revised_response["id"],
            input=(
                "Use read_file to read both /report-v1.md and /report-v2.md in full, "
                "then return only the full /report-v2.md contents verbatim. "
                "Do not research, edit or publish another copy."
            ),
            **session_options,
        )
        if agent_name:
            _assert(
                reread.get("agent_session_id") == session_id,
                "revised read stayed in the original test session",
            )
        _assert(
            _deep_agents_report_matches(reread, report, "/report-v1.md")
            and _deep_agents_report_matches(reread, revised, "/report-v2.md")
            and _deep_agents_answer_matches(reread, revised),
            "read-back and assistant answer both match the entire revised report",
        )
        _assert(
            not {"task", "web_search", "write_file", "edit_file", "publish_report"}
            & set(_tool_call_names(reread)),
            "revised read did not research, edit or export",
        )

        _step("Check a new conversation cannot read that report")
        isolated = _deep_agents_response(
            client,
            stream=stream,
            session_ids=session_ids,
            blob_names=blob_names,
            input=(
                "Use read_file on both /report-v1.md and /report-v2.md. "
                "If they are missing, report that; "
                "do not create or research anything."
            ),
            **session_options,
        )
        if agent_name:
            _assert(
                isolated.get("agent_session_id") == session_id,
                "new conversation used the same test session",
            )
        _assert(
            "read_file" in _tool_call_names(isolated),
            "new conversation checked the file",
        )
        _assert(
            all(
                any(
                    "not found" in output.lower()
                    for output in _deep_agents_file_outputs(isolated, file_path)
                )
                for file_path in ("/report-v1.md", "/report-v2.md")
            ),
            "neither named report exists in the new conversation",
        )
        _assert(
            marker not in json.dumps(isolated)
            and revision_marker not in json.dumps(isolated),
            "new conversation has neither version of the prior report",
        )
        _assert(
            not {
                "task",
                "web_search",
                "write_file",
                "edit_file",
                "publish_report",
                "delete",
            }
            & set(_tool_call_names(isolated)),
            "isolated conversation did not research or modify files",
        )


def check_12_deep_agents() -> None:
    """responses/12_deep_agents: research, revision, Blob downloads, isolation."""
    _check_deep_agents(stream=False)


def check_12_deep_agents_streaming() -> None:
    """responses/12_deep_agents: same research workflow and state over SSE."""
    _check_deep_agents(stream=True)
# Registry
# ---------------------------------------------------------------------------


@dataclass
class Check:
    sample_key: str  # e.g. "responses/01_basic"
    name: str        # short identifier
    fn: Callable[[], None]
    description: str
    server: Optional[SampleServer] = field(default=None, repr=False)

    @property
    def full_name(self) -> str:
        return f"{self.sample_key}::{self.name}"

    @property
    def selector_aliases(self) -> set[str]:
        aliases = {self.sample_key.lower()}
        if "/" not in self.sample_key:
            return aliases

        protocol, folder = self.sample_key.lower().split("/", 1)
        sample_number = folder.split("_", 1)[0]
        aliases.add(folder)
        aliases.add(f"{protocol}/{sample_number}")
        aliases.add(sample_number)
        aliases.add(sample_number.lstrip("0") or "0")
        return aliases


def _build_registry() -> list[Check]:
    return [
        Check("responses/01_basic", "non_streaming", check_01_non_streaming, check_01_non_streaming.__doc__ or ""),
        Check("responses/01_basic", "streaming", check_01_streaming, check_01_streaming.__doc__ or ""),
        Check("responses/02_tools", "non_streaming", check_02_non_streaming, check_02_non_streaming.__doc__ or ""),
        Check("responses/02_tools", "streaming", check_02_streaming, check_02_streaming.__doc__ or ""),
        Check("responses/03_mcp", "remote_mcp", check_03_remote_mcp, check_03_remote_mcp.__doc__ or ""),
        Check("responses/04_foundry_toolbox", "toolbox", check_07_toolbox, check_07_toolbox.__doc__ or ""),
        Check("responses/04_foundry_toolbox", "toolbox_streaming", check_07_toolbox_streaming, check_07_toolbox_streaming.__doc__ or ""),
        Check("responses/05_workflows", "responses_tool_round_trip", check_05_responses_tool_round_trip, check_05_responses_tool_round_trip.__doc__ or ""),
        Check("responses/06_files", "files", check_06_files, check_06_files.__doc__ or ""),
        Check("responses/07_observability", "observability", check_07_observability, check_07_observability.__doc__ or ""),
        Check("responses/08_hitl", "pause_then_resume", check_06_pause_then_resume, check_06_pause_then_resume.__doc__ or ""),
        Check("responses/11_middleware", "approve_json", partial(check_11_middleware, "approve", stream=False), "Native middleware approve (JSON)."),
        Check("responses/11_middleware", "approve_sse", partial(check_11_middleware, "approve", stream=True), "Native middleware approve (SSE)."),
        Check("responses/11_middleware", "edit_json", partial(check_11_middleware, "edit", stream=False), "Native middleware edit (JSON)."),
        Check("responses/11_middleware", "edit_sse", partial(check_11_middleware, "edit", stream=True), "Native middleware edit (SSE)."),
        Check("responses/11_middleware", "reject_json", partial(check_11_middleware, "reject", stream=False), "Native middleware reject (JSON)."),
        Check("responses/11_middleware", "reject_sse", partial(check_11_middleware, "reject", stream=True), "Native middleware reject (SSE)."),
        Check("responses/11_middleware", "message_limit", check_11_message_limit, check_11_message_limit.__doc__ or ""),
        Check("responses/12_deep_agents", "research", check_12_deep_agents, check_12_deep_agents.__doc__ or ""),
        Check("responses/12_deep_agents", "streaming", check_12_deep_agents_streaming, check_12_deep_agents_streaming.__doc__ or ""),
        Check("invocations/01_basic", "multi_turn", check_03_multi_turn, check_03_multi_turn.__doc__ or ""),
        Check("invocations/01_basic", "streaming", check_03_streaming, check_03_streaming.__doc__ or ""),
        Check("invocations/02_tools", "final_text", check_04_final_text, check_04_final_text.__doc__ or ""),
    ]


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
    global _CURRENT_CHECK_NAME, _CURRENT_STEP_TITLE

    previous_check_name = _CURRENT_CHECK_NAME
    previous_step_title = _CURRENT_STEP_TITLE
    _CURRENT_CHECK_NAME = check.full_name
    _CURRENT_STEP_TITLE = ""
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
    finally:
        _CURRENT_CHECK_NAME = previous_check_name
        _CURRENT_STEP_TITLE = previous_step_title


def _filter(registry: list[Check], selectors: list[str]) -> list[Check]:
    if not selectors:
        return registry
    wanted_selectors = {_normalize_selector(s) for s in selectors if s.strip()}
    matched: list[Check] = []
    for chk in registry:
        if chk.selector_aliases.intersection(wanted_selectors):
            matched.append(chk)
    return matched


def _normalize_selector(selector: str) -> str:
    normalized = selector.strip().lower().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.rstrip("/")
    if normalized.endswith("/main.py"):
        normalized = normalized[: -len("/main.py")]
    return normalized


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


def _md_escape(text: str) -> str:
    """Make a string safe to drop inside a single markdown table cell."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    # Pipes break table columns; backslashes need escaping; <br> is fine inline.
    return collapsed.replace("\\", "\\\\").replace("|", "\\|")


def _check_description(check: "Check") -> str:
    """Use only the first paragraph of the docstring as the table description."""
    doc = (check.description or "").strip()
    if not doc:
        return ""
    # Split on the first blank line — keeps the headline sentence(s), drops
    # the long-form prose that some checks (e.g. check_06) append.
    first_para = doc.split("\n\n", 1)[0]
    return re.sub(r"\s+", " ", first_para).strip()


def _fenced_code_block(text: str, language: str = "text") -> list[str]:
    """Return a markdown fenced code block that cannot be closed by ``text``."""
    longest_fence = max(
        (len(match.group(0)) for match in re.finditer(r"`{3,}", text)),
        default=2,
    )
    fence = "`" * max(3, longest_fence + 1)
    body = text.rstrip() or "(no transcript captured)"
    return [f"{fence}{language}", body, fence]


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def _request_input_items(exchange: RawApiExchange) -> list[dict[str, Any]]:
    input_value = exchange.request_json.get("input")
    if not isinstance(input_value, list):
        return []
    return [item for item in input_value if isinstance(item, dict)]


def _response_output_items(exchange: RawApiExchange) -> list[dict[str, Any]]:
    try:
        payload = json.loads(exchange.response_body)
    except json.JSONDecodeError:
        return []
    output = payload.get("output") if isinstance(payload, dict) else None
    if not isinstance(output, list):
        return []
    return [item for item in output if isinstance(item, dict)]


def _approval_request_tool_name(
    previous_exchanges: list[RawApiExchange], approval_request_id: str
) -> str:
    for previous in reversed(previous_exchanges):
        for item in _response_output_items(previous):
            if (
                item.get("type") == "mcp_approval_request"
                and item.get("id") == approval_request_id
            ):
                try:
                    arguments = json.loads(item.get("arguments") or "{}")
                except json.JSONDecodeError:
                    return ""
                value = arguments.get("value") if isinstance(arguments, dict) else None
                if isinstance(value, dict):
                    return str(value.get("tool") or "")
    return ""


def _raw_exchange_context_note(
    exchange: RawApiExchange, previous_exchanges: list[RawApiExchange]
) -> str:
    for item in _request_input_items(exchange):
        item_type = item.get("type")
        if item_type == "mcp_approval_response":
            approval_id = str(item.get("approval_request_id") or "<missing>")
            action = "approves" if item.get("approve") else "rejects"
            tool_name = _approval_request_tool_name(previous_exchanges, approval_id)
            target = (
                f"the proposed `{tool_name}` tool call"
                if tool_name
                else "a previous approval request"
            )
            return (
                f"This resumes the HITL interrupt from the prior turn: it {action} "
                f"{target} with `approval_request_id={approval_id}`."
            )
        if item_type == "function_call_output":
            call_id = str(item.get("call_id") or "<missing>")
            return (
                "This resumes the HITL interrupt through the advanced "
                f"`function_call_output` channel with `call_id={call_id}`."
            )
    return ""


def _append_raw_api_exchange(
    lines: list[str],
    exchange: RawApiExchange,
    previous_exchanges: list[RawApiExchange],
) -> None:
    status = (
        str(exchange.response_status)
        if exchange.response_status is not None
        else "ERROR"
    )
    heading_context = (
        f"{exchange.step_title} — " if exchange.step_title else ""
    )
    lines.append(
        f"### {exchange.ordinal}. `{exchange.check_name}` — "
        f"{heading_context}{exchange.method} `{exchange.path}` ({status})"
    )
    lines.append("")
    if exchange.step_title:
        lines.append(f"- Step: `{exchange.step_title}`")
    lines.append(f"- URL: `{exchange.url}`")
    lines.append(f"- Stream: `{exchange.stream}`")
    lines.append(f"- Timeout: `{exchange.timeout}` seconds")
    context_note = _raw_exchange_context_note(exchange, previous_exchanges)
    if context_note:
        lines.append(f"- Context: {context_note}")
    lines.append("")
    lines.append("**Request JSON**")
    lines.append("")
    lines.extend(_fenced_code_block(_json_text(exchange.request_json), "json"))
    lines.append("")
    if exchange.error:
        lines.append("**Error**")
        lines.append("")
        lines.extend(_fenced_code_block(exchange.error))
        lines.append("")
        return
    lines.append("**Response Headers**")
    lines.append("")
    lines.extend(_fenced_code_block(_json_text(exchange.response_headers), "json"))
    lines.append("")
    response_language, formatted_response = _format_response_body(
        exchange.response_body,
        stream=exchange.stream,
    )
    response_label = "**Response Body (formatted SSE events)**" if exchange.stream else "**Response Body**"
    lines.append(response_label)
    lines.append("")
    lines.extend(_fenced_code_block(formatted_response, response_language))
    lines.append("")


def _write_markdown_report(
    results: list["Result"],
    path: Path,
    *,
    raw_api_exchanges: list[RawApiExchange],
) -> None:
    """Emit a markdown summary and raw API payloads."""
    n_pass = sum(1 for r in results if r.status == "pass")
    n_fail = sum(1 for r in results if r.status == "fail")
    n_skip = sum(1 for r in results if r.status == "skip")
    icons = {"pass": "✅ PASS", "fail": "❌ FAIL", "skip": "⏭️ SKIP"}

    lines: list[str] = []
    lines.append("# Hosting samples E2E report")
    lines.append("")
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append(
        f"**Totals:** {n_pass} passed · {n_fail} failed · {n_skip} skipped "
        f"({len(results)} total)"
    )
    lines.append("")
    lines.append("| # | Test case | What it tests | Result | Duration |")
    lines.append("|---|-----------|---------------|--------|----------|")
    for idx, r in enumerate(results, start=1):
        lines.append(
            "| {n} | `{name}` | {desc} | {res} | {dur:.1f}s |".format(
                n=idx,
                name=_md_escape(r.check.full_name),
                desc=_md_escape(_check_description(r.check)) or "—",
                res=icons.get(r.status, r.status.upper()),
                dur=r.elapsed,
            )
        )

    failures = [r for r in results if r.status == "fail"]
    skips = [r for r in results if r.status == "skip"]
    if failures:
        lines.append("")
        lines.append("## Failure details")
        lines.append("")
        for r in failures:
            lines.append(f"### `{r.check.full_name}`")
            lines.append("")
            lines.append("```")
            lines.append(r.detail or "(no detail captured)")
            lines.append("```")
            lines.append("")
    if skips:
        lines.append("## Skipped")
        lines.append("")
        for r in skips:
            lines.append(f"- `{r.check.full_name}` — {_md_escape(r.detail) or 'skipped'}")
        lines.append("")

    lines.append("## API transcript")
    lines.append("")
    lines.append(
        "Request and response payloads captured directly from the runner's "
        "HTTP calls to each sample host. Response bodies are formatted for "
        "readability; streamed responses are grouped by SSE event with parsed "
        "JSON data payloads."
    )
    lines.append("")
    if raw_api_exchanges:
        previous_exchanges: list[RawApiExchange] = []
        for exchange in raw_api_exchanges:
            _append_raw_api_exchange(lines, exchange, previous_exchanges)
            previous_exchanges.append(exchange)
    else:
        lines.append("(no API calls were made)")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Markdown report written to: {path}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manual end-to-end runner for samples/hosting samples."
    )
    parser.add_argument(
        "samples",
        nargs="*",
        help=(
            "Sample selectors to run (e.g. 01, responses/04, "
            ".\\responses\\04_foundry_toolbox\\, invocations/01_basic). "
            "Default: every check."
        ),
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

    _RAW_API_EXCHANGES.clear()
    results: list[Result] = []
    _section("Run plan")
    print(f"  Working dir : {SAMPLES_DIR}")
    print(f"  Python      : {sys.executable}")
    print(f"  Checks      : {len(selected)}")
    for chk in selected:
        print(f"    - {chk.full_name}")
    print(f"  FOUNDRY_PROJECT_ENDPOINT   set: {bool(os.environ.get('FOUNDRY_PROJECT_ENDPOINT'))}")
    print(f"  TOOLBOX_NAME               set: {bool(os.environ.get('TOOLBOX_NAME'))}")

    for chk in selected:
        result = _run_check(chk)
        results.append(result)
        if result.status == "fail" and args.fail_fast:
            print("\n  (--fail-fast set: stopping)")
            break

    _print_summary(results)
    report_path = Path.cwd() / "run_samples_e2e_report.md"
    _write_markdown_report(
        results,
        report_path,
        raw_api_exchanges=list(_RAW_API_EXCHANGES),
    )
    return 0 if not any(r.status == "fail" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
