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
* for the Foundry-Toolbox sample: ``TOOLBOX_NAME``.

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    """responses/06_files: filesystem tools list and read notes.txt."""
    requires_foundry_endpoint()
    server = start_sample("responses/06_files/main.py")
    try:
        _step("POST /responses (list and read bundled file)")
        resp = _post(
            server,
            "/responses",
            json_body={
                "input": (
                    "You must use list_files and read_text_file to inspect "
                    "notes.txt. Do not answer from memory. Then summarize the "
                    "action items in one short paragraph."
                ),
                "model": "gpt-4o",
            },
            timeout=180.0,
        )
        _assert(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        payload = resp.json()
        _assert(payload["status"] == "completed", "status == completed")
        called_tools = _tool_call_names(payload)
        _kv("called tools", ", ".join(called_tools) or "<none>")
        _assert("list_files" in called_tools, "response invoked list_files")
        _assert("read_text_file" in called_tools, "response invoked read_text_file")
        _assert(
            "function_call_output" in _output_types(payload),
            "output contains a `function_call_output` item",
        )
        text = _response_text(payload).lower()
        _assert("notes" in text or "action" in text, "assistant summarizes notes.txt")
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


# ---------------------------------------------------------------------------
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
