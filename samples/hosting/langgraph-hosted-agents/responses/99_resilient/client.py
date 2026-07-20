"""Client for the resilient LangGraph Responses sample.

Start the agent first:

    python main.py

Then run a resilient background streaming conversation:

    python client.py --background --stream
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from textwrap import indent
from typing import Any
from uuid import uuid4


DEFAULT_BASE_URL = "http://127.0.0.1:8088"
DEFAULT_AUTH_SCOPE = "https://ai.azure.com/.default"
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "incomplete"}
RETRYABLE_HTTP_STATUSES = {408, 409, 424, 429, 500, 502, 503, 504}
# Short per-request timeout for polling GETs. Keeps Ctrl+C responsive during
# [retrying...]: a blocking socket read can't be interrupted on Windows until
# it returns, so we cap each poll request and let the interruptible sleep run.
POLL_REQUEST_TIMEOUT = 5.0
METADATA_STEERABLE_CONVERSATION = "foundry.agent.steerable_conversation"


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


class RetryableResponseError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def _responses_url(base_url: str, response_id: str | None = None) -> str:
    base = base_url.rstrip("/")
    parsed = urllib.parse.urlparse(base)
    path = parsed.path.rstrip("/")
    if response_id is None:
        if path.endswith("/responses"):
            return base
        return urllib.parse.urlunparse(parsed._replace(path=f"{path}/responses"))

    response_path = urllib.parse.quote(response_id, safe="")
    if path.endswith("/responses"):
        path = f"{path}/{response_path}"
    else:
        path = f"{path}/responses/{response_path}"
    return urllib.parse.urlunparse(parsed._replace(path=path))


def _response_action_url(base_url: str, response_id: str, action: str) -> str:
    """Build ``/responses/<id>/<action>`` while preserving the URL query."""
    response_url = _responses_url(base_url, response_id)
    parsed = urllib.parse.urlparse(response_url)
    path = f"{parsed.path.rstrip('/')}/{urllib.parse.quote(action, safe='')}"
    return urllib.parse.urlunparse(parsed._replace(path=path))


def _get_az_token(scope: str) -> str:
    env_token = os.environ.get("AZURE_AI_AUTH_TOKEN")
    if env_token:
        return env_token
    az = shutil.which("az") or shutil.which("az.cmd")
    if az is None:
        raise SystemExit(
            "Azure CLI was not found on PATH. Run `az login`, ensure `az` is on PATH, "
            "or pass --token / set AZURE_AI_AUTH_TOKEN."
        )
    completed = subprocess.run(
        [
            az,
            "account",
            "get-access-token",
            "--scope",
            scope,
            "--query",
            "accessToken",
            "-o",
            "tsv",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _headers(
    url: str,
    *,
    accept: str,
    token: str | None = None,
    auth_scope: str = DEFAULT_AUTH_SCOPE,
    no_auth: bool = False,
) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": accept}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif not no_auth and urllib.parse.urlparse(url).scheme == "https":
        headers["Authorization"] = f"Bearer {_get_az_token(auth_scope)}"
    return headers


def _emit_http_error(status_code: int, body: str) -> None:
    """Print an HTTP failure's status code and body (as JSON when possible).

    Used so expected failure cases (for example a crash with background
    disabled, or a non-terminal ``424``) surface the real status code and
    error payload instead of a generic message.
    """
    print(f"\nresponse status code: {status_code}")
    print("response error json:")
    try:
        print(json.dumps(json.loads(body), indent=2))
    except (ValueError, TypeError):
        print(body)


def _json_request(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
    headers: dict[str, str] | None = None,
    retryable_statuses: set[int] | None = None,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers=headers
        or {"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if retryable_statuses and exc.code in retryable_statuses:
            raise RetryableResponseError(exc.code, body) from exc
        _emit_http_error(exc.code, body)
        raise SystemExit(1) from exc
    return json.loads(body)


def _extract_response(payload: dict[str, Any]) -> dict[str, Any]:
    response = payload.get("response")
    return response if isinstance(response, dict) else payload


def _response_text(response: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks)


def _supports_steering(response: dict[str, Any]) -> bool:
    metadata = response.get("metadata")
    return (
        isinstance(metadata, dict)
        and metadata.get(METADATA_STEERABLE_CONVERSATION) == "true"
    )


def _print_response_started(
    response_id: str,
    *,
    steerable_conversation: bool,
) -> None:
    print(f"{response_id} ===")
    if steerable_conversation:
        print("[running, type 's' for steer, 'c' for cancel]...")
    else:
        print("[running, type 'c' for cancel]...")


def _print_response(response: dict[str, Any], *, full_json: bool = False) -> None:
    if full_json:
        print(json.dumps(response, indent=2))
        return

    response_id = response.get("id") or response.get("response_id")
    status = response.get("status")
    output_types = [item.get("type") for item in response.get("output", [])]
    if response_id:
        print(f"response_id: {response_id}")
    if status and status != "completed":
        print(f"status: {status}")
    if output_types and status != "completed":
        print(f"output: {', '.join(str(t) for t in output_types)}")
    text = _response_text(response).strip()
    if text:
        print("final responses:")
        print(indent(text, "  "))
    error = response.get("error")
    if error:
        print("\nerror:")
        print(json.dumps(error, indent=2))


def _iter_sse(response) -> Iterator[tuple[str, dict[str, Any]]]:
    event_type = "message"
    data_lines: list[str] = []

    def flush() -> tuple[str, dict[str, Any]] | None:
        nonlocal event_type, data_lines
        if not data_lines:
            return None
        raw_data = "\n".join(data_lines)
        current_event = event_type
        event_type = "message"
        data_lines = []
        try:
            return current_event, json.loads(raw_data)
        except json.JSONDecodeError:
            return current_event, {"raw": raw_data}

    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            event = flush()
            if event is not None:
                yield event
            continue
        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())

    event = flush()
    if event is not None:
        yield event


class StreamPrinter:
    def __init__(
        self,
        *,
        full_json: bool = False,
        raw: bool = False,
    ) -> None:
        self.full_json = full_json
        self.raw = raw
        self.response_id: str | None = None
        self.steerable_conversation = False
        self.terminal_response_seen = False
        self.streamed_text_len = 0
        self._streaming_line_open = False
        self._stream_at_line_start = True

    def event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if event_type in {
            "response.completed",
            "response.failed",
            "response.incomplete",
            "response.cancelled",
        }:
            self.terminal_response_seen = True
        if event_type == "response.created":
            response = _extract_response(payload)
            self.response_id = response.get("id") or response.get("response_id")
            self.steerable_conversation = _supports_steering(response)
            if self.response_id:
                _print_response_started(
                    self.response_id,
                    steerable_conversation=self.steerable_conversation,
                )
        if self.raw:
            print(
                json.dumps({"event": event_type, "data": payload}, ensure_ascii=False)
            )
            response = (
                _extract_response(payload)
                if event_type.startswith("response.")
                else None
            )
            return response

        if self.full_json:
            print(f"\nevent: {event_type}")
            print(json.dumps(payload, indent=2))
            return (
                _extract_response(payload)
                if event_type.startswith("response.")
                else None
            )

        if event_type == "response.output_text.delta":
            self.token(str(payload.get("delta", "")))
        elif event_type in {
            "response.completed",
            "response.failed",
            "response.incomplete",
            "response.cancelled",
        }:
            response = _extract_response(payload)
            self.final_response(response)
            return response
        return None

    def token(self, text: str) -> None:
        if not text:
            return
        if not self._streaming_line_open:
            sys.stdout.write("stream:\n")
            self._streaming_line_open = True
        output: list[str] = []
        for character in text:
            if self._stream_at_line_start:
                output.append("  ")
                self._stream_at_line_start = False
            output.append(character)
            if character == "\n":
                self._stream_at_line_start = True
        sys.stdout.write("".join(output))
        sys.stdout.flush()
        self.streamed_text_len += len(text)

    def retrying(self) -> None:
        self.close_streaming_line()
        print("\n[retrying...]\n")

    def close_streaming_line(self) -> None:
        if self._streaming_line_open:
            if not self._stream_at_line_start:
                print()
            self._streaming_line_open = False
            self._stream_at_line_start = True

    def final_response(self, response: dict[str, Any]) -> None:
        self.terminal_response_seen = True
        self.close_streaming_line()
        text = _response_text(response).strip()
        print("final responses:")
        if text:
            print(indent(text, "  "))
        error = response.get("error")
        if error:
            print("\nerror:")
            print(json.dumps(error, indent=2))


def get_response(
    args: argparse.Namespace,
    response_id: str,
    *,
    retryable: bool = False,
    timeout: float | None = None,
) -> dict[str, Any]:
    url = _responses_url(args.base_url, response_id)
    return _json_request(
        "GET",
        url,
        timeout=args.timeout if timeout is None else timeout,
        headers=_headers(
            url,
            accept="application/json",
            token=args.token,
            auth_scope=args.auth_scope,
            no_auth=args.no_auth,
        ),
        retryable_statuses=RETRYABLE_HTTP_STATUSES if retryable else None,
    )


def poll_response(
    args: argparse.Namespace,
    response_id: str,
    *,
    printer: StreamPrinter | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + args.poll_timeout
    last_status = None
    last_http_error: RetryableResponseError | None = None
    printed_text_len = printer.streamed_text_len if printer is not None else 0
    while True:
        try:
            response = get_response(
                args, response_id, retryable=True, timeout=POLL_REQUEST_TIMEOUT
            )
        except RetryableResponseError as exc:
            last_http_error = exc
            if time.monotonic() >= deadline:
                _emit_http_error(exc.status_code, exc.body)
                raise SystemExit(1) from exc
            time.sleep(args.interval)
            continue
        except (
            http.client.HTTPException,
            OSError,
            urllib.error.URLError,
        ) as exc:
            if time.monotonic() >= deadline:
                if last_http_error is not None:
                    _emit_http_error(last_http_error.status_code, last_http_error.body)
                raise SystemExit(
                    f"Timed out waiting for {response_id}; last error: {exc}"
                ) from exc
            time.sleep(args.interval)
            continue
        status = response.get("status")
        if status != last_status and printer is None:
            print(f"status: {status}")
            last_status = status
        if printer is not None:
            text = _response_text(response)
            if len(text) > printed_text_len:
                printer.token(text[printed_text_len:])
                printed_text_len = len(text)
        if status in TERMINAL_STATUSES:
            if printer is None:
                _print_response(response, full_json=args.json)
            else:
                printer.final_response(response)
            return response
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"Timed out waiting for {response_id}; last status: {status}"
            )
        time.sleep(args.interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call the sample Responses API host.")
    parser.add_argument(
        "--base-url",
        "--endpoint",
        dest="base_url",
        default=os.environ.get("RESPONSES_ENDPOINT")
        or os.environ.get("AGENT_RESPONSES_ENDPOINT")
        or os.environ.get("RESPONSES_BASE_URL", DEFAULT_BASE_URL),
        help=f"Agent host base URL. Defaults to {DEFAULT_BASE_URL}.",
    )
    parser.add_argument(
        "--model", help="Optional model field to include in the request."
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run as a resilient background response. Defaults to false.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response via SSE. Defaults to false.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("AZURE_AI_AUTH_TOKEN"),
        help="Bearer token for a remote HTTPS endpoint. Defaults to Azure CLI token lookup.",
    )
    parser.add_argument(
        "--auth-scope",
        default=DEFAULT_AUTH_SCOPE,
        help=f"Azure CLI token scope for remote HTTPS endpoints. Defaults to {DEFAULT_AUTH_SCOPE}.",
    )
    parser.add_argument(
        "--no-auth", action="store_true", help="Do not add Authorization headers."
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="HTTP timeout in seconds."
    )
    parser.add_argument(
        "--store",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store the response so it can be fetched later. Defaults to true.",
    )
    parser.add_argument(
        "--interval", type=float, default=1.0, help="Polling interval in seconds."
    )
    parser.add_argument(
        "--poll-timeout", type=float, default=120.0, help="Polling timeout in seconds."
    )
    parser.add_argument(
        "--token-delay",
        type=_non_negative_float,
        help="Server-side delay in seconds between streamed tokens.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON payloads.")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print each streaming SSE event as one raw JSON object per line.",
    )
    return parser


# ----------------------------------------------------------------------
# Multi-turn interactive mode
# ----------------------------------------------------------------------


def _stdin_reader(input_queue: "queue.Queue[str]") -> None:
    """Continuously read lines from stdin onto a queue (daemon thread)."""
    while True:
        line = sys.stdin.readline()
        if line == "":  # EOF
            break
        input_queue.put(line.rstrip("\n"))


def _wait_for_input(input_queue: "queue.Queue[str]") -> str:
    """Wait for queued stdin while keeping Ctrl+C responsive on Windows."""
    while True:
        try:
            return input_queue.get(timeout=0.2)
        except queue.Empty:
            continue


def _prompt_for_input(input_queue: "queue.Queue[str]") -> str:
    print("You: ", end="", flush=True)
    return _wait_for_input(input_queue)


def _cancel_turn(args: argparse.Namespace, holder: dict[str, Any]) -> None:
    """Cancel the active turn.

    Background: POST ``/responses/<id>/cancel``. Foreground (streaming or
    blocking): disconnect the open connection.
    """
    if holder.get("cancelled"):
        return
    response_id = holder.get("response_id")
    if args.background:
        if not response_id:
            print("[cancel] no response id yet.")
            return
        holder["cancelled"] = True
        cancel_url = _response_action_url(args.base_url, response_id, "cancel")
        print(f"\n[cancel requested for {response_id}]")

        def _submit_cancel() -> None:
            try:
                _json_request(
                    "POST",
                    cancel_url,
                    payload={},
                    timeout=min(args.timeout, 30.0),
                    headers=_headers(
                        cancel_url,
                        accept="application/json",
                        token=args.token,
                        auth_scope=args.auth_scope,
                        no_auth=args.no_auth,
                    ),
                )
            except SystemExit:
                pass  # explicit HTTP error body already printed
            except (TimeoutError, http.client.HTTPException) as exc:
                print(
                    "\n[cancel request was sent, but its response timed out; "
                    f"checking response status: {exc}]"
                )
            except (OSError, urllib.error.URLError) as exc:
                print(f"\n[cancel transport error: {exc}]")

        threading.Thread(target=_submit_cancel, daemon=True).start()
    else:
        holder["cancelled"] = True
        conn = holder.get("conn")
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        print("\n[disconnected]")


def _steer_turn(
    args: argparse.Namespace,
    holder: dict[str, Any],
    input_queue: "queue.Queue[str]",
) -> None:
    """Queue one new turn behind the active steerable background response."""
    if not holder.get("steerable_conversation"):
        print("[steering is not supported by this server]")
        return
    if not args.background:
        print("[steer requires --background]")
        return
    if holder.get("steered_response_id"):
        print("[a steering turn is already queued]")
        return
    response_id = holder.get("response_id")
    if not response_id:
        print("[steer unavailable until the response id is received]")
        return

    print("steer input:")
    steer_input = _wait_for_input(input_queue).strip()
    if not steer_input:
        print("[steer cancelled: empty input]")
        return

    payload: dict[str, Any] = {
        "input": steer_input,
        "background": True,
        "stream": False,
        "store": args.store,
        "previous_response_id": response_id,
    }
    if args.token_delay is not None:
        payload["metadata"] = {"token_delay": str(args.token_delay)}
    if args.model:
        payload["model"] = args.model

    url = _responses_url(args.base_url)
    try:
        response = _extract_response(
            _json_request(
                "POST",
                url,
                payload=payload,
                timeout=min(args.timeout, 30.0),
                headers=_headers(
                    url,
                    accept="application/json",
                    token=args.token,
                    auth_scope=args.auth_scope,
                    no_auth=args.no_auth,
                ),
            )
        )
    except SystemExit:
        return
    except (http.client.HTTPException, OSError, urllib.error.URLError) as exc:
        print(f"[steer transport error: {exc}]")
        return

    steered_response_id = response.get("id") or response.get("response_id")
    if not steered_response_id:
        print("[steer failed: server returned no response id]")
        return
    holder["steered_response_id"] = steered_response_id
    print(
        f"[steer queued: {steered_response_id}; "
        f"status={response.get('status', 'queued')}]"
    )


def _run_stream_turn(
    args: argparse.Namespace, payload: dict[str, Any], holder: dict[str, Any]
) -> None:
    url = _responses_url(args.base_url)
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers=_headers(
            url,
            accept="text/event-stream",
            token=args.token,
            auth_scope=args.auth_scope,
            no_auth=args.no_auth,
        ),
    )
    printer = StreamPrinter(full_json=args.json, raw=args.raw)
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            holder["conn"] = response
            for event_type, event_payload in _iter_sse(response):
                terminal_response = printer.event(event_type, event_payload)
                if printer.response_id:
                    holder["response_id"] = printer.response_id
                    holder["steerable_conversation"] = printer.steerable_conversation
                if event_type == "response.completed":
                    holder["succeeded"] = True
                elif terminal_response is not None and event_type in {
                    "response.failed",
                    "response.incomplete",
                    "response.cancelled",
                }:
                    holder["succeeded"] = False
            if (
                printer.response_id
                and not printer.terminal_response_seen
                and not holder.get("cancelled")
            ):
                printer.retrying()
                terminal_response = poll_response(
                    args, printer.response_id, printer=printer
                )
                holder["succeeded"] = terminal_response.get("status") == "completed"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        _emit_http_error(exc.code, body)
    except (http.client.HTTPException, OSError, urllib.error.URLError) as exc:
        if holder.get("cancelled"):
            print("\n[cancelled]")
        elif printer.response_id:
            printer.retrying()
            terminal_response = poll_response(
                args, printer.response_id, printer=printer
            )
            holder["succeeded"] = terminal_response.get("status") == "completed"
        else:
            print(f"\n[stream error: {exc}]")
    if printer.response_id:
        holder["response_id"] = printer.response_id


def _run_background_turn(
    args: argparse.Namespace, payload: dict[str, Any], holder: dict[str, Any]
) -> None:
    url = _responses_url(args.base_url)
    response = _extract_response(
        _json_request(
            "POST",
            url,
            payload=payload,
            timeout=args.timeout,
            headers=_headers(
                url,
                accept="application/json",
                token=args.token,
                auth_scope=args.auth_scope,
                no_auth=args.no_auth,
            ),
        )
    )
    response_id = response.get("id") or response.get("response_id")
    holder["response_id"] = response_id
    status = response.get("status")
    holder["steerable_conversation"] = _supports_steering(response)
    if response_id:
        _print_response_started(
            response_id,
            steerable_conversation=holder["steerable_conversation"],
        )
    if response_id and status not in TERMINAL_STATUSES:
        printer = StreamPrinter(full_json=args.json, raw=args.raw)
        terminal_response = poll_response(args, response_id, printer=printer)
        holder["succeeded"] = terminal_response.get("status") == "completed"
    else:
        _print_response(response, full_json=args.json)
        holder["succeeded"] = status == "completed"


def _run_existing_background_turn(
    args: argparse.Namespace,
    response_id: str,
    steerable_conversation: bool,
    holder: dict[str, Any],
) -> None:
    """Interactively follow an already-created queued steering response."""
    holder["response_id"] = response_id
    holder["steerable_conversation"] = steerable_conversation
    _print_response_started(
        response_id,
        steerable_conversation=steerable_conversation,
    )
    printer = StreamPrinter(full_json=args.json, raw=args.raw)
    terminal_response = poll_response(args, response_id, printer=printer)
    holder["succeeded"] = terminal_response.get("status") == "completed"


def _run_blocking_turn(
    args: argparse.Namespace, payload: dict[str, Any], holder: dict[str, Any]
) -> None:
    url = _responses_url(args.base_url)
    response = _extract_response(
        _json_request(
            "POST",
            url,
            payload=payload,
            timeout=args.timeout,
            headers=_headers(
                url,
                accept="application/json",
                token=args.token,
                auth_scope=args.auth_scope,
                no_auth=args.no_auth,
            ),
        )
    )
    holder["response_id"] = response.get("id") or response.get("response_id")
    print(f"{holder['response_id']} ===")
    _print_response(response, full_json=args.json)
    holder["succeeded"] = response.get("status") == "completed"


def run_multiturn(args: argparse.Namespace) -> None:
    """Interactive multi-turn conversation.

    All turns use one explicit ``conversation.id`` so the LangGraph
    checkpointer continues the same linear thread.
    """
    input_queue: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(input_queue,), daemon=True).start()

    first_input = _prompt_for_input(input_queue).strip()
    if not first_input:
        print("Exiting multi-turn.")
        return

    if args.stream:
        run_turn = _run_stream_turn
    elif args.background:
        run_turn = _run_background_turn
    else:
        run_turn = _run_blocking_turn

    previous_response_id: str | None = None
    conversation_id = f"resilient-{uuid4().hex}"
    next_input: str | None = first_input
    pending_response_id: str | None = None
    pending_steerable_conversation = False
    turn_index = 0

    while next_input is not None or pending_response_id is not None:
        turn_index += 1
        holder: dict[str, Any] = {
            "response_id": None,
            "succeeded": False,
            "cancelled": False,
            "steered_response_id": None,
            "steerable_conversation": False,
            "conn": None,
            "done": threading.Event(),
        }
        existing_response_id = pending_response_id
        pending_response_id = None

        if existing_response_id is None:
            payload: dict[str, Any] = {
                "input": next_input,
                "background": args.background,
                "stream": args.stream,
                "store": args.store,
                "conversation": {"id": conversation_id},
            }
            if args.token_delay is not None:
                payload["metadata"] = {"token_delay": str(args.token_delay)}
            if previous_response_id:
                payload["previous_response_id"] = previous_response_id
            if args.model:
                payload["model"] = args.model
            turn_label = f"turn {turn_index}"
        else:
            payload = {}
            turn_label = f"turn {turn_index} (steered)"

        print(f"\n=== {turn_label} - ", end="", flush=True)

        def _worker() -> None:
            try:
                if existing_response_id is not None:
                    _run_existing_background_turn(
                        args,
                        existing_response_id,
                        pending_steerable_conversation,
                        holder,
                    )
                else:
                    run_turn(args, payload, holder)
            except SystemExit:
                pass  # error already surfaced
            except Exception as exc:  # noqa: BLE001
                print(f"\n[turn error: {exc}]")
            finally:
                holder["done"].set()

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        # Active phase: the turn runs while we watch for commands.
        while not holder["done"].is_set():
            try:
                cmd = input_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            choice = cmd.strip().lower()
            if choice in ("cancel", "c"):
                _cancel_turn(args, holder)
            elif choice in ("steer", "s"):
                _steer_turn(args, holder, input_queue)
            elif choice:
                if holder.get("steerable_conversation"):
                    print("type s for steer, type c for cancel")
                else:
                    print("type c for cancel")
        worker.join(timeout=2.0)
        steered_response_id = holder.get("steered_response_id")
        if not holder["succeeded"] and not steered_response_id:
            print("\n[turn failed; conversation ended]")
            raise SystemExit(1)

        response_id = holder["response_id"]
        if response_id:
            previous_response_id = response_id

        print(f"=== {turn_label} done ===")

        if steered_response_id:
            pending_response_id = steered_response_id
            pending_steerable_conversation = bool(holder.get("steerable_conversation"))
            next_input = None
            continue

        # Post-turn menu.
        next_input = None
        while next_input is None:
            print("\nType text for a new turn (empty to exit):")
            cmd = _prompt_for_input(input_queue)
            choice = cmd.strip()
            if not choice:
                print("Exiting multi-turn.")
                return
            next_input = choice


def main() -> None:
    args = build_parser().parse_args()
    run_multiturn(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
