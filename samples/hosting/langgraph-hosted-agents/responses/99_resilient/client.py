"""Client for the resilient LangGraph Responses sample.

Start the agent first:

    python main.py

Then run the resilient background streaming call:

    python client.py go

Fetch a stored response later:

    python client.py --get caresp_...
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8088"
DEFAULT_AUTH_SCOPE = "https://ai.azure.com/.default"
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "incomplete"}
RETRYABLE_HTTP_STATUSES = {408, 409, 424, 429, 500, 502, 503, 504}


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
        headers=headers or {"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if retryable_statuses and exc.code in retryable_statuses:
            raise RetryableResponseError(exc.code, body) from exc
        raise SystemExit(f"HTTP {exc.code} from {url}:\n{body}") from exc
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
        print("\nfinal responses:")
        print(text)
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
    def __init__(self, *, full_json: bool = False, raw: bool = False) -> None:
        self.full_json = full_json
        self.raw = raw
        self.response_id: str | None = None
        self.terminal_response_seen = False
        self.streamed_text_len = 0
        self._streaming_line_open = False

    def event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if self.raw:
            print(json.dumps({"event": event_type, "data": payload}, ensure_ascii=False))
            if event_type in {"response.completed", "response.failed", "response.incomplete"}:
                self.terminal_response_seen = True
            response = _extract_response(payload) if event_type.startswith("response.") else None
            if event_type == "response.created" and response is not None:
                self.response_id = response.get("id") or response.get("response_id")
            return response

        if self.full_json:
            print(f"\nevent: {event_type}")
            print(json.dumps(payload, indent=2))
            return _extract_response(payload) if event_type.startswith("response.") else None

        if event_type == "response.created":
            response = _extract_response(payload)
            self.response_id = response.get("id") or response.get("response_id")
            if self.response_id:
                print(f"response_id: {self.response_id}\n")
        elif event_type == "response.output_text.delta":
            self.token(str(payload.get("delta", "")))
        elif event_type in {"response.completed", "response.failed", "response.incomplete"}:
            response = _extract_response(payload)
            self.final_response(response)
            return response
        return None

    def token(self, text: str) -> None:
        if not text:
            return
        if not self._streaming_line_open:
            sys.stdout.write("streaming: \n")
            self._streaming_line_open = True
        sys.stdout.write(text)
        sys.stdout.flush()
        self.streamed_text_len += len(text)

    def retrying(self) -> None:
        self.close_streaming_line()
        print("\n[retrying...]\n")

    def close_streaming_line(self) -> None:
        if self._streaming_line_open:
            print()
            self._streaming_line_open = False

    def final_response(self, response: dict[str, Any]) -> None:
        self.terminal_response_seen = True
        self.close_streaming_line()
        text = _response_text(response).strip()
        print("\nfinal responses:")
        if text:
            print(text)
        error = response.get("error")
        if error:
            print("\nerror:")
            print(json.dumps(error, indent=2))


def create_response(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {
        "input": args.input,
        "background": True,
        "stream": True,
        "store": args.store,
    }
    if args.model:
        payload["model"] = args.model

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
            for event_type, event_payload in _iter_sse(response):
                printer.event(event_type, event_payload)
            if printer.response_id and not printer.terminal_response_seen:
                printer.retrying()
                poll_response(args, printer.response_id, printer=printer)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} from {url}:\n{body}") from exc
    except (http.client.HTTPException, OSError, urllib.error.URLError) as exc:
        if not printer.response_id:
            raise SystemExit(f"Unable to start streaming response: {exc}") from exc
        printer.retrying()
        poll_response(args, printer.response_id, printer=printer)


def get_response(
    args: argparse.Namespace,
    response_id: str,
    *,
    retryable: bool = False,
) -> dict[str, Any]:
    url = _responses_url(args.base_url, response_id)
    return _json_request(
        "GET",
        url,
        timeout=args.timeout,
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
) -> None:
    deadline = time.monotonic() + args.poll_timeout
    last_status = None
    printed_text_len = printer.streamed_text_len if printer is not None else 0
    while True:
        try:
            response = get_response(args, response_id, retryable=True)
        except (
            RetryableResponseError,
            http.client.HTTPException,
            OSError,
            urllib.error.URLError,
        ) as exc:
            if time.monotonic() >= deadline:
                raise SystemExit(f"Timed out waiting for {response_id}; last error: {exc}") from exc
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
            return
        if time.monotonic() >= deadline:
            raise SystemExit(f"Timed out waiting for {response_id}; last status: {status}")
        time.sleep(args.interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call the sample Responses API host.")
    parser.add_argument("input", nargs="?", default="go", help="Input text to send.")
    parser.add_argument(
        "--base-url",
        "--endpoint",
        dest="base_url",
        default=os.environ.get("RESPONSES_ENDPOINT")
        or os.environ.get("AGENT_RESPONSES_ENDPOINT")
        or os.environ.get("RESPONSES_BASE_URL", DEFAULT_BASE_URL),
        help=f"Agent host base URL. Defaults to {DEFAULT_BASE_URL}.",
    )
    parser.add_argument("--model", help="Optional model field to include in the request.")
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
    parser.add_argument("--no-auth", action="store_true", help="Do not add Authorization headers.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--store",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store the response so it can be fetched later. Defaults to true.",
    )
    parser.add_argument("--get", metavar="RESPONSE_ID", help="Fetch a stored response by id.")
    parser.add_argument("--poll", metavar="RESPONSE_ID", help="Poll a stored response until terminal.")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds.")
    parser.add_argument("--poll-timeout", type=float, default=120.0, help="Polling timeout in seconds.")
    parser.add_argument("--json", action="store_true", help="Print full JSON payloads.")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print each streaming SSE event as one raw JSON object per line.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.get and args.poll:
        raise SystemExit("Use either --get or --poll, not both.")
    if args.get:
        _print_response(get_response(args, args.get), full_json=args.json)
    elif args.poll:
        poll_response(args, args.poll)
    else:
        create_response(args)


if __name__ == "__main__":
    main()