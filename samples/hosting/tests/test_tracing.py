# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Check every hosted sample over HTTP with real telemetry exporters.

Run with the samples' requirements and pytest installed::

    python -m pytest \
        samples/hosting/tests/test_tracing.py -q

Each case starts a fresh sample process, substitutes external services, and
receives telemetry on a local HTTP server. The test explicitly enables full
sampling to check export and correlation. It does not check default sampling,
Azure ingestion, project connections, or portal visibility.
"""

from __future__ import annotations

import ast
import gzip
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
from dotenv import dotenv_values
from tracing_scenarios import (
    CASES,
    CONTENT_RECORDING_FLAGS,
    FINAL_TEXT,
    model_responses,
)

SAMPLES = Path(__file__).resolve().parents[2]
TRACE_ID = "1234567890abcdef1234567890abcdef"
USER_TEXT = "Exercise the sample's tools"


def discover_samples(root: Path) -> list[str]:
    """Find host imports and configuration-runner entrypoints across samples/."""
    found = set()
    for source in root.rglob("*"):
        if not source.is_file() or any(
            part.startswith(".") or part in {"tests", "__pycache__"}
            for part in source.relative_to(root).parts
        ):
            continue
        if source.suffix == ".py":
            content = source.read_text(encoding="utf-8-sig")
            if "langchain_azure_ai.agents" not in content:
                continue
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "langchain_azure_ai.agents.hosting"
                ):
                    if any(
                        name.name
                        in {"ResponsesHostServer", "InvocationsHostServer", "main"}
                        for name in node.names
                    ):
                        found.add(source.parent.relative_to(root).as_posix())
                elif isinstance(node, ast.Import) and any(
                    "langchain_azure_ai.agents.hosting" in name.name
                    for name in node.names
                ):
                    found.add(source.parent.relative_to(root).as_posix())
        elif source.name == "Dockerfile" or source.suffix in {".yaml", ".yml"}:
            if "langchain_azure_ai.agents.hosting.run" in source.read_text(
                encoding="utf-8"
            ):
                found.add(source.parent.relative_to(root).as_posix())
    return sorted(found)


def _decode_spans(received: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    spans = []
    for path, body in list(received):
        if path.endswith("/track"):
            for envelope in json.loads(body):
                data = envelope["data"]
                if data["baseType"] in ("RequestData", "RemoteDependencyData"):
                    spans.append(
                        {
                            "name": data["baseData"]["name"],
                            "span_id": data["baseData"]["id"],
                            "trace_id": envelope.get("tags", {}).get("ai.operation.id"),
                            "parent": envelope.get("tags", {}).get(
                                "ai.operation.parentId"
                            ),
                            "attributes": data["baseData"].get("properties", {}),
                        }
                    )
        elif path == "/v1/traces":
            request = ExportTraceServiceRequest.FromString(body)
            for resource in request.resource_spans:
                for scope in resource.scope_spans:
                    for span in scope.spans:
                        spans.append(
                            {
                                "name": span.name,
                                "span_id": span.span_id.hex(),
                                "trace_id": span.trace_id.hex(),
                                "parent": span.parent_span_id.hex(),
                                "attributes": {
                                    a.key: a.value.string_value for a in span.attributes
                                },
                            }
                        )
    return [s for s in spans if s["attributes"].get("gen_ai.operation.name")]


def _invoke(
    client: httpx.Client, protocol: str, payload: dict[str, Any]
) -> dict[str, Any]:
    path = (
        "/responses"
        if protocol == "responses"
        else "/invocations?agent_session_id=trace-demo"
    )
    response = client.post(
        path,
        json=payload,
        headers={
            "traceparent": f"00-{TRACE_ID}-1234567890abcdef-01",
        },
    )
    assert response.status_code in (200, 202), response.text
    result = response.json()
    if protocol == "invocations" and "id" not in result:
        assert "response" in result, result
        return result
    for _ in range(150):
        if result.get("status") in ("completed", "failed", "cancelled", "incomplete"):
            break
        time.sleep(0.1)
        response = client.get(f"/{protocol}/{result['id']}")
        assert response.status_code == 200, response.text
        result = response.json()
    assert result["status"] == "completed", result
    return result


def _content_strings(value: Any) -> list[str]:
    """Read text from both JSON-encoded attributes and nested message parts."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _content_strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _content_strings(item)]
    return [str(value)]


def _assert_recorded_content(
    spans: list[dict[str, Any]], calls: list[dict[str, Any]]
) -> None:
    chats = [
        s["attributes"]
        for s in spans
        if s["attributes"]["gen_ai.operation.name"] == "chat"
    ]
    tools = [
        s["attributes"]
        for s in spans
        if s["attributes"]["gen_ai.operation.name"] == "execute_tool"
    ]
    tool_names = {
        tool["id"]: tool["name"]
        for call in calls
        for tool in call["output"]["data"].get("tool_calls", [])
    }

    def contains(records: list[dict[str, Any]], attribute: str, expected: Any) -> bool:
        actual = [
            text
            for record in records
            for text in _content_strings(record.get(attribute, ""))
        ]
        return all(
            any(text in value for value in actual)
            for text in _content_strings(expected)
        )

    assert contains(chats, "gen_ai.input.messages", USER_TEXT), (
        "User prompt was not exported"
    )
    assert contains(chats, "gen_ai.output.messages", FINAL_TEXT), (
        "Assistant reply was not exported"
    )
    for call in calls:
        for message in call["inputs"]:
            data = message["data"]
            if message["type"] == "system":
                assert contains(chats, "gen_ai.system_instructions", data["content"]), (
                    "System prompt was not exported"
                )
            elif message["type"] == "tool":
                name = tool_names[data["tool_call_id"]]
                matching = [
                    tool for tool in tools if tool.get("gen_ai.tool.name") == name
                ]
                assert contains(matching, "gen_ai.tool.call.result", data["content"]), (
                    "Tool result was not exported",
                    name,
                )
                assert contains(chats, "gen_ai.input.messages", data["content"]), (
                    "Tool response message was not exported",
                    name,
                )
        output = call["output"]["data"]
        if output["content"]:
            assert contains(chats, "gen_ai.output.messages", output["content"]), (
                "Assistant message was not exported"
            )
        for tool_call in output.get("tool_calls", []):
            assert any(
                tool.get("gen_ai.tool.name") == tool_call["name"]
                and json.loads(tool.get("gen_ai.tool.call.arguments", "null"))
                == tool_call["args"]
                for tool in tools
            ), ("Tool arguments were not exported", tool_call)
            assert contains(chats, "gen_ai.output.messages", tool_call["args"]), (
                "Tool call message was not exported",
                tool_call,
            )


def test_all_hosted_samples_have_a_scenario() -> None:
    discovered = set(discover_samples(SAMPLES))
    assert discovered, "No hosted samples discovered"
    assert discovered == set(CASES), (
        f"Add scenarios in tracing_scenarios.py for: {discovered - CASES.keys()}; "
        f"remove stale scenarios: {CASES.keys() - discovered}"
    )


def test_discovery_includes_new_sample_locations(tmp_path: Path) -> None:
    direct = tmp_path / "future-agent"
    direct.mkdir()
    (direct / "app.py").write_text(
        "from langchain_azure_ai.agents.hosting import ResponsesHostServer as Host\n",
        encoding="utf-8",
    )
    configured = tmp_path / "another" / "agent"
    configured.mkdir(parents=True)
    (configured / "Dockerfile").write_text(
        'CMD ["python", "-m", "langchain_azure_ai.agents.hosting.run"]',
        encoding="utf-8",
    )
    assert discover_samples(tmp_path) == ["another/agent", "future-agent"]


@pytest.mark.parametrize("sample", discover_samples(SAMPLES))
def test_sample_defaults_keep_tracing_enabled(sample: str) -> None:
    forbidden = {
        "OTEL_SDK_DISABLED": {"true"},
        "OTEL_TRACES_SAMPLER": {"always_off", "parentbased_always_off"},
        "OTEL_TRACES_EXPORTER": {"none"},
    }
    for path in (SAMPLES / sample).rglob("*"):
        settings: dict[str, Any] = {}
        if path.name == ".env.example":
            settings.update(dotenv_values(path, interpolate=False))
        elif path.suffix == ".py" and "tests" not in path.parts:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "setdefault"
                    and len(node.args) >= 2
                    and all(isinstance(arg, ast.Constant) for arg in node.args[:2])
                ):
                    settings[node.args[0].value] = node.args[1].value
        for name, disabled_values in forbidden.items():
            assert str(settings.get(name, "")).lower() not in disabled_values, (
                path,
                name,
                settings.get(name),
            )


@pytest.mark.parametrize("sample", discover_samples(SAMPLES))
@pytest.mark.parametrize(
    ("destination", "content_recording"),
    [
        pytest.param("azure", False, id="azure-default"),
        pytest.param("otlp", False, id="otlp-default"),
        pytest.param("disabled", False, id="disabled"),
        pytest.param("azure", True, id="azure-content"),
        pytest.param("otlp", True, id="otlp-content"),
    ],
)
def test_hosted_sample_exports_traces(
    sample: str, destination: str, content_recording: bool, tmp_path: Path
) -> None:
    assert sample in CASES, "Add a scenario in tracing_scenarios.py"
    protocol, scenario = CASES[sample]
    messages = model_responses(scenario)
    tool_names = {call["name"] for message in messages for call in message.tool_calls}
    approvals = scenario in {"approval", "email", "resilient"}
    expected = {
        "chat": len(messages),
        "execute_tool": len(tool_names),
        "invoke_agent": 2 if approvals else 1,
    }
    received: list[tuple[str, bytes]] = []

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["content-length"]))
            if self.headers.get("content-encoding") == "gzip":
                body = gzip.decompress(body)
            received.append((self.path, body))
            result = b""
            if self.path.endswith("/track"):
                count = len(json.loads(body))
                result = json.dumps(
                    {"itemsReceived": count, "itemsAccepted": count, "errors": []}
                ).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)

        def log_message(self, *_: Any) -> None:
            pass

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(
            (
                "OTEL_",
                "APPLICATIONINSIGHTS_",
                "APPLICATION_INSIGHTS_",
                "FOUNDRY_",
                "AZURE_AI_",
                "AZURE_TRACING_",
                "LANGSMITH_",
                "LANGCHAIN_",
            )
        )
    }
    env.update(
        PORT=str(port),
        PYTHONIOENCODING="utf-8",
        PYTHON_DOTENV_DISABLED="1",
        AGENTSERVER_STATE_ROOT=str(tmp_path),
        OTEL_BSP_SCHEDULE_DELAY="100",
        OTEL_TRACES_SAMPLER="always_on",
        APPLICATIONINSIGHTS_STATSBEAT_DISABLED_ALL="true",
        FOUNDRY_PROJECT_ENDPOINT="https://example.invalid/api/projects/tracing-test",
        AZURE_AI_MODEL_DEPLOYMENT_NAME="tracing-test-model",
        GITHUB_PAT="unused-test-token",
        TOOLBOX_NAME="tracing-test",
        REPORT_STORAGE_ACCOUNT_URL="https://example.invalid",
        REPORT_STORAGE_CONTAINER="reports",
    )
    defaults = dotenv_values(SAMPLES / sample / ".env.example", interpolate=False)
    for name in CONTENT_RECORDING_FLAGS:
        value = "true" if content_recording else defaults.get(name)
        if value is not None:
            env[name] = value
    with ThreadingHTTPServer(("127.0.0.1", 0), Collector) as collector:
        threading.Thread(target=collector.serve_forever, daemon=True).start()
        target = f"http://127.0.0.1:{collector.server_port}"
        if destination == "otlp":
            env["OTEL_EXPORTER_OTLP_ENDPOINT"] = target
            env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = f"{target}/v1/traces"
            env["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
        else:
            env["APPLICATIONINSIGHTS_CONNECTION_STRING"] = (
                "InstrumentationKey=00000000-0000-0000-0000-000000000001;"
                f"IngestionEndpoint={target}/"
            )
            env["APPLICATION_INSIGHTS_CONNECTION_STRING"] = env[
                "APPLICATIONINSIGHTS_CONNECTION_STRING"
            ]
        if destination == "disabled":
            env["OTEL_SDK_DISABLED"] = "true"
        with (tmp_path / "host.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("tracing_runner.py")),
                    sample,
                ],
                cwd=tmp_path,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                with httpx.Client(
                    base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False
                ) as client:
                    for _ in range(150):
                        assert proc.poll() is None, (tmp_path / "host.log").read_text(
                            encoding="utf-8"
                        )
                        try:
                            client.get("/health")
                            break
                        except httpx.ConnectError:
                            time.sleep(0.2)
                    else:
                        pytest.fail("Sample startup timed out")
                    payload = (
                        {
                            "input": USER_TEXT,
                            "conversation": "trace-demo",
                            "background": scenario == "resilient",
                            "store": True,
                        }
                        if protocol == "responses"
                        else {
                            "message": USER_TEXT,
                            "background": scenario == "resilient",
                        }
                    )
                    first = _invoke(client, protocol, payload)
                    if approvals:
                        approval = next(
                            i
                            for i in first["output"]
                            if i["type"] == "mcp_approval_request"
                        )
                        payload["input" if protocol == "responses" else "message"] = [
                            {
                                "type": "mcp_approval_response",
                                "approval_request_id": approval["id"],
                                "approve": True,
                            }
                        ]
                        payload[
                            "previous_response_id"
                            if protocol == "responses"
                            else "previous_invocation_id"
                        ] = first["id"]
                        first = _invoke(client, protocol, payload)
                    assert FINAL_TEXT in json.dumps(first)
                    observed = json.loads(
                        (tmp_path / "model-calls.json").read_text(encoding="utf-8")
                    )
                    for name in CONTENT_RECORDING_FLAGS:
                        if name in env:
                            assert observed["content_recording"][name] == env[name]
                    records_content = "true" in observed["content_recording"].values()
                    for _ in range(10 if destination == "disabled" else 100):
                        spans = _decode_spans(received)
                        counts = {
                            op: sum(
                                s["attributes"]["gen_ai.operation.name"] == op
                                for s in spans
                            )
                            for op in expected
                        }
                        if all(counts[op] >= count for op, count in expected.items()):
                            if records_content:
                                try:
                                    _assert_recorded_content(spans, observed["calls"])
                                except AssertionError:
                                    time.sleep(0.1)
                                    continue
                            break
                        time.sleep(0.1)
                    if destination == "disabled":
                        assert spans == []
                    else:
                        # The host and sample instrumentors may both emit spans.
                        assert all(
                            counts[op] >= count for op, count in expected.items()
                        ), (counts, expected, spans)
                        assert {
                            s["attributes"].get("gen_ai.tool.name") for s in spans
                        } >= tool_names
                        assert all(
                            s["trace_id"] == TRACE_ID and s["parent"] for s in spans
                        )
                        if records_content:
                            _assert_recorded_content(spans, observed["calls"])
                        elif set(observed["content_recording"].values()) == {"false"}:
                            attributes = [s["attributes"] for s in spans]
                            for attribute in (
                                "gen_ai.input.messages",
                                "gen_ai.output.messages",
                            ):
                                recorded = _content_strings(
                                    [a.get(attribute, "") for a in attributes]
                                )
                                assert not any(
                                    USER_TEXT in text or FINAL_TEXT in text
                                    for text in recorded
                                ), "Disabled content capture exported message text"
            finally:
                proc.terminate()
                proc.wait(timeout=15)
                collector.shutdown()
                (tmp_path / "exported-spans.json").write_text(
                    json.dumps(
                        {
                            "sample": sample,
                            "destination": destination,
                            "content_recording_requested": content_recording,
                            "spans": _decode_spans(received),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
