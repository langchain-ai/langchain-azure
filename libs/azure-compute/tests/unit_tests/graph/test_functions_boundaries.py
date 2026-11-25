# Copyright (c) Microsoft. All rights reserved.

"""Functions factory, serialization, route-alias, and lifecycle boundaries."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace

import azure.functions as func
import pytest

from langchain_azure_compute.graph import azure_functions as af
from tests.unit_tests.graph.test_functions_auth import Sdk, registration


@dataclass
class Row:
    value: int


@pytest.mark.parametrize(
    "obj,expected",
    [
        (Row(2), {"value": 2}),
        (SimpleNamespace(model_dump=lambda: {"value": 3}), {"value": 3}),
        (SimpleNamespace(type="ai", content="text"), {"type": "ai", "content": "text"}),
        (
            SimpleNamespace(
                type="tool", content="text", tool_calls=[1], tool_call_id="id"
            ),
            {
                "type": "tool",
                "content": "text",
                "tool_calls": [1],
                "tool_call_id": "id",
            },
        ),
        (7, "7"),
    ],
)
def test_http_json_serialization(obj, expected):
    assert af._json_default(obj) == expected


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://host/api/workflow/g/run", "http://host/api"),
        ("http://host/api/graphs/g/start", "http://host/api"),
        ("http://host/api/health", "http://host/api"),
        ("http://host/other/?q=1", "http://host/other"),
    ],
)
def test_api_base_extraction(url, expected):
    assert af._extract_api_base(url) == expected


def test_factory_registration_shapes_and_invalid_inputs():
    reg = registration()
    assert af._normalize_registrations([reg]) == {"wave2": reg}
    assert af._normalize_registrations({"wave2": reg}) == {"wave2": reg}
    assert af._normalize_registrations(reg.graph, default_name="wave2")
    assert af._coerce_to_registration(reg) is reg
    with pytest.raises(ValueError, match="name"):
        af._coerce_to_registration(SimpleNamespace())
    with pytest.raises(ValueError, match="Provide"):
        af.create_function_app()
    with pytest.raises(ValueError, match="least one"):
        af.create_function_app([], Sdk())
    with pytest.raises(ValueError, match="2, 3 or 4"):
        af.create_function_app(reg, Sdk(), respond_authorizer=lambda caller: True)
    with pytest.raises(ValueError, match="mapping keys"):
        af.create_function_app({"wrong": reg}, Sdk())


@pytest.mark.parametrize(
    "route",
    [
        "graphs/missing/start",
        "graphs/missing/status/iid",
        "graphs/missing/respond/iid",
    ],
)
def test_unknown_graph_routes(route):
    app = af.create_function_app(registration(), Sdk(), require_instance_owner=False)
    req = func.HttpRequest(
        method="GET" if "/status/" in route else "POST",
        url="http://localhost/api/" + route,
        body=b"{}",
    )
    assert af.dispatch_function_request(app, req).status_code == 404


@pytest.mark.parametrize("kind", ["status", "respond"])
def test_unknown_shared_instance(kind):
    app = af.create_function_app(registration(), Sdk(), require_instance_owner=False)
    req = func.HttpRequest(
        method="GET" if kind == "status" else "POST",
        url=f"http://localhost/api/graphs/wave2/{kind}/missing",
        body=b"",
    )
    assert af.dispatch_function_request(app, req).status_code == 404


@pytest.mark.parametrize("body", [b"\xff", b"{", b""])
def test_invalid_start_body_is_validation_error(body):
    app = af.create_function_app(registration(), Sdk(), require_instance_owner=False)
    req = func.HttpRequest(
        method="POST",
        url="http://localhost/api/graphs/wave2/start",
        body=body,
    )
    assert af.dispatch_function_request(app, req).status_code == 400


def test_workflow_aliases_failed_status_and_empty_resume():
    reg, sdk = registration(), Sdk()
    app = af.create_function_app(reg, sdk, require_instance_owner=False)
    sdk.schedule_new_orchestration("run", {"thread": "t"}, "iid")
    for route in ("workflow/wave2/respond/iid", "workflow/wave2/respond/iid/i1"):
        req = func.HttpRequest(
            method="POST", url="http://localhost/api/" + route, body=b""
        )
        assert af.dispatch_function_request(app, req).status_code == 200
    sdk.states["iid"].runtime_status.name = "FAILED"
    req = func.HttpRequest(
        method="GET",
        url="http://localhost/api/workflow/wave2/status/iid",
        body=b"",
    )
    response = af.dispatch_function_request(app, req)
    assert response.status_code == 200
    assert b"Workflow execution failed" in response.get_body()
    req = func.HttpRequest(
        method="POST",
        url="http://localhost/api/workflow/wave2/run",
        body=b'{"input":{"value":"v"},"instance_id":"new","node_max_concurrency":2}',
    )
    assert af.dispatch_function_request(app, req).status_code == 202


def test_owned_response_query_cannot_override_version():
    sdk = Sdk()
    app = af.create_function_app(registration(), sdk, require_instance_owner=False)
    sdk.schedule_new_orchestration("run", {"thread": "t"}, "iid")
    req = func.HttpRequest(
        method="POST",
        url="http://localhost/api/graphs/wave2/respond/iid/i1?version=v2",
        body=b"true",
    )
    assert af.dispatch_function_request(app, req).status_code == 400


def test_local_serve_finalizes_without_holding_worker_lock(monkeypatch):
    app = af.create_function_app(registration(), Sdk(), require_instance_owner=False)
    calls = []

    class Server:
        def __init__(self, *args, **kwargs):
            self.base_url = "http://127.0.0.1:7071"

        def start(self):
            calls.append("start")

        def stop(self):
            calls.append("stop")

    class Event:
        def wait(self, *args):
            raise KeyboardInterrupt

    monkeypatch.setattr(af, "FunctionAppHttpServer", Server)
    monkeypatch.setattr(
        af, "threading", SimpleNamespace(Event=Event, Lock=threading.Lock)
    )
    app.serve()
    assert calls == ["start", "stop"]


def test_owned_sdk_lifecycle_and_options_are_resolved_once(monkeypatch):
    instances = []

    class Client:
        def __init__(self, **options):
            self.options = options
            self.closed = False
            instances.append(self)

        def close(self):
            self.closed = True
            raise RuntimeError("close failed")

    monkeypatch.setattr(af, "DurableTaskSchedulerClient", Client)
    state = af._FunctionAppState({"host_address": "127.0.0.1:8080"})
    sdk = state.get_sdk()
    assert state.get_sdk() is sdk
    assert sdk.options == {"host_address": "127.0.0.1:8080"}
    state.stop()
    assert sdk.closed
    assert state._client_sdk is None
    state = af._FunctionAppState("127.0.0.1:8080")
    assert state._resolve_options()["host_address"] == "127.0.0.1:8080"
    monkeypatch.setenv("DTS_ENDPOINT", "127.0.0.1:8080")
    monkeypatch.delenv("DURABLE_TASK_SCHEDULER_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("DTS_CONNECTION_STRING", raising=False)
    assert af._FunctionAppState()._resolve_options()["host_address"] == "127.0.0.1:8080"


def test_stop_swallows_worker_failure_outside_lock():
    state = af._FunctionAppState()

    def stop():
        assert state._lock.acquire(blocking=False)
        state._lock.release()
        raise RuntimeError("stop failed")

    state._worker = SimpleNamespace(stop=stop)
    state.stop()
    assert state._worker is None


def test_connection_options_are_snapshotted_and_cannot_switch_a_live_client(
    monkeypatch,
):
    options = {"host_address": "first:8080"}
    state = af._FunctionAppState(options)
    options["host_address"] = "second:8080"
    assert state._resolve_options()["host_address"] == "first:8080"

    class Client:
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(af, "DurableTaskSchedulerClient", Client)
    original = state.get_sdk()
    with pytest.raises(ValueError, match="connection options"):
        state.add_registrations(
            {"wave2": registration()},
            sdk_or_options={"host_address": "second:8080"},
        )
    assert not state.registrations
    assert state.get_sdk() is original
    state.stop()
