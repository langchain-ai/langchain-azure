# Copyright (c) Microsoft. All rights reserved.

"""HTTP transport failures, fail-closed callbacks, and atomic worker replacement."""

from __future__ import annotations

import http.client
from types import SimpleNamespace as NS

import azure.functions as func
import pytest

from langchain_azure_compute.graph import RunError
from langchain_azure_compute.graph import azure_functions as af
from tests.unit_tests.graph.test_functions_auth import Sdk, registration


def request(path, body=b"", method="POST"):
    return func.HttpRequest(method=method, url="http://host/api/" + path, body=body)


def test_boolean_integer_and_recursive_resume_validation():
    for value in ("true", "yes", 1):
        assert af._parse_bool(value)
    for value in (None, [], {}, 1.5):
        with pytest.raises(ValueError):
            af._parse_bool(value)
    with pytest.raises(ValueError):
        af._parse_int("-1", "timeout")
    assert af._has_reserved_resume_keys([{"checkpoint_bundle": {}}])
    assert not af._has_reserved_resume_keys([{"version": 2}])


def test_worker_replacement_and_start_cleanup_errors(monkeypatch):
    reg = registration()
    monkeypatch.setattr(reg, "register", lambda worker: None)
    stopped = []

    class Worker:
        def __init__(self, **options):
            pass

        def start(self):
            pass

        def stop(self):
            stopped.append(self)
            raise RuntimeError("stop failed")

    monkeypatch.setattr(af, "DurableTaskSchedulerWorker", Worker)
    monkeypatch.setattr(af, "DurableTaskSchedulerClient", lambda **kw: Sdk())
    state = af._FunctionAppState({}, start_worker=True)
    state.registrations = {"wave2": reg}
    state.get_sdk()
    old = state._worker
    with state._lock:
        state._ensure_worker_locked({})
    assert state._worker is old
    second = registration()
    second.name = "second"
    monkeypatch.setattr(second, "register", lambda worker: None)
    state.add_registrations({"second": second}, sdk_or_options={})
    assert state._worker is not old and old in stopped
    with pytest.raises(ValueError, match="already registered"):
        state.add_registrations({"wave2": registration()})
    state._owns_client = True
    state._client_sdk = NS()
    state.stop()

    class Broken(Worker):
        def start(self):
            raise ValueError("start failed")

    monkeypatch.setattr(af, "DurableTaskSchedulerWorker", Broken)
    state = af._FunctionAppState({}, start_worker=True)
    state.registrations = {"wave2": reg}
    with pytest.raises(ValueError, match="start failed"):
        state.ensure_worker()
    assert state._worker is None


@pytest.mark.parametrize("failure", [None, ValueError, RunError])
def test_respond_infers_interrupt_and_sanitizes_validation_errors(monkeypatch, failure):
    reg, sdk = registration(), Sdk()
    calls = []

    def resume(*args):
        if failure:
            raise failure("private detail")
        calls.append(args)

    run = NS(instance_id="iid", thread_id="t", status=lambda: "RUNNING", resume=resume)
    monkeypatch.setattr(reg, "client", lambda client: NS(attach_run=lambda iid: run))
    app = af.create_function_app(reg, sdk, require_instance_owner=False)
    response = af.dispatch_function_request(
        app, request("graphs/wave2/respond/iid", b'{"approved": true}')
    )
    assert response.status_code == (
        400 if failure is ValueError else 409 if failure else 200
    )
    assert b"private detail" not in response.get_body()
    if not failure:
        assert calls == [({"approved": True},)]


def test_callback_and_run_conflicts_are_fail_closed(monkeypatch):
    reg, sdk = registration(), Sdk()
    sdk.schedule_new_orchestration("run", {"thread": "t", "owner": "alice"}, "iid")

    def broken(*args):
        raise RuntimeError("private detail")

    app = af.create_function_app(
        reg, sdk, identity_provider=lambda req: "bob", respond_authorizer=broken
    )
    assert (
        af.dispatch_function_request(
            app, request("graphs/wave2/respond/iid", b'{"value": true}')
        ).status_code
        == 403
    )

    def conflict(*args, **kwargs):
        raise RunError("private conflict")

    monkeypatch.setattr(reg, "client", lambda sdk: NS(start=conflict))
    app = af.create_function_app(reg, sdk, require_instance_owner=False)
    assert (
        af.dispatch_function_request(
            app, request("graphs/wave2/start", b'{"input": {}}')
        ).status_code
        == 409
    )
    af.stop_function_app(NS())


def test_dispatch_prefixes_non_http_triggers_and_missing_routes():
    app = func.FunctionApp()
    app._function_builders.append(NS(build=lambda auth: NS(get_trigger=lambda: None)))
    for url in ("http://host/api", "http://host/other", "http://host/api/other"):
        req = func.HttpRequest(method="GET", url=url, body=b"")
        assert af.dispatch_function_request(app, req).status_code == 404


def test_http_server_invalid_lengths_dispatch_errors_empty_body_and_headers(
    monkeypatch,
):
    app = NS()

    def dispatch(app, req):
        if req.url.endswith("boom"):
            raise RuntimeError("private detail")
        return func.HttpResponse(
            body=b"", headers={"Content-Type": "text/plain", "X-Trace": "ok"}
        )

    monkeypatch.setattr(af, "dispatch_function_request", dispatch)
    server = af.FunctionAppHttpServer(app).start()
    server.start()
    try:
        for path, length, expected in (
            ("/ok", "bad", 400),
            ("/ok", "1000001", 413),
            ("/boom", "0", 500),
            ("/ok", "0", 200),
        ):
            conn = http.client.HTTPConnection(server.host, server.port, timeout=5)
            try:
                conn.request("POST", path, body=b"", headers={"Content-Length": length})
                response = conn.getresponse()
                assert response.status == expected
                assert b"private detail" not in response.read()
                if expected == 200:
                    assert response.getheader("X-Trace") == "ok"
            finally:
                conn.close()
    finally:
        server.stop()
        server.stop()


def test_registration_factory_and_empty_app_lifecycle():
    reg = registration()
    app = reg.create_function_app(Sdk(), require_instance_owner=False)
    assert app._langgraph_state.registrations[reg.name] is reg
    with pytest.raises(ValueError, match="Provide"):
        af.LangGraphFunctionApp()
