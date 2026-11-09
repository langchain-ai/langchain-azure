# Copyright (c) Microsoft. All rights reserved.

"""Regression tests for Functions hosting, ownership, and HTTP boundaries."""

from __future__ import annotations

import importlib.util
import inspect
import io
import json
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, TypedDict

import azure.functions as func
import pytest
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from langchain_azure_compute.graph import Limits, attach_durability
from langchain_azure_compute.graph import azure_functions as af
from langchain_azure_compute.graph._checkpoints import _checkpoint_id, _get_state
from langchain_azure_compute.graph.runner import (
    make_orchestrator,
    pack,
    run_node,
    run_step,
    run_tool,
    unpack,
)


class State(TypedDict):
    value: str


class OwnedState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    value: str


@pytest.mark.parametrize("durable_nodes", [False, ["agent"]])
def test_owned_workflow_advances_through_tool_node_commits_and_resume(durable_nodes):
    @tool
    def echo(value: str) -> str:
        """Echo a value."""
        return value

    def agent(state):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "echo",
                            "args": {"value": state["value"]},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    def review(state):
        decision = interrupt("Approve?")
        return {"value": decision["responder"]}

    builder = StateGraph(OwnedState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([echo]))
    builder.add_node("review", review)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "tools")
    builder.add_edge("tools", "review")
    builder.add_edge("review", END)
    reg = attach_durability(
        builder.compile(),
        name="owned_flow",
        durable_nodes=durable_nodes,
    )
    sdk = Sdk()
    run = reg.client(sdk).start({"value": "hi"}, owner="alice")

    class Context:
        def call_activity(self, name, input, **kwargs):
            return ("activity", name, dict(input))

        def when_all(self, activities):
            return ("all", activities)

        def wait_for_external_event(self, name):
            return ("event", name)

        def set_custom_status(self, status):
            pass

    seen = []

    def execute(work):
        if work[0] == "all":
            return [execute(item) for item in work[1]]
        if work[0] == "event":
            return {"value": pack({"approved": True, "responder": "manager"})}
        _, name, req = work
        assert req["owner"] == "alice"
        if name == reg.step_name:
            seen.append(req["kind"])
            return run_step(
                reg.graph,
                reg.limits,
                req,
                tool_config=reg.tool_config,
                node_config=reg.node_config,
            )
        if name == reg.tool_name:
            return run_tool(reg.graph, req)
        assert name == reg.node_name
        return run_node(reg.graph, req)

    gen = make_orchestrator(reg.limits)(Context(), sdk.scheduled[0])
    work = next(gen)
    for _ in range(40):
        try:
            work = gen.send(execute(work))
        except StopIteration as result:
            assert result.value["status"] == "done"
            break
    else:
        pytest.fail("Owned workflow did not complete")
    assert "tools" in seen and "resume" in seen
    if durable_nodes:
        assert "nodes" in seen
    snap = _get_state(reg.graph, {"configurable": {"thread_id": run.thread_id}})
    assert snap.values["value"] == "manager"
    assert snap.metadata["durable_owner"] == "alice"


def registration():
    builder = StateGraph(State)
    builder.add_node("echo", lambda s: s)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return attach_durability(builder.compile(), name="wave2", version="v1")


class Sdk:
    def __init__(self):
        self.states = {}
        self.scheduled = []
        self.events = []

    def get_orchestration_state(self, instance_id, fetch_payloads=False):
        return self.states.get(instance_id)

    def schedule_new_orchestration(self, name, input, instance_id):
        self.scheduled.append(input)
        self.states[instance_id] = SimpleNamespace(
            runtime_status=SimpleNamespace(name="RUNNING"),
            serialized_input=json.dumps(input),
            serialized_custom_status=json.dumps(
                {"interrupts": [{"id": "i1", "value": pack("Approve?")}]}
            ),
            serialized_output=None,
        )

    def raise_orchestration_event(self, instance_id, event_name, data=None):
        self.events.append((instance_id, event_name, data))


def request(route="start", body=None, caller="alice", query=""):
    return func.HttpRequest(
        method="POST" if not route.startswith("status") else "GET",
        url=f"http://127.0.0.1/api/graphs/wave2/{route}{query}",
        headers={"caller": caller} if caller else {},
        params={},
        route_params={},
        body=json.dumps(body).encode() if body is not None else b"",
    )


def app_for(reg, sdk, **kwargs):
    return af.create_function_app(
        reg, sdk, identity_provider=lambda req: req.headers.get("caller"), **kwargs
    )


def dispatch(app, *args, **kwargs):
    return af.dispatch_function_request(app, request(*args, **kwargs))


def start(app, caller="alice", instance_id="EXP-7"):
    return dispatch(
        app,
        body={"input": {"value": "hello"}, "instance_id": instance_id},
        caller=caller,
    )


def test_worker_defaults_are_opt_in():
    for fn in (
        af.create_function_app,
        af.LangGraphFunctionApp,
        af.register_function_routes,
        registration().register_function_routes,
        registration().create_function_app,
    ):
        assert inspect.signature(fn).parameters["start_worker"].default is False


def test_anonymous_requires_explicit_opt_out(monkeypatch):
    monkeypatch.delenv("WEBSITE_AUTH_ENABLED", raising=False)
    reg, sdk = registration(), Sdk()
    with pytest.raises(ValueError, match="ANONYMOUS requires explicit"):
        af.create_function_app(reg, sdk, auth_level=func.AuthLevel.ANONYMOUS)
    app = af.create_function_app(
        reg, sdk, auth_level=func.AuthLevel.ANONYMOUS, require_instance_owner=False
    )
    assert start(app, caller=None).status_code == 202


@pytest.mark.parametrize("identity_source", ["callback", "easyauth"])
def test_anonymous_identity_source_enforces_owner(monkeypatch, identity_source):
    monkeypatch.delenv("WEBSITE_AUTH_ENABLED", raising=False)
    reg, sdk = registration(), Sdk()
    if identity_source == "callback":
        app = app_for(reg, sdk, auth_level=func.AuthLevel.ANONYMOUS)
        assert start(app).status_code == 202
        assert dispatch(app, "status/EXP-7", caller="bob").status_code == 403
    else:
        monkeypatch.setenv("WEBSITE_AUTH_ENABLED", "true")
        app = af.create_function_app(reg, sdk, auth_level=func.AuthLevel.ANONYMOUS)
        assert start(app, caller=None).status_code == 401
        sdk.schedule_new_orchestration(
            "run", {"owner": "alice", "thread": "t"}, "EXP-7"
        )
        req = func.HttpRequest(
            method="GET",
            url="http://localhost/api/graphs/wave2/status/EXP-7",
            headers={"x-ms-client-principal-id": "bob"},
            body=b"",
        )
        assert af.dispatch_function_request(app, req).status_code == 403


def test_durable_owner_overrides_poisoned_cache_and_missing_state():
    sdk = Sdk()
    state = af._FunctionAppState(sdk)
    sdk.schedule_new_orchestration("run", {"owner": "alice"}, "EXP-7")
    state.owner_by_instance["EXP-7"] = "bob"
    assert af._resolve_durable_owner(state, sdk, "EXP-7") == "alice"
    sdk.states.clear()
    assert af._resolve_durable_owner(state, sdk, "EXP-7") is None


@pytest.mark.parametrize("route", ["status/EXP-7", "respond/EXP-7/i1"])
def test_purged_instance_cannot_reuse_another_owners_cached_thread(route):
    reg, sdk = registration(), Sdk()
    app = app_for(reg, sdk)
    assert start(app, caller="alice").status_code == 202
    cached_thread = app._langgraph_state.thread_by_instance["EXP-7"]
    sdk.states.clear()
    bob = reg.client(sdk).start(
        {"value": "bob"},
        instance_id="EXP-7",
        owner="bob",
    )
    assert bob.thread_id != cached_thread
    response = dispatch(app, route, body={"approved": True}, caller="bob")
    assert response.status_code == 200
    assert json.loads(response.get_body())["threadId"] == bob.thread_id


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_client_reuse_checks_owner_even_with_explicit_thread(asynchronous):
    reg, sdk = registration(), Sdk()
    sdk.schedule_new_orchestration(
        "run", {"owner": "alice", "thread": "private"}, "EXP-7"
    )
    if asynchronous:

        class AsyncSdk:
            async def get_orchestration_state(self, *args, **kwargs):
                return sdk.get_orchestration_state(*args, **kwargs)

        client = reg.async_client(AsyncSdk())
        with pytest.raises(PermissionError):
            await client.start(
                {"value": "attack"},
                instance_id="EXP-7",
                thread_id="private",
                owner="bob",
            )
        run = await client.start({}, instance_id="EXP-7", owner="alice")
    else:
        client = reg.client(sdk)
        with pytest.raises(PermissionError):
            client.start(
                {"value": "attack"},
                instance_id="EXP-7",
                thread_id="private",
                owner="bob",
            )
        run = client.start({}, instance_id="EXP-7", owner="alice")
    assert run.thread_id == "private"
    assert len(sdk.scheduled) == 1


@pytest.mark.parametrize("race_stage", ["reuse", "schedule"])
def test_start_race_never_caches_attacker(race_stage):
    reg = registration()

    class RacingSdk(Sdk):
        calls = 0

        def get_orchestration_state(self, instance_id, fetch_payloads=False):
            self.calls += 1
            if self.calls == 2 and race_stage == "reuse":
                super().schedule_new_orchestration(
                    "run", {"owner": "alice", "thread": "private"}, instance_id
                )
            return super().get_orchestration_state(instance_id, fetch_payloads)

        def schedule_new_orchestration(self, name, input, instance_id):
            super().schedule_new_orchestration(
                name, {"owner": "alice", "thread": "private"}, instance_id
            )

    sdk = RacingSdk()
    app = app_for(reg, sdk)
    assert start(app, caller="bob").status_code == 403
    assert app._langgraph_state.owner_by_instance.get("EXP-7") != "bob"
    assert dispatch(app, "status/EXP-7", caller="bob").status_code == 403


@pytest.mark.parametrize("arity", [2, 3])
def test_respond_policy_separates_owner_and_stamps_identity(arity):
    reg, sdk = registration(), Sdk()
    policy = (
        (lambda caller, iid: caller == "manager")
        if arity == 2
        else (lambda caller, iid, owner: caller == "manager" and caller != owner)
    )
    app = app_for(reg, sdk, respond_authorizer=policy)
    assert start(app).status_code == 202
    value = {"approved": True, "reviewer": "vp", "responder": "vp", "version": 2}
    assert dispatch(app, "respond/EXP-7/i1", body=value).status_code == 403
    assert dispatch(app, "respond/EXP-7/i1", body=value, caller=None).status_code == 401
    assert (
        dispatch(app, "respond/EXP-7/i1", body=value, caller="manager").status_code
        == 200
    )
    stamped = unpack(sdk.events[0][2]["value"])
    assert stamped == {**value, "reviewer": "manager", "responder": "manager"}
    assert dispatch(app, "status/EXP-7", caller="manager").status_code == 403


def test_default_respond_owner_and_server_stamp():
    reg, sdk = registration(), Sdk()
    app = app_for(reg, sdk)
    assert start(app).status_code == 202
    assert (
        dispatch(
            app, "respond/EXP-7/i1", body={"approved": True}, caller="bob"
        ).status_code
        == 403
    )
    assert (
        dispatch(
            app, "respond/EXP-7/i1", body={"value": {"approved": True, "version": 2}}
        ).status_code
        == 200
    )
    assert unpack(sdk.events[0][2]["value"]) == {
        "approved": True,
        "version": 2,
        "responder": "alice",
        "reviewer": "alice",
    }


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_owner_threads_are_namespaced_and_checkpoint_owner_is_guarded(
    asynchronous,
):
    reg, sdk = registration(), Sdk()
    if asynchronous:

        class AsyncSdk:
            async def get_orchestration_state(self, *args, **kwargs):
                return sdk.get_orchestration_state(*args, **kwargs)

            async def schedule_new_orchestration(self, *args, **kwargs):
                return sdk.schedule_new_orchestration(*args, **kwargs)

        client = reg.async_client(AsyncSdk())
        run = await client.start({"value": "alice"}, instance_id="EXP-7", owner="alice")
    else:
        client = reg.client(sdk)
        run = client.start({"value": "alice"}, instance_id="EXP-7", owner="alice")
    assert run.thread_id != "EXP-7"
    payload = sdk.scheduled[-1]
    run_step(reg.graph, Limits(), payload["req"])
    snap = _get_state(reg.graph, {"configurable": {"thread_id": run.thread_id}})
    assert snap.metadata["durable_owner"] == "alice"
    sdk.states.clear()  # DTS purge must not make retained checkpoints public.
    with pytest.raises(PermissionError):
        if asynchronous:
            await client.start({"value": "bob"}, thread_id=run.thread_id, owner="bob")
        else:
            client.start({"value": "bob"}, thread_id=run.thread_id, owner="bob")
    with pytest.raises(PermissionError):
        if asynchronous:
            await client.start({"value": "anonymous"}, thread_id=run.thread_id)
        else:
            client.start({"value": "anonymous"}, thread_id=run.thread_id)
    if asynchronous:
        other = await client.start({"value": "bob"}, instance_id="EXP-7", owner="bob")
    else:
        other = client.start({"value": "bob"}, instance_id="EXP-7", owner="bob")
    assert other.thread_id != run.thread_id


def test_step_checks_checkpoint_owner_before_execution():
    reg = registration()
    req = {
        "thread": "private",
        "kind": "start",
        "expect": None,
        "input": pack({"value": "alice"}),
        "owner": "alice",
        "instance_id": "A",
    }
    run_step(reg.graph, Limits(), req)
    snap = _get_state(reg.graph, {"configurable": {"thread_id": "private"}})
    with pytest.raises(PermissionError):
        run_step(
            reg.graph,
            Limits(),
            {
                **req,
                "expect": _checkpoint_id(snap),
                "owner": "bob",
                "instance_id": "B",
            },
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_concurrency", "oops"),
        ("node_max_concurrency", {}),
        ("timeout", "oops"),
        ("version", "v404"),
        ("instance_id", "../../private"),
        ("instance_id", "x" * 129),
        ("instance_id", ""),
        ("instance_id", 42),
        ("reuse_existing", "maybe"),
        ("wait", "maybe"),
    ],
)
def test_start_validation_is_400(field, value):
    reg, sdk = registration(), Sdk()
    app = app_for(reg, sdk)
    resp = dispatch(app, body={"input": {"value": "hi"}, field: value})
    assert resp.status_code == 400
    assert not sdk.scheduled


@pytest.mark.parametrize("value", ["false", "0", "no", "", False])
def test_false_reuse_existing_is_not_truthy(value):
    reg, sdk = registration(), Sdk()
    app = app_for(reg, sdk)
    assert start(app).status_code == 202
    assert (
        dispatch(
            app,
            body={
                "input": {"value": "new"},
                "instance_id": "EXP-7",
                "reuse_existing": value,
            },
        ).status_code
        == 202
    )
    assert len(sdk.scheduled) == 2


def test_status_urls_quote_instance_ids():
    reg, sdk = registration(), Sdk()
    result = json.loads(start(app_for(reg, sdk), instance_id="alice:EXP@7").get_body())
    assert "alice%3AEXP%407" in result["statusQueryGetUri"]


@pytest.mark.parametrize("route", ["start", "respond/EXP-7/i1"])
def test_request_body_cap(route):
    reg, sdk = registration(), Sdk()
    app = app_for(reg, sdk)
    start(app)
    resp = dispatch(app, route, body={"input": "x" * 1_000_001})
    assert resp.status_code == 413


@pytest.mark.parametrize("key", ["__packed__", "__resume_map__", "checkpoint_bundle"])
def test_internal_resume_envelopes_rejected(key):
    reg, sdk = registration(), Sdk()
    app = app_for(reg, sdk)
    start(app)
    assert (
        dispatch(app, "respond/EXP-7/i1", body={"value": {key: "attack"}}).status_code
        == 400
    )
    assert not sdk.events


def test_server_errors_are_sanitized():
    reg, sdk = registration(), Sdk()
    app = app_for(reg, sdk)

    def fail(*args, **kwargs):
        raise RuntimeError("connection-string=private; secret internal traceback")

    sdk.schedule_new_orchestration = fail
    resp = start(app)
    assert resp.status_code == 500
    assert b"private" not in resp.get_body()
    assert b"traceback" not in resp.get_body()


def test_owner_and_thread_caches_are_bounded():
    state = af._FunctionAppState(Sdk())
    for cache in (state.owner_by_instance, state.thread_by_instance):
        assert isinstance(cache, OrderedDict)
        for i in range(1100):
            cache[str(i)] = "v"
        assert len(cache) <= 1024
        assert "0" not in cache


def test_stop_releases_lock_before_external_callbacks():
    state = af._FunctionAppState()
    acquired = []

    def check_lock():
        ok = state._lock.acquire(blocking=False)
        acquired.append(ok)
        if ok:
            state._lock.release()

    state._worker = SimpleNamespace(stop=check_lock)
    state._client_sdk = SimpleNamespace(close=check_lock)
    state._owns_client = True
    state.stop()
    assert acquired == [True, True]
    assert state._worker is None and state._client_sdk is None
    state.stop()
    assert len(acquired) == 2


def test_embedded_worker_threads_are_daemon_before_start(monkeypatch):
    class Worker:
        def __init__(self, **kwargs):
            pass

        def start(self):
            self._runLoop = threading.Thread(target=lambda: None)
            self._runLoop.start()
            self._runLoop.join()

        def stop(self):
            pass

    monkeypatch.setattr(af, "DurableTaskSchedulerWorker", Worker)
    state = af._FunctionAppState({}, start_worker=True)
    worker = state.ensure_worker()
    assert worker._runLoop.daemon
    state.stop()


def test_failed_worker_start_rolls_back_registrations(monkeypatch):
    class Worker:
        def __init__(self, **kwargs):
            pass

        def add_orchestrator(self, *args, **kwargs):
            pass

        def add_activity(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("worker failed")

        def stop(self):
            pass

    monkeypatch.setattr(af, "DurableTaskSchedulerWorker", Worker)
    app = func.FunctionApp()
    with pytest.raises(RuntimeError, match="worker failed"):
        af.register_function_routes(app, registration(), {}, start_worker=True)
    assert not getattr(app, "_langgraph_state", None)
    assert not app._function_builders


def test_route_registration_failure_does_not_publish_state(monkeypatch):
    app = func.FunctionApp()
    original = app.route
    calls = 0

    def broken_route(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("route failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(app, "route", broken_route)
    with pytest.raises(RuntimeError, match="route failed"):
        af.register_function_routes(app, registration(), Sdk())
    assert not getattr(app, "_langgraph_state", None)
    assert not app._function_builders


def test_incremental_registrations_preserve_security_and_do_not_duplicate_routes():
    reg, sdk = registration(), Sdk()
    app = app_for(reg, sdk, auth_level=func.AuthLevel.ANONYMOUS)
    builders = list(app._function_builders)
    extra = registration()
    extra.name = "additional"
    af.register_function_routes(app, extra)
    assert len(app._langgraph_state.registrations) == 2
    assert app._function_builders == builders
    with pytest.raises(ValueError, match="security cannot change"):
        af.register_function_routes(app, reg, require_instance_owner=False)
    assert start(app, caller=None).status_code == 401
    assert start(app).status_code == 202


def test_external_client_is_reusable_after_stop_without_being_closed():
    reg, sdk = registration(), Sdk()
    app = app_for(reg, sdk)
    app.stop_worker()
    assert start(app).status_code == 202


@pytest.mark.parametrize("lookup", ["missing", "error", "no_owner"])
def test_custom_respond_policy_fails_closed_on_owner_lookup(lookup):
    reg, sdk = registration(), Sdk()
    calls = []
    app = app_for(
        reg,
        sdk,
        respond_authorizer=lambda *args: calls.append(args) or True,
    )
    if lookup == "error":

        def fail(*args, **kwargs):
            raise OSError("private error")

        sdk.get_orchestration_state = fail
    elif lookup == "no_owner":
        sdk.schedule_new_orchestration("run", {"thread": "t"}, "EXP-7")
    assert dispatch(app, "respond/EXP-7/i1", body={"approved": True}).status_code == 403
    assert not calls and not sdk.events


def test_checkpoint_owner_survives_orchestrator_steps():
    from langchain_azure_compute.graph.runner import OrchestratorState

    state = OrchestratorState.from_payload(
        {
            "thread": "t",
            "owner": "alice",
            "step_name": "langgraph.wave2.step",
            "req": {"thread": "t", "kind": "step", "expect": "checkpoint"},
        }
    )
    assert state.apply_step_options_to({})["owner"] == "alice"
    assert state.apply_context_to({})["owner"] == "alice"
    assert state.to_can_payload()["req"]["owner"] == "alice"


def test_incremental_worker_failure_preserves_existing_state(monkeypatch):
    app = af.create_function_app(registration(), {}, require_instance_owner=False)
    state = app._langgraph_state
    original_regs = state.registrations.copy()
    original_builders = list(app._function_builders)
    new_reg = registration()
    new_reg.name = "additional"

    class Worker:
        def __init__(self, **kwargs):
            raise RuntimeError("startup failed")

    monkeypatch.setattr(af, "DurableTaskSchedulerWorker", Worker)
    with pytest.raises(RuntimeError, match="startup failed"):
        af.register_function_routes(
            app, new_reg, {}, start_worker=True, require_instance_owner=False
        )
    assert app._langgraph_state is state
    assert state.registrations == original_regs
    assert app._function_builders == original_builders
    assert state._want_worker is False


def sample_module(sample):
    path = Path(__file__).resolve().parents[5] / "samples" / "langgraph-durabletask"
    path = path / sample / "sample.py"
    if not path.exists():
        pytest.skip("Samples are not included in the isolated release test layout")
    spec = importlib.util.spec_from_file_location(f"wave2_{sample}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "sample",
    [
        "09_two_process_crash_recovery",
        "10_azure_functions_hitl",
    ],
)
@pytest.mark.parametrize(
    "endpoint",
    [
        "127.0.0.1:8080",
        "Endpoint=http://127.0.0.1:8080;TaskHub=parent-hub;Authentication=None",
    ],
)
def test_child_scheduler_environment_is_exact(sample, endpoint, monkeypatch):
    mod = sample_module(sample)
    conflicts = {
        "DURABLE_TASK_SCHEDULER_CONNECTION_STRING": "Endpoint=http://bad:9999;TaskHub=bad",
        "DTS_CONNECTION_STRING": "Endpoint=http://other:8888;TaskHub=other",
        "DTS_ENDPOINT": "bad:1234",
        "DTS_EMULATOR": "bad:5555",
        "DTS_TASKHUB": "bad-hub",
        "TASKHUB_NAME": "other-hub",
    }
    for key, value in conflicts.items():
        monkeypatch.setenv(key, value)
    environments = []

    def spawn(*args, **kwargs):
        environments.append(kwargs["env"])
        return SimpleNamespace(stdout=io.StringIO("WORKER_READY pid=77\n"))

    monkeypatch.setattr(mod.subprocess, "Popen", spawn)
    _, pid = mod._spawn_worker_subprocess(
        "worker",
        "wave2",
        endpoint=endpoint,
        taskhub="parent-hub",
    )
    assert pid == 77
    env = environments[0]
    assert env["DTS_TASKHUB"] == "parent-hub"
    expected_key = (
        "DURABLE_TASK_SCHEDULER_CONNECTION_STRING"
        if "Endpoint=" in endpoint
        else "DTS_ENDPOINT"
    )
    assert env[expected_key] == endpoint
    for key in conflicts:
        if key not in (expected_key, "DTS_TASKHUB"):
            assert key not in env
    assert os.environ["DTS_TASKHUB"] == "bad-hub"


@pytest.mark.parametrize(
    "sample",
    [
        "09_two_process_crash_recovery",
        "10_azure_functions_hitl",
    ],
)
def test_sample_endpoint_resolution_matches_sdk_precedence(sample, monkeypatch):
    mod = sample_module(sample)
    for key in (
        "DURABLE_TASK_SCHEDULER_CONNECTION_STRING",
        "DTS_CONNECTION_STRING",
        "DTS_ENDPOINT",
        "DTS_EMULATOR",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DTS_EMULATOR", "true")
    assert mod._resolve_endpoint() == "127.0.0.1:8080"
    connection = "Endpoint=http://127.0.0.1:8080;TaskHub=parent;Authentication=None"
    monkeypatch.setenv("DTS_CONNECTION_STRING", connection)
    monkeypatch.setenv("DTS_ENDPOINT", "bad:8888")
    assert mod._resolve_endpoint() == connection
    preferred = "Endpoint=http://127.0.0.1:9999;TaskHub=preferred;Authentication=None"
    monkeypatch.setenv("DURABLE_TASK_SCHEDULER_CONNECTION_STRING", preferred)
    assert mod._resolve_endpoint() == preferred


def test_sample_host_defaults_and_verified_responder(monkeypatch):
    monkeypatch.setenv("START_EMBEDDED_WORKER", "0")
    monkeypatch.setenv("LOCAL_DEMO_IDENTITY", "1")
    mod = sample_module("10_azure_functions_hitl")._load_function_app_module()
    assert mod.app._langgraph_state._want_worker is False
    assert (
        inspect.signature(mod.build_expense_function_app)
        .parameters["start_worker"]
        .default
        is False
    )
    sdk = Sdk()
    app = mod.build_expense_function_app(dts_options=sdk)

    def call(route, body, principal):
        return af.dispatch_function_request(
            app,
            func.HttpRequest(
                method="POST",
                url=f"http://localhost/api/graphs/expense_approval/{route}",
                headers={"x-demo-user": principal},
                body=json.dumps(body).encode(),
            ),
        )

    assert (
        call(
            "start",
            {
                "input": {"expense_id": "EXP-7", "employee": "Alice", "amount": 12},
                "instance_id": "EXP-7",
            },
            "demo-user",
        ).status_code
        == 202
    )
    body = {"approved": True, "reviewer": "spoofed"}
    assert call("respond/EXP-7/i1", body, "demo-user").status_code == 403
    assert call("respond/EXP-7/i1", body, "vp-finance").status_code == 200
    assert unpack(sdk.events[0][2]["value"])["responder"] == "vp-finance"
    assert mod._authorize_expense_response("vp-finance", "EXP-7", "demo-user")
    assert not mod._authorize_expense_response("vp-finance", "EXP-7", "vp-finance")
    monkeypatch.delenv("LOCAL_DEMO_IDENTITY")
    monkeypatch.setenv("APPROVER_PRINCIPAL_IDS", "manager-id, finance-id")
    assert mod._authorize_expense_response("manager-id", "EXP-7", "employee-id")
    assert not mod._authorize_expense_response("other-id", "EXP-7", "employee-id")


def test_embedded_worker_process_exits_and_runs_atexit():
    script = """
import threading
from langchain_azure_compute.graph import azure_functions as af
class Worker:
    def __init__(self, **kwargs):
        self.done = threading.Event()
    def start(self):
        self._runLoop = threading.Thread(target=self.done.wait)
        self._runLoop.start()
    def stop(self):
        self.done.set()
        self._runLoop.join(timeout=1)
        print('ATEXIT_STOP', flush=True)
af.DurableTaskSchedulerWorker = Worker
state = af._FunctionAppState({}, start_worker=True)
state.ensure_worker()
print('READY', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "READY" in result.stdout and "ATEXIT_STOP" in result.stdout


def test_http_server_enforces_cap_before_reading_body():
    import http.client

    app = app_for(registration(), Sdk())
    with af.FunctionAppHttpServer(app) as server:
        conn = http.client.HTTPConnection(server.host, server.port, timeout=5)
        try:
            # Advertise an oversized body but send no bytes: must reject before read.
            conn.request(
                "POST",
                "/api/graphs/wave2/start",
                headers={"Content-Length": "1000001"},
            )
            resp = conn.getresponse()
            assert resp.status == 413
            resp.read()
        finally:
            conn.close()
