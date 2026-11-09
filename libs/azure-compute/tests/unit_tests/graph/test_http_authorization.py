# Copyright (c) Microsoft. All rights reserved.

"""Batch HITL responses and graph-scoped owner/approver authorization."""

from __future__ import annotations

import json
from typing import TypedDict

import azure.functions as func
import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from langchain_azure_compute.graph import attach_durability
from langchain_azure_compute.graph import azure_functions as af
from langchain_azure_compute.graph._config import _check_payload
from langchain_azure_compute.graph.runner import RunError, pack, unpack
from tests.unit_tests.graph.test_functions_auth import (
    Sdk,
    app_for,
    dispatch,
    registration,
)
from tests.unit_tests.graph.test_real_backend import real_runtime as real_runtime


def pending_sdk(name="langgraph.wave2.run"):
    sdk = Sdk()
    sdk.schedule_new_orchestration(name, {"owner": "alice", "thread": "t"}, "EXP-7")
    sdk.states["EXP-7"].name = name
    sdk.states["EXP-7"].serialized_custom_status = json.dumps(
        {
            "interrupts": [
                {"id": "int-1", "value": pack({"question": "first"})},
                {"id": "int-2", "value": pack({"question": "second"})},
            ],
            "ckpt": "c",
            "step": 1,
        }
    )
    return sdk


@pytest.mark.parametrize(
    "values",
    [
        {"int-1": {"approved": True}, "int-2": {"approved": False}},
        {"int-1": {"responder": "spoof", "reviewer": "spoof"}, "int-2": False},
        {"int-1": True},
    ],
)
def test_batch_response_stamps_each_dict_without_changing_interrupt_ids(values):
    sdk = pending_sdk()
    app = app_for(registration(), sdk)
    response = dispatch(app, "respond/EXP-7", body={"value": values})
    assert response.status_code == 200
    assert len(sdk.events) == len(values)
    for _, event_name, event in sdk.events:
        interrupt_id = event_name.removeprefix("resume:")
        expected = values[interrupt_id]
        if isinstance(expected, dict):
            expected = {**expected, "responder": "alice", "reviewer": "alice"}
        assert unpack(event["value"]) == expected


def test_empty_single_payload_is_stamped_without_becoming_an_interrupt_map():
    sdk = pending_sdk()
    sdk.states["EXP-7"].serialized_custom_status = json.dumps(
        {
            "interrupts": [{"id": "int-1", "value": pack("question")}],
        }
    )
    app = app_for(registration(), sdk)
    assert dispatch(app, "respond/EXP-7", body={"value": {}}).status_code == 200
    assert unpack(sdk.events[0][2]["value"]) == {
        "responder": "alice",
        "reviewer": "alice",
    }


def test_real_parallel_http_approval_and_graph_scoped_status_read(real_runtime):
    class State(TypedDict):
        first: dict
        second: dict

    _, sdk, worker = real_runtime
    builder = StateGraph(State)
    for name in ("first", "second"):

        def review(state, _name=name):
            return {_name: interrupt({"node": _name})}

        builder.add_node(name, review)
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    reg = attach_durability(builder.compile(), name="http_approvals")

    def policy(caller, instance_id, owner, graph_name):
        return caller == "manager" and owner == "alice" and graph_name == reg.name

    app = af.create_function_app(
        reg,
        sdk,
        identity_provider=lambda req: req.headers.get("caller"),
        respond_authorizer=policy,
    )
    reg.register(worker)
    worker.start()
    run = reg.client(sdk).start({"first": {}, "second": {}}, owner="alice")
    run.wait_pending(timeout=20)
    pending = run.pending_interrupts()
    assert len(pending) == 2
    url = f"http://host/api/graphs/{reg.name}/status/{run.instance_id}"
    response = af.dispatch_function_request(
        app,
        func.HttpRequest(
            method="GET", url=url, headers={"caller": "manager"}, body=b""
        ),
    )
    assert response.status_code == 200
    values = {
        intr.interrupt_id: {"approved": intr.value["node"] == "first"}
        for intr in pending
    }
    response = af.dispatch_function_request(
        app,
        func.HttpRequest(
            method="POST",
            url=url.replace("/status/", "/respond/"),
            headers={"caller": "manager"},
            body=json.dumps({"value": values}).encode(),
        ),
    )
    assert response.status_code == 200
    result = run.wait_result(timeout=20)
    for name in ("first", "second"):
        assert result[name] == {
            "approved": name == "first",
            "responder": "manager",
            "reviewer": "manager",
        }


@pytest.mark.parametrize("arity", [2, 3, 4, "optional4", "variadic"])
def test_approvers_can_read_status_and_policy_receives_graph_scope(arity):
    calls = []

    def two(caller, instance_id):
        calls.append((caller, instance_id))
        return caller == "manager"

    def three(caller, instance_id, owner):
        calls.append((caller, instance_id, owner))
        return caller == "manager" and owner == "alice"

    def four(caller, instance_id, owner, graph_name):
        calls.append((caller, instance_id, owner, graph_name))
        return caller == "manager" and graph_name == "wave2"

    def optional_four(caller, instance_id, owner, graph_name=None):
        return four(caller, instance_id, owner, graph_name)

    def variadic(*args):
        calls.append(args)
        return args[0] == "manager"

    policy = {
        2: two,
        3: three,
        4: four,
        "optional4": optional_four,
        "variadic": variadic,
    }[arity]
    sdk = pending_sdk()
    app = app_for(registration(), sdk, respond_authorizer=policy)
    assert dispatch(app, "status/EXP-7", caller="manager").status_code == 200
    assert (
        dispatch(
            app, "respond/EXP-7/int-1", body={"value": True}, caller="manager"
        ).status_code
        == 200
    )
    expected = ("manager", "EXP-7", "alice", "wave2")
    count = 4 if arity in (4, "optional4") else 2 if arity == 2 else 3
    assert calls == [expected[:count], expected[:count]]
    calls.clear()
    assert dispatch(app, "status/EXP-7", caller="alice").status_code == 200
    assert calls == []


@pytest.mark.parametrize("policy_result", [False, "truthy", "exception"])
def test_approver_reads_fail_closed_on_rejection_or_policy_error(policy_result):
    def policy(caller, instance_id, owner, graph_name):
        if policy_result == "exception":
            raise RuntimeError("private detail")
        return policy_result

    app = app_for(registration(), pending_sdk(), respond_authorizer=policy)
    response = dispatch(app, "status/EXP-7", caller="manager")
    assert response.status_code == 403
    assert b"private detail" not in response.get_body()
    assert dispatch(app, "status/EXP-7", caller=None).status_code == 401


@pytest.mark.parametrize("route", ["status/EXP-7", "respond/EXP-7/int-1", "start"])
@pytest.mark.parametrize("auth", ["owner", "approver", "local"])
def test_routes_reject_instances_from_other_graphs_even_with_matching_owner(
    route, auth
):
    sdk = pending_sdk("langgraph.hr@v1.run")
    kwargs = {}
    if auth == "approver":
        kwargs["respond_authorizer"] = lambda *args: True
    elif auth == "local":
        kwargs["require_instance_owner"] = False
    app = app_for(registration(), sdk, **kwargs)
    body = (
        {"input": {}, "instance_id": "EXP-7"} if route == "start" else {"value": True}
    )
    response = dispatch(app, route, body=body)
    assert response.status_code in (403, 404)
    assert not sdk.events and len(sdk.scheduled) == 1


@pytest.mark.parametrize(
    "name",
    [
        None,
        "langgraph.wave2.run",
        "langgraph.wave2@v1.run",
        "langgraph.wave2@v1.2.19.run",
    ],
)
def test_matching_or_legacy_metadata_names_preserve_owner_access(name):
    app = app_for(registration(), pending_sdk(name))
    assert dispatch(app, "status/EXP-7").status_code == 200


@pytest.mark.parametrize("name", ["langgraph.wave2.extra.run", "", 42])
def test_unknown_or_malformed_graph_metadata_is_not_authorized(name):
    app = app_for(registration(), pending_sdk(name))
    assert dispatch(app, "status/EXP-7").status_code == 403


def test_approver_policy_cannot_read_an_instance_without_a_verified_owner():
    sdk = pending_sdk()
    sdk.states["EXP-7"].serialized_input = json.dumps({"thread": "t"})
    app = app_for(registration(), sdk, respond_authorizer=lambda *args: True)
    assert dispatch(app, "status/EXP-7", caller="manager").status_code == 403


def test_five_argument_approver_policy_is_rejected():
    def policy(a, b, c, d, e):
        return True

    with pytest.raises(ValueError, match="2, 3 or 4"):
        app_for(registration(), pending_sdk(), respond_authorizer=policy)


@pytest.mark.parametrize(
    "payload,inline",
    [
        ({"checkpoint_bundle": {"data": "x" * 950_001}}, True),
        ({"output": "x" * 950_001}, False),
        (["x" * 950_001], False),
        ({"checkpoint_bundle": None, "output": "x" * 950_001}, False),
    ],
)
def test_payload_advice_distinguishes_checkpoint_bundles_from_activity_data(
    payload, inline
):
    with pytest.raises(RunError) as error:
        _check_payload(payload, "activity output")
    advice = str(error.value)
    assert "payload_store" not in advice
    if inline:
        assert "persistent checkpointer" in advice
        assert all(
            name in advice for name in ("SqliteSaver", "PostgresSaver", "CosmosDBSaver")
        )
    else:
        assert "persistent checkpointer" not in advice
        assert "tool/node" in advice and "external storage" in advice
