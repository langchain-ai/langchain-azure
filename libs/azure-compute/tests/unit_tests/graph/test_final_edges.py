# Copyright (c) Microsoft. All rights reserved.

"""Checkpoint coordinate, optional-bundle and legacy replay edge contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from langchain_azure_compute.graph import _api as api
from langchain_azure_compute.graph import _checkpoints as cp
from langchain_azure_compute.graph import _config as config
from langchain_azure_compute.graph import runner as r
from tests.unit_tests.graph.test_api_edges import AsyncSdk
from tests.unit_tests.graph.test_checkpoint_edges import snapshot
from tests.unit_tests.graph.test_orchestration_edges import Context, initial
from tests.unit_tests.graph.test_wave2 import Sdk, registration


def test_persistent_checkpoint_migration_does_not_use_inline_completion_cache():
    reg = registration()
    reg.graph.checkpointer = NS(get_tuple=lambda cfg: None)
    result = r.run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "migrate",
            "expect": None,
        },
    )
    assert result["status"] == "done"


def test_resume_upgrade_values_are_forwarded_to_the_next_step():
    ctx = Context()
    gen = r.make_orchestrator(r.Limits())(ctx, initial())
    next(gen)
    gen.send(
        {
            "status": "interrupted",
            "ckpt": "c",
            "interrupt": {"id": "i", "value": r.pack(1)},
        }
    )
    step = gen.send(
        {
            "value": r.pack(1),
            "upgrade": {
                "version": "v2",
                "as_node": "next",
                "migrate_values": r.pack({"n": 2}),
            },
        }
    )
    assert step.input["as_node"] == "next"
    assert step.input["migrate_values"] == r.pack({"n": 2})
    gen.close()


def test_mid_start_input_parent_and_missing_loop_parent():
    assert cp._is_mid_start(
        NS(),
        snapshot(metadata={"source": "input"}, parent_config=cp._config("t", "expect")),
        "expect",
        [],
    )
    assert not cp._is_mid_start(
        NS(), snapshot(metadata={"source": "loop"}, next=("node",)), "expect", []
    )
    node = NS(bound=None)
    assert config._node_candidates(NS(nodes={"node": node}), "node") == [node]


@pytest.mark.parametrize(
    "user_config", [{"configurable": {}, "tags": ["test"]}, {"tags": ["test"]}]
)
def test_isolated_node_accepts_empty_user_configurable_mapping(
    monkeypatch, user_config
):
    graph = NS(
        nodes={
            "node": NS(
                bound=NS(func=lambda: None, afunc=None, invoke=lambda value, cfg: value)
            )
        }
    )
    monkeypatch.setattr(cp, "_prepare_executable_tasks", lambda *a: {})
    status, value = cp._exec_single_node(
        graph,
        snapshot(),
        "id",
        "node",
        None,
        "t",
        [],
        run_config=user_config,
    )
    assert status == "ok" and value == {"n": 1}


def test_inspection_does_not_claim_a_version_migration_jump():
    reg = registration()
    report = r.run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "inspect",
            "expect": None,
        },
    )
    assert report["status"] == "done" and "jumped" not in report


def test_sync_upgrade_without_version_and_legacy_result():
    reg, sdk = registration(), Sdk()
    reg.version = None
    reg.upgrade_on_continue_as_new = True
    run = reg.client(sdk).start({"value": "new"}, instance_id="iid")
    assert "target_version" not in sdk.scheduled[-1]
    sdk.wait_for_orchestration_completion = lambda *a, **kw: NS(
        runtime_status=NS(name="COMPLETED"),
        serialized_output=json.dumps({"gen": 0, "steps": 1, "result": r.pack(7)}),
    )
    assert run.wait_result() == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_missing_inline_bundle_during_start_is_not_serialized(
    monkeypatch, asynchronous
):
    reg, sdk = registration(), AsyncSdk() if asynchronous else Sdk()
    reg.inline_checkpoints = True
    reg.graph.invoke({"value": "old"}, api._config("t"))
    monkeypatch.setattr(api, "_export_checkpoint_bundle", lambda *a: None)
    client = reg.async_client(sdk) if asynchronous else reg.client(sdk)
    result = client.start({"value": "new"}, thread_id="t", instance_id="iid")
    if asynchronous:
        await result
    assert "checkpoint_bundle" not in sdk.scheduled[-1]["req"]
