# Copyright (c) Microsoft. All rights reserved.

"""Synchronous and asynchronous run-client boundary and resume contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langgraph.types import Command

from langchain_azure_compute.graph import RunError
from langchain_azure_compute.graph._api import AsyncRun, PendingInterrupt, Run
from langchain_azure_compute.graph.runner import pack, unpack
from tests.unit_tests.graph.test_wave2 import Sdk, registration


def state(status="RUNNING", custom=None, output=None):
    return SimpleNamespace(
        runtime_status=SimpleNamespace(name=status),
        serialized_input=json.dumps({"thread": "t"}),
        serialized_custom_status=json.dumps(custom) if custom is not None else None,
        serialized_output=json.dumps(output) if output is not None else None,
        failure_details=SimpleNamespace(message="run failed"),
    )


class AsyncSdk(Sdk):
    async def get_orchestration_state(self, *args, **kwargs):
        return super().get_orchestration_state(*args, **kwargs)

    async def schedule_new_orchestration(self, *args, **kwargs):
        return super().schedule_new_orchestration(*args, **kwargs)

    async def raise_orchestration_event(self, *args, **kwargs):
        return super().raise_orchestration_event(*args, **kwargs)

    async def wait_for_orchestration_completion(self, iid, **kwargs):
        return self.states.get(iid)


@pytest.mark.parametrize("shape", ["command", "receipts", "id", "value", "empty"])
def test_sync_resume_shapes_and_migration_data(shape):
    reg, sdk = registration(), Sdk()
    items = [{"id": "i1", "value": pack("prompt")}]
    if shape in ("command", "receipts"):
        items.append({"id": "i2", "value": pack("second")})
    sdk.states["r"] = state(custom={"interrupts": items, "step": 2, "ckpt": "c"})
    run = Run(reg, sdk, "r", "t")
    metadata = {"version": "v1", "as_node": "review", "values": {"value": "updated"}}
    if shape == "command":
        run.resume(Command(resume={"i1": True, "i2": False}), **metadata)
    elif shape == "receipts":
        run.resume({p: True for p in run.pending_interrupts()}, **metadata)
    elif shape == "id":
        run.resume("i1", True, **metadata)
    elif shape == "value":
        run.resume(Command(resume=True), **metadata)
    else:
        run.resume(**metadata)
    assert len(sdk.events) == len(items)
    for _, name, data in sdk.events:
        assert name.startswith("resume:")
        assert data["version"] == "v1" and data["as_node"] == "review"
        assert data["step"] == 2 and data["ckpt"] == "c"
        assert unpack(data["migrate_values"]) == {"value": "updated"}


def test_sync_command_stale_receipt_and_ambiguous_resume():
    reg, sdk = registration(), Sdk()
    sdk.states["r"] = state(
        custom={
            "interrupts": [
                {"id": "i1", "value": pack(1)},
                {"id": "i2", "value": pack(2)},
            ],
            "step": 2,
            "ckpt": "current",
        }
    )
    run = Run(reg, sdk, "r", "t")
    stale = PendingInterrupt("i1", 1, step=1, ckpt="old")
    run.resume(Command(resume={stale: True}))
    assert not sdk.events
    for value in (Command(resume=True), None):
        with pytest.raises(ValueError, match="multiple"):
            run.resume(value)
    with pytest.raises(ValueError, match="unknown"):
        run.resume(Command(resume={PendingInterrupt("wrong", None): True}))
    sdk.states.clear()
    with pytest.raises(RunError):
        run.resume(Command(resume=True))


@pytest.mark.asyncio
async def test_async_receipt_resume_all_migration_fields():
    reg, sdk = registration(), AsyncSdk()
    sdk.states["r"] = state(
        custom={
            "pending": {"id": "i1"},
            "step": 3,
            "ckpt": "c",
        }
    )
    run = AsyncRun(reg, sdk, "r", "t")
    receipt = await run.pending_interrupt()
    assert receipt.value is None
    await run.resume(
        receipt,
        True,
        version="v1",
        as_node="review",
        values={"value": "updated"},
    )
    payload = sdk.events[0][2]
    assert payload["version"] == "v1" and payload["as_node"] == "review"
    assert payload["step"] == 3 and payload["ckpt"] == "c"
    assert unpack(payload["migrate_values"]) == {"value": "updated"}
    assert unpack(payload["value"]) is True
    sdk.events.clear()
    await run.resume()
    assert unpack(sdk.events[0][2]["value"]) is None
    sdk.states.clear()
    assert await run.status() == "UNKNOWN"
    with pytest.raises(RunError):
        await run.resume(True)


@pytest.mark.parametrize("change", ["step", "ckpt"])
@pytest.mark.asyncio
async def test_async_wait_pending_identifies_new_checkpoint_or_step(change):
    reg, sdk = registration(), AsyncSdk()
    sdk.states["r"] = state(
        custom={
            "pending": {"id": "i1", "value": pack(1)},
            "step": 2,
            "ckpt": "new",
        }
    )
    run = AsyncRun(reg, sdk, "r", "t")
    previous = PendingInterrupt(
        "i1",
        1,
        step=1 if change == "step" else 2,
        ckpt="old" if change == "ckpt" else "new",
    )
    assert await run.wait_pending(previous, timeout=0.1)
    sdk.states["r"] = state("COMPLETED")
    assert await run.wait_pending(timeout=0.1) is None
    sdk.states["r"] = state("FAILED")
    with pytest.raises(RunError, match="run failed"):
        await run.wait_pending(timeout=0.1)


@pytest.mark.parametrize("status", ["FAILED", "TERMINATED", "COMPLETED", None])
@pytest.mark.asyncio
async def test_async_wait_result_terminal_states(status):
    reg, sdk = registration(), AsyncSdk()
    if status is not None:
        sdk.states["r"] = state(
            status,
            output={
                "gen": 1,
                "steps": 2,
                "version": "v1",
                "result": pack({"value": "result"}),
            },
        )
    run = AsyncRun(reg, sdk, "r", "t")
    if status == "COMPLETED":
        assert await run.wait_result(0.1) == {"value": "result"}
        assert run.generations == 2 and run.steps == 2 and run.version == "v1"
    else:
        with pytest.raises(RunError):
            await run.wait_result(0.1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_concurrency", 0),
        ("node_max_concurrency", 0),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_client_concurrency_validation(field, value, asynchronous):
    reg = registration()
    client = reg.async_client(AsyncSdk()) if asynchronous else reg.client(Sdk())
    with pytest.raises(ValueError):
        if asynchronous:
            await client.start({"value": "v"}, **{field: value})
        else:
            client.start({"value": "v"}, **{field: value})


@pytest.mark.asyncio
async def test_async_start_forwards_run_configuration_context_and_caps():
    reg, sdk = registration(), AsyncSdk()
    run = await reg.async_client(sdk).start(
        {"value": "v"},
        config={
            "configurable": {"thread_id": "explicit"},
            "max_concurrency": 2,
            "tags": ["tag"],
        },
        node_max_concurrency=3,
        context={"user": "u"},
        version="v1",
    )
    payload = sdk.scheduled[0]
    assert run.thread_id == "explicit"
    assert payload["version"] == "v1"
    assert payload["max_concurrency"] == 2 and payload["node_max_concurrency"] == 3
    assert unpack(payload["run_config"])["tags"] == ["tag"]
    assert payload["context"]
