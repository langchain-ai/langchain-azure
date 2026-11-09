# Copyright (c) Microsoft. All rights reserved.

"""SDK configuration, registered activity wrappers, and client edge contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest
from langgraph.graph import StateGraph

from langchain_azure_compute.graph import _api as api
from langchain_azure_compute.graph.runner import RunError, pack, unpack
from tests.unit_tests.graph.test_client_boundaries import AsyncSdk, state
from tests.unit_tests.graph.test_functions_auth import Sdk, registration


@pytest.fixture
def scheduler_env(monkeypatch):
    for key in (
        "DTS_ENDPOINT",
        "DTS_EMULATOR",
        "DTS_CONNECTION_STRING",
        "DURABLE_TASK_SCHEDULER_CONNECTION_STRING",
        "DTS_TASKHUB",
        "TASKHUB_NAME",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize(
    "variable,value",
    [
        ("DTS_ENDPOINT", "127.0.0.1:8080"),
        ("DTS_EMULATOR", "true"),
        (
            "DTS_ENDPOINT",
            "Endpoint=http://127.0.0.1:8080;TaskHub=hub;Authentication=None",
        ),
        (
            "DTS_CONNECTION_STRING",
            "Endpoint=http://127.0.0.1:8080;TaskHub=hub;Authentication=None",
        ),
        (
            "DURABLE_TASK_SCHEDULER_CONNECTION_STRING",
            "Endpoint=http://127.0.0.1:8080;TaskHub=hub;Authentication=None",
        ),
    ],
)
def test_all_scheduler_environment_sources(variable, value, scheduler_env, monkeypatch):
    monkeypatch.setenv(variable, value)
    assert api.sdk_options()["host_address"] == "127.0.0.1:8080"


@pytest.mark.parametrize(
    "mode",
    [
        "ManagedIdentity",
        "WorkloadIdentity",
        "Environment",
        "AzureCli",
        "AzurePowerShell",
        "DefaultAzure",
    ],
)
@pytest.mark.parametrize("credential_given", [False, True])
def test_each_credential_mode_preserves_explicit_credentials(
    mode,
    credential_given,
    scheduler_env,
    monkeypatch,
):
    import azure.identity

    created = []

    def credential(**kwargs):
        result = NS(options=kwargs)
        created.append(result)
        return result

    for name in (
        "ManagedIdentityCredential",
        "WorkloadIdentityCredential",
        "EnvironmentCredential",
        "AzureCliCredential",
        "AzurePowerShellCredential",
        "DefaultAzureCredential",
    ):
        monkeypatch.setattr(azure.identity, name, credential)
    supplied = object() if credential_given else None
    result = api.sdk_options(
        connection_string=(
            f"Endpoint=https://host.durabletask.io;TaskHub=hub;Authentication={mode};"
            "ClientID=client;ignored;Unknown=value"
        ),
        token_credential=supplied,
    )
    assert result["token_credential"] is (supplied if credential_given else created[0])
    assert len(created) == (0 if credential_given else 1)
    assert result["secure_channel"]


def test_sdk_options_invalid_and_fallback_values(scheduler_env, monkeypatch):
    with pytest.raises(ValueError, match="required"):
        api.sdk_options()
    with pytest.raises(ValueError, match="unsupported"):
        api.sdk_options(connection_string="Endpoint=localhost:8080;Authentication=typo")
    monkeypatch.setenv("DTS_ENDPOINT", "127.0.0.1:8080")
    assert (
        api.sdk_options(connection_string="Unknown=x;bad")["host_address"]
        == "127.0.0.1:8080"
    )
    monkeypatch.delenv("DTS_ENDPOINT")
    monkeypatch.setenv("DTS_EMULATOR", "yes")
    assert (
        api.sdk_options(connection_string="Unknown=x")["host_address"]
        == "127.0.0.1:8080"
    )
    assert api._extract_hostname("[::1]:8080") == "::1"
    assert api._strip_scheme("GRPCS://host/") == ("host", True)
    assert api._strip_scheme("grpc://host/") == ("host", False)
    assert repr(api.PendingInterrupt("interrupt", 2)).startswith("PendingInterrupt(")


class Worker:
    def __init__(self, logger=None):
        self.activities = {}
        self._logger = logger
        self._is_running = False
        self._work_item_filters = None
        self.filters_called = False

    def add_activity(self, function):
        self.activities[function.__name__] = function

    def add_orchestrator(self, function):
        self.orchestrator = function

    def use_work_item_filters(self):
        self.filters_called = True
        raise RuntimeError("optional filtering unavailable")


@pytest.mark.parametrize("logging", [False, True])
def test_activity_wrappers_delegate_log_and_transport_failures(monkeypatch, logging):
    logs = []
    logger = NS(info=logs.append, warning=logs.append) if logging else None
    reg = registration()
    worker = Worker(logger)
    reg.register(worker)
    assert worker.filters_called
    ctx = NS(instance_id="instance")
    monkeypatch.setattr(api, "_get_state", lambda *a, **k: NS(next=("a", "b")))
    for kind, report in (
        ("step", {"status": "running"}),
        ("nodes", {"status": "nodes", "node_tasks": [{"name": "n"}]}),
        ("tools", {"status": "tools", "node": "tools", "tool_calls": [{"name": "t"}]}),
    ):
        monkeypatch.setattr(api, "run_step", lambda *a, **k: report)
        assert (
            worker.activities[reg.step_name](ctx, {"thread": "t", "kind": kind})
            == report
        )

    def fail(*a, **k):
        raise RunError("invalid step")

    monkeypatch.setattr(api, "run_step", lambda *a, **k: {"status": "running"})
    assert (
        worker.activities[reg.step_name](ctx, {"thread": "", "kind": "step"})["status"]
        == "running"
    )
    monkeypatch.setattr(api, "run_step", fail)
    assert worker.activities[reg.step_name](NS(), {"thread": ""})["status"] == "error"

    def user_fail(*a, **k):
        raise ValueError("user error")

    monkeypatch.setattr(api, "run_step", user_fail)
    with pytest.raises(ValueError):
        worker.activities[reg.step_name](ctx, {"kind": "step", "thread": "t"})
    for name, attr in ((reg.tool_name, "run_tool"), (reg.node_name, "run_node")):
        monkeypatch.setattr(api, attr, lambda *a, **k: {"status": "ok"})
        assert (
            worker.activities[name](ctx, {"call": {"t": "bad", "b64": ""}})["status"]
            == "ok"
        )
        monkeypatch.setattr(api, attr, fail)
        assert worker.activities[name](ctx, {})["status"] == "error"
        monkeypatch.setattr(api, attr, user_fail)
        with pytest.raises(ValueError):
            worker.activities[name](ctx, {})
    error = ValueError("handled")
    error._durable_tool_error_info = {"handled": True, "content": "safe"}

    def transport(*a, **k):
        raise error

    monkeypatch.setattr(api, "run_tool", transport)
    with pytest.raises(RuntimeError, match="DURABLE_HANDLED_TOOL_ERROR"):
        worker.activities[reg.tool_name](ctx, {"call": pack({"name": "t"})})
    if logging:
        assert any("instance" in line for line in logs)


def test_logging_read_failures_and_versioned_delegates(monkeypatch):
    reg = registration()
    reg.previous_versions["v1"] = reg.graph
    worker = Worker(NS(info=lambda text: None, warning=lambda text: None))
    worker._is_running = True
    reg.register(worker)
    assert not worker.filters_called

    def fail(*args, **kwargs):
        raise OSError("checkpoint unavailable")

    monkeypatch.setattr(api, "_get_state", fail)
    monkeypatch.setattr(api, "run_step", lambda *a, **k: {"status": "done"})
    assert (
        worker.activities["langgraph.wave2@v1.step"](
            NS(),
            {
                "kind": "resume",
                "thread": "t",
                "checkpoint_bundle": {},
            },
        )["status"]
        == "done"
    )
    for kind in ("tool", "node"):
        monkeypatch.setattr(api, "run_" + kind, lambda *a, **k: {"status": "ok"})
        assert (
            worker.activities["langgraph.wave2@v1." + kind](NS(), {})["status"] == "ok"
        )
    graph = reg.graph
    graph._durable_orig_interrupt_before = ["echo"]
    graph._durable_orig_interrupt_after = ["echo"]
    reg._configure_single_graph(
        graph, reg.tool_config, reg.node_config, overwrite_config=False
    )


@pytest.mark.parametrize("field", ["serialized_input", "serialized_output"])
def test_attach_resolves_payload_thread_and_ignores_malformed_payloads(field):
    reg, sdk = registration(), Sdk()
    meta = state()
    meta.serialized_input = "invalid"
    meta.serialized_output = "[]"
    setattr(meta, field, json.dumps({"thread": "actual"}))
    sdk.states["iid"] = meta
    assert reg.client(sdk).attach_run("iid").thread_id == "actual"
    meta.serialized_input, meta.serialized_output = "{}", "invalid"
    assert reg.client(sdk).attach_run("iid").thread_id == "iid"
    assert reg.client(NS()).attach_run("iid", "hint").thread_id == "hint"


def test_client_reuse_payloads_missing_and_invalid_owner():
    for payload in ("invalid", "[]", "{}"):
        with pytest.raises(PermissionError):
            api._existing_run_input(NS(serialized_input=payload), "alice")


def test_uncompiled_current_and_previous_graph_are_rejected():
    from tests.unit_tests.graph.test_functions_auth import State

    builder = StateGraph(State)
    with pytest.raises(ValueError, match="uncompiled"):
        api.attach_durability(builder, name="bad")
    with pytest.raises(ValueError, match="uncompiled"):
        api.attach_durability(
            registration().graph, name="bad", previous_versions={"old": builder}
        )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_reuse_fetches_missing_payload_and_preserves_checkpoint_owner(async_mode):
    reg, sdk = registration(), Sdk()
    sdk.states["iid"] = state()
    sdk.states["iid"].serialized_input = None
    calls = []

    class Sync:
        def get_orchestration_state(self, iid, fetch_payloads=False):
            calls.append(fetch_payloads)
            return state() if fetch_payloads else sdk.states["iid"]

    class Async:
        async def get_orchestration_state(self, iid, fetch_payloads=False):
            return Sync().get_orchestration_state(iid, fetch_payloads)

    if async_mode:
        run = await reg.async_client(Async()).start({}, instance_id="iid")
    else:
        run = reg.client(Sync()).start({}, instance_id="iid")
    assert run.thread_id == "t" and calls == [False, True]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_pending_without_value_and_wait_timeout(async_mode):
    reg, sdk = registration(), AsyncSdk() if async_mode else Sdk()
    sdk.states["iid"] = state(custom={"pending": {"id": "i"}, "step": 1, "ckpt": "c"})
    run = (
        api.AsyncRun(reg, sdk, "iid", "t")
        if async_mode
        else api.Run(reg, sdk, "iid", "t")
    )
    if async_mode:
        assert (await run.pending_interrupt()).value is None
        with pytest.raises(TimeoutError):
            await run.wait_pending(
                api.PendingInterrupt("i", None, step=1, ckpt="c"), timeout=0.025
            )
        sdk.states["iid"] = state(custom={})
        assert await run.pending_interrupts() == []
    else:
        assert run.pending_interrupt().value is None
        with pytest.raises(TimeoutError):
            run.wait_pending(
                api.PendingInterrupt("i", None, step=1, ckpt="c"), timeout=0.025
            )
        sdk.states["iid"] = state(custom={})
        assert run.pending_interrupts() == [] and run.pending_interrupt() is None


def test_resume_static_id_mapping_validation_and_stale_receipt():
    reg, sdk = registration(), Sdk()
    sdk.states["iid"] = state(
        custom={"pending": {"id": "i", "value": pack(1)}, "ckpt": "new"}
    )
    run = api.Run(reg, sdk, "iid", "t")
    run.resume("i")
    assert unpack(sdk.events[0][2]["value"]) is None
    run.resume({api.PendingInterrupt("i", 1, ckpt="old"): True})
    assert len(sdk.events) == 1
    with pytest.raises(ValueError, match="unknown"):
        run.resume(api.STATIC_INTERRUPT_PREFIX + "unknown")
    with pytest.raises(ValueError, match="unknown"):
        run.resume({api.PendingInterrupt("unknown", 1): True})


@pytest.mark.asyncio
async def test_async_resume_value_with_no_checkpoint_coordinates():
    reg, sdk = registration(), AsyncSdk()
    sdk.states["iid"] = state(custom={"pending": {"id": "i", "value": pack(True)}})
    run = api.AsyncRun(reg, sdk, "iid", "t")
    await run.resume(True)
    assert sdk.events[0][2] == {"value": pack(True)}
    await run.resume(api.PendingInterrupt("i", None))
    assert sdk.events[1][2] == {"value": pack(None)}


def test_sync_cache_eviction_config_concurrency_and_empty_reuse():
    reg, sdk = registration(), Sdk()
    reg._active_thread_instances.update({str(i): str(i) for i in range(256)})
    reg.client(sdk).start({}, config={"max_concurrency": 2})
    assert "0" not in reg._active_thread_instances
    assert len(reg._active_thread_instances) == 256
    reg.graph.interrupt_before_nodes = ()
    reg.graph.interrupt_after_nodes = ()
    api.attach_durability(reg.graph, name="tuple_pauses")
    sdk.states["empty"] = state()
    sdk.states["empty"].serialized_input = "{}"
    assert reg.client(sdk).start({}, instance_id="empty").thread_id == "empty"
    sdk.states["absent_fields"] = NS(serialized_input=None, serialized_output=None)
    assert reg.client(sdk).attach_run("absent_fields").thread_id == "absent_fields"


def test_pending_large_values_are_hydrated_from_matching_checkpoint_tasks(monkeypatch):
    reg, sdk = registration(), Sdk()
    run = api.Run(reg, sdk, "iid", "t")
    snap = NS(
        tasks=(
            NS(interrupts=()),
            NS(
                interrupts=(
                    NS(id="other", value=1),
                    NS(id="i", value=2),
                )
            ),
        ),
        config=api._config("t"),
        metadata={},
    )
    monkeypatch.setattr(api, "_get_state", lambda *args: snap)
    assert run._resolve_interrupt({"id": "i"}).value == 2
    assert run._resolve_interrupt({"id": "missing"}).value is None


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, "v1"])
async def test_async_upgrade_payload_and_legacy_result(version):
    reg, sdk = registration(), AsyncSdk()
    reg.version = version
    reg.upgrade_on_continue_as_new = True
    run = await reg.async_client(sdk).start({}, instance_id="iid")
    assert sdk.scheduled[-1]["upgrade_on_can"] is True
    sdk.states["iid"] = NS(
        runtime_status=NS(name="COMPLETED"),
        serialized_output=json.dumps({"gen": 0, "steps": 1, "result": pack(7)}),
    )

    async def completion(*args, **kwargs):
        return sdk.states["iid"]

    sdk.wait_for_orchestration_completion = completion
    assert await run.wait_result() == 7
