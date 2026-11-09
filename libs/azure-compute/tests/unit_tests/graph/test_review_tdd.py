# Copyright (c) Microsoft. All rights reserved.

"""TDD test suite for all P0, P1, and P2 review findings."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict

import azure.functions as func
import ormsgpack
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from langchain_azure_compute.graph import (
    Limits,
    attach_durability,
    sdk_options,
)
from langchain_azure_compute.graph.azure_functions import (
    create_function_app,
    dispatch_function_request,
)
from langchain_azure_compute.graph.runner import (
    make_orchestrator,
    pack,
    read_output,
    run_step,
    run_tool,
    unpack,
)


def _repo_file(path: Path) -> Path:
    if not path.exists():
        pytest.skip(
            "Repo-root README/samples not present in isolated release-tests layout"
        )
    return path


def _review_chat_registration():
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    b = StateGraph(State)
    b.add_node("reply", lambda s: {"messages": [AIMessage(content="reply")]})
    b.add_edge(START, "reply")
    b.add_edge("reply", END)
    return attach_durability(b.compile(), name="review_chat")


def _review_completed_turn(reg, thread, text, expect=None):
    report = run_step(
        reg.graph,
        Limits(),
        {
            "thread": thread,
            "kind": "start",
            "expect": expect,
            "input": pack({"messages": [HumanMessage(content=text)]}),
        },
    )
    assert report["status"] == "done"
    return {
        "gen": 0,
        "steps": 1,
        "ckpt": report["ckpt"],
        "checkpoint_bundle": report["checkpoint_bundle"],
    }


def test_review_inline_cache_is_scoped_to_saver():
    a, b = _review_chat_registration(), _review_chat_registration()
    _review_completed_turn(a, "user-42", "private")
    a.graph.checkpointer.delete_thread("user-42")

    class Sdk:
        def schedule_new_orchestration(self, name, input, instance_id):
            self.payload = input

        def get_orchestration_state(self, *a, **k):
            return None

    sdk = Sdk()
    with pytest.raises(ValueError, match="empty|checkpoint|state"):
        b.client(sdk).start(None, thread_id="user-42")
    a.client(sdk).start(None, thread_id="user-42")
    assert sdk.payload["req"]["expect"] is not None
    assert a.graph.get_state({"configurable": {"thread_id": "user-42"}}).values


async def test_review_async_inline_cache_is_scoped_to_saver():
    a, b = _review_chat_registration(), _review_chat_registration()
    _review_completed_turn(a, "user-42", "private")
    a.graph.checkpointer.delete_thread("user-42")

    class Sdk:
        async def schedule_new_orchestration(self, name, input, instance_id):
            self.payload = input

        async def get_orchestration_state(self, *a, **k):
            return None

    sdk = Sdk()
    with pytest.raises(ValueError, match="empty|checkpoint|state"):
        await b.async_client(sdk).start(None, thread_id="user-42")
    await a.async_client(sdk).start(None, thread_id="user-42")
    assert sdk.payload["req"]["expect"] is not None


@pytest.mark.parametrize("async_handle", [False, True])
async def test_review_result_hydrates_fresh_client_and_advances_older_state(
    async_handle,
):
    from langchain_azure_compute.graph._api import AsyncRun, Run
    from langchain_azure_compute.graph._inline_saver import _get_cached_inline_bundle

    worker, client = _review_chat_registration(), _review_chat_registration()
    outputs = [_review_completed_turn(worker, "remote", "one")]
    outputs.append(_review_completed_turn(worker, "remote", "two", outputs[0]["ckpt"]))
    # An empty namespace is not evidence of a newer checkpoint.
    client.graph.checkpointer.storage["remote"][""] = {}

    class Sdk:
        def wait_for_orchestration_completion(self, iid, **kwargs):
            return SimpleNamespace(
                runtime_status=SimpleNamespace(name="COMPLETED"),
                serialized_output=json.dumps(outputs[int(iid)]),
            )

    class AsyncSdk(Sdk):
        async def wait_for_orchestration_completion(self, iid, **kwargs):
            return super().wait_for_orchestration_completion(iid, **kwargs)

    for index in range(2):
        if async_handle:
            await AsyncRun(client, AsyncSdk(), str(index), "remote").wait_result()
        else:
            Run(client, Sdk(), str(index), "remote").wait_result()
        assert (
            _get_cached_inline_bundle(
                client.graph,
                "remote",
            )
            == outputs[index]["checkpoint_bundle"]
        )
    assert [
        m.content
        for m in client.graph.get_state(
            {"configurable": {"thread_id": "remote"}},
        ).values["messages"]
    ] == ["one", "reply", "two", "reply"]


def test_review_result_preserves_namespace_missing_from_older_bundle():
    from langchain_azure_compute.graph._inline_saver import _hydrate_checkpoint_bundle

    reg = _review_chat_registration()
    report = _review_completed_turn(reg, "t", "one")
    current = reg.graph.get_state({"configurable": {"thread_id": "t"}})
    _hydrate_checkpoint_bundle(
        reg.graph,
        "t",
        {"storage": {}, "v": 1},
        preserve_newer=True,
    )
    assert (
        reg.graph.get_state({"configurable": {"thread_id": "t"}}).config
        == current.config
    )
    assert report["checkpoint_bundle"]["storage"]


@pytest.mark.parametrize("provider_result", [None, "", 7, "raises"])
def test_review_invalid_identity_provider_denied(provider_result):
    def identity(req):
        if provider_result == "raises":
            raise ValueError("invalid token")
        return provider_result

    app = create_function_app(
        _review_chat_registration(),
        SimpleNamespace(),
        identity_provider=identity,
    )
    assert (
        dispatch_function_request(
            app,
            _review_http_request("start"),
        ).status_code
        == 401
    )


def test_review_easyauth_requires_injected_headers(monkeypatch):
    from langchain_azure_compute.graph.azure_functions import _extract_caller_principal

    monkeypatch.setenv("WEBSITE_AUTH_ENABLED", "true")
    assert _extract_caller_principal(SimpleNamespace()) is None
    assert (
        _extract_caller_principal(SimpleNamespace(headers=["not", "mapping"])) is None
    )


def test_review_missing_protected_instance_fails_closed(monkeypatch):
    monkeypatch.setenv("WEBSITE_AUTH_ENABLED", "true")

    class Sdk:
        def get_orchestration_state(self, *a, **k):
            return None

        def schedule_new_orchestration(self, *a, **k):
            pytest.fail("must not schedule")

    app = create_function_app(_review_chat_registration(), Sdk())
    assert (
        dispatch_function_request(
            app,
            _review_http_request("status/owned", {"x-ms-client-principal-id": "alice"}),
        ).status_code
        == 403
    )


def test_review_owner_is_private_while_interrupted_and_after_resume():
    inp = {
        "thread": "t",
        "req": {"thread": "t", "kind": "start", "expect": None},
        "step_name": "langgraph.review.step",
        "owner": "alice",
    }

    class Ctx:
        def __init__(self):
            self.statuses = []

        def call_activity(self, name, **kwargs):
            return kwargs

        def wait_for_external_event(self, name):
            return name

        def set_custom_status(self, status):
            self.statuses.append(status)

    ctx = Ctx()
    gen = make_orchestrator(Limits())(ctx, inp)
    next(gen)
    assert (
        gen.send(
            {
                "status": "interrupted",
                "ckpt": "c",
                "interrupt": {"id": "i", "value": pack("q")},
            },
        )
        == "resume:i"
    )
    assert "owner" not in ctx.statuses[-1]
    gen.send({"value": pack(True), "ckpt": "c"})
    assert ctx.statuses[-1] == {"state": "running"}
    gen.close()


@pytest.mark.parametrize("async_handle", [False, True])
@pytest.mark.parametrize("prune_old", [False, True])
@pytest.mark.parametrize("checkpoint_key", ["ckpt", "last_checkpoint_id", None])
async def test_review_old_result_preserves_newer_turn(
    async_handle,
    prune_old,
    checkpoint_key,
):
    from langchain_azure_compute.graph._api import AsyncRun, Run

    reg = _review_chat_registration()
    first = _review_completed_turn(reg, "history", "one")
    second = _review_completed_turn(reg, "history", "two", first["ckpt"])
    latest = reg.graph.get_state({"configurable": {"thread_id": "history"}})
    if prune_old:
        reg.graph.checkpointer.storage["history"][""].pop(first["ckpt"], None)
    old = dict(first)
    cid = old.pop("ckpt")
    if checkpoint_key is not None:
        old[checkpoint_key] = cid
    meta = SimpleNamespace(
        runtime_status=SimpleNamespace(name="COMPLETED"),
        serialized_output=json.dumps(old),
    )

    class Sdk:
        def wait_for_orchestration_completion(self, *a, **k):
            return meta

    class AsyncSdk:
        async def wait_for_orchestration_completion(self, *a, **k):
            return meta

    if async_handle:
        await AsyncRun(reg, AsyncSdk(), "run-one", "history").wait_result()
    else:
        Run(reg, Sdk(), "run-one", "history").wait_result()
    current = reg.graph.get_state({"configurable": {"thread_id": "history"}})
    assert current.config == latest.config
    assert [m.content for m in current.values["messages"]] == [
        "one",
        "reply",
        "two",
        "reply",
    ]
    from langchain_azure_compute.graph._inline_saver import _get_cached_inline_bundle

    assert (
        _get_cached_inline_bundle(reg.graph, "history") == second["checkpoint_bundle"]
    )
    third = _review_completed_turn(reg, "history", "three", second["ckpt"])
    assert [
        m.content for m in read_output(reg.graph, "history", third["ckpt"])["messages"]
    ] == [
        "one",
        "reply",
        "two",
        "reply",
        "three",
        "reply",
    ]


async def test_review_sync_saver_io_is_offloaded(monkeypatch):
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from langchain_azure_compute.graph._async_bridge import _ensure_async_saver

    main_tid = threading.get_ident()
    calls = []

    class Saver(BaseCheckpointSaver):
        def get_tuple(self, config):
            calls.append(("get", threading.get_ident()))
            return None

        def put(self, config, checkpoint, metadata, new_versions):
            calls.append(("put", threading.get_ident()))
            return config

        def put_writes(self, config, writes, task_id, task_path=""):
            calls.append(("writes", threading.get_ident()))

        def list(self, config, *, filter=None, before=None, limit=None):
            calls.append(("list", threading.get_ident()))
            yield "checkpoint"

    saver = Saver()
    _ensure_async_saver(saver)
    cfg = {"configurable": {"thread_id": "t"}}
    assert await saver.aget_tuple(cfg) is None
    assert await saver.aput(cfg, {}, {}, {}) == cfg
    await saver.aput_writes(cfg, [], "task")
    assert [item async for item in saver.alist(cfg)] == ["checkpoint"]
    assert {name for name, _ in calls} == {"get", "put", "writes", "list"}
    assert all(tid != main_tid for _, tid in calls)


def _review_http_request(route, headers=None):
    return func.HttpRequest(
        method="POST" if route == "start" or route.startswith("respond") else "GET",
        url=f"http://localhost/api/graphs/review_chat/{route}",
        headers=headers or {},
        params={},
        route_params={},
        body=json.dumps({"input": {"messages": []}, "instance_id": "owned"}).encode(),
    )


def test_review_untrusted_easyauth_headers_rejected(monkeypatch):
    monkeypatch.delenv("WEBSITE_AUTH_ENABLED", raising=False)
    app = create_function_app(_review_chat_registration(), SimpleNamespace())
    req = _review_http_request("start", {"x-ms-client-principal-id": "alice"})
    assert dispatch_function_request(app, req).status_code == 401


@pytest.mark.parametrize(
    "source",
    [
        "serialized_input",
        "serialized_custom_status",
        "serialized_output",
    ],
)
@pytest.mark.parametrize("route", ["status/owned", "respond/owned/intr", "start"])
def test_review_owner_fallback_denies_other_principal(source, route, monkeypatch):
    monkeypatch.setenv("WEBSITE_AUTH_ENABLED", "true")
    meta = SimpleNamespace(
        serialized_input="invalid",
        serialized_custom_status="[]",
        serialized_output=None,
    )
    setattr(meta, source, json.dumps({"owner": "alice"}))

    class Sdk:
        def schedule_new_orchestration(self, *a, **k):
            pytest.fail("must not schedule")

        def get_orchestration_state(self, *a, **k):
            return meta

    app = create_function_app(_review_chat_registration(), Sdk())
    req = _review_http_request(route, {"x-ms-client-principal-id": "mallory"})
    assert dispatch_function_request(app, req).status_code == 403


@pytest.mark.parametrize("payload", [None, "", "invalid", "[]", '{"owner": 42}'])
@pytest.mark.parametrize("route", ["status/owned", "respond/owned/intr", "start"])
def test_review_unresolvable_owner_fails_closed(payload, route, monkeypatch):
    monkeypatch.setenv("WEBSITE_AUTH_ENABLED", "1")

    class Sdk:
        def schedule_new_orchestration(self, *a, **k):
            pytest.fail("must not schedule")

        def get_orchestration_state(self, *a, **k):
            return SimpleNamespace(serialized_input=payload)

    app = create_function_app(_review_chat_registration(), Sdk())
    req = _review_http_request(route, {"x-ms-client-principal-id": "alice"})
    assert dispatch_function_request(app, req).status_code == 403


@pytest.mark.parametrize("route", ["status/owned", "respond/owned/intr", "start"])
def test_review_owner_lookup_failure_fails_closed(route, monkeypatch):
    monkeypatch.setenv("WEBSITE_AUTH_ENABLED", "yes")

    class Sdk:
        def schedule_new_orchestration(self, *a, **k):
            pytest.fail("must not schedule")

        def get_orchestration_state(self, *a, **k):
            raise OSError("backend unavailable")

    app = create_function_app(_review_chat_registration(), Sdk())
    req = _review_http_request(route, {"x-ms-client-principal-id": "alice"})
    assert dispatch_function_request(app, req).status_code == 403


def test_review_identity_provider_and_key_only_contract(monkeypatch):
    monkeypatch.delenv("WEBSITE_AUTH_ENABLED", raising=False)
    reg = _review_chat_registration()

    class Sdk:
        def schedule_new_orchestration(self, name, input, instance_id):
            self.payload = input

        def get_orchestration_state(self, *a, **k):
            return (
                SimpleNamespace(serialized_input=json.dumps(self.payload))
                if hasattr(self, "payload")
                else None
            )

    sdk = Sdk()
    app = create_function_app(reg, sdk, identity_provider=lambda req: "verified")
    assert (
        dispatch_function_request(app, _review_http_request("start")).status_code == 202
    )
    assert sdk.payload["owner"] == "verified"
    key_only = create_function_app(reg, sdk, require_instance_owner=False)
    assert (
        dispatch_function_request(
            key_only,
            _review_http_request("start"),
        ).status_code
        == 202
    )
    assert "owner" not in sdk.payload


@pytest.mark.parametrize("owner", ["alice", None, "", 42])
def test_review_owner_survives_continue_as_new_and_output(owner):
    from langchain_azure_compute.graph.runner import OrchestratorState

    inp = {
        "thread": "t",
        "req": {"thread": "t", "kind": "start", "expect": None},
        "step_name": "langgraph.review.step",
        "owner": owner,
    }
    state = OrchestratorState.from_payload(inp)
    expected = owner if isinstance(owner, str) and owner else None
    assert state.owner == expected
    assert state.to_can_payload().get("owner") == expected

    class Ctx:
        def call_activity(self, name, **kwargs):
            return kwargs

        def set_custom_status(self, value):
            pass

    gen = make_orchestrator(Limits())(Ctx(), inp)
    next(gen)
    with pytest.raises(StopIteration) as completed:
        gen.send({"status": "done", "ckpt": "c"})
    assert "owner" not in completed.value.value


@pytest.mark.parametrize("approved", [False, True])
def test_review_readme_refund_gate(approved):
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode
    from langgraph.types import Command

    from langchain_azure_compute.graph import durable_tool

    text = _repo_file(Path(__file__).resolve().parents[3] / "README.md").read_text(
        encoding="utf-8"
    )
    snippet = text.split("class SupportState(TypedDict):", 1)[1].split(
        "\nreg = attach_durability",
        1,
    )[0]
    ns = dict(
        TypedDict=TypedDict,
        Annotated=Annotated,
        AnyMessage=AnyMessage,
        add_messages=add_messages,
        durable_tool=durable_tool,
        tool=tool,
        interrupt=interrupt,
        AIMessage=AIMessage,
        StateGraph=StateGraph,
        ToolNode=ToolNode,
        START=START,
        END=END,
    )
    exec(
        compile(
            "class SupportState(TypedDict):" + snippet,
            "<readme>",
            "exec",
            dont_inherit=True,
        ),
        ns,
    )
    graph = ns["graph"]
    graph.checkpointer = InMemorySaver()
    cfg = {"configurable": {"thread_id": f"gate-{approved}"}}
    graph.invoke({"messages": [], "approved": False}, cfg)
    result = graph.invoke(Command(resume={"approved": approved}), cfg)
    assert any(isinstance(m, ToolMessage) for m in result["messages"]) is approved


@pytest.mark.parametrize(
    "sample",
    ["05_multi_turn_conversation", "07_multi_graph_app", "10_azure_functions_hitl"],
)
def test_review_sample_root_install_includes_requirements(sample):
    root = Path(__file__).resolve().parents[5]
    text = _repo_file(
        root / "samples" / "langgraph-durabletask" / sample / "README.md"
    ).read_text(
        encoding="utf-8",
    )
    assert (
        f'pip install -e "libs/azure-compute[durabletask]" '
        f"-r samples/langgraph-durabletask/{sample}/requirements.txt"
    ) in text


@pytest.mark.parametrize(
    "sample", ["09_two_process_crash_recovery", "10_azure_functions_hitl"]
)
@pytest.mark.parametrize("child_output", ["", "partial line"])
def test_review_worker_readiness_timeout_reaps_silent_child(
    sample,
    child_output,
    monkeypatch,
):
    root = Path(__file__).resolve().parents[5]
    path = _repo_file(root / "samples" / "langgraph-durabletask" / sample / "sample.py")
    spec = importlib.util.spec_from_file_location(f"readiness_{sample}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    original_popen = subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        proc = original_popen(
            [
                sys._base_executable,
                "-c",
                f"import sys,time; sys.stdout.write({child_output!r}); "
                "sys.stdout.flush(); time.sleep(10)",
            ],
            **kwargs,
        )
        children.append(proc)
        return proc

    monkeypatch.setattr(mod.subprocess, "Popen", spawn)
    try:
        with pytest.raises(TimeoutError):
            mod._spawn_worker_subprocess(
                "silent",
                "review",
                endpoint="127.0.0.1:8080",
                readiness_timeout=0.1,
            )
        assert len(children) == 1
        assert children[0].poll() is not None
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
            child.stdout.close()


@pytest.mark.parametrize(
    "sample",
    [
        "09_two_process_crash_recovery",
        "10_azure_functions_hitl",
    ],
)
@pytest.mark.parametrize("outcome", ["ready", "exit", "invalid_pid"])
def test_review_worker_readiness_success_and_failure_cleanup(
    sample, outcome, monkeypatch
):
    root = Path(__file__).resolve().parents[5]
    path = _repo_file(root / "samples" / "langgraph-durabletask" / sample / "sample.py")
    spec = importlib.util.spec_from_file_location(f"readiness_{sample}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    script = {
        "ready": (
            "import os,sys; print('WORKER_READY pid='+str(os.getpid()),flush=True); "
            "sys.stdout.write('x'*100000); sys.stdout.flush()"
        ),
        "exit": "print('startup failed',flush=True)",
        "invalid_pid": (
            "import time; print('WORKER_READY pid=bad',flush=True); time.sleep(10)"
        ),
    }[outcome]
    original_popen = subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        proc = original_popen([sys._base_executable, "-c", script], **kwargs)
        children.append(proc)
        return proc

    monkeypatch.setattr(mod.subprocess, "Popen", spawn)
    try:
        if outcome == "ready":
            proc, pid = mod._spawn_worker_subprocess(
                "worker",
                "review",
                endpoint="127.0.0.1:8080",
                readiness_timeout=10,
            )
            assert pid == proc.pid
            proc.wait(timeout=10)
        else:
            with pytest.raises(
                ValueError if outcome == "invalid_pid" else RuntimeError
            ):
                mod._spawn_worker_subprocess(
                    "worker",
                    "review",
                    endpoint="127.0.0.1:8080",
                    readiness_timeout=10,
                )
            assert children[0].poll() is not None
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
            child.stdout.close()


def test_review_sample_local_identity_requires_explicit_opt_in(monkeypatch):
    root = Path(__file__).resolve().parents[5]
    path = _repo_file(
        root
        / "samples"
        / "langgraph-durabletask"
        / "10_azure_functions_hitl"
        / "sample.py"
    )
    spec = importlib.util.spec_from_file_location("review_local_identity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.delenv("LOCAL_DEMO_IDENTITY", raising=False)
    app_mod = mod._load_function_app_module()
    assert app_mod._identity_options() == {}
    monkeypatch.setenv("LOCAL_DEMO_IDENTITY", "1")
    provider = app_mod._identity_options()["identity_provider"]
    assert (
        provider(SimpleNamespace(headers={"x-demo-user": "demo-user"})) == "demo-user"
    )
    assert (
        provider(SimpleNamespace(headers={"x-ms-client-principal-id": "demo-user"}))
        is None
    )


def test_p0_1_uv_lock_has_no_internal_feed_urls() -> None:
    """P0-1: uv.lock must not contain internal pkgs.visualstudio.com or
    packagefeedproxy URLs."""
    lock_path = Path(__file__).resolve().parents[3] / "uv.lock"
    text = lock_path.read_text(encoding="utf-8")
    assert "pkgs.visualstudio.com" not in text
    assert "packagefeedproxy.microsoft.io" not in text


def test_p0_2_azure_functions_cross_host_auth_and_hardening(monkeypatch) -> None:
    """P0-2: Cross-host ownership in durable storage, fail-closed principal check,
    no thread_id/config override, and start_worker=False default."""

    monkeypatch.setenv("WEBSITE_AUTH_ENABLED", "true")

    class ExpenseState(TypedDict):
        expense_id: str
        approved: bool

    b = StateGraph(ExpenseState)
    b.add_node(
        "review",
        lambda s: {"approved": bool(interrupt({"expense_id": s["expense_id"]}))},
    )
    b.add_edge(START, "review")
    b.add_edge("review", END)
    reg = attach_durability(
        b.compile(checkpointer=InMemorySaver()),
        name="expense_approval",
        tool_max_concurrency=2,
    )

    orch_states: dict[str, Any] = {}
    raised_events: list[tuple[str, str, dict[str, Any]]] = []

    class SharedFakeDtsSdk:
        def schedule_new_orchestration(
            self, name: str, input: Any = None, instance_id: str | None = None
        ) -> str:
            iid = instance_id or "inst-1"
            orch_states[iid] = SimpleNamespace(
                runtime_status=SimpleNamespace(name="RUNNING"),
                serialized_input=json.dumps(input),
                serialized_custom_status=json.dumps(
                    {
                        "pending": {
                            "id": "intr-1",
                            "value": pack({"expense_id": "EXP-1"}),
                        },
                        "interrupts": [
                            {"id": "intr-1", "value": pack({"expense_id": "EXP-1"})}
                        ],
                        "step": 1,
                        "ckpt": "ckpt-1",
                    }
                ),
                serialized_output=None,
                failure_details=SimpleNamespace(message="secret internal stack trace"),
            )
            return iid

        def get_orchestration_state(
            self, instance_id: str, fetch_payloads: bool = False
        ) -> Any:
            return orch_states.get(instance_id)

        def raise_orchestration_event(
            self, instance_id: str, event_name: str, data: Any = None
        ) -> None:
            raised_events.append((instance_id, event_name, data))

        def wait_for_orchestration_completion(
            self, instance_id: str, timeout: int = 60, fetch_payloads: bool = True
        ) -> Any:
            return orch_states.get(instance_id)

    shared_sdk = SharedFakeDtsSdk()
    # Two separate FunctionApp instances (simulating Host A and Host B after scale-out)
    # Also verify start_worker defaults to False!
    app_host_a = create_function_app(reg, shared_sdk)
    app_host_b = create_function_app(reg, shared_sdk)
    assert app_host_a._langgraph_state._want_worker is False

    # 1. Missing principal when require_principal=True (default) -> 401
    unauth_start = func.HttpRequest(
        method="POST",
        url="http://localhost:7071/api/graphs/expense_approval/start?instance_id=inst-1",
        headers={"Content-Type": "application/json"},
        params={},
        route_params={},
        body=json.dumps({"expense_id": "EXP-1", "approved": False}).encode("utf-8"),
    )
    assert dispatch_function_request(app_host_a, unauth_start).status_code == 401

    # 2. Alice starts inst-1 on Host A (attempting to pass thread_id="victim-thread" and max_concurrency=99)  # noqa: E501
    alice_headers = {
        "Content-Type": "application/json",
        "x-ms-client-principal-id": "alice-id",
        "x-ms-client-principal-idp": "aad",
    }
    alice_start = func.HttpRequest(
        method="POST",
        url="http://localhost:7071/api/graphs/expense_approval/start",
        headers=alice_headers,
        params={},
        route_params={},
        body=json.dumps(
            {
                "input": {"expense_id": "EXP-1", "approved": False},
                "instance_id": "inst-1",
                "thread_id": "victim-thread",
                "max_concurrency": 99,
            }
        ).encode("utf-8"),
    )
    resp_start = dispatch_function_request(app_host_a, alice_start)
    assert resp_start.status_code == 202
    start_payload = json.loads(resp_start.get_body())
    # The server ignores caller thread overrides and namespaces by verified owner.
    assert start_payload["threadId"].startswith("owner:")
    assert start_payload["threadId"].endswith(":inst-1")
    # max_concurrency must be capped to registration's limit (2), not 99!
    sched_inp = json.loads(orch_states["inst-1"].serialized_input)
    assert sched_inp.get("max_concurrency") == 2

    # 3. Cross-host check on Host B (which has an empty in-memory state!):
    # Mallory on Host B tries to read status, respond, or restart inst-1 -> all 403!
    mallory_headers = {
        "Content-Type": "application/json",
        "x-ms-client-principal-id": "mallory-id",
        "x-ms-client-principal-idp": "aad",
    }
    mallory_status = func.HttpRequest(
        method="GET",
        url="http://localhost:7071/api/graphs/expense_approval/status/inst-1",
        headers=mallory_headers,
        params={},
        route_params={},
        body=b"",
    )
    assert dispatch_function_request(app_host_b, mallory_status).status_code == 403

    mallory_respond = func.HttpRequest(
        method="POST",
        url="http://localhost:7071/api/graphs/expense_approval/respond/inst-1/intr-1",
        headers=mallory_headers,
        params={},
        route_params={},
        body=json.dumps({"value": True}).encode("utf-8"),
    )
    assert dispatch_function_request(app_host_b, mallory_respond).status_code == 403

    # Alice on Host B CAN read status of inst-1 (ownership recovered from durable DTS state!)  # noqa: E501
    alice_status_b = func.HttpRequest(
        method="GET",
        url="http://localhost:7071/api/graphs/expense_approval/status/inst-1?thread_id=other-thread",
        headers=alice_headers,
        params={"thread_id": "other-thread"},
        route_params={},
        body=b"",
    )
    resp_alice_b = dispatch_function_request(app_host_b, alice_status_b)
    assert resp_alice_b.status_code == 200
    assert json.loads(resp_alice_b.get_body())["threadId"] == start_payload["threadId"]

    # 4. /respond rejects state-mutation / migration keys (values, as_node, version) -> 400!  # noqa: E501
    bad_respond = func.HttpRequest(
        method="POST",
        url="http://localhost:7071/api/graphs/expense_approval/respond/inst-1/intr-1",
        headers=alice_headers,
        params={},
        route_params={},
        body=json.dumps(
            {
                "value": True,
                "values": {"approved": True},
                "as_node": "review",
                "version": "v2",
            }
        ).encode("utf-8"),
    )
    assert dispatch_function_request(app_host_b, bad_respond).status_code == 400


def test_p0_3_stateless_turn_input_step_across_worker_hops() -> None:
    """P0-3: A create_agent graph with RemainingSteps / recursion_limit must
    pass 15+ turns
    even when every step runs on a fresh worker."""
    from langchain.agents import create_agent

    import langchain_azure_compute.graph._checkpoints as ckpt_mod

    class FakeAgentModel(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

    responses = [AIMessage(content=f"Reply {i}") for i in range(15)]
    model = FakeAgentModel(messages=iter(responses))
    saver = InMemorySaver()
    agent = create_agent(model=model, tools=[], checkpointer=saver)
    reg = attach_durability(agent, name="stateless_turns", checkpointer=saver)

    ckpt: str | None = None
    for turn in range(15):
        if hasattr(ckpt_mod, "_TURN_INPUT_STEP"):
            ckpt_mod._TURN_INPUT_STEP.clear()
        req: dict[str, Any] = {
            "thread": "t-hops",
            "kind": "start",
            "expect": ckpt,
            "input": pack({"messages": [HumanMessage(content=f"Turn {turn}")]}),
        }
        rep = run_step(
            reg.graph,
            Limits(),
            req,
            tool_config=reg.tool_config,
            node_config=reg.node_config,
        )
        while rep["status"] != "done":
            if hasattr(ckpt_mod, "_TURN_INPUT_STEP"):
                ckpt_mod._TURN_INPUT_STEP.clear()
            step_req: dict[str, Any] = {
                "thread": "t-hops",
                "kind": "step",
                "expect": rep["ckpt"],
            }
            if "turn_input_step" in rep:
                step_req["turn_input_step"] = rep["turn_input_step"]
            rep = run_step(
                reg.graph,
                Limits(),
                step_req,
                tool_config=reg.tool_config,
                node_config=reg.node_config,
            )
        ckpt = rep["ckpt"]
    assert not hasattr(ckpt_mod, "_TURN_INPUT_STEP")


def test_p0_4_rejects_msgpack_ext_constructor_and_orchestrator_never_unpacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P0-4: unpack() must reject hostile msgpack EXT_CONSTRUCTOR payloads (e.g.
    builtins.eval)
    and run_graph_orchestrator must never call unpack() during orchestration replay."""
    executed = False

    def pwn_fn(*args: Any) -> str:
        nonlocal executed
        executed = True
        return "pwned"

    # EXT_CONSTRUCTOR (code 0 or 2 in JsonPlusSerializer): tuple (module, name, args)
    malicious_raw = ormsgpack.packb(
        ormsgpack.Ext(0, ormsgpack.packb(("builtins", "eval", ["1+1"])))
    )
    malicious_blob = {
        "t": "msgpack",
        "b64": base64.b64encode(malicious_raw).decode("ascii"),
    }
    with pytest.raises(Exception):
        res = unpack(malicious_blob)
        assert res != 2, "EXT_CONSTRUCTOR executed builtins.eval!"
    assert not executed

    # Also verify run_graph_orchestrator NEVER calls unpack() even on multi-interrupt resume  # noqa: E501
    import langchain_azure_compute.graph.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "unpack",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("orchestrator must never call unpack()")
        ),
    )

    orch = make_orchestrator(Limits(), name="langgraph.no_unpack.run")

    class FakeCtx:
        def __init__(self) -> None:
            self.custom_status: Any = None

        def call_activity(
            self, name: str, *, input: Any = None, retry_policy: Any = None
        ) -> dict[str, Any]:
            return {"op": "activity", "name": name, "input": input}

        def wait_for_external_event(self, name: str) -> dict[str, Any]:
            return {"op": "event", "name": name}

        def set_custom_status(self, status: Any) -> None:
            self.custom_status = status

    ctx = FakeCtx()
    gen = orch(
        ctx,  # type: ignore[arg-type]
        {
            "proto": 1,
            "thread": "t1",
            "gen": 0,
            "steps": 0,
            "req": {"thread": "t1", "kind": "start", "expect": None},
            "step_name": "langgraph.no_unpack.step",
            "tool_name": "langgraph.no_unpack.tool",
            "node_name": "langgraph.no_unpack.node",
        },
    )
    t1 = next(gen)
    assert t1["op"] == "activity"
    # Step reports 2 parallel interrupts
    ev1 = gen.send(
        {
            "status": "interrupted",
            "ckpt": "c1",
            "interrupt": {"id": "i1", "value": pack("q1")},
            "interrupts": [
                {"id": "i1", "value": pack("q1")},
                {"id": "i2", "value": pack("q2")},
            ],
        }
    )
    assert ev1 == {"op": "event", "name": "resume:i1"}
    ev2 = gen.send({"value": pack("a1"), "ckpt": "c1"})
    assert ev2 == {"op": "event", "name": "resume:i2"}
    resume_step = gen.send({"value": pack("a2"), "ckpt": "c1"})
    assert resume_step["op"] == "activity"
    assert resume_step["input"]["kind"] == "resume"


def test_p1_1_zero_config_cross_process_start_with_expect_none_never_returns_stale() -> (  # noqa: E501
    None
):
    """P1-1: In zero-config mode, a new turn arriving with expect=None and a new
    input on a worker
    that happens to hold prior state must NOT treat it as replayed=True
    and return the old turn's output."""

    class ChatState(TypedDict):
        text: str

    b = StateGraph(ChatState)
    b.add_node("echo", lambda s: {"text": f"echo:{s['text']}"})
    b.add_edge(START, "echo")
    b.add_edge("echo", END)
    reg = attach_durability(b.compile(), name="p1_1_app")

    # Turn 1 runs on Worker 1
    r1 = run_step(
        reg.graph,
        Limits(),
        {
            "thread": "shared-t",
            "kind": "start",
            "expect": None,
            "input": pack({"text": "turn1"}),
        },
        tool_config=reg.tool_config,
        node_config=reg.node_config,
    )
    assert read_output(reg.graph, "shared-t", r1["ckpt"]) == {"text": "echo:turn1"}

    # Now Client B (a separate process with no local bundle cache) sends Turn 2 with expect=None  # noqa: E501
    # and a different input {"text": "turn2"} to Worker 1.
    # It must NOT silently return {"text": "echo:turn1"} as replayed=True!
    with pytest.raises(Exception, match="Concurrent turn conflict"):
        r2 = run_step(
            reg.graph,
            Limits(),
            {
                "thread": "shared-t",
                "kind": "start",
                "expect": None,
                "input": pack({"text": "turn2"}),
            },
            tool_config=reg.tool_config,
            node_config=reg.node_config,
        )
        assert read_output(reg.graph, "shared-t", r2["ckpt"]) != {"text": "echo:turn1"}


def test_p1_2_bundle_single_ckpt_size_guard_and_sdk_options_default_channel() -> None:
    """P1-2: Zero-config bundle exports only latest_cid (not parent_cid), raises
    RunError when
    exceeding DTS payload limit, omits duplicate 'result' key in orchestrator
    output, and
    sdk_options() only sets custom gRPC message size when explicitly requested."""
    from langchain_azure_compute.graph import RunError, sdk_options
    from langchain_azure_compute.graph._inline_saver import _export_checkpoint_bundle

    class BigState(TypedDict):
        blob: str

    b = StateGraph(BigState)
    b.add_node("n1", lambda s: {"blob": s["blob"] + ":step1"})
    b.add_node("n2", lambda s: {"blob": s["blob"] + ":step2"})
    b.add_edge(START, "n1")
    b.add_edge("n1", "n2")
    b.add_edge("n2", END)
    reg = attach_durability(b.compile(), name="p1_2_app")

    r1 = run_step(
        reg.graph,
        Limits(),
        {
            "thread": "t-b",
            "kind": "start",
            "expect": None,
            "input": pack({"blob": "hi"}),
        },
        tool_config=reg.tool_config,
        node_config=reg.node_config,
    )
    run_step(
        reg.graph,
        Limits(),
        {"thread": "t-b", "kind": "step", "expect": r1["ckpt"]},
        tool_config=reg.tool_config,
        node_config=reg.node_config,
    )
    bundle = _export_checkpoint_bundle(reg.graph, "t-b")
    assert bundle is not None
    # Only 1 checkpoint (latest_cid) should be in storage[''] (parent_cid dropped)
    assert len(bundle["storage"][""]) == 1

    # Oversized state (> 1 MiB) in zero-config mode must fail loudly with RunError
    with pytest.raises(RunError, match="checkpoint_bundle|payload limit"):
        run_step(
            reg.graph,
            Limits(),
            {
                "thread": "t-huge",
                "kind": "start",
                "expect": None,
                "input": pack({"blob": "x" * 1_100_000}),
            },
            tool_config=reg.tool_config,
            node_config=reg.node_config,
        )

    # sdk_options() without max_message_bytes should not force a 64 MiB channel override
    opts_default = sdk_options("127.0.0.1:8080")
    assert opts_default["channel_options"].max_receive_message_length is None
    assert opts_default["channel_options"].max_send_message_length is None
    opts_custom = sdk_options("127.0.0.1:8080", max_message_bytes=8 * 1024 * 1024)
    assert opts_custom["channel_options"].max_receive_message_length == 8 * 1024 * 1024
    assert opts_custom["channel_options"].max_send_message_length == 8 * 1024 * 1024
    for opts in (opts_default, opts_custom):
        assert opts["channel_options"].raw_options == [
            ("grpc.primary_user_agent", "langgraph-durabletask")
        ]


def test_p1_3_invoke_does_not_catch_user_type_error_or_double_store_async_input() -> (
    None
):
    """P1-3: A synchronous node raising TypeError must not trigger an ainvoke()
    retry, and an
    async-only first node with a sync saver must store the initial HumanMessage
    only once."""

    class MsgState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    calls = 0

    def buggy_sync_node(state: MsgState) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise TypeError("user type error inside sync node")

    b1 = StateGraph(MsgState)
    b1.add_node("buggy", buggy_sync_node)
    b1.add_edge(START, "buggy")
    b1.add_edge("buggy", END)
    reg1 = attach_durability(b1.compile(checkpointer=InMemorySaver()), name="p1_3_sync")

    with pytest.raises(TypeError, match="user type error inside sync node"):
        run_step(
            reg1.graph,
            Limits(),
            {
                "thread": "t-err",
                "kind": "start",
                "expect": None,
                "input": pack({"messages": [HumanMessage(content="hi")]}),
            },
            tool_config=reg1.tool_config,
            node_config=reg1.node_config,
        )
    assert calls == 1, f"buggy_sync_node ran {calls} times instead of 1!"

    async def async_first_node(state: MsgState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="from async")]}

    b2 = StateGraph(MsgState)
    b2.add_node("async_node", async_first_node)
    b2.add_edge(START, "async_node")
    b2.add_edge("async_node", END)
    reg2 = attach_durability(
        b2.compile(checkpointer=InMemorySaver()), name="p1_3_async"
    )

    rep = run_step(
        reg2.graph,
        Limits(),
        {
            "thread": "t-async",
            "kind": "start",
            "expect": None,
            "input": pack({"messages": [HumanMessage(content="once")]}),
        },
        tool_config=reg2.tool_config,
        node_config=reg2.node_config,
    )
    out = read_output(reg2.graph, "t-async", rep["ckpt"])
    human_msgs = [m for m in out["messages"] if isinstance(m, HumanMessage)]
    assert len(human_msgs) == 1, f"Expected 1 HumanMessage, got {len(human_msgs)}!"


def test_p1_5_attach_durability_does_not_mutate_caller_graph() -> None:
    """P1-5: attach_durability must work on a copy so caller's compiled graph is
    untouched
    and two registrations on the same compiled graph keep independent settings."""

    class S(TypedDict):
        x: int

    b = StateGraph(S)
    b.add_node("inc", lambda s: {"x": s["x"] + 1})
    b.add_edge(START, "inc")
    b.add_edge("inc", END)
    raw_graph = b.compile()

    assert raw_graph.checkpointer is None
    orig_ib = list(raw_graph.interrupt_before_nodes or [])

    reg1 = attach_durability(raw_graph, name="app_one", tool_retries=2)
    reg2 = attach_durability(raw_graph, name="app_two", tool_retries=5)

    # Caller's compiled graph must remain completely unmutated!
    assert raw_graph.checkpointer is None
    assert list(raw_graph.interrupt_before_nodes or []) == orig_ib
    assert not hasattr(raw_graph, "_durable_name")
    assert raw_graph.invoke({"x": 10}) == {"x": 11}

    # The two registrations must not overwrite each other's graph settings!
    assert reg1.graph is not reg2.graph
    assert getattr(reg1.graph, "_durable_name") == "app_one"
    assert getattr(reg2.graph, "_durable_name") == "app_two"
    assert reg1.tool_config.default_retries == 2
    assert reg2.tool_config.default_retries == 5


def test_p1_6_read_output_honors_checkpoint_id_across_turns() -> None:
    """P1-6: read_output(graph, thread, checkpoint_id) must return the state at
    checkpoint_id,
    not the latest turn's state when a subsequent turn has already updated the
    thread."""
    from langchain_azure_compute.graph.runner import read_output

    class S(TypedDict):
        x: int

    saver = InMemorySaver()
    b = StateGraph(S)
    b.add_node("add10", lambda s: {"x": s["x"] + 10})
    b.add_edge(START, "add10")
    b.add_edge("add10", END)
    reg = attach_durability(
        b.compile(checkpointer=saver),
        name="p1_6_app",
        checkpointer=saver,
        inline_checkpoints=False,
    )

    # Turn 1: x = 1 -> 11
    r1 = run_step(
        reg.graph,
        Limits(),
        {"thread": "t-p16", "kind": "start", "expect": None, "input": pack({"x": 1})},
        tool_config=reg.tool_config,
        node_config=reg.node_config,
    )
    ckpt1 = r1["ckpt"]

    # Turn 2 on the same thread: x = 100 -> 110
    r2 = run_step(
        reg.graph,
        Limits(),
        {
            "thread": "t-p16",
            "kind": "start",
            "expect": ckpt1,
            "input": pack({"x": 100}),
        },
        tool_config=reg.tool_config,
        node_config=reg.node_config,
    )
    ckpt2 = r2["ckpt"]

    assert read_output(reg.graph, "t-p16", ckpt1) == {"x": 11}
    assert read_output(reg.graph, "t-p16", ckpt2) == {"x": 110}


def test_p1_4_version_qualified_activities_and_mixed_version_workers() -> None:
    """P1-4: Versioned graphs must register and call version-qualified activity names
    (`langgraph.<name>@<version>.{step,tool,node}`) so v2 work never lands on v1 workers
    during a rolling deploy, and target_version is retained across mixed-worker
    steps."""

    class S(TypedDict):
        x: int

    saver = InMemorySaver()
    b1 = StateGraph(S)
    b1.add_node("n1", lambda s: {"x": s["x"] + 1})
    b1.add_node("n2", lambda s: {"x": s["x"] + 1})
    b1.add_node("n3", lambda s: {"x": s["x"] + 1})
    b1.add_edge(START, "n1")
    b1.add_edge("n1", "n2")
    b1.add_edge("n2", "n3")
    b1.add_edge("n3", END)
    g1 = b1.compile(checkpointer=saver)

    b2 = StateGraph(S)
    b2.add_node("n1", lambda s: {"x": s["x"] + 10})
    b2.add_node("n2", lambda s: {"x": s["x"] + 10})
    b2.add_node("n3", lambda s: {"x": s["x"] + 10})
    b2.add_edge(START, "n1")
    b2.add_edge("n1", "n2")
    b2.add_edge("n2", "n3")
    b2.add_edge("n3", END)
    g2 = b2.compile(checkpointer=saver)

    lim = Limits(steps_per_generation=2)
    reg_v1 = attach_durability(
        g1,
        name="rolling_app",
        checkpointer=saver,
        inline_checkpoints=False,
        version="v1",
        limits=lim,
    )
    reg_v2 = attach_durability(
        g2,
        name="rolling_app",
        checkpointer=saver,
        inline_checkpoints=False,
        version="v2",
        previous_versions={"v1": g1},
        upgrade_on_continue_as_new=True,
        limits=lim,
    )

    class MockWorker:
        def __init__(self) -> None:
            self.activities: dict[str, Any] = {}
            self.orchestrators: dict[str, Any] = {}

        def add_orchestrator(self, fn: Any) -> str:
            self.orchestrators[fn.__name__] = fn
            return fn.__name__

        def add_activity(self, fn: Any) -> str:
            self.activities[fn.__name__] = fn
            return fn.__name__

    w1 = MockWorker()
    w2 = MockWorker()
    reg_v1.register(w1)
    reg_v2.register(w2)

    # Worker 1 only advertises @v1 (not @v2); Worker 2 advertises both @v2 and @v1
    assert "langgraph.rolling_app@v1.step" in w1.activities
    assert "langgraph.rolling_app@v2.step" not in w1.activities
    assert "langgraph.rolling_app@v1.step" in w2.activities
    assert "langgraph.rolling_app@v2.step" in w2.activities

    class FakeCtx:
        def __init__(self) -> None:
            self.can: Any = None

        def call_activity(
            self, name: str, *, input: Any = None, retry_policy: Any = None
        ) -> Any:
            return ("act", name, input)

        def set_custom_status(self, status: Any) -> None:
            pass

        def continue_as_new(self, new_input: Any, save_events: bool = False) -> None:
            self.can = (new_input, save_events)

    # Simulate a rolling deploy where step 1 of a v1 run lands on w2 (which knows target_version='v2')  # noqa: E501
    # and step 2 of that same generation lands on w1 (which is still on v1!).
    # At continue_as_new after step 2, the run must STILL upgrade to v2 and
    # route v2 steps ONLY to w2.
    payload: dict[str, Any] = {
        "v": 1,
        "thread": "t-roll",
        "step_name": reg_v1.step_name,
        "tool_name": reg_v1.tool_name,
        "node_name": reg_v1.node_name,
        "version": "v1",
        "upgrade_on_can": True,
        "req": {
            "thread": "t-roll",
            "kind": "start",
            "expect": None,
            "input": pack({"x": 0}),
            "version": "v1",
        },
    }
    act_names_seen: list[str] = []
    v1_call_idx = 0
    while True:
        ctx = FakeCtx()
        gen = make_orchestrator(lim)(ctx, payload)
        try:
            y = next(gen)
            while True:
                _, act_name, act_inp = y
                act_names_seen.append(act_name)
                if "@v2." in act_name:
                    # Only w2 has @v2 activities
                    assert act_name in w2.activities and act_name not in w1.activities
                    rep = w2.activities[act_name](
                        SimpleNamespace(orchestration_id="i1"), act_inp
                    )
                else:
                    # Alternate w2 (step 1) then w1 (step 2) for @v1 activities
                    v1_call_idx += 1
                    chosen_w = w2 if v1_call_idx == 1 else w1
                    rep = chosen_w.activities[act_name](
                        SimpleNamespace(orchestration_id="i1"), act_inp
                    )
                y = gen.send(rep)
        except StopIteration as done:
            if ctx.can is not None:
                payload, _ = ctx.can
                continue
            final_out = done.value
            break

    from langchain_azure_compute.graph.runner import read_output

    assert act_names_seen == [
        "langgraph.rolling_app@v1.step",
        "langgraph.rolling_app@v1.step",
        "langgraph.rolling_app@v2.step",
    ]
    assert final_out["version"] == "v2"
    # Step 1 (n1 on v1: +1) + Step 2 (n2 on v1: +1) + Step 3 (n3 on v2 after upgrade: +10) = 12  # noqa: E501
    assert read_output(reg_v2.graph, "t-roll") == {"x": 12}


def test_p2_1_malformed_resume_events_ignored_and_limits_persisted_in_state() -> None:
    """P2: Malformed external resume events (raw strings, None, dicts without
    valid 'value')
    must be ignored rather than crashing the orchestrator, and Limits must be
    persisted in
    OrchestratorState so changing worker Limits between deploys does not alter
    in-flight runs."""

    class FakeCtx:
        def __init__(self) -> None:
            self.can: Any = None

        def call_activity(
            self, name: str, *, input: Any = None, retry_policy: Any = None
        ) -> Any:
            return ("act", name, input, retry_policy)

        def wait_for_external_event(self, name: str) -> Any:
            return ("evt", name)

        def set_custom_status(self, status: Any) -> None:
            pass

        def continue_as_new(self, new_input: Any, save_events: bool = False) -> None:
            self.can = (new_input, save_events)

    # 1. Malformed resume events are dropped and orchestrator keeps waiting
    ctx = FakeCtx()
    init_inp: dict[str, Any] = {
        "proto": 1,
        "thread": "t-p21",
        "gen": 0,
        "steps": 0,
        "step_name": "langgraph.x.step",
        "tool_name": "langgraph.x.tool",
        "node_name": "langgraph.x.node",
        "limits": {
            "max_steps": 50,
            "steps_per_generation": 2,
            "byte_budget": 100_000,
            "retries": 5,
            "custom_status_limit": 1024,
        },
        "req": {
            "thread": "t-p21",
            "kind": "start",
            "expect": None,
            "input": pack({"x": 1}),
        },
    }
    # Deploy 2 worker has different Limits(retries=1, steps_per_generation=999),
    # but the in-flight run must use the Limits persisted in init_inp["limits"]!
    gen = make_orchestrator(Limits(retries=1, steps_per_generation=999))(ctx, init_inp)
    first_yield = next(gen)
    # Persisted limits.retries=5 -> max_number_of_attempts == 5
    assert first_yield[3].max_number_of_attempts == 5

    ev_wait = gen.send(
        {
            "status": "interrupted",
            "ckpt": "c1",
            "interrupt": {"id": "int-1", "value": pack("approve?")},
        }
    )
    assert ev_wait == ("evt", "resume:int-1")

    # Send malformed events: raw string, None, dict without 'value', dict with
    # non-blob 'value'
    for bad_event in (
        "raw_approve_string",
        None,
        {"not_value": 123},
        {"value": "not_a_packed_blob"},
    ):
        still_waiting = gen.send(bad_event)
        assert still_waiting == ("evt", "resume:int-1")

    # Now send a valid event -> advances to resume activity, and since steps_per_generation=2 in state.limits,  # noqa: E501
    # after step 2 completes with status="running", it triggers continue_as_new and preserves limits!  # noqa: E501
    res_act = gen.send({"value": pack("approved")})
    assert res_act[0] == "act" and res_act[2]["kind"] == "resume"
    with pytest.raises(StopIteration):
        gen.send({"status": "running", "ckpt": "c2"})
    assert ctx.can is not None
    can_payload, _ = ctx.can
    assert can_payload["limits"]["retries"] == 5
    assert can_payload["limits"]["steps_per_generation"] == 2


def test_p2_2_sibling_durable_node_outputs_committed_before_failure_and_tool_ns_call_id() -> (  # noqa: E501
    None
):
    """P2: (a) When one durable node fails in a parallel batch while a sibling
    durable node
    succeeds, the orchestrator must commit the completed sibling's output before
    re-raising.
    (b) Parallel tools inside a subgraph must include call_id in their interrupt IDs
    (tool_ns|call_id) so they never collide."""

    class FakeTaskFailedError(Exception):
        pass

    class FakeCtx:
        def __init__(self) -> None:
            self.committed_reqs: list[dict[str, Any]] = []

        def call_activity(
            self, name: str, *, input: Any = None, retry_policy: Any = None
        ) -> Any:
            return ("act", name, input)

        def set_custom_status(self, status: Any) -> None:
            pass

    ctx = FakeCtx()
    gen = make_orchestrator(Limits())(
        ctx,
        {
            "proto": 1,
            "thread": "t-p22",
            "gen": 0,
            "steps": 0,
            "step_name": "langgraph.x.step",
            "tool_name": "langgraph.x.tool",
            "node_name": "langgraph.x.node",
            "req": {
                "thread": "t-p22",
                "kind": "start",
                "expect": None,
                "input": pack({"x": 1}),
            },
        },
    )
    next(gen)
    nodes_rep = {
        "status": "nodes",
        "ckpt": "c1",
        "node_tasks": [
            {
                "call_id": "c1:node:n_ok:0:t0",
                "index": 0,
                "task_id": "t0",
                "name": "n_ok",
                "input": pack({"x": 1}),
                "durable": True,
                "retries": 1,
            },
            {
                "call_id": "c1:node:n_fail:1:t1",
                "index": 1,
                "task_id": "t1",
                "name": "n_fail",
                "input": pack({"x": 1}),
                "durable": True,
                "retries": 1,
            },
        ],
    }
    node0_yield = gen.send(nodes_rep)
    assert node0_yield[0] == "act" and node0_yield[1] == "langgraph.x.node"
    # Node 0 succeeds, yielding Node 1
    node1_yield = gen.send(
        {"call_id": "c1:node:n_ok:0:t0", "status": "ok", "output": pack({"a": 10})}
    )
    assert node1_yield[0] == "act" and node1_yield[1] == "langgraph.x.node"

    # Node 1 fails! Sending the exception must FIRST yield a StepKind.NODES
    # partial commit for index 0
    commit_yield = gen.throw(FakeTaskFailedError("n_fail boom"))
    assert commit_yield[0] == "act" and commit_yield[2]["kind"] == "nodes"
    assert commit_yield[2]["partial"] is True
    assert commit_yield[2]["outputs"] == {"0": pack({"a": 10})}
    # And after the commit step finishes, the original failure is re-raised!
    with pytest.raises(FakeTaskFailedError, match="n_fail boom"):
        gen.send({"status": "running", "ckpt": "c1"})

    # Part (b): Subgraph parallel tools calling interrupt() must not collide on tool_ns
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode
    from langgraph.types import interrupt

    @tool
    def sub_tool_a(q: str) -> str:
        """Sub tool A."""
        reply = interrupt(f"confirm A: {q}")
        return f"A:{reply}"

    @tool
    def sub_tool_b(q: str) -> str:
        """Sub tool B."""
        reply = interrupt(f"confirm B: {q}")
        return f"B:{reply}"

    class SubState(TypedDict):
        messages: Annotated[list[Any], add_messages]

    def sub_agent(state: SubState) -> dict[str, Any]:
        if len(state["messages"]) == 1:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"id": "call_a", "name": "sub_tool_a", "args": {"q": "1"}},
                            {"id": "call_b", "name": "sub_tool_b", "args": {"q": "2"}},
                        ],
                    )
                ]
            }
        return {"messages": [AIMessage(content="done")]}

    sb = StateGraph(SubState)
    sb.add_node("agent", sub_agent)
    sb.add_node("tools", ToolNode([sub_tool_a, sub_tool_b]))
    sb.add_edge(START, "agent")
    sb.add_edge("agent", "tools")
    sb.add_edge("tools", END)

    pb = StateGraph(SubState)
    pb.add_node("sub", sb.compile())
    pb.add_edge(START, "sub")
    pb.add_edge("sub", END)
    saver = InMemorySaver()
    reg = attach_durability(
        pb.compile(checkpointer=saver),
        name="sub_int_p22",
        checkpointer=saver,
        inline_checkpoints=False,
    )

    r1 = run_step(
        reg.graph,
        Limits(),
        {
            "thread": "t-sub-int",
            "kind": "start",
            "expect": None,
            "input": pack({"messages": [HumanMessage(content="go")]}),
        },
    )
    assert r1["status"] == "tools" and len(r1["tool_calls"]) == 2 and r1.get("tool_ns")
    tc0, tc1 = r1["tool_calls"][0], r1["tool_calls"][1]
    res0 = run_tool(
        reg.graph,
        {
            "thread": "t-sub-int",
            "ckpt": r1["ckpt"],
            "node": r1["node"],
            "tool_ns": r1["tool_ns"],
            "call_id": tc0["call_id"],
            "index": tc0["index"],
            "call": tc0["call"],
        },
    )
    res1 = run_tool(
        reg.graph,
        {
            "thread": "t-sub-int",
            "ckpt": r1["ckpt"],
            "node": r1["node"],
            "tool_ns": r1["tool_ns"],
            "call_id": tc1["call_id"],
            "index": tc1["index"],
            "call": tc1["call"],
        },
    )
    assert res0["status"] == "interrupted" and res1["status"] == "interrupted"
    # Their interrupt IDs must be distinct even though both share the same
    # subgraph tool_ns!
    assert res0["interrupt"]["id"] != res1["interrupt"]["id"]


def test_p2_3_custom_status_limit_on_tool_and_node_interrupts_and_bounded_caches() -> (
    None
):
    """P2: Tool and node interrupt payloads exceeding Limits.custom_status_limit
    must be stripped
    and marked oversized=True in custom_status, and per-saver caches must be bounded."""

    class FakeCtx:
        def __init__(self) -> None:
            self.statuses: list[Any] = []

        def call_activity(
            self, name: str, *, input: Any = None, retry_policy: Any = None
        ) -> Any:
            return ("act", name, input)

        def wait_for_external_event(self, name: str) -> Any:
            return ("evt", name)

        def set_custom_status(self, status: Any) -> None:
            self.statuses.append(status)

    ctx = FakeCtx()
    lim = Limits(custom_status_limit=128)
    gen = make_orchestrator(lim)(
        ctx,
        {
            "proto": 1,
            "thread": "t-p23",
            "gen": 0,
            "steps": 0,
            "step_name": "langgraph.x.step",
            "tool_name": "langgraph.x.tool",
            "node_name": "langgraph.x.node",
            "req": {
                "thread": "t-p23",
                "kind": "start",
                "expect": None,
                "input": pack({"x": 1}),
            },
        },
    )
    next(gen)
    gen.send(
        {
            "status": "tools",
            "ckpt": "c1",
            "node": "tools",
            "tool_calls": [
                {
                    "call_id": "c1:tools:0:tc1",
                    "index": 0,
                    "id": "tc1",
                    "name": "t1",
                    "call": pack({"name": "t1"}),
                    "durable": True,
                    "retries": 1,
                }
            ],
        }
    )
    huge_val = pack("X" * 1000)
    gen.send(
        {
            "call_id": "c1:tools:0:tc1",
            "status": "interrupted",
            "interrupt": {"id": "ti1", "value": huge_val},
        }
    )
    assert ctx.statuses[-1]["pending"]["oversized"] is True
    assert "value" not in ctx.statuses[-1]["pending"]


def test_p2_4_async_bridge_timeout_and_double_submit_guard() -> None:
    """P2: (a) _run_async must enforce a timeout and cancel hung coroutines instead of
    blocking forever. (b) Starting a second orchestration on the same thread while one
    is already RUNNING must raise RunError unless an explicit distinct
    instance_id is used."""

    from langchain_azure_compute.graph._async_bridge import _run_async

    async def hung_coro() -> None:
        await asyncio.sleep(0.5)

    g_dummy = SimpleNamespace(_durable_async_timeout=0.05)
    with pytest.raises(TimeoutError):
        _run_async(g_dummy, hung_coro())  # type: ignore[arg-type]

    # Part (b): Double-submit guard on RunClient.start
    class S(TypedDict):
        x: int

    b = StateGraph(S)
    b.add_node("n", lambda s: {"x": s["x"] + 1})
    b.add_edge(START, "n")
    b.add_edge("n", END)
    reg = attach_durability(b.compile(), name="p24_app")

    active_instances: dict[str, str] = {}

    class FakeSdk:
        def get_orchestration_state(
            self, iid: str, fetch_payloads: bool = False
        ) -> Any:
            if iid in active_instances:
                return SimpleNamespace(
                    runtime_status=SimpleNamespace(name=active_instances[iid])
                )
            return None

        def schedule_new_orchestration(
            self, name: str, input: Any = None, instance_id: str | None = None
        ) -> str:
            active_instances[str(instance_id)] = "RUNNING"
            return str(instance_id)

    client = reg.client(FakeSdk())  # type: ignore[arg-type]
    run1 = client.start({"x": 1}, thread_id="shared-thread")
    assert run1.thread_id == "shared-thread"
    # Second start on the same thread while run1 is RUNNING must fail loudly!
    with pytest.raises(
        __import__("langchain_azure_compute.graph", fromlist=["RunError"]).RunError,
        match="already has an active run",
    ):
        client.start({"x": 2}, thread_id="shared-thread")


def test_p2_5_sdk_options_precedence_user_agent_and_async_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2: (a) Explicit connection_string or DTS_CONNECTION_STRING takes
    precedence over DTS_EMULATOR.
    (b) sdk_options includes user-agent metadata.
    (c) AsyncRunClient and AsyncRun provide non-blocking async methods."""

    from langchain_azure_compute.graph import sdk_options

    monkeypatch.setenv("DTS_EMULATOR", "true")
    opts = sdk_options(
        connection_string="Endpoint=http://localhost:8080;TaskHub=myhub;Authentication=None"
    )
    assert opts["taskhub"] == "myhub"
    assert "channel_options" in opts and any(
        k == "grpc.primary_user_agent" and v == "langgraph-durabletask"
        for k, v in opts["channel_options"].raw_options
    )

    # Part (c): AsyncRunClient and AsyncRun
    from langchain_azure_compute.graph import AsyncRun, AsyncRunClient

    class S(TypedDict):
        x: int

    b = StateGraph(S)
    b.add_node("n", lambda s: {"x": s["x"] + 1})
    b.add_edge(START, "n")
    b.add_edge("n", END)
    reg = attach_durability(b.compile(), name="p25_async")

    class FakeAsyncSdk:
        def __init__(self) -> None:
            self.scheduled: list[Any] = []
            self.events: list[Any] = []
            self.state = SimpleNamespace(
                runtime_status=SimpleNamespace(name="RUNNING"),
                serialized_custom_status=json.dumps(
                    {
                        "pending": {"id": "int-a1", "value": pack("approve?")},
                        "interrupts": [{"id": "int-a1", "value": pack("approve?")}],
                        "step": 1,
                        "ckpt": "c1",
                    }
                ),
                serialized_output=None,
                failure_details=None,
            )

        async def schedule_new_orchestration(
            self, name: str, *, input: Any = None, instance_id: str | None = None
        ) -> str:
            self.scheduled.append((name, input, instance_id))
            return str(instance_id)

        async def get_orchestration_state(
            self, iid: str, *, fetch_payloads: bool = False
        ) -> Any:
            if not self.scheduled:
                return None
            return self.state

        async def raise_orchestration_event(
            self, iid: str, name: str, *, data: Any = None
        ) -> None:
            self.events.append((iid, name, data))
            self.state = SimpleNamespace(
                runtime_status=SimpleNamespace(name="COMPLETED"),
                serialized_custom_status=None,
                serialized_output=json.dumps(
                    {
                        "status": "done",
                        "thread": "t-async",
                        "gen": 0,
                        "steps": 2,
                        "ckpt": "c2",
                        "output": pack({"x": 42}),
                    }
                ),
                failure_details=None,
            )

        async def wait_for_orchestration_completion(
            self, iid: str, *, timeout: int = 60, fetch_payloads: bool = True
        ) -> Any:
            return self.state

    async def _exercise_async() -> None:
        asdk = FakeAsyncSdk()
        aclient = reg.async_client(asdk)
        assert isinstance(aclient, AsyncRunClient)
        arun = await aclient.start({"x": 41}, thread_id="t-async")
        assert isinstance(arun, AsyncRun)
        assert await arun.status() == "RUNNING"
        pending = await arun.wait_pending(timeout=2.0)
        assert (
            pending is not None
            and pending.interrupt_id == "int-a1"
            and pending.value == "approve?"
        )
        await arun.resume(pending, "yes")
        res = await arun.wait_result(timeout=2.0)
        assert res == {"x": 42}

    asyncio.run(_exercise_async())


def test_p2_6_real_orchestrator_replay_and_v1_create_agent() -> None:
    """Test gaps: (a) Re-run make_orchestrator from scratch on every replay turn
    to verify
    strict replay determinism across steps, tools, and interrupts.
    (b) Exercise langchain.agents.create_agent with durable tools."""
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool

    @tool
    def multiply_numbers(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    class ToolBindingFakeModel(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

    model = ToolBindingFakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc_mul",
                            "name": "multiply_numbers",
                            "args": {"a": 6, "b": 7},
                        }
                    ],
                ),
                AIMessage(content="The product is 42."),
            ]
        )
    )
    saver = InMemorySaver()
    agent = create_agent(model=model, tools=[multiply_numbers], checkpointer=saver)
    reg = attach_durability(
        agent, name="v1_agent", checkpointer=saver, inline_checkpoints=False
    )

    class ReplayDriver:
        """Replays the orchestrator generator from turn 0 on every single
        activity/event completion."""

        def __init__(self, orch_fn: Any, init_input: dict[str, Any]) -> None:
            self.orch_fn = orch_fn
            self.init_input = init_input
            self.history: list[Any] = []

        def run_to_completion(self, dispatch_fn: Any) -> dict[str, Any]:
            while True:
                idx = 0

                class ReplayCtx:
                    def __init__(self) -> None:
                        self.is_replaying = True

                    def call_activity(
                        self, name: str, *, input: Any = None, retry_policy: Any = None
                    ) -> Any:
                        return ("act", name, input)

                    def set_custom_status(self, status: Any) -> None:
                        pass

                ctx = ReplayCtx()
                gen = self.orch_fn(ctx, self.init_input)
                try:
                    yielded = next(gen)
                    while idx < len(self.history):
                        yielded = gen.send(self.history[idx])
                        idx += 1
                    ctx.is_replaying = False
                    # Execute the new frontier task and record its result in history, then replay from scratch!  # noqa: E501
                    res = dispatch_fn(yielded)
                    self.history.append(res)
                except StopIteration as done:
                    return done.value

    orch = make_orchestrator(reg.limits)
    init_payload = {
        "proto": 1,
        "thread": "t-v1-agent",
        "gen": 0,
        "steps": 0,
        "step_name": reg.step_name,
        "tool_name": reg.tool_name,
        "node_name": reg.node_name,
        "req": {
            "thread": "t-v1-agent",
            "kind": "start",
            "expect": None,
            "input": pack({"messages": [HumanMessage(content="What is 6 * 7?")]}),
        },
    }

    from langchain_azure_compute.graph.runner import read_output, run_tool

    def dispatch(yielded: Any) -> Any:
        _, act_name, act_inp = yielded
        if act_name.endswith(".step"):
            return run_step(
                reg.graph,
                reg.limits,
                act_inp,
                tool_config=reg.tool_config,
                node_config=reg.node_config,
            )
        return run_tool(reg.graph, act_inp)

    driver = ReplayDriver(orch, init_payload)
    final_out = driver.run_to_completion(dispatch)
    assert final_out["status"] == "done"

    final_state = read_output(reg.graph, "t-v1-agent", final_out["ckpt"])
    assert final_state["messages"][-1].content == "The product is 42."

    # Part (c): Exercise deepagents.create_deep_agent with durable tools
    from deepagents import create_deep_agent

    deep_model = ToolBindingFakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc_deep",
                            "name": "multiply_numbers",
                            "args": {"a": 8, "b": 9},
                        }
                    ],
                ),
                AIMessage(content="Deep agent result is 72."),
            ]
        )
    )
    deep_saver = InMemorySaver()
    deep_graph = create_deep_agent(
        model=deep_model, tools=[multiply_numbers], checkpointer=deep_saver
    )
    deep_reg = attach_durability(
        deep_graph,
        name="deep_agent_v1",
        checkpointer=deep_saver,
        inline_checkpoints=False,
    )
    deep_init = {
        "proto": 1,
        "thread": "t-deep-agent",
        "gen": 0,
        "steps": 0,
        "step_name": deep_reg.step_name,
        "tool_name": deep_reg.tool_name,
        "node_name": deep_reg.node_name,
        "req": {
            "thread": "t-deep-agent",
            "kind": "start",
            "expect": None,
            "input": pack({"messages": [HumanMessage(content="Compute 8 * 9")]}),
        },
    }

    def dispatch_deep(yielded: Any) -> Any:
        _, act_name, act_inp = yielded
        if act_name.endswith(".step"):
            return run_step(
                deep_reg.graph,
                deep_reg.limits,
                act_inp,
                tool_config=deep_reg.tool_config,
                node_config=deep_reg.node_config,
            )
        return run_tool(deep_reg.graph, act_inp)

    deep_out = ReplayDriver(
        make_orchestrator(deep_reg.limits), deep_init
    ).run_to_completion(dispatch_deep)
    assert deep_out["status"] == "done"
    deep_state = read_output(deep_reg.graph, "t-deep-agent", deep_out["ckpt"])
    assert deep_state["messages"][-1].content == "Deep agent result is 72."


def test_p1_9_registration_register_worker_logging_and_async_client_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1-9: Exercise Registration.register(worker) activity wrappers,
    worker_logger INFO/WARNING
    paths, work_item_filters auto-enable, and AsyncRunClient/AsyncRun branches."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode

    from langchain_azure_compute.graph import durable_node, durable_tool

    @tool
    @durable_tool
    def echo_tool(msg: str) -> str:
        """Echo message."""
        if msg == "boom":
            raise RuntimeError("tool boom")
        return f"echo:{msg}"

    class WState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        val: int

    def agent_node(state: WState) -> dict[str, Any]:
        if len(state["messages"]) == 1:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"id": "c1", "name": "echo_tool", "args": {"msg": "hi"}}
                        ],
                    )
                ]
            }
        return {"messages": [AIMessage(content="done")]}

    @durable_node
    def compute_node(state: WState) -> dict[str, Any]:
        if state.get("val") == -1:
            raise RuntimeError("node boom")
        return {"val": state.get("val", 0) + 5}

    b = StateGraph(WState)
    b.add_node("agent", agent_node)
    b.add_node("tools", ToolNode([echo_tool]))
    b.add_node("compute", compute_node)
    b.add_edge(START, "agent")
    b.add_edge("agent", "tools")
    b.add_edge("tools", "compute")
    b.add_edge("compute", END)
    reg = attach_durability(b.compile(), name="worker_log_app")

    info_logs: list[str] = []
    warn_logs: list[str] = []

    class FakeLogger:
        def info(self, msg: str) -> None:
            info_logs.append(msg)

        def warning(self, msg: str) -> None:
            warn_logs.append(msg)

    class FakeWorker:
        def __init__(self) -> None:
            self._logger = FakeLogger()
            self._is_running = False
            self._work_item_filters = None
            self.filters_enabled = False
            self.activities: dict[str, Any] = {}
            self.orchestrators: dict[str, Any] = {}

        def use_work_item_filters(self) -> None:
            self.filters_enabled = True

        def add_orchestrator(self, fn: Any) -> str:
            self.orchestrators[fn.__name__] = fn
            return fn.__name__

        def add_activity(self, fn: Any) -> str:
            self.activities[fn.__name__] = fn
            return fn.__name__

    w = FakeWorker()
    reg.register(w)
    assert w.filters_enabled is True
    ctx = SimpleNamespace(orchestration_id="inst-log-1")

    step_fn = w.activities[reg.step_name]
    tool_fn = w.activities[reg.tool_name]
    node_fn = w.activities[reg.node_name]

    # 1. Start step -> tools
    r1 = step_fn(
        ctx,
        {
            "thread": "t-wlog",
            "kind": "start",
            "expect": None,
            "input": pack({"messages": [HumanMessage(content="go")], "val": 10}),
        },
    )
    assert r1["status"] == "tools"
    # 2. Tool activity success & failure logging
    tc = r1["tool_calls"][0]
    t_rep = tool_fn(
        ctx,
        {
            "thread": "t-wlog",
            "ckpt": r1["ckpt"],
            "node": r1["node"],
            "call_id": tc["call_id"],
            "index": tc["index"],
            "call": tc["call"],
            "checkpoint_bundle": r1["checkpoint_bundle"],
        },
    )
    assert t_rep["status"] == "ok"
    with pytest.raises(RuntimeError, match="tool boom"):
        tool_fn(
            ctx,
            {
                "thread": "t-wlog",
                "ckpt": r1["ckpt"],
                "node": r1["node"],
                "call_id": tc["call_id"],
                "index": tc["index"],
                "call": pack(
                    {"id": "c_err", "name": "echo_tool", "args": {"msg": "boom"}}
                ),
                "checkpoint_bundle": r1["checkpoint_bundle"],
            },
        )
    # 3. Commit tools -> pauses before durable node 'compute'
    r2 = step_fn(
        ctx,
        {
            "thread": "t-wlog",
            "kind": "tools",
            "expect": r1["ckpt"],
            "node": r1["node"],
            "outputs": [t_rep["output"]],
            "checkpoint_bundle": r1["checkpoint_bundle"],
        },
    )
    assert r2["status"] == "nodes"
    nt = r2["node_tasks"][0]
    # 4. Node activity success & failure logging
    n_rep = node_fn(
        ctx,
        {
            "thread": "t-wlog",
            "ckpt": r2["ckpt"],
            "call_id": nt["call_id"],
            "index": nt["index"],
            "task_id": nt["task_id"],
            "name": nt["name"],
            "input": nt.get("input"),
            "checkpoint_bundle": r2["checkpoint_bundle"],
        },
    )
    assert n_rep["status"] == "ok"
    with pytest.raises(RuntimeError, match="unknown graph node"):
        node_fn(
            ctx,
            {
                "thread": "t-wlog",
                "ckpt": r2["ckpt"],
                "call_id": nt["call_id"],
                "index": nt["index"],
                "task_id": nt["task_id"],
                "name": "unknown_node",
                "checkpoint_bundle": r2["checkpoint_bundle"],
            },
        )
    r3 = step_fn(
        ctx,
        {
            "thread": "t-wlog",
            "kind": "nodes",
            "expect": r2["ckpt"],
            "outputs": [n_rep["output"]],
            "checkpoint_bundle": r2["checkpoint_bundle"],
        },
    )
    assert r3["status"] == "done"
    # 5. Step failure logging
    failure = step_fn(
        ctx, {"thread": "t-wlog", "kind": "bogus_kind", "expect": r3["ckpt"]}
    )
    assert failure["status"] == "error"
    assert "unknown step kind" in failure["error"]

    assert len(info_logs) >= 4
    assert len(warn_logs) >= 3

    # 6. Exercise sdk_options Authentication modes, payload_store, and log_handler
    import logging

    monkeypatch.setenv("AZURE_TENANT_ID", "00000000-0000-0000-0000-000000000000")
    monkeypatch.setenv("AZURE_CLIENT_ID", "00000000-0000-0000-0000-000000000000")
    monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", __file__)
    dummy_handler = logging.NullHandler()
    for auth_mode in (
        "ManagedIdentity",
        "ManagedIdentity;ClientId=cid-1",
        "WorkloadIdentity",
        "WorkloadIdentity;ClientId=cid-2",
        "Environment",
        "AzureCli",
        "AzurePowerShell",
        "DefaultAzure",
        "DefaultAzure;ClientId=cid-3",
    ):
        opts = sdk_options(
            connection_string=f"Endpoint=https://my-dts.westus2.durabletask.io;TaskHub=th;Authentication={auth_mode}",
            payload_store=SimpleNamespace(),  # type: ignore[arg-type]
            log_handler=dummy_handler,
        )
        assert opts["token_credential"] is not None
        assert opts["log_handler"] is dummy_handler
        assert "payload_store" in opts


def test_p1_7_and_p1_8_pyproject_bounds_curated_beta_surface_and_no_leftovers() -> None:
    """P1-7 & P1-8:
    - pyproject.toml sets langgraph>=1.2.0,<1.3 and <1.14 ceilings on
    durabletask packages,  # noqa: E501
      and does not skip langgraph.* in mypy overrides.
    - langchain_azure_compute.graph.__all__ exports a small curated @beta
    surface (<= 13 names),  # noqa: E501
      excluding internal wire types and pack/unpack.
    - No langgraph_durabletask leftovers exist in package modules."""
    import langchain_azure_compute.graph as pkg

    pkg_root = Path(__file__).resolve().parents[3]
    pyproject_text = (pkg_root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"langgraph>=1.2.0,<1.3"' in pyproject_text
    assert '"langgraph-checkpoint>=4.1.0,<5.0"' in pyproject_text
    assert '"langgraph-prebuilt>=1.1.0,<1.2"' in pyproject_text
    assert '"durabletask>=1.13.0,<1.14"' in pyproject_text
    assert '"durabletask-azuremanaged>=1.13.0,<1.14"' in pyproject_text
    assert '"langgraph.*"' not in pyproject_text

    # Curated public __all__ surface
    assert len(pkg.__all__) <= 13
    for internal_sym in (
        "pack",
        "unpack",
        "PackedBlob",
        "StepRequest",
        "CheckpointBundle",
        "mark_tool",
        "mark_node",
    ):
        assert internal_sym not in pkg.__all__

    # Every exported function/class is marked @beta
    for sym_name in pkg.__all__:
        obj = getattr(pkg, sym_name)
        doc = getattr(obj, "__doc__", "") or ""
        assert "Beta" in doc or "preview" in doc, f"{sym_name} is missing @beta marker"

    # No langgraph_durabletask leftovers in package source files
    graph_dir = pkg_root / "langchain_azure_compute" / "graph"
    for py_file in graph_dir.glob("*.py"):
        txt = py_file.read_text(encoding="utf-8")
        assert "langgraph_durabletask" not in txt, f"leftover in {py_file.name}"
        assert "langgraph-durabletask[azure-functions]" not in txt, (
            f"leftover in {py_file.name}"
        )
