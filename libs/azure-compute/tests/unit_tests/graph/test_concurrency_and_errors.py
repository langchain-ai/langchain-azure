# Copyright (c) Microsoft. All rights reserved.

"""Concurrency, retry isolation, interrupt races, and payload limits."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections import Counter, OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from langchain_azure_compute.graph import (
    Limits,
    RunError,
    attach_durability,
    sdk_options,
)
from langchain_azure_compute.graph._api import AsyncRun, Run
from langchain_azure_compute.graph._async_bridge import _get_bg_loop, _run_async
from langchain_azure_compute.graph._config import OrchestratorState
from langchain_azure_compute.graph._inline_saver import (
    _export_checkpoint_bundle,
    _hydrate_checkpoint_bundle,
)
from langchain_azure_compute.graph.runner import (
    DURABLE_TOOL_ERROR_KEY,
    _extract_failure_info,
    make_orchestrator,
    pack,
    read_output,
    run_step,
    run_tool,
    unpack,
)
from tests.unit_tests.graph.test_orchestrator import Ctx, start_input


def _drive_tools(reg, request):
    report = run_step(reg.graph, reg.limits, request)
    for _ in range(12):
        if report["status"] == "done":
            return report
        req = {"thread": request["thread"], "expect": report["ckpt"], "kind": "step"}
        if report["status"] == "tools":
            outputs = []
            for call in report["tool_calls"]:
                result = run_tool(
                    reg.graph,
                    {
                        **call,
                        "thread": request["thread"],
                        "ckpt": report["ckpt"],
                        "node": report["node"],
                    },
                )
                outputs.append(result["output"])
            req.update(kind="tools", node=report["node"], outputs=outputs)
        report = run_step(reg.graph, reg.limits, req)
    pytest.fail("graph did not complete")


@pytest.mark.parametrize("decision", ["reject", "respond", "edit"])
@pytest.mark.parametrize("send", [True, False])
def test_hitl_schedules_only_approved_calls(decision, send):
    executed = []

    @tool
    def transfer(amount: int) -> str:
        """Transfer the approved amount."""
        executed.append(amount)
        return str(amount)

    calls = [
        {
            "name": "transfer",
            "args": {"amount": 100},
            "id": "unsafe",
            "type": "tool_call",
        },
        {"name": "transfer", "args": {"amount": 1}, "id": "safe", "type": "tool_call"},
        {"name": "transfer", "args": {"amount": 2}, "id": "safe2", "type": "tool_call"},
    ]
    pending = calls[1:]
    messages = [AIMessage(content="", tool_calls=calls)]
    if decision == "edit":
        edited = {**calls[0], "args": {"amount": 3}}
        pending = [edited, *pending]
        if not send:
            messages = [AIMessage(content="", tool_calls=pending)]
    else:
        messages.append(ToolMessage(content=f"human:{decision}", tool_call_id="unsafe"))

    builder = StateGraph(MessagesState)
    builder.add_node("human", lambda s: {"messages": messages})
    builder.add_node("tools", ToolNode([transfer]))
    builder.add_edge(START, "human")
    if send:
        builder.add_conditional_edges(
            "human", lambda s: [Send("tools", [call]) for call in pending], ["tools"]
        )
    else:
        builder.add_edge("human", "tools")
    builder.add_edge("tools", END)
    reg = attach_durability(builder.compile(), name="hitl")
    _drive_tools(
        reg,
        {
            "kind": "start",
            "thread": "t",
            "expect": None,
            "input": pack({"messages": []}),
        },
    )
    assert sorted(executed) == ([1, 2, 3] if decision == "edit" else [1, 2])
    output = read_output(reg.graph, "t")
    counts = Counter(
        m.tool_call_id for m in output["messages"] if isinstance(m, ToolMessage)
    )
    assert counts == {"unsafe": 1, "safe": 1, "safe2": 1}


@pytest.mark.parametrize("memory", [False, True])
def test_stale_expect_is_not_replay_of_another_turn(memory):
    executed = []

    @tool
    def record(value: str) -> str:
        """Record a turn's value."""
        executed.append(value)
        return value

    def model(state):
        text = state["messages"][-1].content
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "record", "args": {"value": text}, "id": text}
                    ],
                )
            ]
        }

    saver = (
        InMemorySaver()
        if memory
        else SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    )
    builder = StateGraph(MessagesState)
    builder.add_node("model", model)
    builder.add_node("tools", ToolNode([record]))
    builder.add_edge(START, "model")
    builder.add_edge("model", "tools")
    builder.add_edge("tools", END)
    reg = attach_durability(builder.compile(checkpointer=saver), name="turns")
    first = _drive_tools(
        reg,
        {
            "kind": "start",
            "thread": "t",
            "expect": None,
            "instance_id": "A",
            "input": pack({"messages": [HumanMessage(content="A")]}),
        },
    )
    base = first["ckpt"]
    req_b = {
        "kind": "start",
        "thread": "t",
        "expect": base,
        "instance_id": "B",
        "input": pack({"messages": [HumanMessage(content="B")]}),
    }
    report = run_step(reg.graph, reg.limits, req_b)
    assert report["status"] == "tools"
    metadata = saver.get_tuple({"configurable": {"thread_id": "t"}}).metadata
    assert metadata["durable_instance_id"] == "B"
    assert metadata["durable_input_sig"]
    assert metadata["durable_expect"] == base
    assert run_step(reg.graph, reg.limits, req_b)["replayed"]
    for other in [
        {**req_b, "instance_id": "C"},
        {**req_b, "input": pack({"messages": [HumanMessage(content="C")]})},
    ]:
        with pytest.raises(RunError, match="Concurrent turn conflict"):
            run_step(reg.graph, reg.limits, other)
    assert executed == ["A"]
    _drive_tools(reg, req_b)
    assert executed == ["A", "B"]
    assert [
        m.content
        for m in read_output(reg.graph, "t")["messages"]
        if isinstance(m, HumanMessage)
    ] == ["A", "B"]


def _one_tool_reg(fn, args, **node_options):
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode([fn], **node_options))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    reg = attach_durability(builder.compile(), name="errors")
    report = run_step(
        reg.graph,
        reg.limits,
        {
            "kind": "start",
            "thread": "t",
            "expect": None,
            "input": pack(
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[{"name": fn.name, "args": args, "id": "call"}],
                        )
                    ]
                }
            ),
        },
    )
    return (
        reg,
        report,
        {
            **report["tool_calls"][0],
            "thread": "t",
            "ckpt": report["ckpt"],
            "node": "tools",
        },
    )


def test_invalid_tool_arguments_are_an_error_message_without_retry():
    executed = []

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        executed.append((a, b))
        return a + b

    reg, report, req = _one_tool_reg(add, {"a": "not-a-number", "b": 2})
    result = run_tool(reg.graph, req)
    message = unpack(result["output"])
    assert result["status"] == "ok"
    assert isinstance(message, ToolMessage) and message.status == "error"
    assert "a" in message.content
    assert not executed
    _drive_tools(
        reg,
        {
            "kind": "tools",
            "thread": "t",
            "expect": report["ckpt"],
            "node": "tools",
            "outputs": [result["output"]],
        },
    )
    assert not executed


class MyCustomError(Exception):
    pass


@pytest.mark.parametrize(
    "policy", [(MyCustomError,), lambda e: f"custom:{type(e).__name__}"]
)
def test_exhausted_custom_error_uses_real_exception(policy):
    @tool
    def fail() -> str:
        """Fail transiently."""
        raise MyCustomError("failed")

    reg, report, req = _one_tool_reg(fail, {}, handle_tool_errors=policy)
    for _ in range(3):
        with pytest.raises(Exception) as failure:
            run_tool(reg.graph, req)
    # Reproduce the SDK's type-and-message-only transport, losing the Python instance.
    info = _extract_failure_info(failure.value)
    sdk_failure = SimpleNamespace(
        details=SimpleNamespace(error_type=info["error_type"], message=info["message"])
    )
    transported = _extract_failure_info(sdk_failure)
    _drive_tools(
        reg,
        {
            "kind": "tools",
            "thread": "t",
            "expect": report["ckpt"],
            "node": "tools",
            "outputs": [pack({DURABLE_TOOL_ERROR_KEY: transported})],
        },
    )
    message = read_output(reg.graph, "t")["messages"][-1]
    assert message.status == "error"
    assert "MyCustomError" in message.content


def test_unmatched_custom_error_is_not_converted_to_a_tool_message():
    @tool
    def fail() -> str:
        """Fail with an unhandled error."""
        raise ValueError("unhandled")

    reg, report, req = _one_tool_reg(fail, {}, handle_tool_errors=(MyCustomError,))
    with pytest.raises(Exception) as failure:
        run_tool(reg.graph, req)
    info = _extract_failure_info(failure.value)
    with pytest.raises(Exception, match="unhandled"):
        run_step(
            reg.graph,
            reg.limits,
            {
                "kind": "tools",
                "thread": "t",
                "expect": report["ckpt"],
                "node": "tools",
                "outputs": [pack({DURABLE_TOOL_ERROR_KEY: info})],
            },
        )


def _echo_reg(**kwargs):
    builder = StateGraph(dict)
    builder.add_node("echo", lambda s: s)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return attach_durability(builder.compile(), name="echo", **kwargs)


def test_final_bundle_does_not_duplicate_state_and_can_is_single_copy():
    reg = _echo_reg()
    report = run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "start",
            "expect": None,
            "input": pack({"text": "x" * 420_000}),
        },
    )
    assert not report.get("output")
    assert len(json.dumps(report).encode()) < 950_000
    state = OrchestratorState.from_payload(
        {
            **start_input(),
            "checkpoint_bundle": report["checkpoint_bundle"],
        }
    )
    payload = state.to_can_payload()
    assert bool(payload.get("checkpoint_bundle")) != bool(
        payload["req"].get("checkpoint_bundle")
    )
    restored = OrchestratorState.from_payload(payload)
    assert restored.checkpoint_bundle == report["checkpoint_bundle"]


@pytest.mark.parametrize("async_handle", [False, True])
async def test_wait_result_reads_hydrated_bundle_when_output_is_null(async_handle):
    worker, client = _echo_reg(), _echo_reg()
    report = run_step(
        worker.graph,
        worker.limits,
        {
            "thread": "t",
            "kind": "start",
            "expect": None,
            "input": pack({"text": "value"}),
        },
    )
    terminal = SimpleNamespace(
        runtime_status=SimpleNamespace(name="COMPLETED"),
        serialized_output=json.dumps({**report, "gen": 0, "steps": 1, "output": None}),
    )
    if async_handle:

        class Sdk:
            async def wait_for_orchestration_completion(self, *a, **k):
                return terminal

        output = await AsyncRun(client, Sdk(), "r", "t").wait_result()
    else:
        sdk = SimpleNamespace(
            wait_for_orchestration_completion=lambda *a, **k: terminal
        )
        output = Run(client, sdk, "r", "t").wait_result()
    assert output == {"text": "value"}


@pytest.mark.parametrize("where", ["report", "commit", "can", "start"])
def test_entire_payload_budget_is_checked(where):
    oversized = {"padding": "x" * 950_000}
    with pytest.raises(RunError, match="tool/node|external storage"):
        if where == "report":
            reg = _echo_reg(inline_checkpoints=False, checkpointer=InMemorySaver())
            run_step(
                reg.graph,
                reg.limits,
                {"thread": "t", "kind": "inspect", "expect": None, **oversized},
            )
        elif where == "commit":
            ctx = Ctx()
            gen = make_orchestrator(Limits())(ctx, start_input())
            next(gen)
            gen.send(
                {
                    "status": "tools",
                    "ckpt": "c",
                    "node": "tools",
                    "tool_calls": [
                        {
                            "call_id": "c",
                            "index": 0,
                            "durable": True,
                            "call": pack({}),
                            "retries": 1,
                        }
                    ],
                }
            )
            gen.send({"status": "ok", "output": pack(oversized)})
        elif where == "can":
            state = OrchestratorState.from_payload(start_input())
            state.request.update(oversized)
            state.to_can_payload()
        else:
            _echo_reg().client(
                SimpleNamespace(schedule_new_orchestration=lambda *a, **k: None)
            ).start(oversized)


@pytest.mark.parametrize("status", ["COMPLETED", "FAILED", "TERMINATED"])
async def test_terminal_runs_never_report_sticky_interrupts(status):
    meta = SimpleNamespace(
        runtime_status=SimpleNamespace(name=status),
        serialized_custom_status=json.dumps(
            {"pending": {"id": "i", "value": pack("answered")}}
        ),
    )
    reg = _echo_reg()
    sync = Run(
        reg, SimpleNamespace(get_orchestration_state=lambda *a, **k: meta), "r", "t"
    )
    assert sync.pending_interrupt() is None
    assert sync.pending_interrupts() == []

    class Sdk:
        async def get_orchestration_state(self, *a, **k):
            return meta

    async_run = AsyncRun(reg, Sdk(), "r", "t")
    assert await async_run.pending_interrupt() is None
    assert await async_run.pending_interrupts() == []


def test_cleared_status_is_explicit_and_owner_is_not_public():
    ctx = Ctx()
    inp = {**start_input(), "owner": "principal"}
    gen = make_orchestrator(Limits())(ctx, inp)
    next(gen)
    gen.send(
        {
            "status": "interrupted",
            "ckpt": "c",
            "interrupt": {"id": "i", "value": pack("approve")},
        }
    )
    gen.send({"value": pack(True)})
    assert ctx.status[-1] == {"state": "running"}
    with pytest.raises(StopIteration) as completed:
        gen.send({"status": "done", "ckpt": "d", "output": pack("answer")})
    assert ctx.status[-1] == {"state": "completed"}
    assert all("owner" not in status for status in ctx.status)
    assert "owner" not in completed.value.value
    state = OrchestratorState.from_payload(inp)
    assert state.to_can_payload()["owner"] == "principal"


class CustomSerde:
    def __init__(self):
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        self.inner = JsonPlusSerializer()

    def dumps_typed(self, value):
        kind, raw = self.inner.dumps_typed(value)
        return f"custom:{kind}", raw

    def loads_typed(self, pair):
        return self.inner.loads_typed((pair[0].removeprefix("custom:"), pair[1]))


@pytest.mark.parametrize("inline", [False, True])
def test_caller_serializer_is_preserved_and_bundle_roundtrips(inline):
    serde = CustomSerde()
    saver = InMemorySaver(serde=serde)
    reg = _echo_reg(checkpointer=saver, inline_checkpoints=inline)
    run_step(
        reg.graph,
        reg.limits,
        {"thread": "t", "kind": "start", "expect": None, "input": pack({"value": 1})},
    )
    assert saver.serde is serde
    if inline:
        bundle = _export_checkpoint_bundle(reg.graph, "t")
        saver.delete_thread("t")
        _hydrate_checkpoint_bundle(reg.graph, "t", bundle)
    assert read_output(reg.graph, "t") == {"value": 1}


@dataclass
class CustomValue:
    value: str


def test_allowed_msgpack_modules_restores_custom_values():
    _echo_reg(allowed_msgpack_modules=[(__name__, "CustomValue")])
    assert unpack(pack(CustomValue("one"))) == CustomValue("one")


async def test_async_start_uses_saver_own_running_loop():
    loop = asyncio.get_running_loop()

    class LoopSaver(BaseCheckpointSaver):
        async def aget_tuple(self, config):
            assert asyncio.get_running_loop() is loop
            return None

    reg = _echo_reg(checkpointer=LoopSaver(), inline_checkpoints=False)

    class Sdk:
        async def schedule_new_orchestration(self, *a, **k):
            pass

    await reg.async_client(Sdk()).start({"value": 1}, thread_id="t")


def test_async_bridge_never_mutates_saver_loop_or_lock():
    original_loop = asyncio.new_event_loop()
    lock = asyncio.Lock()
    saver = SimpleNamespace(loop=original_loop, lock=lock)

    async def value():
        return 42

    assert _run_async(SimpleNamespace(checkpointer=saver), value()) == 42
    assert saver.loop is original_loop and saver.lock is lock
    assert _get_bg_loop().is_running()
    original_loop.close()


def test_async_bridge_called_on_background_loop_does_not_deadlock():
    async def outer():
        async def value():
            return 42

        graph = SimpleNamespace(checkpointer=None, _durable_async_timeout=0.05)
        return _run_async(graph, value())

    future = asyncio.run_coroutine_threadsafe(outer(), _get_bg_loop())
    assert future.result(timeout=2) == 42


def test_limits_exposes_async_timeout_and_fifo_cache():
    reg = _echo_reg(limits=Limits(async_timeout_seconds=0.01))
    assert reg.graph._durable_async_timeout == 0.01
    for i in range(258):
        run_step(
            reg.graph,
            reg.limits,
            {
                "thread": str(i),
                "kind": "start",
                "expect": None,
                "input": pack({"value": i}),
            },
        )
    starts = reg.graph.checkpointer._durable_completed_starts
    expects = reg.graph.checkpointer._durable_completed_expects
    assert isinstance(starts, OrderedDict) and isinstance(expects, OrderedDict)
    assert len(starts) == len(expects) == 256
    assert next(iter(starts))[0] == next(iter(expects))[0] == "2"


def test_sdk_options_taskhub_name_and_insecure_credentials(monkeypatch):
    monkeypatch.delenv("DTS_TASKHUB", raising=False)
    monkeypatch.setenv("TASKHUB_NAME", "configured")
    assert sdk_options("127.0.0.1:8080")["taskhub"] == "configured"
    assert sdk_options("127.0.0.1:8080", taskhub="explicit")["taskhub"] == "explicit"
    with pytest.raises(ValueError, match="insecure channel"):
        sdk_options(
            "scheduler.example.com", token_credential=object(), secure_channel=False
        )
    assert (
        sdk_options("localhost:8080", token_credential=object(), secure_channel=False)[
            "secure_channel"
        ]
        is False
    )


async def test_default_instance_ids_are_deterministic_for_same_base_and_input():
    reg = _echo_reg()

    class Sdk:
        def get_orchestration_state(self, *a, **k):
            return None

        def schedule_new_orchestration(self, name, input, instance_id):
            self.payload = input

    sdk = Sdk()
    a = reg.client(sdk).start({"value": 1}, thread_id="t", reuse_existing=False)
    b = reg.client(sdk).start({"value": 1}, thread_id="t", reuse_existing=False)
    c = reg.client(sdk).start({"value": 2}, thread_id="t", reuse_existing=False)
    assert a.instance_id == b.instance_id != c.instance_id
    assert a.instance_id.startswith("r-")
    assert sdk.payload["req"]["instance_id"] == c.instance_id

    class AsyncSdk:
        async def schedule_new_orchestration(self, *a, **k):
            pass

    async_b = await reg.async_client(AsyncSdk()).start(
        {"value": 1}, thread_id="t", reuse_existing=False
    )
    assert async_b.instance_id == b.instance_id


def test_full_state_tool_task_with_all_calls_answered_is_completed_without_execution():
    executed = []

    @tool
    def dangerous() -> str:
        """Perform a sensitive operation."""
        executed.append(True)
        return "executed"

    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode([dangerous]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    reg = attach_durability(builder.compile(), name="answered")
    _drive_tools(
        reg,
        {
            "thread": "t",
            "kind": "start",
            "expect": None,
            "input": pack(
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[{"name": "dangerous", "args": {}, "id": "c"}],
                        ),
                        ToolMessage(content="rejected", tool_call_id="c"),
                    ]
                }
            ),
        },
    )
    assert not executed


def test_payload_budget_uses_actual_sdk_encoding_not_compact_json():
    from durabletask.serialization import JsonDataConverter

    from langchain_azure_compute.graph._config import _check_payload

    payload = {"tiny_items": [""] * 260_000}
    assert len(JsonDataConverter().serialize(payload).encode("utf-8")) > 950_000
    with pytest.raises(RunError, match="payload limit"):
        _check_payload(payload, "many items")


def test_oversized_step_result_is_checked_in_addition_to_its_bundle(monkeypatch):
    from langchain_azure_compute.graph import _inline_saver

    builder = StateGraph(dict)
    builder.add_node("expand", lambda s: {"text": "x" * 720_000})
    builder.add_edge(START, "expand")
    builder.add_edge("expand", END)
    reg = attach_durability(builder.compile(), name="expand")
    monkeypatch.setattr(_inline_saver, "MAX_BUNDLE_BYTES", 2_000_000)
    with pytest.raises(RunError, match="activity result.*payload limit"):
        run_step(
            reg.graph,
            reg.limits,
            {
                "thread": "t",
                "kind": "start",
                "expect": None,
                "input": pack({"text": ""}),
            },
        )


def test_registered_step_conflict_is_non_retryable_to_the_sdk():
    reg = _echo_reg()

    class Worker:
        def __init__(self):
            self.activities = {}

        def add_activity(self, fn):
            self.activities[fn.__name__] = fn

        def add_orchestrator(self, fn):
            self.orchestrator = fn

    worker = Worker()
    reg.register(worker)
    fn = worker.activities[reg.step_name]
    ctx = SimpleNamespace(instance_id="first")
    fn(
        ctx,
        {"thread": "t", "kind": "start", "expect": None, "input": pack({"value": 1})},
    )
    conflict = fn(
        SimpleNamespace(instance_id="second"),
        {
            "thread": "t",
            "kind": "start",
            "expect": None,
            "input": pack({"value": 2}),
        },
    )
    assert conflict["status"] == "error"
    orch_ctx = Ctx()
    gen = worker.orchestrator(orch_ctx, start_input())
    next(gen)
    with pytest.raises(RunError, match="Concurrent turn conflict"):
        gen.send(conflict)
    assert len(orch_ctx.calls) == 1


@pytest.mark.parametrize("decision", ["reject", "respond", "edit"])
def test_real_create_agent_hitl_never_executes_rejected_or_responded_calls(decision):
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    executed = []

    @tool
    def transfer(amount: int) -> str:
        """Transfer an approved amount."""
        executed.append(amount)
        return str(amount)

    @tool
    def safe(value: int) -> str:
        """Perform an operation requiring no human review."""
        return str(value)

    class Model(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = Model(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "transfer", "args": {"amount": 100}, "id": "unsafe"},
                        {"name": "safe", "args": {"value": 1}, "id": "safe"},
                        {"name": "safe", "args": {"value": 2}, "id": "safe2"},
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )
    agent = create_agent(
        model=model,
        tools=[transfer, safe],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "transfer": {
                        "allowed_decisions": ["approve", "reject", "respond", "edit"]
                    },
                }
            )
        ],
    )
    reg = attach_durability(agent, name="real_hitl")
    report = run_step(
        reg.graph,
        reg.limits,
        {
            "thread": "t",
            "kind": "start",
            "expect": None,
            "input": pack({"messages": [HumanMessage(content="transfer")]}),
        },
    )
    for _ in range(8):
        if report["status"] == "interrupted":
            break
        report = run_step(
            reg.graph,
            reg.limits,
            {
                "thread": "t",
                "kind": "step",
                "expect": report["ckpt"],
            },
        )
    assert report["status"] == "interrupted"
    assert not executed
    response = {"type": decision, "message": f"human:{decision}"}
    if decision == "edit":
        response["edited_action"] = {"name": "transfer", "args": {"amount": 3}}
    _drive_tools(
        reg,
        {
            "thread": "t",
            "kind": "resume",
            "expect": report["ckpt"],
            "resume": pack({"decisions": [response]}),
        },
    )
    assert executed == ([3] if decision == "edit" else [])
    messages = read_output(reg.graph, "t")["messages"]
    assert Counter(m.tool_call_id for m in messages if isinstance(m, ToolMessage)) == {
        "unsafe": 1,
        "safe": 1,
        "safe2": 1,
    }
