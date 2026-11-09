# Copyright (c) Microsoft. All rights reserved.

"""Real gRPC worker/client tests, including durable replay and external events."""

from __future__ import annotations

import operator
import socket
import threading
import uuid
from collections import Counter
from typing import Annotated, Any, TypedDict

import pytest
from durabletask.client import TaskHubGrpcClient
from durabletask.testing import create_test_backend
from durabletask.worker import TaskHubGrpcWorker
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command, Send, interrupt

from langchain_azure_compute.graph import Limits, attach_durability, durable_tool


@pytest.fixture
def real_runtime():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    backend = create_test_backend(port=port)
    client = TaskHubGrpcClient(host_address=f"127.0.0.1:{port}")
    worker = TaskHubGrpcWorker(host_address=f"127.0.0.1:{port}")
    try:
        yield backend, client, worker
    finally:
        worker.stop()
        close = getattr(client, "close", None)
        if close:
            close()
        backend.stop()
        backend.reset()


class FanoutState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    approved: bool


def test_parallel_tools_retry_interrupt_and_idempotent_replay(real_runtime):
    _, sdk, worker = real_runtime
    attempts = Counter()
    call_ids = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    @durable_tool(retries=2)
    @tool
    def booking(kind: str, runtime: ToolRuntime) -> str:
        """Perform a booking, recording the server-derived idempotency key."""
        with lock:
            attempts[kind] += 1
            attempt = attempts[kind]
            call_ids.append((kind, runtime.config["metadata"]["durable_call_id"]))
        if attempt == 1:
            barrier.wait(timeout=5)
        if kind == "car" and attempt == 1:
            raise RuntimeError("transient outage")
        return kind

    builder = StateGraph(FanoutState)
    builder.add_node(
        "plan",
        lambda s: {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "booking",
                            "args": {"kind": k},
                            "id": k,
                            "type": "tool_call",
                        }
                        for k in ("flight", "car")
                    ],
                )
            ]
        },
    )
    builder.add_node("tools", ToolNode([booking]))
    builder.add_node("review", lambda s: {"approved": bool(interrupt("Approve?"))})
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "tools")
    builder.add_edge("tools", "review")
    builder.add_edge("review", END)
    reg = attach_durability(builder.compile(), name="real_fanout")
    reg.register(worker)
    worker.start()
    client = reg.client(sdk)
    iid = "real-" + uuid.uuid4().hex
    inp = {"messages": [], "approved": False}
    run = client.start(inp, instance_id=iid, owner="alice")
    pending = run.wait_pending(timeout=20)
    assert pending is not None and pending.value == "Approve?"
    assert len(run.pending_interrupts()) == 1
    run.resume(pending, True)
    output = run.wait_result(timeout=20)
    assert output["approved"]
    assert run.pending_interrupts() == []
    assert attempts == {"flight": 1, "car": 2}
    assert len({key for kind, key in call_ids if kind == "car"}) == 1
    assert len({key for _, key in call_ids}) == 2
    replay = client.start(inp, instance_id=iid, owner="alice")
    history_before = sdk.get_orchestration_history(iid)
    assert history_before
    assert replay.wait_result(timeout=10) == output
    assert attempts == {"flight": 1, "car": 2}
    assert sdk.get_orchestration_history(iid) == history_before


class AgentModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


@pytest.mark.parametrize("decision", ["reject", "respond"])
def test_real_create_agent_send_hitl_never_executes_denied_tool(real_runtime, decision):
    _, sdk, worker = real_runtime
    executed = []

    @tool
    def dangerous() -> str:
        """Perform a dangerous mutation."""
        executed.append("dangerous")
        return "mutated"

    @tool
    def safe() -> str:
        """Perform a safe read."""
        executed.append("safe")
        return "read"

    model = AgentModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "dangerous",
                            "args": {},
                            "id": "d1",
                            "type": "tool_call",
                        },
                        {"name": "safe", "args": {}, "id": "s1", "type": "tool_call"},
                    ],
                ),
                AIMessage(content="Done"),
            ]
        )
    )
    agent = create_agent(
        model,
        [dangerous, safe],
        middleware=[
            HumanInTheLoopMiddleware(interrupt_on={"dangerous": True, "safe": False}),
        ],
    )
    reg = attach_durability(agent, name="real_hitl_" + decision)
    reg.register(worker)
    worker.start()
    run = reg.client(sdk).start({"messages": [HumanMessage(content="Do it")]})
    pending = run.wait_pending(timeout=20)
    assert pending is not None
    choice = {"type": decision}
    if decision == "respond":
        choice["message"] = "Do not execute"
    run.resume(pending, {"decisions": [choice]})
    output = run.wait_result(timeout=20)
    assert executed == ["safe"]
    assert (
        len(
            [
                m
                for m in output["messages"]
                if isinstance(m, ToolMessage) and m.tool_call_id == "d1"
            ]
        )
        == 1
    )
    assert run.pending_interrupts() == []


class RolloverState(TypedDict):
    n: int
    history: Annotated[list[int], operator.add]


def test_real_multi_generation_rollover_and_replay_determinism(real_runtime):
    backend, sdk, worker = real_runtime
    executed = []

    def advance(state):
        n = state["n"] + 1
        executed.append(n)
        return {"n": n, "history": [n]}

    builder = StateGraph(RolloverState)
    builder.add_node("advance", advance)
    builder.add_edge(START, "advance")
    builder.add_conditional_edges("advance", lambda s: "advance" if s["n"] < 6 else END)
    reg = attach_durability(
        builder.compile(),
        name="real_rollover",
        limits=Limits(steps_per_generation=2),
    )
    reg.register(worker)
    worker.start()
    client = reg.client(sdk)
    run = client.start({"n": 0, "history": []}, instance_id="rollover")
    output = run.wait_result(timeout=20)
    assert output == {"n": 6, "history": [1, 2, 3, 4, 5, 6]}
    assert executed == [1, 2, 3, 4, 5, 6]
    state = sdk.get_orchestration_state(run.instance_id, fetch_payloads=True)
    import json

    assert json.loads(state.serialized_output)["gen"] >= 2
    before = list(executed)
    history_before = sdk.get_orchestration_history(run.instance_id)
    assert history_before
    assert client.start({}, instance_id="rollover").wait_result(timeout=10) == output
    assert executed == before
    assert sdk.get_orchestration_history(run.instance_id) == history_before
    second = client.start({"n": 0, "history": []}, instance_id="rollover-second")
    assert second.wait_result(timeout=20) == output
    assert executed == before + before


class SendState(TypedDict):
    results: Annotated[list[int], operator.add]


def test_real_send_parallel_node_fanout(real_runtime):
    _, sdk, worker = real_runtime
    builder = StateGraph(SendState)
    builder.add_node("map", lambda s: {"results": [s["item"] * 2]})
    builder.add_conditional_edges(
        START,
        lambda s: [
            Send("map", {"item": 1}),
            Send("map", {"item": 2}),
        ],
    )
    builder.add_edge("map", END)
    reg = attach_durability(builder.compile(), name="real_send", durable_nodes=True)
    reg.register(worker)
    worker.start()
    output = reg.client(sdk).start({"results": []}).wait_result(timeout=20)
    assert sorted(output["results"]) == [2, 4]


@pytest.mark.parametrize("migration_kind", ["auto", "explicit"])
def test_real_version_upgrade_preserves_checkpoint_state(real_runtime, migration_kind):
    _, sdk, worker = real_runtime

    def build(version):
        builder = StateGraph(RolloverState)
        builder.add_node("prep", lambda s: {"n": s["n"] + 1, "history": [1]})
        approval = "approval_old" if version == "v1" else "approval_new"
        if migration_kind == "explicit":
            builder.add_node(
                approval,
                lambda s: {
                    "history": [2] if interrupt("Approve?") else [0],
                },
            )
        builder.add_node(
            "finish",
            lambda s: {
                "n": s["n"] * (2 if version == "v1" else 10),
                "history": [3],
            },
        )
        builder.add_edge(START, "prep")
        if migration_kind == "explicit":
            builder.add_edge("prep", approval)
            builder.add_edge(approval, "finish")
        else:
            builder.add_edge("prep", "finish")
        builder.add_edge("finish", END)
        return builder.compile()

    old, current = build("v1"), build("v2")
    migration = (
        (lambda state: {"n": state["n"] + 100})
        if migration_kind == "auto"
        else (
            lambda state, snap, resume: {
                "as_node": "approval_new",
                "values": {"history": [9]},
            }
        )
    )
    reg = attach_durability(
        current,
        name="real_upgrade_" + migration_kind,
        version="v2",
        previous_versions={"v1": old},
        migrations={("v1", "v2"): migration},
        upgrade_on_continue_as_new=migration_kind == "auto",
        limits=Limits(steps_per_generation=1 if migration_kind == "auto" else 250),
    )
    reg.register(worker)
    worker.start()
    run = reg.client(sdk).start({"n": 4, "history": []}, version="v1")
    if migration_kind == "explicit":
        pending = run.wait_pending(timeout=20)
        assert pending is not None
        run.upgrade("v2", receipt=pending, resume=True)
    output = run.wait_result(timeout=20)
    assert output["n"] == (1050 if migration_kind == "auto" else 50)
    assert run.version == "v2"
    assert run.pending_interrupts() == []


@pytest.mark.parametrize("kind", ["tools", "nodes", "nodes_both"])
def test_real_parallel_activity_interrupts_and_partial_completion(real_runtime, kind):
    _, sdk, worker = real_runtime
    trace = []
    if kind == "tools":

        @durable_tool(retries=2)
        @tool
        def first() -> str:
            """Ask for two sequential approvals."""
            a = interrupt({"activity": "first", "round": 1})
            b = interrupt({"activity": "first", "round": 2})
            trace.append("first")
            return f"{a}:{b}"

        @durable_tool(retries=2)
        @tool
        async def second() -> str:
            """Ask for a separate approval."""
            a = interrupt({"activity": "second", "round": 1})
            trace.append("second")
            return str(a)

        builder = StateGraph(FanoutState)
        builder.add_node(
            "plan",
            lambda s: {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": n, "args": {}, "id": n, "type": "tool_call"}
                            for n in ("first", "second")
                        ],
                    )
                ]
            },
        )
        builder.add_node("tools", ToolNode([first, second]))
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "tools")
        builder.add_edge("tools", END)
        reg = attach_durability(builder.compile(), name="parallel_tool_approvals")
        initial = {"messages": [], "approved": False}
    else:
        builder = StateGraph(SendState)

        def first(state):
            a = interrupt({"activity": "first", "round": 1})
            b = interrupt({"activity": "first", "round": 2})
            trace.append("first")
            return {"results": [int(a) + int(b)]}

        async def second(state):
            if kind == "nodes_both":
                interrupt({"activity": "second", "round": 1})
            trace.append("second")
            return {"results": [10]}

        builder.add_node("first", first)
        builder.add_node("second", second)
        builder.add_edge(START, "first")
        builder.add_edge(START, "second")
        builder.add_edge("first", END)
        builder.add_edge("second", END)
        reg = attach_durability(
            builder.compile(), name="parallel_node_approvals", durable_nodes=True
        )
        initial = {"results": []}
    reg.register(worker)
    worker.start()
    run = reg.client(sdk).start(initial)
    previous = None
    for _ in range(4):
        receipt = run.wait_pending(previous, timeout=20)
        if receipt is None:
            break
        assert receipt.value["round"] in (1, 2)
        run.resume(receipt, 1)
        previous = receipt
    else:
        pytest.fail("Approvals did not converge")
    output = run.wait_result(timeout=20)
    assert run.pending_interrupts() == []
    assert trace.count("first") == 1 and trace.count("second") == 1
    if kind.startswith("nodes"):
        assert sorted(output["results"]) == [2, 10]
    else:
        assert len([m for m in output["messages"] if isinstance(m, ToolMessage)]) == 2


def test_real_empty_tool_batch_and_local_only_tool_specs(real_runtime):
    _, sdk, worker = real_runtime

    @tool
    def local() -> str:
        """Read a local value."""
        return "local"

    builder = StateGraph(FanoutState)
    builder.add_node("tools", ToolNode([local]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    reg = attach_durability(builder.compile(), name="empty_tools")
    reg.register(worker)
    worker.start()
    empty = (
        reg.client(sdk)
        .start({"messages": [AIMessage(content="no calls")], "approved": False})
        .wait_result(timeout=20)
    )
    assert len(empty["messages"]) == 1


def test_tool_runtime_reader_writer_and_command_composition():
    from types import SimpleNamespace

    from langchain_azure_compute.graph import _checkpoints as cp

    @tool
    def mutate(runtime: ToolRuntime) -> Command:
        """Exercise legacy channel callbacks with state merging."""
        child_config = var_child_runnable_config.get()
        assert child_config is not None
        conf = child_config["configurable"]
        read, send = conf["__pregel_read"], conf["__pregel_send"]
        assert read("n") == 1
        assert read("__root__")["n"] == 1
        send([("__resume__", None), ("data", {"a": None, "b": 2})])
        send([("data", {"c": 3}), ("n", 2)])
        assert read("data") == {"b": 2, "c": 3}
        assert read("n") == 2
        return Command(
            update={"messages": [ToolMessage(content="ok", tool_call_id="call")]}
        )

    node = ToolNode([mutate])
    graph = SimpleNamespace(builder=SimpleNamespace(state_schema=None), store=None)
    result, command = cp._exec_single_tool(
        graph,
        node,
        {"n": 1, "data": {"a": 1}, "messages": []},
        {"name": "mutate", "args": {}, "id": "call", "type": "tool_call"},
        "thread",
        "",
        [],
        task_input_type="dict",
        run_config={"tags": ["tag"], "recursion_limit": 3},
    )
    assert result == "ok" and command.update["n"] == 2
    assert command.update["data"] == {"a": None, "b": 2, "c": 3}


@pytest.mark.parametrize("local_count", [1, 2])
@pytest.mark.parametrize("local_interrupt", [False, True])
def test_real_mixed_local_tools_commit_or_reject_unsupported_interrupts(
    real_runtime,
    local_count,
    local_interrupt,
):
    _, sdk, worker = real_runtime

    @durable_tool
    @tool
    def remote() -> str:
        """Read a durable value."""
        return "remote"

    @tool
    def local() -> str:
        """Read a local value."""
        if local_interrupt:
            interrupt("Local approval")
        return "local"

    builder = StateGraph(FanoutState)
    builder.add_node("tools", ToolNode([remote, local]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    reg = attach_durability(
        builder.compile(),
        name="mixed_tools",
        durable_tools={"remote": True, "local": False},
        limits=Limits(retries=1),
    )
    reg.register(worker)
    worker.start()
    calls = [{"name": "remote", "args": {}, "id": "remote", "type": "tool_call"}]
    calls += [
        {"name": "local", "args": {}, "id": f"local-{i}", "type": "tool_call"}
        for i in range(local_count)
    ]
    run = reg.client(sdk).start(
        {"messages": [AIMessage(content="", tool_calls=calls)], "approved": False}
    )
    if local_interrupt:
        from langchain_azure_compute.graph import RunError

        with pytest.raises(RunError, match="mark it durable"):
            run.wait_result(timeout=20)
    else:
        result = run.wait_result(timeout=20)
        answers = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(answers) == local_count + 1


def test_real_large_interrupt_values_are_hydrated_from_checkpoint(real_runtime):
    _, sdk, worker = real_runtime
    builder = StateGraph(FanoutState)
    builder.add_node("review", lambda s: {"approved": interrupt("x" * 20_000)})
    builder.add_edge(START, "review")
    builder.add_edge("review", END)
    reg = attach_durability(
        builder.compile(),
        name="large_approval",
        limits=Limits(custom_status_limit=1000),
    )
    reg.register(worker)
    worker.start()
    run = reg.client(sdk).start({"messages": [], "approved": False})
    pending = run.wait_pending(timeout=20)
    assert pending.value == "x" * 20_000
    run.resume(pending, True)
    assert run.wait_result(timeout=20)["approved"] is True


def test_real_nested_local_tool_does_not_get_scheduled_as_durable(real_runtime):
    _, sdk, worker = real_runtime

    @tool
    def remote() -> str:
        """Should never run."""
        raise AssertionError("Unexpected remote tool")

    @tool
    def local() -> str:
        """Read locally."""
        return "local"

    child = StateGraph(FanoutState)
    child.add_node("tools", ToolNode([remote, local]))
    child.add_edge(START, "tools")
    child.add_edge("tools", END)
    root = StateGraph(FanoutState)
    root.add_node("child", child.compile())
    root.add_edge(START, "child")
    root.add_edge("child", END)
    reg = attach_durability(
        root.compile(),
        name="nested_local",
        durable_tools={"remote": True, "local": False},
    )
    reg.register(worker)
    worker.start()
    result = (
        reg.client(sdk)
        .start(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "local",
                                "args": {},
                                "id": "local",
                                "type": "tool_call",
                            },
                        ],
                    )
                ],
                "approved": False,
            }
        )
        .wait_result(timeout=20)
    )
    assert result["messages"][-1].content == "local"


def test_real_parallel_tool_nodes_roll_over_between_partial_commits(real_runtime):
    class State(TypedDict):
        a: Annotated[list[BaseMessage], add_messages]
        b: Annotated[list[BaseMessage], add_messages]

    _, sdk, worker = real_runtime

    @durable_tool
    @tool
    def echo(value: str) -> str:
        """Echo a value."""
        return value

    builder = StateGraph(State)
    for name in ("a", "b"):
        builder.add_node(name, ToolNode([echo], messages_key=name))
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    reg = attach_durability(
        builder.compile(),
        name="parallel_tool_nodes",
        limits=Limits(steps_per_generation=1),
    )
    reg.register(worker)
    worker.start()
    data = {
        name: [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "echo",
                        "args": {"value": name},
                        "id": name,
                        "type": "tool_call",
                    },
                ],
            )
        ]
        for name in ("a", "b")
    }
    run = reg.client(sdk).start(data)
    output = run.wait_result(timeout=20)
    assert output["a"][-1].content == "a" and output["b"][-1].content == "b"
    assert run.generations > 1
