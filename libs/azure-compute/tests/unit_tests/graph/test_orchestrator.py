# Copyright (c) Microsoft. All rights reserved.

"""The orchestrator generator driven by a fake context (deterministic, no DTS)."""

import pytest
from durabletask import task

from langchain_azure_compute.graph.runner import (
    Limits,
    RunError,
    make_orchestrator,
    pack,
    unpack,
)


@pytest.mark.parametrize("source", ["endpoint", "environment", "connection_string"])
@pytest.mark.parametrize(
    "endpoint",
    [
        "https://my-scheduler.westus2.durabletask.io",
        "grpcs://my-scheduler.westus2.durabletask.io/",
        "my-scheduler.westus2.durabletask.io",
        "https://MY-SCHEDULER.WESTUS2.DURABLETASK.IO:443/",
    ],
)
def test_sdk_options_defaults_hosted_dts_to_default_azure_credential(
    monkeypatch, source, endpoint
):
    from unittest.mock import Mock

    from langchain_azure_compute.graph import sdk_options

    credential = object()
    factory = Mock(return_value=credential)
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", factory)
    for name in (
        "DURABLE_TASK_SCHEDULER_CONNECTION_STRING",
        "DTS_CONNECTION_STRING",
        "DTS_ENDPOINT",
        "DTS_EMULATOR",
        "DTS_TASKHUB",
    ):
        monkeypatch.delenv(name, raising=False)

    if source == "endpoint":
        opts = sdk_options(endpoint, taskhub="prod-hub")
    elif source == "environment":
        monkeypatch.setenv("DTS_ENDPOINT", endpoint)
        monkeypatch.setenv("DTS_TASKHUB", "prod-hub")
        opts = sdk_options()
    else:
        opts = sdk_options(connection_string=f"Endpoint={endpoint};TaskHub=prod-hub")

    factory.assert_called_once_with()
    assert opts["token_credential"] is credential
    assert opts["secure_channel"] is True
    assert opts["taskhub"] == "prod-hub"
    assert opts["host_address"] == endpoint.removeprefix("https://").removeprefix(
        "grpcs://"
    ).rstrip("/")


@pytest.mark.parametrize("authentication", ["None", "Anonymous", "Emulator", ""])
def test_sdk_options_preserves_explicit_anonymous_authentication(
    monkeypatch, authentication
):
    from unittest.mock import Mock

    from langchain_azure_compute.graph import sdk_options

    factory = Mock()
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", factory)
    opts = sdk_options(
        connection_string=(
            "Endpoint=my-scheduler.westus2.durabletask.io;TaskHub=prod-hub;"
            f"Authentication={authentication}"
        )
    )

    factory.assert_not_called()
    assert opts["token_credential"] is None
    assert opts["secure_channel"] is True


def test_sdk_options_preserves_explicit_token_credential(monkeypatch):
    from unittest.mock import Mock

    from langchain_azure_compute.graph import sdk_options

    factory = Mock()
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", factory)
    credential = object()
    opts = sdk_options(
        "https://my-scheduler.westus2.durabletask.io",
        token_credential=credential,
    )

    factory.assert_not_called()
    assert opts["token_credential"] is credential
    assert opts["secure_channel"] is True


@pytest.mark.parametrize(
    ("endpoint", "secure"),
    [
        ("127.0.0.1:8080", False),
        ("http://localhost:8080/", False),
        ("dts-emulator:8080", False),
        ("https://scheduler.example.com", True),
        ("https://durabletask.io", True),
        ("https://scheduler.durabletask.io.example.com", True),
    ],
)
def test_sdk_options_does_not_default_credentials_for_other_hosts(
    monkeypatch, endpoint, secure
):
    from unittest.mock import Mock

    from langchain_azure_compute.graph import sdk_options

    factory = Mock()
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", factory)
    opts = sdk_options(endpoint)

    factory.assert_not_called()
    assert opts["token_credential"] is None
    assert opts["secure_channel"] is secure


class ActTask(tuple):
    is_complete = False
    is_failed = False
    _parent = None

    def __new__(cls, val):
        inst = super().__new__(cls, val)
        inst._parents = {}
        inst._parent = None
        inst._result = None
        inst._exception = None
        return inst

    def get_result(self):
        return self._result

    def get_exception(self):
        return self._exception


class Ctx:
    def __init__(self):
        (
            self.calls,
            self.call_names,
            self.policies,
            self.events,
            self.status,
            self.can,
        ) = [], [], [], [], [], None
        self.yields = []
        self.scheduled_at_yield = []

    def call_activity(self, name, input=None, retry_policy=None):
        self.calls.append(input)
        self.call_names.append(name)
        self.policies.append(retry_policy)
        return ActTask(("act", name, input))

    def wait_for_external_event(self, name):
        self.events.append(name)
        return ("evt", name)

    def set_custom_status(self, value):
        self.status.append(value)

    def continue_as_new(self, new_input, save_events=False):
        self.can = (new_input, save_events)


def start_input(max_concurrency=None):
    inp = {
        "thread": "t",
        "gen": 0,
        "steps": 0,
        "step_name": "langgraph.x.step",
        "tool_name": "langgraph.x.tool",
        "req": {"kind": "start", "expect": None},
    }
    if max_concurrency is not None:
        inp["max_concurrency"] = max_concurrency
    return inp


def run(limits, reports, events=(), *, max_concurrency=None):
    ctx = Ctx()
    gen = make_orchestrator(limits)(ctx, start_input(max_concurrency=max_concurrency))
    reports, events = list(reports), list(events)
    try:
        yielded = next(gen)
        while True:
            ctx.yields.append(yielded)
            ctx.scheduled_at_yield.append(len(ctx.calls))
            if isinstance(yielded, task.WhenAllTask):
                val = [reports.pop(0) for _ in yielded.get_tasks()]
            elif yielded[0] == "act":
                val = reports.pop(0)
            else:
                val = events.pop(0)
            yielded = gen.send(val)
    except StopIteration as done:
        return ctx, done.value


def R(ckpt: str) -> dict[str, str]:
    return {"status": "running", "ckpt": ckpt}


DONE = {"status": "done", "ckpt": "z"}


def test_done_output_has_steps_generation_and_checkpoint():
    ctx, out = run(Limits(), [R("a"), R("b"), DONE])
    assert (
        out["status"] == "done"
        and out["steps"] == 3
        and out["gen"] == 0
        and out["ckpt"] == "z"
    )
    assert "result" not in out
    assert [c["expect"] for c in ctx.calls] == [None, "a", "b"]


def test_continue_as_new_after_step_limit_carries_checkpoint_and_counters():
    ctx, out = run(Limits(steps_per_generation=2), [R("a"), R("b")])
    new_input, save_events = ctx.can
    assert out is None and save_events is False
    assert new_input["proto"] == 1
    assert new_input["gen"] == 1 and new_input["steps"] == 2
    assert new_input["req"] == {"thread": "t", "kind": "step", "expect": "b"}
    assert new_input["tool_name"] == "langgraph.x.tool"


def test_orchestrator_rejects_unsupported_protocol_version():
    ctx = Ctx()
    gen = make_orchestrator(Limits())(ctx, {**start_input(), "proto": 99})
    with pytest.raises(RunError, match="unsupported orchestrator protocol version"):
        next(gen)


def test_continue_as_new_on_byte_budget_even_with_few_steps():
    big = {"status": "running", "ckpt": "a", "pad": "x" * 5000}
    ctx, out = run(Limits(byte_budget=4000, steps_per_generation=10_000), [big])
    assert ctx.can is not None and ctx.can[0]["gen"] == 1


def test_interrupt_waits_for_resume_event_and_carries_value():
    interrupted = {
        "status": "interrupted",
        "ckpt": "a",
        "interrupt": {"id": "iid", "value": {}},
    }
    ctx, out = run(Limits(), [interrupted, DONE], events=[{"value": pack(True)}])
    assert ctx.events == ["resume:iid"]
    assert ctx.status[0]["pending"]["id"] == "iid" and ctx.status[-1] is None
    assert ctx.calls[1]["kind"] == "resume" and ctx.calls[1]["expect"] == "a"


def test_parallel_interrupts_wait_for_each_branch_and_pack_resume_map():
    i1, i2 = (
        {"id": "int-left", "value": pack("L")},
        {"id": "int-right", "value": pack("R")},
    )
    interrupted = {
        "status": "interrupted",
        "ckpt": "a",
        "interrupt": i1,
        "interrupts": [i1, i2],
    }
    ctx, out = run(
        Limits(),
        [interrupted, DONE],
        events=[{"value": pack("ok-L")}, {"value": pack("ok-R")}],
    )
    assert ctx.events == ["resume:int-left", "resume:int-right"]
    assert ctx.calls[1]["kind"] == "resume" and ctx.calls[1]["expect"] == "a"
    assert unpack(ctx.calls[1]["resume"]) == {"int-left": "ok-L", "int-right": "ok-R"}


def test_durable_tool_calls_scheduled_individually_with_per_tool_retry_policy():
    tools_report = {
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
                "retries": 4,
            },
            {
                "call_id": "c1:tools:1:tc2",
                "index": 1,
                "id": "tc2",
                "name": "t2",
                "call": pack({"name": "t2"}),
                "durable": True,
                "retries": 2,
            },
        ],
    }
    t_out1 = {"call_id": "c1:tools:0:tc1", "output": pack("out1")}
    t_out2 = {"call_id": "c1:tools:1:tc2", "output": pack("out2")}
    ctx, out = run(Limits(), [tools_report, t_out1, t_out2, DONE])
    assert out["status"] == "done"
    assert ctx.call_names == [
        "langgraph.x.step",
        "langgraph.x.tool",
        "langgraph.x.tool",
        "langgraph.x.step",
    ]
    assert ctx.policies[1].max_number_of_attempts == 4
    assert ctx.policies[2].max_number_of_attempts == 2
    assert ctx.calls[3]["kind"] == "tools" and ctx.calls[3]["expect"] == "c1"
    assert [unpack(o) for o in ctx.calls[3]["outputs"]] == ["out1", "out2"]


def test_step_limit_fails_the_run():
    with pytest.raises(RunError, match="step limit"):
        run(Limits(max_steps=2, steps_per_generation=10_000), [R("a"), R("b"), R("c")])


def test_orchestrator_handles_tool_interrupt_and_skips_local_tools_in_mixed_turn():
    tools_report = {
        "status": "tools",
        "ckpt": "c1",
        "node": "tools",
        "tool_calls": [
            {
                "call_id": "c1:tools:0:tc_loc",
                "index": 0,
                "id": "tc_loc",
                "name": "loc_t",
                "call": pack({"name": "loc_t"}),
                "durable": False,
                "retries": 1,
            },
            {
                "call_id": "c1:tools:1:tc_dur",
                "index": 1,
                "id": "tc_dur",
                "name": "dur_t",
                "call": pack({"name": "dur_t"}),
                "durable": True,
                "retries": 3,
            },
        ],
    }
    t_int = {
        "call_id": "c1:tools:1:tc_dur",
        "status": "interrupted",
        "interrupt": {"id": "tint-1", "value": pack("approve?")},
    }
    t_ok = {"call_id": "c1:tools:1:tc_dur", "status": "ok", "output": pack("dur_out")}
    ctx, out = run(
        Limits(),
        [tools_report, t_int, t_ok, DONE],
        events=[{"value": pack("approved")}],
    )
    assert out["status"] == "done"
    # Only the durable tool was scheduled as langgraph.x.tool (twice: once interrupted,
    # once resumed)
    assert ctx.call_names == [
        "langgraph.x.step",
        "langgraph.x.tool",
        "langgraph.x.tool",
        "langgraph.x.step",
    ]
    assert ctx.events == ["resume:tint-1"]
    assert [unpack(r) for r in ctx.calls[2]["resumes"]] == ["approved"]
    assert ctx.calls[3]["kind"] == "tools"
    assert (
        unpack(ctx.calls[3]["outputs"]["1"]) == "dur_out"
        and "0" not in ctx.calls[3]["outputs"]
    )


def test_orchestrator_forwards_bundle_across_steps_tools_and_can():
    b1 = {"v": 1, "storage": {"": {"c1": []}}, "writes": [], "blobs": []}
    b2 = {"v": 1, "storage": {"": {"c2": []}}, "writes": [], "blobs": []}
    r1 = {
        "status": "tools",
        "ckpt": "c1",
        "node": "tools",
        "checkpoint_bundle": b1,
        "tool_calls": [
            {
                "call_id": "c1:tools:0:tc1",
                "index": 0,
                "id": "tc1",
                "name": "t1",
                "call": pack({"name": "t1"}),
                "durable": True,
                "retries": 2,
            },
        ],
    }
    t_out = {"call_id": "c1:tools:0:tc1", "status": "ok", "output": pack("ok")}
    r2 = {"status": "running", "ckpt": "c2", "checkpoint_bundle": b2}
    ctx, out = run(Limits(steps_per_generation=2), [r1, t_out, r2])
    assert out is None and ctx.can is not None
    # Tool activity and tool-commit step both received b1
    assert ctx.calls[1]["checkpoint_bundle"] == b1
    assert ctx.calls[2]["checkpoint_bundle"] == b1
    # continue_as_new carries b2 in the next generation's request
    next_inp = ctx.can[0]
    assert next_inp["req"]["checkpoint_bundle"] == b2

    # Next generation completes and surfaces output + checkpoint_bundle
    ctx2 = Ctx()
    gen2 = make_orchestrator(Limits())(ctx2, next_inp)
    yielded = next(gen2)
    assert yielded[2]["checkpoint_bundle"] == b2
    with pytest.raises(StopIteration) as done_exc:
        gen2.send(
            {
                "status": "done",
                "ckpt": "c3",
                "checkpoint_bundle": b2,
                "output": pack({"answer": 42}),
            }
        )
    assert unpack(done_exc.value.value["output"]) == {"answer": 42}
    assert done_exc.value.value["checkpoint_bundle"] == b2


def test_vacation_booking_mid_toolnode_failure_and_idempotent_client_retry(tmp_path):
    """Vacation booking graph (plan_trip -> ToolNode([book_flight, book_hotel.

    book_car]) -> review -> confirm)
    proving:
    1. Plain LangGraph checkpointer alone double-books FLIGHT and HOTEL when book_car
       fails mid-ToolNode
       (because ToolNode is a single superstep).
    2. DTS per-tool activities (langgraph.<name>.tool) record book_flight and book_hotel
       in DTS history and retry
       ONLY book_car -> bookings.log has 3 lines (1 FLIGHT, 1 HOTEL, 1 CAR).
    3. RunClient.start(..., instance_id=..., reuse_existing=True) deduplicates client
       retries both while RUNNING and after COMPLETED.
    """
    import json
    import sqlite3
    from types import SimpleNamespace
    from typing import Annotated, NotRequired, TypedDict

    from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    from langchain_azure_compute.graph import attach_durability, durable_tool
    from langchain_azure_compute.graph.runner import run_step, run_tool

    class VacationState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        destination: str
        confirmations: NotRequired[list[str]]
        confirmed: NotRequired[bool]

    def build_vacation_graph(log_path, saver, *, fail_car_once: bool = True):
        import threading

        seq = {"n": 0, "car_attempts": 0}
        log_lock = threading.Lock()
        flight_done = threading.Event()
        hotel_done = threading.Event()

        def external_booking(kind: str) -> str:
            with log_lock:
                seq["n"] += 1
                confirmation = f"{kind}-{1000 + seq['n']}"
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{kind} BOOKED: {confirmation}\n")
                return confirmation

        @durable_tool(retries=3)
        @tool
        def book_flight(destination: str) -> str:
            """Book a flight."""
            flight_done.clear()
            hotel_done.clear()
            out = external_booking("FLIGHT")
            flight_done.set()
            return out

        @durable_tool(retries=3)
        @tool
        def book_hotel(destination: str) -> str:
            """Book a hotel."""
            flight_done.wait(timeout=5)
            out = external_booking("HOTEL")
            hotel_done.set()
            return out

        @durable_tool(retries=3)
        @tool
        def book_car(destination: str) -> str:
            """Book a rental car."""
            hotel_done.wait(timeout=5)
            seq["car_attempts"] += 1
            if fail_car_once and seq["car_attempts"] == 1:
                raise RuntimeError("worker crashed while booking car")
            return external_booking("CAR")

        def plan_trip(state: VacationState):
            dest = state["destination"]
            return {
                "messages": [
                    AIMessage(
                        content=f"Planning vacation to {dest}",
                        tool_calls=[
                            {
                                "name": "book_flight",
                                "args": {"destination": dest},
                                "id": "flight-1",
                                "type": "tool_call",
                            },
                            {
                                "name": "book_hotel",
                                "args": {"destination": dest},
                                "id": "hotel-1",
                                "type": "tool_call",
                            },
                            {
                                "name": "book_car",
                                "args": {"destination": dest},
                                "id": "car-1",
                                "type": "tool_call",
                            },
                        ],
                    )
                ]
            }

        def review(state: VacationState):
            confs = [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
            return {
                "confirmations": confs,
                "messages": [AIMessage(content="Reviewed: " + ", ".join(confs))],  # type: ignore[arg-type]
            }

        def confirm(state: VacationState):
            return {"confirmed": True, "messages": [AIMessage(content="Confirmed")]}

        b = StateGraph(VacationState)
        b.add_node("plan", plan_trip)
        b.add_node("tools", ToolNode([book_flight, book_hotel, book_car]))
        b.add_node("review", review)
        b.add_node("confirm", confirm)
        b.add_edge(START, "plan")
        b.add_edge("plan", "tools")
        b.add_edge("tools", "review")
        b.add_edge("review", "confirm")
        b.add_edge("confirm", END)
        return b.compile(checkpointer=saver)

    # 1. Contrast: Plain LangGraph + SqliteSaver alone double-books FLIGHT and HOTEL on
    # mid-ToolNode crash!
    plain_log = tmp_path / "plain_bookings.log"
    plain_conn = sqlite3.connect(tmp_path / "plain.db", check_same_thread=False)
    plain_graph = build_vacation_graph(
        plain_log, SqliteSaver(plain_conn), fail_car_once=True
    )
    plain_cfg = {"configurable": {"thread_id": "user42"}}
    with pytest.raises(RuntimeError, match="worker crashed while booking car"):
        plain_graph.invoke(
            {"messages": [HumanMessage(content="Book Paris")], "destination": "Paris"},
            plain_cfg,
        )
    # Resuming from SqliteSaver re-runs the entire ToolNode from the 'plan' checkpoint
    # -> 5 bookings (2 FLIGHT, 2 HOTEL, 1 CAR)!
    plain_graph.invoke(None, plain_cfg)
    plain_conn.close()
    plain_lines = plain_log.read_text(encoding="utf-8").splitlines()
    assert len(plain_lines) == 5
    assert [line.split()[0] for line in plain_lines] == [
        "FLIGHT",
        "HOTEL",
        "FLIGHT",
        "HOTEL",
        "CAR",
    ]

    # 2. Scenario A + Scenario B with DTS (langgraph-durabletask):
    dts_log = tmp_path / "bookings.log"
    dts_conn = sqlite3.connect(tmp_path / "dts.db", check_same_thread=False)
    reg = attach_durability(
        build_vacation_graph(dts_log, SqliteSaver(dts_conn), fail_car_once=True),
        name="vacation",
    )

    class FakeSdk:
        def __init__(self):
            self.scheduled = []
            self.states = {}

        def get_orchestration_state(self, iid, fetch_payloads=False):
            return self.states.get(iid)

        def schedule_new_orchestration(self, name, input=None, instance_id=None):
            self.scheduled.append((name, input, instance_id))
            self.states[instance_id] = SimpleNamespace(
                runtime_status=SimpleNamespace(name="RUNNING"),
                serialized_input=json.dumps(input),
                serialized_output=None,
                failure_details=None,
            )

        def wait_for_orchestration_completion(
            self, iid, timeout=60, fetch_payloads=True
        ):
            return self.states[iid]

    sdk = FakeSdk()
    client = reg.client(sdk)
    init_inp = {
        "messages": [HumanMessage(content="Book Paris")],
        "destination": "Paris",
    }

    # Client 1 starts run with explicit instance_id + thread_id
    run1 = client.start(init_inp, instance_id="user42-vacation-1", thread_id="user42")
    assert len(sdk.scheduled) == 1
    assert run1.instance_id == "user42-vacation-1" and run1.thread_id == "user42"

    # Scenario B (in-flight retry): Client 1 crashes while RUNNING; Client 2 retries
    # with same instance_id
    run2_inflight = client.start(
        init_inp, instance_id="user42-vacation-1", thread_id="user42"
    )
    assert len(sdk.scheduled) == 1
    assert (
        run2_inflight.instance_id == "user42-vacation-1"
        and run2_inflight.thread_id == "user42"
    )

    # Drive the orchestrator against real run_step / run_tool (with book_car failing on
    # attempt 1 and retrying)
    ctx = Ctx()
    gen = make_orchestrator(reg.limits)(ctx, sdk.scheduled[0][1])

    def _exec_act(act_name, act_inp):
        if act_name == reg.step_name:
            return run_step(reg.graph, reg.limits, act_inp, tool_config=reg.tool_config)
        try:
            return run_tool(reg.graph, act_inp)
        except RuntimeError:
            # DTS retries only the failed tool activity (book_car), NOT book_flight or
            # book_hotel!
            return run_tool(reg.graph, act_inp)

    try:
        yielded = next(gen)
        while True:
            if isinstance(yielded, task.WhenAllTask):
                res = [_exec_act(t[1], t[2]) for t in yielded.get_tasks()]
            else:
                kind, act_name, act_inp = yielded
                assert kind == "act"
                res = _exec_act(act_name, act_inp)
            yielded = gen.send(res)
    except StopIteration as done:
        orch_out = done.value

    sdk.states["user42-vacation-1"] = SimpleNamespace(
        runtime_status=SimpleNamespace(name="COMPLETED"),
        serialized_input=json.dumps(sdk.scheduled[0][1]),
        serialized_output=json.dumps(orch_out),
        failure_details=None,
    )

    # Scenario B (post-completion retry): Client retries after COMPLETED (both with and
    # without explicit thread_id)
    run3_done = client.start(
        init_inp, instance_id="user42-vacation-1", thread_id="user42"
    )
    run4_idem = client.start(init_inp, instance_id="user42-vacation-1")
    assert len(sdk.scheduled) == 1
    assert (
        run4_idem.instance_id == "user42-vacation-1" and run4_idem.thread_id == "user42"
    )

    out1 = run1.wait_result()
    out2 = run2_inflight.wait_result()
    out3 = run3_done.wait_result()
    out4 = run4_idem.wait_result()
    dts_conn.close()

    assert out1["confirmed"] is True
    assert (
        out1["confirmations"]
        == out2["confirmations"]
        == out3["confirmations"]
        == out4["confirmations"]
    )
    assert out1["confirmations"] == ["FLIGHT-1001", "HOTEL-1002", "CAR-1003"]

    dts_lines = dts_log.read_text(encoding="utf-8").splitlines()
    assert dts_lines == [
        "FLIGHT BOOKED: FLIGHT-1001",
        "HOTEL BOOKED: HOTEL-1002",
        "CAR BOOKED: CAR-1003",
    ]


def test_orchestrator_schedules_durable_tools_and_isolates_interrupt():
    """Verify that make_orchestrator schedules all 3 tool activities BEFORE yielding.

    the first task,
    yields each task in index order without WhenAllTask, and if one tool interrupts,
    completed sibling
    tools are NOT re-scheduled on resume."""
    tools_report = {
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
                "retries": 2,
            },
            {
                "call_id": "c1:tools:1:tc2",
                "index": 1,
                "id": "tc2",
                "name": "t2",
                "call": pack({"name": "t2"}),
                "durable": True,
                "retries": 3,
            },
            {
                "call_id": "c1:tools:2:tc3",
                "index": 2,
                "id": "tc3",
                "name": "t3",
                "call": pack({"name": "t3"}),
                "durable": True,
                "retries": 4,
            },
        ],
    }
    ctx = Ctx()
    gen = make_orchestrator(Limits())(ctx, start_input())
    first_yield = next(gen)
    assert first_yield[0] == "act" and first_yield[1] == "langgraph.x.step"

    # Send the tools_report: orchestrator must schedule all 3 tool activities BEFORE
    # yielding the first tool task
    t0_yield = gen.send(tools_report)
    assert isinstance(t0_yield, ActTask) and t0_yield[2]["index"] == 0
    assert ctx.call_names == [
        "langgraph.x.step",
        "langgraph.x.tool",
        "langgraph.x.tool",
        "langgraph.x.tool",
    ]
    assert [c["call_id"] for c in ctx.calls[1:4]] == [
        "c1:tools:0:tc1",
        "c1:tools:1:tc2",
        "c1:tools:2:tc3",
    ]

    # Tool 0 and Tool 2 succeed; Tool 1 interrupts
    t1_yield = gen.send(
        {"call_id": "c1:tools:0:tc1", "status": "ok", "output": pack("out0")}
    )
    assert isinstance(t1_yield, ActTask) and t1_yield[2]["index"] == 1
    t2_yield = gen.send(
        {
            "call_id": "c1:tools:1:tc2",
            "status": "interrupted",
            "interrupt": {"id": "int-t2", "value": pack("approve-t2?")},
        }
    )
    assert isinstance(t2_yield, ActTask) and t2_yield[2]["index"] == 2
    evt_yield = gen.send(
        {"call_id": "c1:tools:2:tc3", "status": "ok", "output": pack("out2")}
    )
    assert evt_yield == ("evt", "resume:int-t2")

    # After resume event, ONLY Tool 1 (index 1) is re-scheduled
    resumed_tool_yield = gen.send({"value": pack("yes-t2")})
    assert (
        isinstance(resumed_tool_yield, ActTask)
        and resumed_tool_yield[1] == "langgraph.x.tool"
    )
    assert resumed_tool_yield[2]["index"] == 1
    assert [unpack(r) for r in resumed_tool_yield[2]["resumes"]] == ["yes-t2"]
    assert (
        len(ctx.calls) == 5
    )  # 1 step + 3 initial tools + 1 resumed tool (siblings not re-run!)

    commit_yield = gen.send(
        {"call_id": "c1:tools:1:tc2", "status": "ok", "output": pack("out1")}
    )
    assert commit_yield[1] == "langgraph.x.step" and commit_yield[2]["kind"] == "tools"
    assert [unpack(o) for o in commit_yield[2]["outputs"]] == ["out0", "out1", "out2"]


def test_orchestrator_tool_max_concurrency_sequential_and_bounded_batches():
    """Verify max_concurrency=1 schedules 1 tool at a time sequentially, and.

    max_concurrency=2
    schedules 4 tools in two batches of 2 before yielding each task in index order."""
    four_calls = [
        {
            "call_id": f"c1:tools:{i}:tc{i}",
            "index": i,
            "id": f"tc{i}",
            "name": f"t{i}",
            "call": pack({"name": f"t{i}"}),
            "durable": True,
            "retries": 2,
        }
        for i in range(4)
    ]
    t_outs = [
        {"call_id": f"c1:tools:{i}:tc{i}", "status": "ok", "output": pack(f"res{i}")}
        for i in range(4)
    ]

    # 1. max_concurrency=1 -> 4 sequential single-activity yields (scheduled_at_yield
    # increments 1 by 1)
    rep_seq = {
        "status": "tools",
        "ckpt": "c1",
        "node": "tools",
        "max_concurrency": 1,
        "tool_calls": four_calls,
    }
    ctx1, out1 = run(Limits(), [rep_seq, *t_outs, DONE])
    assert out1["status"] == "done"
    assert ctx1.scheduled_at_yield == [1, 2, 3, 4, 5, 6]
    assert [unpack(o) for o in ctx1.calls[-1]["outputs"]] == [
        "res0",
        "res1",
        "res2",
        "res3",
    ]

    # 2. max_concurrency=2 -> two batches of size 2 (batch 1 schedules 2 tasks -> 3
    # total; batch 2 schedules 2 tasks -> 5 total)
    rep_b2 = {
        "status": "tools",
        "ckpt": "c1",
        "node": "tools",
        "tool_calls": four_calls,
    }
    ctx2, out2 = run(Limits(), [rep_b2, *t_outs, DONE], max_concurrency=2)
    assert out2["status"] == "done"
    assert ctx2.scheduled_at_yield == [1, 3, 3, 5, 5, 6]
    assert ctx2.calls[-1]["max_concurrency"] == 2
    assert [unpack(o) for o in ctx2.calls[-1]["outputs"]] == [
        "res0",
        "res1",
        "res2",
        "res3",
    ]


def test_orchestrator_catches_exhausted_tool_failure_on_handle_errors():
    """Verify make_orchestrator catches exhausted tool exceptions when.

    handle_errors=True (converting
    them to __durable_tool_error__ payloads for run_step) and re-raises when
    handle_errors=False.
    """
    rep_handled = {
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
                "retries": 2,
                "handle_errors": True,
            },
        ],
    }
    ctx = Ctx()
    gen = make_orchestrator(Limits())(ctx, start_input())
    next(gen)
    gen.send(rep_handled)
    commit_yield = gen.throw(RuntimeError("exhausted tool error"))
    assert commit_yield[1] == "langgraph.x.step" and commit_yield[2]["kind"] == "tools"
    err_payload = unpack(commit_yield[2]["outputs"][0])
    assert err_payload == {
        "__durable_tool_error__": {
            "error_type": "RuntimeError",
            "message": "exhausted tool error",
        }
    }

    # When handle_errors=False, exception propagates out of make_orchestrator
    rep_unhandled = {
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
                "retries": 2,
                "handle_errors": False,
            },
        ],
    }
    ctx2 = Ctx()
    gen2 = make_orchestrator(Limits())(ctx2, start_input())
    next(gen2)
    gen2.send(rep_unhandled)
    with pytest.raises(RuntimeError, match="unhandled tool failure"):
        gen2.throw(RuntimeError("unhandled tool failure"))


def test_orchestrator_durable_nodes_parallel_batch_max_concurrency_and_interrupt():
    """Verify make_orchestrator schedules status='nodes' tasks in parallel before.

    yielding each task
    in index order, respects node_max_concurrency, and handles interrupt() inside a
    durable node.
    """
    three_nodes = [
        {
            "call_id": f"c1:node:worker:{i}:tid{i}",
            "index": i,
            "task_id": f"tid{i}",
            "name": "worker",
            "input": pack({"item": i}),
            "durable": True,
            "retries": 3,
        }
        for i in range(3)
    ]
    n_outs = [
        {
            "call_id": f"c1:node:worker:{i}:tid{i}",
            "status": "ok",
            "output": pack({"res": i}),
        }
        for i in range(3)
    ]
    rep_nodes = {"status": "nodes", "ckpt": "c1", "node_tasks": three_nodes}

    # 1. Default parallel fan-out schedules all 3 node tasks before yielding index 0
    ctx, out = run(Limits(), [rep_nodes, *n_outs, DONE])
    assert out["status"] == "done"
    assert ctx.scheduled_at_yield == [1, 4, 4, 4, 5]
    assert ctx.calls[-1]["kind"] == "nodes"
    assert [unpack(o) for o in ctx.calls[-1]["outputs"]] == [
        {"res": 0},
        {"res": 1},
        {"res": 2},
    ]

    # 2. Bounded concurrency=2 -> batch of 2 (1 + 2 = 3 scheduled) + batch of 1 (3 + 1 =
    # 4 scheduled)
    rep_b2 = {
        "status": "nodes",
        "ckpt": "c1",
        "max_concurrency": 2,
        "node_tasks": three_nodes,
    }
    ctx2, out2 = run(Limits(), [rep_b2, *n_outs, DONE])
    assert out2["status"] == "done"
    assert ctx2.scheduled_at_yield == [1, 3, 3, 4, 5]

    # 3. Interrupt inside node task 1 -> only node task 1 re-runs after resume
    ctx3 = Ctx()
    gen = make_orchestrator(Limits())(ctx3, start_input())
    next(gen)
    n0_yield = gen.send(rep_nodes)
    assert isinstance(n0_yield, ActTask) and n0_yield[2]["index"] == 0
    assert len(ctx3.calls) == 4  # 1 step + all 3 node tasks scheduled in parallel!
    n1_yield = gen.send(n_outs[0])
    assert isinstance(n1_yield, ActTask) and n1_yield[2]["index"] == 1
    n2_yield = gen.send(
        {
            "call_id": three_nodes[1]["call_id"],
            "status": "interrupted",
            "interrupt": {"id": "n-int-1", "value": pack("ask-n1")},
        }
    )
    assert isinstance(n2_yield, ActTask) and n2_yield[2]["index"] == 2
    ev_yield = gen.send(n_outs[2])
    assert ev_yield == ("evt", "resume:n-int-1")
    re_yield = gen.send({"value": pack("approved-n1")})
    assert isinstance(re_yield, ActTask) and re_yield[1] == "langgraph.x.node"
    assert re_yield[2]["index"] == 1 and [
        unpack(r) for r in re_yield[2]["resumes"]
    ] == ["approved-n1"]
    commit_yield = gen.send(n_outs[1])
    assert commit_yield[1] == "langgraph.x.step" and commit_yield[2]["kind"] == "nodes"


def test_orchestrator_version_upgrade_on_continue_as_new_and_on_resume():
    """Verify make_orchestrator upgrades version='v1' -> 'v2' on continue_as_new when.

    target_version='v2'
    is reported, and also upgrades on resume when version='v2' is passed in the resume
    event.
    """
    # Part 1: Auto-upgrade on continue_as_new
    ctx = Ctx()
    gen = make_orchestrator(Limits(steps_per_generation=1))(
        ctx, {**start_input(), "version": "v1"}
    )
    next(gen)
    with pytest.raises(StopIteration):
        gen.send({"status": "running", "ckpt": "c1", "target_version": "v2"})
    assert ctx.can is not None
    can_inp, save_ev = ctx.can
    assert save_ev is False
    assert can_inp["version"] == "v2"
    assert can_inp["req"]["version"] == "v2"
    assert can_inp["req"]["migrate_from"] == "v1"

    # Part 2: Explicit upgrade on resume event
    ctx2 = Ctx()
    gen2 = make_orchestrator(Limits())(ctx2, {**start_input(), "version": "v1"})
    next(gen2)
    ev_y = gen2.send(
        {
            "status": "interrupted",
            "ckpt": "c1",
            "interrupt": {"id": "i1", "value": pack("q?")},
        }
    )
    assert ev_y == ("evt", "resume:i1")
    res_y = gen2.send(
        {
            "value": pack("answer"),
            "version": "v2",
            "as_node": "new_gate",
            "migrate_values": pack({"flag": True}),
        }
    )
    assert res_y[1] == "langgraph.x@v2.step"
    req_resume = res_y[2]
    assert req_resume["kind"] == "resume"
    assert req_resume["migrate_from"] == "v1"
    assert req_resume["version"] == "v2"
    assert req_resume["as_node"] == "new_gate"
    assert unpack(req_resume["migrate_values"]) == {"flag": True}


def test_orchestrator_version_upgrade_during_tool_and_node_interrupts():
    """Verify make_orchestrator handles version/upgrade payloads when resuming from a.

    tool-level
    or node-level interrupt (both in-place version upgrade and as_node topology jump).
    """
    # 1. Tool-level interrupt with in-place version upgrade (jumped=False)
    tools_rep = {
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
                "retries": 2,
            },
        ],
    }
    ctx = Ctx()
    gen = make_orchestrator(Limits())(ctx, {**start_input(), "version": "v1"})
    next(gen)
    gen.send(tools_rep)
    ev = gen.send(
        {
            "call_id": "c1:tools:0:tc1",
            "status": "interrupted",
            "interrupt": {"id": "ti1", "value": pack("ask")},
        }
    )
    assert ev == ("evt", "resume:ti1")
    mig_yield = gen.send({"value": pack("yes"), "version": "v2"})
    assert mig_yield[1] == "langgraph.x@v2.step" and mig_yield[2]["kind"] == "migrate"
    assert mig_yield[2]["migrate_from"] == "v1" and mig_yield[2]["version"] == "v2"
    tool_retry_yield = gen.send({"status": "tools", "ckpt": "c1@mig", "jumped": False})
    assert (
        tool_retry_yield[1] == "langgraph.x@v2.tool"
        and tool_retry_yield[2]["version"] == "v2"
    )
    commit_yield = gen.send(
        {"call_id": "c1:tools:0:tc1", "status": "ok", "output": pack("ok")}
    )
    assert (
        commit_yield[1] == "langgraph.x@v2.step"
        and commit_yield[2]["kind"] == "tools"
        and commit_yield[2]["version"] == "v2"
    )

    # 2. Node-level interrupt with as_node jump (jumped=True skips node retry & commit)
    nodes_rep = {
        "status": "nodes",
        "ckpt": "c2",
        "node_tasks": [
            {
                "call_id": "c2:node:n1:0:tid1",
                "index": 0,
                "task_id": "tid1",
                "name": "n1",
                "durable": True,
                "retries": 2,
            },
        ],
    }
    ctx2 = Ctx()
    gen2 = make_orchestrator(Limits())(ctx2, {**start_input(), "version": "v1"})
    next(gen2)
    gen2.send(nodes_rep)
    ev2 = gen2.send(
        {
            "call_id": "c2:node:n1:0:tid1",
            "status": "interrupted",
            "interrupt": {"id": "ni1", "value": pack("ask-n")},
        }
    )
    assert ev2 == ("evt", "resume:ni1")
    mig_y2 = gen2.send(
        {"value": pack("jump"), "upgrade": {"version": "v2", "as_node": "after_n1"}}
    )
    assert (
        mig_y2[1] == "langgraph.x@v2.step"
        and mig_y2[2]["kind"] == "migrate"
        and mig_y2[2]["as_node"] == "after_n1"
    )
    with pytest.raises(StopIteration) as done_exc:
        gen2.send({"status": "done", "ckpt": "c3", "jumped": True})
    assert (
        done_exc.value.value["version"] == "v2" and done_exc.value.value["ckpt"] == "c3"
    )


def test_orchestrator_parallel_tools_yielded_in_index_order_on_failure():
    """Verify make_orchestrator schedules all batch tasks up front, then yields each.

    task in index order
    without WhenAllTask so an early tool failure with handle_errors=True awaits slow
    sibling tasks cleanly.
    """
    ctx = Ctx()
    failed_t = ActTask(("act", "t0", {}))
    sibling_t = ActTask(("act", "t1", {}))
    tasks_iter = iter([failed_t, sibling_t])
    ctx.call_activity = lambda name, input=None, retry_policy=None: (
        ctx.calls.append(input) or next(tasks_iter)
        if name.endswith(".tool")
        else ActTask(("act", name, input))
    )
    rep = {
        "status": "tools",
        "ckpt": "c1",
        "node": "tools",
        "tool_calls": [
            {
                "call_id": "c1:0",
                "index": 0,
                "id": "0",
                "name": "t0",
                "call": pack({}),
                "durable": True,
                "retries": 1,
                "handle_errors": True,
            },
            {
                "call_id": "c1:1",
                "index": 1,
                "id": "1",
                "name": "t1",
                "call": pack({}),
                "durable": True,
                "retries": 1,
                "handle_errors": True,
            },
        ],
    }
    gen = make_orchestrator(Limits())(ctx, start_input())
    next(gen)
    y0 = gen.send(rep)
    # Both tool activities were scheduled before yielding y0 (failed_t), with no
    # WhenAllTask parent attached!
    assert y0 is failed_t
    assert len(ctx.calls) == 2
    assert [c["call_id"] for c in ctx.calls] == ["c1:0", "c1:1"]
    assert failed_t._parent is None and not failed_t._parents
    assert sibling_t._parent is None and not sibling_t._parents

    # Tool 0 fails -> orchestrator catches ValueError and yields sibling_t (index 1)
    # next
    y1 = gen.throw(ValueError("failed tool 0"))
    assert y1 is sibling_t
    commit_y = gen.send(
        {"call_id": "c1:1", "status": "ok", "output": pack("sibling_ok")}
    )
    assert commit_y[1] == "langgraph.x.step" and commit_y[2]["kind"] == "tools"
    outputs = [unpack(o) for o in commit_y[2]["outputs"]]
    assert outputs[0] == {
        "__durable_tool_error__": {
            "error_type": "ValueError",
            "message": "failed tool 0",
        }
    }
    assert outputs[1] == "sibling_ok"


def test_continue_as_new_every_single_step_preserves_all_fields():
    """Run with Limits(steps_per_generation=1) across a real graph combining
    @durable_tool, @durable_node, interrupt(), config/context, and version upgrade so
    continue_as_new
    fires after every single step and _build_can_payload preserves every field."""
    from dataclasses import dataclass
    from operator import add
    from typing import Annotated, Any, TypedDict

    from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from langgraph.prebuilt.tool_node import ToolRuntime
    from langgraph.runtime import Runtime
    from langgraph.types import interrupt

    from langchain_azure_compute.graph import (
        attach_durability,
        durable_node,
        durable_tool,
    )
    from langchain_azure_compute.graph.runner import (
        read_output,
        run_node,
        run_step,
        run_tool,
    )

    @dataclass
    class T1Ctx:
        env: str

    class T1State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        log: Annotated[list[str], add]

    @durable_tool(retries=2)
    @tool
    def t1_tool(x: int, runtime: ToolRuntime[Any, Any]) -> str:
        """Tool reading config and context."""
        tag = (runtime.config.get("tags") or ["none"])[0]
        return f"tool:{runtime.context.env}:{tag}:{x}"

    def agent(s: T1State):
        if any(isinstance(m, ToolMessage) for m in s["messages"]):
            return {
                "messages": [AIMessage(content="tools-done")],
                "log": ["agent:after_tool"],
            }
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "t1_tool",
                            "args": {"x": 7},
                            "id": "tc1",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            "log": ["agent:call_tool"],
        }

    @durable_node(retries=2)
    def dnode(s: T1State, config: RunnableConfig, runtime: Runtime[T1Ctx]):
        uid = (config.get("configurable") or {}).get("uid", "none")
        answer = interrupt("approve-dnode?")
        return {"log": [f"dnode:{runtime.context.env}:{uid}:{answer}"]}

    def build_t1(ver_label: str):
        b = StateGraph(T1State, context_schema=T1Ctx)
        b.add_node("agent", agent)  # type: ignore[arg-type]
        b.add_node("tools", ToolNode([t1_tool]))
        b.add_node("dnode", dnode)  # type: ignore[arg-type]
        b.add_node("finish", lambda s: {"log": [f"finish:{ver_label}"]})  # type: ignore[arg-type]
        b.add_edge(START, "agent")
        b.add_conditional_edges(
            "agent",
            lambda s: "tools" if s["messages"][-1].tool_calls else "dnode",
            {"tools": "tools", "dnode": "dnode"},
        )
        b.add_edge("tools", "agent")
        b.add_edge("dnode", "finish")
        b.add_edge("finish", END)
        return b.compile(checkpointer=InMemorySaver())

    g1 = build_t1("v1")
    g2 = build_t1("v2")
    # Share checkpointer so we can inspect final state easily, while also exercising
    # inline checkpoint_bundle
    g2.checkpointer = g1.checkpointer
    lim = Limits(steps_per_generation=1)
    reg = attach_durability(
        g2,
        name="t1_torture",
        limits=lim,
        version="v2",
        previous_versions={"v1": g1},
        migrations={
            ("v1", "v2"): lambda state: {"log": state["log"] + ["migrated:v1->v2"]}
        },
        upgrade_on_continue_as_new=True,
        tool_max_concurrency=2,
        node_max_concurrency=2,
    )

    scheduled = []

    class FakeSdk:
        def get_orchestration_state(self, iid, fetch_payloads=False):
            return None

        def schedule_new_orchestration(self, name, input=None, instance_id=None):
            scheduled.append(input)

    reg.client(FakeSdk()).start(
        {"messages": [HumanMessage(content="go")], "log": []},
        thread_id="t-t1",
        version="v1",
        config={"configurable": {"uid": "u42"}, "tags": ["t1tag"]},
        context=T1Ctx(env="prod"),
    )

    cur_payload = scheduled[0]
    generations = 0
    while True:
        generations += 1
        ctx = Ctx()
        gen = make_orchestrator(lim)(ctx, cur_payload)
        try:
            yielded = next(gen)
            while True:
                if isinstance(yielded, task.WhenAllTask):
                    raise AssertionError("single tool/node should yield ActTask")
                elif yielded[0] == "act":
                    _, act_name, act_inp = yielded
                    if act_name.endswith(".step"):
                        val = run_step(reg.graph, lim, act_inp)
                    elif act_name.endswith(".tool"):
                        val = run_tool(reg.graph, act_inp)
                    else:
                        val = run_node(reg.graph, act_inp)
                else:
                    val = {"value": pack("yes-t1")}
                yielded = gen.send(val)
        except StopIteration as done:
            if ctx.can is not None:
                cur_payload, save_ev = ctx.can
                assert save_ev is False
                assert "run_config" in cur_payload and "context" in cur_payload
                assert (
                    cur_payload.get("max_concurrency") == 2
                    and cur_payload.get("node_max_concurrency") == 2
                )
                continue
            final_out = done.value
            break

    assert generations >= 3
    assert final_out["version"] == "v2"
    out_state = read_output(reg.graph, "t-t1")
    assert out_state["log"] == [
        "agent:call_tool",
        "migrated:v1->v2",
        "agent:after_tool",
        "dnode:prod:u42:yes-t1",
        "finish:v2",
    ]
    tmsgs = [m.content for m in out_state["messages"] if isinstance(m, ToolMessage)]
    assert tmsgs == ["tool:prod:t1tag:7"]


def test_duplicate_resume_retry_on_sequential_interrupts_is_deduplicated_by_ckpt():
    """Verify that when a node (or @durable_tool / @durable_node) has 2 sequential.

    interrupt() calls
    sharing the same interrupt_id, a duplicate client retry of run.resume(r1, 'ans1')
    (both buffered
    in-flight before pause #2 and called post-advance after pause #2 is visible) is
    discarded rather
    than answering interrupt #2 with interrupt #1's value."""
    import json
    from types import SimpleNamespace
    from typing import TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    from langchain_azure_compute.graph import attach_durability
    from langchain_azure_compute.graph.runner import run_step

    class SeqState(TypedDict):
        out: str

    def two_questions(state: SeqState) -> SeqState:
        a1 = interrupt("q1")
        a2 = interrupt("q2")
        return {"out": f"{a1}+{a2}"}

    b = StateGraph(SeqState)
    b.add_node("ask", two_questions)
    b.add_edge(START, "ask")
    b.add_edge("ask", END)
    reg = attach_durability(b.compile(checkpointer=InMemorySaver()), name="seq_dup")

    state_holder = {"cs": None, "status": "RUNNING", "output": None}
    event_queue = []

    class FakeSdk:
        def schedule_new_orchestration(self, name, input=None, instance_id=None):
            self.init_payload = input

        def get_orchestration_state(self, iid, fetch_payloads=False):
            return SimpleNamespace(
                runtime_status=SimpleNamespace(name=state_holder["status"]),
                serialized_custom_status=state_holder["cs"],
                serialized_output=state_holder["output"],
                failure_details=None,
            )

        def raise_orchestration_event(self, iid, name, data=None):
            event_queue.append((name, data))

        def wait_for_orchestration_completion(
            self, iid, timeout=60, fetch_payloads=True
        ):
            return self.get_orchestration_state(iid, fetch_payloads=fetch_payloads)

    sdk = FakeSdk()
    run_h = reg.client(sdk).start({"out": ""})

    ctx = Ctx()
    gen = make_orchestrator(reg.limits)(ctx, sdk.init_payload)

    # Step 1: runs until interrupt #1 ("q1")
    y = next(gen)
    assert y[0] == "act" and y[1] == reg.step_name
    rep1 = run_step(reg.graph, reg.limits, y[2])
    assert rep1["status"] == "interrupted"
    y = gen.send(rep1)
    assert y == ("evt", "resume:" + rep1["interrupt"]["id"])
    state_holder["cs"] = json.dumps(ctx.status[-1])

    r1 = run_h.pending_interrupt()
    assert r1 is not None and r1.value == "q1" and r1.ckpt == rep1["ckpt"]

    # Client calls run.resume(r1, "ans1") TWICE back-to-back (simulating HTTP timeout
    # retry before custom_status updates)
    run_h.resume(r1, "ans1")
    run_h.resume(r1, "ans1")
    assert len(event_queue) == 2
    assert event_queue[0][1]["ckpt"] == r1.ckpt
    assert event_queue[1][1]["ckpt"] == r1.ckpt

    # Orchestrator consumes first event -> runs step -> hits interrupt #2 ("q2")
    _, ev1_data = event_queue.pop(0)
    y = gen.send(ev1_data)
    assert y[0] == "act" and y[1] == reg.step_name
    rep2 = run_step(reg.graph, reg.limits, y[2])
    assert rep2["status"] == "interrupted" and rep2["ckpt"] != rep1["ckpt"]
    y = gen.send(rep2)
    assert y == ("evt", "resume:" + rep2["interrupt"]["id"])
    state_holder["cs"] = json.dumps(ctx.status[-1])

    # Buffered duplicate ev1_data arrives at pause #2: orchestrator discards it and
    # keeps waiting on "resume:<id>"!
    _, dup_ev1_data = event_queue.pop(0)
    y_still_waiting = gen.send(dup_ev1_data)
    assert y_still_waiting == ("evt", "resume:" + rep2["interrupt"]["id"])

    # Now r2 is visible in custom_status; a late post-advance retry of run.resume(r1,
    # "ans1") is a no-op!
    r2 = run_h.pending_interrupt()
    assert r2 is not None and r2.value == "q2" and r2.ckpt == rep2["ckpt"]
    run_h.resume(r1, "ans1")
    assert len(event_queue) == 0

    # Client resumes r2 with "ans2" -> orchestrator accepts it and finishes with
    # "ans1+ans2"!
    run_h.resume(r2, "ans2")
    assert len(event_queue) == 1
    _, ev2_data = event_queue.pop(0)
    y = gen.send(ev2_data)
    assert y[0] == "act" and y[1] == reg.step_name
    rep3 = run_step(reg.graph, reg.limits, y[2])
    assert rep3["status"] == "done"
    with pytest.raises(StopIteration) as done_exc:
        gen.send(rep3)
    state_holder["status"] = "COMPLETED"
    state_holder["output"] = json.dumps(done_exc.value.value)
    assert run_h.wait_result() == {"out": "ans1+ans2"}
