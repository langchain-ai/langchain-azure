# Copyright (c) Microsoft. All rights reserved.

import asyncio
import sqlite3
from operator import add
from typing import Annotated, Any, NotRequired, TypedDict

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from langchain_azure_compute.graph import (
    CUSTOM_STATUS_INLINE_LIMIT,
    MAX_GRAPH_NAME_LENGTH,
    Limits,
    ToolConfig,
    attach_durability,
    durable_node,
    durable_tool,
    mark_tool,
    pack,
    read_output,
    unpack,
)
from langchain_azure_compute.graph.runner import (
    _checkpoint_id,
    _config,
    run_node,
    run_step,
    run_tool,
)
from tests.unit_tests.graph.conversation import conversation_graph, initial_state

LIM = Limits()
TCFG = ToolConfig(default_durable=True)


def run_steps_until_pause(graph, thread, inp=None):
    """Do by hand what the orchestrator does (including per-tool durable activities).

    returns the list of reports.
    """
    attach_durability(graph, name="drv")
    expect = _checkpoint_id(graph.get_state(_config(thread)))
    request = {
        "thread": thread,
        "kind": "start",
        "expect": expect,
        "input": pack(inp if inp is not None else initial_state()),
    }
    reports = []
    while True:
        report = run_step(graph, LIM, request)
        reports.append(report)
        while report["status"] == "tools":
            outputs = []
            for tc in report["tool_calls"]:
                t_res = run_tool(
                    graph,
                    {
                        "thread": thread,
                        "ckpt": report["ckpt"],
                        "node": report["node"],
                        "call_id": tc["call_id"],
                        "index": tc["index"],
                        "call": tc["call"],
                    },
                )
                outputs.append(t_res["output"])
            report = run_step(
                graph,
                LIM,
                {
                    "thread": thread,
                    "kind": "tools",
                    "expect": report["ckpt"],
                    "node": report["node"],
                    "outputs": outputs,
                },
            )
            reports.append(report)
        if report["status"] == "done":
            return reports
        if report["status"] == "interrupted":
            request = {
                "thread": thread,
                "kind": "resume",
                "expect": report["ckpt"],
                "resume": pack(True),
            }
        else:
            request = {"thread": thread, "kind": "step", "expect": report["ckpt"]}


def test_conversation_runs_to_completion_one_superstep_per_activity():
    effects = []
    graph = conversation_graph(InMemorySaver(), 3, effects)
    reports = run_steps_until_pause(graph, "t1")
    final = read_output(graph, "t1", reports[-1]["ckpt"])
    assert final["turn"] == 3 and effects == [2, 3, 4]
    assert sum(r["status"] == "interrupted" for r in reports) == 3
    assert sum(r["status"] == "tools" for r in reports) == 3
    assert "result" not in reports[-1]
    assert all(not r["replayed"] for r in reports)


def test_redelivered_step_does_not_run_the_node_twice():
    effects = []
    graph = conversation_graph(InMemorySaver(), 1, effects)
    attach_durability(graph, name="r", durable_tools=False)
    first = {
        "thread": "t2",
        "kind": "start",
        "expect": None,
        "input": pack(initial_state()),
    }
    run_step(graph, LIM, first)
    request = None
    while True:
        ckpt = graph.get_state({"configurable": {"thread_id": "t2"}}).config[
            "configurable"
        ]["checkpoint_id"]
        report = run_step(
            graph, LIM, request or {"thread": "t2", "kind": "step", "expect": ckpt}
        )
        if report["status"] == "interrupted":
            request = {
                "thread": "t2",
                "kind": "resume",
                "expect": report["ckpt"],
                "resume": pack(True),
            }
            report = run_step(graph, LIM, request)
            tool_req = {"thread": "t2", "kind": "step", "expect": report["ckpt"]}
            run_step(graph, LIM, tool_req)
            before = list(effects)
            again = run_step(graph, LIM, tool_req)
            assert again["replayed"] and effects == before
            break
        request = None


def test_stale_expectation_reports_current_state_without_stepping():
    graph = conversation_graph(InMemorySaver(), 1)
    report = run_step(
        graph,
        LIM,
        {
            "thread": "t3",
            "kind": "start",
            "expect": None,
            "input": pack(initial_state()),
        },
    )
    stale = run_step(
        graph,
        LIM,
        {"thread": "t3", "kind": "step", "expect": "not-the-current-checkpoint"},
    )
    assert stale["replayed"] and stale["ckpt"] == report["ckpt"]


def test_graph_without_checkpointer_is_rejected():
    with pytest.raises(ValueError, match="checkpointer"):
        attach_durability(conversation_graph(False, 1), name="x")
    with pytest.raises(ValueError, match="checkpointer"):
        attach_durability(
            conversation_graph(None, 1), name="x", inline_checkpoints=False
        )
    reg = attach_durability(conversation_graph(None, 1), name="x_auto")
    assert (
        isinstance(reg.graph.checkpointer, InMemorySaver)
        and reg.inline_checkpoints is True
    )


def test_uncompiled_state_graph_builder_is_rejected():
    class S(TypedDict):
        x: int

    b = StateGraph(S)
    b.add_node("n", lambda s: s)
    b.add_edge(START, "n")
    b.add_edge("n", END)
    with pytest.raises(ValueError, match="uncompiled"):
        attach_durability(b, name="x")


def test_graph_name_validation_and_human_readable_registration_names():
    """Verify Registration produces langgraph.<name>.{run,step,tool} and enforces.

    MAX_GRAPH_NAME_LENGTH (64)
    and alphanumeric/[_.-] character constraints."""
    assert MAX_GRAPH_NAME_LENGTH == 64
    g = conversation_graph(InMemorySaver(), 1)

    max_name = "a" * MAX_GRAPH_NAME_LENGTH
    reg = attach_durability(g, name=max_name)
    assert reg.orchestrator_name == f"langgraph.{max_name}.run"
    assert reg.step_name == f"langgraph.{max_name}.step"
    assert reg.tool_name == f"langgraph.{max_name}.tool"

    reg_punct = attach_durability(g, name="support_agent-v1.prod")
    assert reg_punct.orchestrator_name == "langgraph.support_agent-v1.prod.run"
    assert reg_punct.step_name == "langgraph.support_agent-v1.prod.step"
    assert reg_punct.tool_name == "langgraph.support_agent-v1.prod.tool"

    # 65 characters exceeds MAX_GRAPH_NAME_LENGTH
    with pytest.raises(ValueError, match="maximum allowed length is 64"):
        attach_durability(g, name="a" * (MAX_GRAPH_NAME_LENGTH + 1))

    # Empty, whitespace, leading punctuation, or invalid characters raise ValueError
    for bad_name in (
        "",
        "   ",
        "bad name",
        "bad/name",
        "-leading_dash",
        ".leading_dot",
        "bad:colon",
    ):
        with pytest.raises(ValueError, match="graph name"):
            attach_durability(g, name=bad_name)


def test_state_survives_a_new_saver_instance_over_the_same_sqlite_file(tmp_path):
    effects = []
    path = tmp_path / "cp.db"
    saver = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
    graph = conversation_graph(saver, 2, effects)
    request = {
        "thread": "t4",
        "kind": "start",
        "expect": None,
        "input": pack(initial_state()),
    }
    while run_step(graph, LIM, request)["status"] == "running":
        ckpt = graph.get_state({"configurable": {"thread_id": "t4"}}).config[
            "configurable"
        ]["checkpoint_id"]
        request = {"thread": "t4", "kind": "step", "expect": ckpt}
    saver2 = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
    graph2 = conversation_graph(saver2, 2, effects)
    state = graph2.get_state({"configurable": {"thread_id": "t4"}})
    assert state.values["turn"] == 0 and state.tasks[0].interrupts
    report = run_step(
        graph2,
        LIM,
        {
            "thread": "t4",
            "kind": "resume",
            "expect": state.config["configurable"]["checkpoint_id"],
            "resume": pack(True),
        },
    )
    assert report["status"] in ("running", "tools") and effects == []


def test_mid_start_crash_runs_node_1_on_retry_without_duplicating_input():
    class S(TypedDict):
        items: Annotated[list[str], add]

    attempts = 0

    def node1(s: S):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("node 1 failed after input checkpoint was written")
        return {"items": ["node1"]}

    b = StateGraph(S)
    b.add_node("n1", node1)
    b.add_edge(START, "n1")
    b.add_edge("n1", END)
    g = b.compile(checkpointer=InMemorySaver())

    req = {
        "thread": "t-crash",
        "kind": "start",
        "expect": None,
        "input": pack({"items": ["user-in"]}),
    }
    with pytest.raises(RuntimeError, match="node 1 failed"):
        run_step(g, LIM, req)

    # Confirm LangGraph already wrote checkpoints for this thread before node 1 finished
    assert _checkpoint_id(g.get_state(_config("t-crash"))) is not None

    # Retrying the same start request must NOT return replayed=True; it must run node 1
    # without duplicating user-in
    rep = run_step(g, LIM, req)
    assert not rep["replayed"] and rep["status"] == "done" and attempts == 2
    assert read_output(g, "t-crash") == {"items": ["user-in", "node1"]}

    # A subsequent redelivery after completion DOES return replayed=True
    rep2 = run_step(g, LIM, req)
    assert rep2["replayed"] and attempts == 2


def test_thread_id_reuse_starts_new_turn_instead_of_returning_stale_state():
    class S(TypedDict):
        items: Annotated[list[str], add]

    fail_turn2_once = False

    def node1(s: S):
        nonlocal fail_turn2_once
        if fail_turn2_once:
            fail_turn2_once = False
            raise RuntimeError("turn 2 transient failure")
        return {"items": [f"processed-{len(s['items'])}"]}

    b = StateGraph(S)
    b.add_node("n1", node1)
    b.add_edge(START, "n1")
    b.add_edge("n1", END)
    g = b.compile(checkpointer=InMemorySaver())

    # Turn 1 on thread "shared-thread"
    run_steps_until_pause(g, "shared-thread", {"items": ["turn1"]})
    assert read_output(g, "shared-thread") == {"items": ["turn1", "processed-1"]}

    # Turn 2 on the SAME thread "shared-thread", with a mid-start crash and retry!
    expect_before_t2 = _checkpoint_id(g.get_state(_config("shared-thread")))
    assert expect_before_t2 is not None
    req_t2 = {
        "thread": "shared-thread",
        "kind": "start",
        "expect": expect_before_t2,
        "input": pack({"items": ["turn2"]}),
    }
    fail_turn2_once = True
    with pytest.raises(RuntimeError, match="turn 2 transient failure"):
        run_step(g, LIM, req_t2)

    rep_t2 = run_step(g, LIM, req_t2)
    assert not rep_t2["replayed"] and rep_t2["status"] == "done"
    assert read_output(g, "shared-thread") == {
        "items": ["turn1", "processed-1", "turn2", "processed-3"]
    }


def test_custom_status_inline_limit_omits_oversized_interrupt_value():
    class S(TypedDict):
        answer: str

    big_payload = {"prompt": "x" * (CUSTOM_STATUS_INLINE_LIMIT + 4096)}

    def ask(s: S):
        return {"answer": interrupt(big_payload)}

    b = StateGraph(S)
    b.add_node("ask", ask)
    b.add_edge(START, "ask")
    b.add_edge("ask", END)
    g = b.compile(checkpointer=InMemorySaver())

    rep = run_step(
        g,
        LIM,
        {
            "thread": "t-big-int",
            "kind": "start",
            "expect": None,
            "input": pack({"answer": ""}),
        },
    )
    assert rep["status"] == "interrupted"
    assert (
        "value" not in rep["interrupt"]
        and rep["interrupt"].get("in_checkpoint") is True
    )


def test_parallel_interrupts_surfaced_and_resumed_via_map_and_command():
    class S(TypedDict):
        items: Annotated[list[str], add]

    def left(s: S):
        v = interrupt({"branch": "left"})
        return {"items": [f"L:{v}"]}

    def right(s: S):
        v = interrupt({"branch": "right"})
        return {"items": [f"R:{v}"]}

    b = StateGraph(S)
    b.add_node("left", left)
    b.add_node("right", right)
    b.add_edge(START, "left")
    b.add_edge(START, "right")
    b.add_edge("left", END)
    b.add_edge("right", END)
    g = b.compile(checkpointer=InMemorySaver())

    rep = run_step(
        g,
        LIM,
        {
            "thread": "t-par-int",
            "kind": "start",
            "expect": None,
            "input": pack({"items": []}),
        },
    )
    assert rep["status"] == "interrupted"
    assert len(rep["interrupts"]) == 2
    branches = {unpack(i["value"])["branch"]: i["id"] for i in rep["interrupts"]}
    assert set(branches) == {"left", "right"}

    resume_map = {branches["left"]: "okL", branches["right"]: "okR"}
    rep2 = run_step(
        g,
        LIM,
        {
            "thread": "t-par-int",
            "kind": "resume",
            "expect": rep["ckpt"],
            "resume": pack(Command(resume=resume_map)),
        },
    )
    assert rep2["status"] == "done"
    assert sorted(read_output(g, "t-par-int")["items"]) == ["L:okL", "R:okR"]


def test_per_tool_durable_activity_fault_injection_unit():
    """3 tools in one ToolNode superstep; tool #2 fails on attempt 1 and succeeds on.

    attempt 2; tool #1 runs once.
    """

    class S(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    counts = {"t1": 0, "t2": 0, "t3": 0}

    @durable_tool(retries=3)
    @tool
    def t1(x: int) -> str:
        """Tool 1."""
        counts["t1"] += 1
        return f"r1:{x}"

    @durable_tool(retries=4)
    @tool
    def t2(x: int) -> str:
        """Tool 2."""
        counts["t2"] += 1
        if counts["t2"] == 1:
            raise RuntimeError("tool #2 transient failure")
        return f"r2:{x}"

    @durable_tool(retries=2)
    @tool
    def t3(x: int) -> str:
        """Tool 3."""
        counts["t3"] += 1
        return f"r3:{x}"

    def model_node(s: S):
        if any(isinstance(m, ToolMessage) for m in s["messages"]):
            return {"messages": [AIMessage(content="all-done")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "t1",
                            "args": {"x": 10},
                            "id": "call-1",
                            "type": "tool_call",
                        },
                        {
                            "name": "t2",
                            "args": {"x": 20},
                            "id": "call-2",
                            "type": "tool_call",
                        },
                        {
                            "name": "t3",
                            "args": {"x": 30},
                            "id": "call-3",
                            "type": "tool_call",
                        },
                    ],
                )
            ]
        }

    b = StateGraph(S)
    b.add_node("model", model_node)
    b.add_node("tools", ToolNode([t1, t2, t3]))
    b.add_conditional_edges(
        "model",
        lambda s: "tools" if s["messages"][-1].tool_calls else END,
        {"tools": "tools", END: END},
    )
    b.add_edge(START, "model")
    b.add_edge("tools", "model")
    g = b.compile(checkpointer=InMemorySaver())
    attach_durability(g, name="multi_tool")

    rep = run_step(
        g,
        LIM,
        {
            "thread": "t-tools",
            "kind": "start",
            "expect": None,
            "input": pack({"messages": [HumanMessage(content="go")]}),
        },
    )
    assert rep["status"] == "tools" and len(rep["tool_calls"]) == 3
    assert [tc["retries"] for tc in rep["tool_calls"]] == [3, 4, 2]

    outputs = []
    for tc in rep["tool_calls"]:
        t_req = {
            "thread": "t-tools",
            "ckpt": rep["ckpt"],
            "node": rep["node"],
            "call_id": tc["call_id"],
            "index": tc["index"],
            "call": tc["call"],
        }
        try:
            res = run_tool(g, t_req)
        except RuntimeError:
            # Retry only the failed tool activity (tool #2), NOT tool #1!
            res = run_tool(g, t_req)
        outputs.append(res["output"])

    assert counts == {"t1": 1, "t2": 2, "t3": 1}
    rep_after_tools = run_step(
        g,
        LIM,
        {
            "thread": "t-tools",
            "kind": "tools",
            "expect": rep["ckpt"],
            "node": "tools",
            "outputs": outputs,
        },
    )
    assert rep_after_tools["status"] == "running"
    rep_final = run_step(
        g, LIM, {"thread": "t-tools", "kind": "step", "expect": rep_after_tools["ckpt"]}
    )
    assert rep_final["status"] == "done"
    final_msgs = read_output(g, "t-tools")["messages"]
    assert [m.content for m in final_msgs if isinstance(m, ToolMessage)] == [
        "r1:10",
        "r2:20",
        "r3:30",
    ]


def test_compiled_subgraph_node_and_subgraph_interrupt_resume():
    """Verify compiled subgraphs run natively as graph nodes including subgraph.

    interrupts.
    """

    class SubState(TypedDict):
        trace: Annotated[list[str], add]
        val: int

    def sub1(s: SubState):
        bonus = interrupt({"confirm_val": s["val"]})
        return {"trace": ["sub1"], "val": s["val"] + int(bonus)}

    def sub2(s: SubState):
        return {"trace": ["sub2"], "val": s["val"] * 2}

    sb = StateGraph(SubState)
    sb.add_node("s1", sub1)
    sb.add_node("s2", sub2)
    sb.add_edge(START, "s1")
    sb.add_edge("s1", "s2")
    sb.add_edge("s2", END)
    subgraph = sb.compile()

    pb = StateGraph(SubState)
    pb.add_node(
        "outer_start", lambda s: {"trace": ["outer_start"], "val": s["val"] + 1}
    )
    pb.add_node("subgraph_node", subgraph)
    pb.add_node("outer_end", lambda s: {"trace": ["outer_end"]})
    pb.add_edge(START, "outer_start")
    pb.add_edge("outer_start", "subgraph_node")
    pb.add_edge("subgraph_node", "outer_end")
    pb.add_edge("outer_end", END)
    parent = pb.compile(checkpointer=InMemorySaver())
    attach_durability(parent, name="subgraph_test")

    r1 = run_step(
        parent,
        LIM,
        {
            "thread": "t-sub",
            "kind": "start",
            "expect": None,
            "input": pack({"trace": [], "val": 4}),
        },
    )
    assert r1["status"] == "running"
    r2 = run_step(
        parent, LIM, {"thread": "t-sub", "kind": "step", "expect": r1["ckpt"]}
    )
    assert r2["status"] == "interrupted" and unpack(r2["interrupt"]["value"]) == {
        "confirm_val": 5
    }
    r3 = run_step(
        parent,
        LIM,
        {"thread": "t-sub", "kind": "resume", "expect": r2["ckpt"], "resume": pack(10)},
    )
    assert r3["status"] == "running"
    r4 = run_step(
        parent, LIM, {"thread": "t-sub", "kind": "step", "expect": r3["ckpt"]}
    )
    assert r4["status"] == "done"
    out = read_output(parent, "t-sub")
    assert (
        out["val"] == 30
        and "sub1" in out["trace"]
        and "sub2" in out["trace"]
        and out["trace"][-1] == "outer_end"
    )


def test_async_nodes_and_output_schema_projection():
    class Inp(TypedDict):
        text: str

    class Full(TypedDict):
        text: str
        secret_internal: NotRequired[str]
        reply: NotRequired[str]

    class Out(TypedDict):
        text: str
        reply: str

    async def classify(s: Full):
        await asyncio.sleep(0)
        return {
            "text": s["text"].strip(),
            "secret_internal": "hidden",
            "reply": "ok:" + s["text"].strip(),
        }

    b = StateGraph(Full, input_schema=Inp, output_schema=Out)
    b.add_node("classify", classify)
    b.add_edge(START, "classify")
    b.add_edge("classify", END)
    g = b.compile(
        checkpointer=SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    )
    run_steps_until_pause(g, "t-async", {"text": " hi "})
    assert read_output(g, "t-async") == {"text": "hi", "reply": "ok:hi"}


def test_interrupt_inside_durable_tool_unit():
    """Verify interrupt() called inside a durable tool pauses with interrupt payload.

    and resumes cleanly.
    """

    class S(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    @durable_tool(retries=2)
    @tool
    def pay(amount: int) -> str:
        """Pay amount with human approval."""
        ok = interrupt({"confirm_amount": amount})
        return f"charged:{amount}:{ok}"

    def model_node(s: S):
        if any(isinstance(m, ToolMessage) for m in s["messages"]):
            return {"messages": [AIMessage(content="done")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "pay",
                            "args": {"amount": 500},
                            "id": "tc-pay",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    b = StateGraph(S)
    b.add_node("model", model_node)
    b.add_node("tools", ToolNode([pay]))
    b.add_edge(START, "model")
    b.add_conditional_edges(
        "model",
        lambda s: "tools" if s["messages"][-1].tool_calls else END,
        {"tools": "tools", END: END},
    )
    b.add_edge("tools", "model")
    g = b.compile(checkpointer=InMemorySaver())
    attach_durability(g, name="bug1_tool_int")

    r1 = run_step(
        g,
        LIM,
        {
            "thread": "t-b1",
            "kind": "start",
            "expect": None,
            "input": pack({"messages": [HumanMessage(content="pay")]}),
        },
    )
    assert r1["status"] == "tools" and len(r1["tool_calls"]) == 1
    tc = r1["tool_calls"][0]
    t_req = {
        "thread": "t-b1",
        "ckpt": r1["ckpt"],
        "node": r1["node"],
        "call_id": tc["call_id"],
        "index": tc["index"],
        "call": tc["call"],
    }
    t1 = run_tool(g, t_req)
    assert t1["status"] == "interrupted" and unpack(t1["interrupt"]["value"]) == {
        "confirm_amount": 500
    }

    t_req["resumes"] = [pack("yes")]
    t2 = run_tool(g, t_req)
    assert t2["status"] == "ok"
    r2 = run_step(
        g,
        LIM,
        {
            "thread": "t-b1",
            "kind": "tools",
            "expect": r1["ckpt"],
            "node": r1["node"],
            "outputs": [t2["output"]],
        },
    )
    r3 = run_step(g, LIM, {"thread": "t-b1", "kind": "step", "expect": r2["ckpt"]})
    assert r3["status"] == "done"
    assert [
        m.content
        for m in read_output(g, "t-b1")["messages"]
        if isinstance(m, ToolMessage)
    ] == ["charged:500:yes"]


def test_multiple_sequential_interrupts_in_one_node_idempotency():
    """Verify multiple sequential interrupt() calls inside a single node share raw.

    checkpoint_id;
    retrying the first resume after it already advanced to interrupt #2 must be detected
    as replayed.
    """

    class S(TypedDict):
        out: str

    def two_interrupts(s: S):
        v1 = interrupt("first_q")
        v2 = interrupt(f"second_q_after_{v1}")
        return {"out": f"{v1}+{v2}"}

    b = StateGraph(S)
    b.add_node("n", two_interrupts)
    b.add_edge(START, "n")
    b.add_edge("n", END)
    g = b.compile(checkpointer=InMemorySaver())
    attach_durability(g, name="bug2_seq_int")

    r1 = run_step(
        g,
        LIM,
        {"thread": "t-b2", "kind": "start", "expect": None, "input": pack({"out": ""})},
    )
    assert (
        r1["status"] == "interrupted" and unpack(r1["interrupt"]["value"]) == "first_q"
    )

    resume1_req = {
        "thread": "t-b2",
        "kind": "resume",
        "expect": r1["ckpt"],
        "resume": pack("ans1"),
    }
    r2 = run_step(g, LIM, resume1_req)
    assert (
        r2["status"] == "interrupted"
        and unpack(r2["interrupt"]["value"]) == "second_q_after_ans1"
    )
    assert r2["ckpt"] != r1["ckpt"]

    # Simulate worker crash-retry of resume1_req: must NOT feed "ans1" into the second
    # interrupt!
    r2_retry = run_step(g, LIM, resume1_req)
    assert r2_retry["replayed"] is True and r2_retry["status"] == "interrupted"
    assert unpack(r2_retry["interrupt"]["value"]) == "second_q_after_ans1"

    r3 = run_step(
        g,
        LIM,
        {
            "thread": "t-b2",
            "kind": "resume",
            "expect": r2["ckpt"],
            "resume": pack("ans2"),
        },
    )
    assert r3["status"] == "done" and read_output(g, "t-b2") == {"out": "ans1+ans2"}


def test_async_sqlite_saver_across_turns_and_running_loop(tmp_path):
    """Verify AsyncSqliteSaver works both when called inside a running event loop and.

    across turns on worker threads.
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    class S(TypedDict):
        items: Annotated[list[str], add]

    async def step1(s: S):
        v = interrupt("approve?")
        return {"items": [f"s1:{v}"]}

    async def step2(s: S):
        return {"items": ["s2"]}

    b = StateGraph(S)
    b.add_node("s1", step1)
    b.add_node("s2", step2)
    b.add_edge(START, "s1")
    b.add_edge("s1", "s2")
    b.add_edge("s2", END)

    db_path = str(tmp_path / "async_unit.db")

    async def run_inside_loop():
        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            g = b.compile(checkpointer=saver)
            attach_durability(g, name="bug3_async")
            r1 = run_step(
                g,
                LIM,
                {
                    "thread": "t-b3",
                    "kind": "start",
                    "expect": None,
                    "input": pack({"items": ["init"]}),
                },
            )
            assert r1["status"] == "interrupted"
            r2 = run_step(
                g,
                LIM,
                {
                    "thread": "t-b3",
                    "kind": "resume",
                    "expect": r1["ckpt"],
                    "resume": pack("yes"),
                },
            )
            assert r2["status"] == "running"
            r3 = run_step(
                g, LIM, {"thread": "t-b3", "kind": "step", "expect": r2["ckpt"]}
            )
            assert r3["status"] == "done"
            return read_output(g, "t-b3")

    assert asyncio.run(run_inside_loop()) == {"items": ["init", "s1:yes", "s2"]}

    # Also test when AsyncSqliteSaver is created in one asyncio.run() and used from sync
    # worker threads later
    db_path2 = str(tmp_path / "async_unit2.db")

    async def make_detached_saver():
        conn = await aiosqlite.connect(db_path2)
        return AsyncSqliteSaver(conn), conn

    saver2, conn2 = asyncio.run(make_detached_saver())
    try:
        g2 = b.compile(checkpointer=saver2)
        attach_durability(g2, name="bug3_detached")
        r1 = run_step(
            g2,
            LIM,
            {
                "thread": "t-b3-2",
                "kind": "start",
                "expect": None,
                "input": pack({"items": ["start2"]}),
            },
        )
        r2 = run_step(
            g2,
            LIM,
            {
                "thread": "t-b3-2",
                "kind": "resume",
                "expect": r1["ckpt"],
                "resume": pack("ok2"),
            },
        )
        r3 = run_step(
            g2, LIM, {"thread": "t-b3-2", "kind": "step", "expect": r2["ckpt"]}
        )
        assert r3["status"] == "done" and read_output(g2, "t-b3-2") == {
            "items": ["start2", "s1:ok2", "s2"]
        }
    finally:
        asyncio.run(conn2.close())


def test_parallel_and_subgraph_durable_tool_nodes():
    """Verify durable ToolNode running in parallel with another node OR inside a.

    compiled subgraph
    still emits status='tools' and executes durable tool activities."""

    class SPar(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        log: Annotated[list[str], add]

    @durable_tool(retries=4)
    @tool
    def calc(x: int) -> str:
        """Calc."""
        return f"c:{x}"

    def model_p(s: SPar):
        if any(isinstance(m, ToolMessage) for m in s["messages"]):
            return {"messages": [AIMessage(content="done")], "log": ["model_done"]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calc",
                            "args": {"x": 7},
                            "id": "tc1",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            "log": ["model_call"],
        }

    bp = StateGraph(SPar)
    bp.add_node("model", model_p)
    bp.add_node("tools", ToolNode([calc]))
    bp.add_node("audit", lambda s: {"log": ["audited"]})
    bp.add_edge(START, "model")
    bp.add_conditional_edges(
        "model", lambda s: ["tools", "audit"] if s["messages"][-1].tool_calls else END
    )
    bp.add_edge("tools", "model")
    bp.add_edge("audit", END)
    gp = bp.compile(checkpointer=InMemorySaver())
    attach_durability(gp, name="bug4_par")

    r1 = run_step(
        gp,
        LIM,
        {
            "thread": "t-b4p",
            "kind": "start",
            "expect": None,
            "input": pack({"messages": [HumanMessage(content="hi")], "log": []}),
        },
    )
    assert r1["status"] == "tools" and r1["tool_calls"][0]["retries"] == 4
    tc = r1["tool_calls"][0]
    t_res = run_tool(
        gp,
        {
            "thread": "t-b4p",
            "ckpt": r1["ckpt"],
            "node": r1["node"],
            "call_id": tc["call_id"],
            "index": tc["index"],
            "call": tc["call"],
        },
    )
    r2 = run_step(
        gp,
        LIM,
        {
            "thread": "t-b4p",
            "kind": "tools",
            "expect": r1["ckpt"],
            "node": r1["node"],
            "outputs": [t_res["output"]],
        },
    )
    while r2["status"] != "done":
        r2 = run_step(
            gp, LIM, {"thread": "t-b4p", "kind": "step", "expect": r2["ckpt"]}
        )
    out_p = read_output(gp, "t-b4p")
    assert out_p["log"] == ["model_call", "audited", "model_done"]

    # Subgraph with durable ToolNode
    class SSub(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    @durable_tool(retries=3)
    @tool
    def sub_tool(x: int) -> str:
        """Sub tool."""
        return f"sub:{x}"

    def sub_model(s: SSub):
        if any(isinstance(m, ToolMessage) for m in s["messages"]):
            return {"messages": [AIMessage(content="sub_done")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "sub_tool",
                            "args": {"x": 9},
                            "id": "stc1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    sb = StateGraph(SSub)
    sb.add_node("sub_model", sub_model)
    sb.add_node("sub_tools", ToolNode([sub_tool]))
    sb.add_edge(START, "sub_model")
    sb.add_conditional_edges(
        "sub_model",
        lambda s: "sub_tools" if s["messages"][-1].tool_calls else END,
        {"sub_tools": "sub_tools", END: END},
    )
    sb.add_edge("sub_tools", "sub_model")

    pb = StateGraph(SSub)
    pb.add_node("outer", lambda s: {"messages": [HumanMessage(content="start")]})
    pb.add_node("sub", sb.compile())
    pb.add_edge(START, "outer")
    pb.add_edge("outer", "sub")
    pb.add_edge("sub", END)
    parent = pb.compile(checkpointer=InMemorySaver())
    attach_durability(parent, name="bug4_sub")

    sr1 = run_step(
        parent,
        LIM,
        {
            "thread": "t-b4s",
            "kind": "start",
            "expect": None,
            "input": pack({"messages": []}),
        },
    )
    sr2 = run_step(
        parent, LIM, {"thread": "t-b4s", "kind": "step", "expect": sr1["ckpt"]}
    )
    assert sr2["status"] == "tools" and sr2["node"] == "sub:sub_tools"
    stc = sr2["tool_calls"][0]
    st_res = run_tool(
        parent,
        {
            "thread": "t-b4s",
            "ckpt": sr2["ckpt"],
            "node": sr2["node"],
            "tool_ns": sr2.get("tool_ns"),
            "tool_ckpt": sr2.get("tool_ckpt"),
            "call_id": stc["call_id"],
            "index": stc["index"],
            "call": stc["call"],
        },
    )
    sr3 = run_step(
        parent,
        LIM,
        {
            "thread": "t-b4s",
            "kind": "tools",
            "expect": sr2["ckpt"],
            "node": sr2["node"],
            "tool_ns": sr2.get("tool_ns"),
            "outputs": [st_res["output"]],
        },
    )
    sr4 = run_step(
        parent, LIM, {"thread": "t-b4s", "kind": "step", "expect": sr3["ckpt"]}
    )
    assert sr4["status"] == "done"
    assert [m.content for m in read_output(parent, "t-b4s")["messages"]] == [
        "start",
        "",
        "sub:9",
        "sub_done",
    ]


def test_mixed_durable_and_local_tools_in_same_turn_and_string_option():
    """Verify ToolConfig.from_options('local') does not iterate over characters, and.

    mixed durable+local
    tools in the same ToolNode turn run local tools inline during commit while
    scheduling only durable tools.
    """
    tcfg_loc = ToolConfig.from_options(durable_tools="local")
    assert tcfg_loc.default_durable is False and not tcfg_loc.policy
    with pytest.raises(ValueError, match="unsupported tool policy value"):
        ToolConfig.from_options(durable_tools="invalid_mode")

    class SMix(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    @mark_tool(durable=False)
    @tool
    def loc_t(x: int) -> str:
        """Local tool."""
        return f"loc:{x}"

    @durable_tool(retries=3)
    @tool
    def dur_t(x: int) -> str:
        """Durable tool."""
        return f"dur:{x}"

    def model_m(s: SMix):
        if any(isinstance(m, ToolMessage) for m in s["messages"]):
            return {"messages": [AIMessage(content="done")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "loc_t",
                            "args": {"x": 1},
                            "id": "c1",
                            "type": "tool_call",
                        },
                        {
                            "name": "dur_t",
                            "args": {"x": 2},
                            "id": "c2",
                            "type": "tool_call",
                        },
                    ],
                )
            ]
        }

    bm = StateGraph(SMix)
    bm.add_node("model", model_m)
    bm.add_node("tools", ToolNode([loc_t, dur_t]))
    bm.add_edge(START, "model")
    bm.add_conditional_edges(
        "model",
        lambda s: "tools" if s["messages"][-1].tool_calls else END,
        {"tools": "tools", END: END},
    )
    bm.add_edge("tools", "model")
    gm = bm.compile(checkpointer=InMemorySaver())
    attach_durability(gm, name="bug5_mix")

    r1 = run_step(
        gm,
        LIM,
        {
            "thread": "t-b5",
            "kind": "start",
            "expect": None,
            "input": pack({"messages": [HumanMessage(content="go")]}),
        },
    )
    assert [(tc["name"], tc["durable"]) for tc in r1["tool_calls"]] == [
        ("loc_t", False),
        ("dur_t", True),
    ]
    dur_tc = r1["tool_calls"][1]
    t_res = run_tool(
        gm,
        {
            "thread": "t-b5",
            "ckpt": r1["ckpt"],
            "node": r1["node"],
            "call_id": dur_tc["call_id"],
            "index": dur_tc["index"],
            "call": dur_tc["call"],
        },
    )
    r2 = run_step(
        gm,
        LIM,
        {
            "thread": "t-b5",
            "kind": "tools",
            "expect": r1["ckpt"],
            "node": r1["node"],
            "outputs": {str(dur_tc["index"]): t_res["output"]},
        },
    )
    r3 = run_step(gm, LIM, {"thread": "t-b5", "kind": "step", "expect": r2["ckpt"]})
    assert r3["status"] == "done"
    assert [
        m.content
        for m in read_output(gm, "t-b5")["messages"]
        if isinstance(m, ToolMessage)
    ] == ["loc:1", "dur:2"]


def test_static_interrupt_before_and_interrupt_after_breakpoints():
    """Verify user-configured compile(interrupt_before=[...], interrupt_after=[...]).

    pauses at static breakpoints.
    """

    class SStatic(TypedDict):
        trace: Annotated[list[str], add]
        x: int

    bs = StateGraph(SStatic)
    bs.add_node("a", lambda s: {"trace": ["a"], "x": s["x"] + 1})
    bs.add_node("b", lambda s: {"trace": ["b"], "x": s["x"] * 10})
    bs.add_node("c", lambda s: {"trace": ["c"], "x": s["x"] + 5})
    bs.add_edge(START, "a")
    bs.add_edge("a", "b")
    bs.add_edge("b", "c")
    bs.add_edge("c", END)
    gs = bs.compile(
        checkpointer=InMemorySaver(), interrupt_before=["b"], interrupt_after=["b"]
    )
    attach_durability(gs, name="bug6_static")

    rs1 = run_step(
        gs,
        LIM,
        {
            "thread": "t-b6",
            "kind": "start",
            "expect": None,
            "input": pack({"trace": [], "x": 1}),
        },
    )
    assert rs1["status"] == "interrupted"
    assert unpack(rs1["interrupt"]["value"]) == {
        "breakpoint": "interrupt_before",
        "nodes": ["b"],
        "next": ["b"],
    }
    assert read_output(gs, "t-b6") == {"trace": ["a"], "x": 2}

    rs2 = run_step(
        gs,
        LIM,
        {
            "thread": "t-b6",
            "kind": "resume",
            "expect": rs1["ckpt"],
            "resume": pack(None),
        },
    )
    assert rs2["status"] == "interrupted"
    assert unpack(rs2["interrupt"]["value"]) == {
        "breakpoint": "interrupt_after",
        "nodes": ["b"],
        "next": ["c"],
    }
    assert read_output(gs, "t-b6") == {"trace": ["a", "b"], "x": 20}

    rs3 = run_step(
        gs,
        LIM,
        {
            "thread": "t-b6",
            "kind": "resume",
            "expect": rs2["ckpt"],
            "resume": pack(None),
        },
    )
    assert rs3["status"] == "done"
    assert read_output(gs, "t-b6") == {"trace": ["a", "b", "c"], "x": 25}


def test_checkpointerless_graph_survives_worker_switch_across_steps_and_tools():
    """A graph compiled with NO checkpointer (builder.compile()) auto-installs.

    InMemorySaver with
    inline_checkpoints=True and transfers checkpoint_bundle across separate worker
    instances.
    """
    effects_a: list[int] = []
    effects_b: list[int] = []
    # Worker A and Worker B each compile with checkpointer=None (distinct ephemeral
    # InMemorySavers)
    worker_a_graph = conversation_graph(None, 1, effects_a)
    worker_b_graph = conversation_graph(None, 1, effects_b)
    reg_a = attach_durability(worker_a_graph, name="ephemeral_handoff")
    reg_b = attach_durability(worker_b_graph, name="ephemeral_handoff")
    assert reg_a.inline_checkpoints is True and reg_b.inline_checkpoints is True
    assert worker_a_graph.checkpointer is not worker_b_graph.checkpointer

    # Step 1 (start -> model) and Step 2 (approval interrupt) on Worker A
    r1 = run_step(
        worker_a_graph,
        LIM,
        {
            "thread": "t-eph",
            "kind": "start",
            "expect": None,
            "input": pack(initial_state()),
        },
    )
    assert r1["status"] == "running" and "checkpoint_bundle" in r1
    r2 = run_step(
        worker_a_graph,
        LIM,
        {
            "thread": "t-eph",
            "kind": "step",
            "expect": r1["ckpt"],
            "checkpoint_bundle": r1["checkpoint_bundle"],
        },
    )
    assert r2["status"] == "interrupted" and "checkpoint_bundle" in r2

    # Worker A crashes/stops here! Worker B (whose InMemorySaver is completely empty)
    # handles resume, tool, and completion
    assert _checkpoint_id(worker_b_graph.get_state(_config("t-eph"))) is None
    r3 = run_step(
        worker_b_graph,
        LIM,
        {
            "thread": "t-eph",
            "kind": "resume",
            "expect": r2["ckpt"],
            "resume": pack(True),
            "checkpoint_bundle": r2["checkpoint_bundle"],
        },
    )
    assert r3["status"] == "tools" and "checkpoint_bundle" in r3

    # Even if the durable tool runs on a third fresh graph instance (Worker C),
    # checkpoint_bundle hydrates it
    worker_c_graph = conversation_graph(None, 1, effects_b)
    attach_durability(worker_c_graph, name="ephemeral_handoff")
    tc = r3["tool_calls"][0]
    t_res = run_tool(
        worker_c_graph,
        {
            "thread": "t-eph",
            "ckpt": r3["ckpt"],
            "node": r3["node"],
            "call_id": tc["call_id"],
            "index": tc["index"],
            "call": tc["call"],
            "checkpoint_bundle": r3["checkpoint_bundle"],
        },
    )
    assert t_res["status"] == "ok" and effects_b == [2] and effects_a == []

    r4 = run_step(
        worker_b_graph,
        LIM,
        {
            "thread": "t-eph",
            "kind": "tools",
            "expect": r3["ckpt"],
            "node": r3["node"],
            "outputs": [t_res["output"]],
            "checkpoint_bundle": r3["checkpoint_bundle"],
        },
    )
    assert r4["status"] == "running"
    r5 = run_step(
        worker_b_graph,
        LIM,
        {
            "thread": "t-eph",
            "kind": "step",
            "expect": r4["ckpt"],
            "checkpoint_bundle": r4["checkpoint_bundle"],
        },
    )
    assert r5["status"] == "done" and "output" in r5
    final_val = unpack(r5["output"])
    assert final_val["turn"] == 1 and final_val["approved"] is True


def test_async_durable_tools_and_local_tool_max_concurrency_unit():
    """Verify async @tool execution via run_tool (including interrupt() inside an async.

    tool)
    and max_concurrency (None, 1, 2) on local tools and ToolConfig /
    attach_durability().
    """
    import threading
    import time

    with pytest.raises(ValueError, match="max_concurrency"):
        ToolConfig(max_concurrency=0)

    class SAsync(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    @durable_tool(retries=2)
    @tool
    async def async_Approve(x: int) -> str:
        """Async tool that calls interrupt()."""
        await asyncio.sleep(0.01)
        ok = interrupt({"async_confirm": x})
        return f"async:{x}:{ok}"

    def model_a(s: SAsync):
        if any(isinstance(m, ToolMessage) for m in s["messages"]):
            return {"messages": [AIMessage(content="done")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "async_Approve",
                            "args": {"x": 42},
                            "id": "ac1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    ba = StateGraph(SAsync)
    ba.add_node("model", model_a)
    ba.add_node("tools", ToolNode([async_Approve]))
    ba.add_edge(START, "model")
    ba.add_conditional_edges(
        "model",
        lambda s: "tools" if s["messages"][-1].tool_calls else END,
        {"tools": "tools", END: END},
    )
    ba.add_edge("tools", "model")
    ga = ba.compile(checkpointer=InMemorySaver())
    reg_a = attach_durability(ga, name="async_tool_u", tool_max_concurrency=2)
    assert reg_a.tool_config.max_concurrency == 2

    ra1 = run_step(
        ga,
        LIM,
        {
            "thread": "t-au",
            "kind": "start",
            "expect": None,
            "input": pack({"messages": [HumanMessage(content="go")]}),
        },
    )
    assert ra1["status"] == "tools" and ra1["max_concurrency"] == 2
    tc = ra1["tool_calls"][0]
    t_req = {
        "thread": "t-au",
        "ckpt": ra1["ckpt"],
        "node": ra1["node"],
        "call_id": tc["call_id"],
        "index": tc["index"],
        "call": tc["call"],
    }
    ta_int = run_tool(ga, t_req)
    assert ta_int["status"] == "interrupted" and unpack(
        ta_int["interrupt"]["value"]
    ) == {"async_confirm": 42}
    t_req["resumes"] = [pack("ok")]
    ta_ok = run_tool(ga, t_req)
    assert ta_ok["status"] == "ok"

    # Local tools in run_step(kind="tools") honoring max_concurrency=1 vs
    # max_concurrency=2
    active = {"cur": 0, "peak": 0}
    order: list[str] = []
    lock = threading.Lock()

    @mark_tool(durable=False)
    @tool
    def loc_work(idx: int) -> str:
        """Local work."""
        with lock:
            active["cur"] += 1
            active["peak"] = max(active["peak"], active["cur"])
            order.append(f"start:{idx}")
        time.sleep(0.03)
        with lock:
            order.append(f"end:{idx}")
            active["cur"] -= 1
        return f"loc:{idx}"

    @durable_tool
    @tool
    def dur_anchor() -> str:
        """Durable anchor."""
        return "anchor"

    def model_loc(s: SAsync):
        if any(isinstance(m, ToolMessage) for m in s["messages"]):
            return {"messages": [AIMessage(content="done")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "dur_anchor",
                            "args": {},
                            "id": "da",
                            "type": "tool_call",
                        },
                        *[
                            {
                                "name": "loc_work",
                                "args": {"idx": i},
                                "id": f"lw-{i}",
                                "type": "tool_call",
                            }
                            for i in range(4)
                        ],
                    ],
                )
            ]
        }

    bl = StateGraph(SAsync)
    bl.add_node("model", model_loc)
    bl.add_node("tools", ToolNode([dur_anchor, loc_work]))
    bl.add_edge(START, "model")
    bl.add_conditional_edges(
        "model",
        lambda s: "tools" if s["messages"][-1].tool_calls else END,
        {"tools": "tools", END: END},
    )
    bl.add_edge("tools", "model")

    for mc, expected_peak in ((1, 1), (2, 2)):
        active["cur"] = active["peak"] = 0
        order.clear()
        gl = bl.compile(checkpointer=InMemorySaver())
        attach_durability(gl, name=f"loc_mc_{mc}", tool_max_concurrency=mc)
        rl1 = run_step(
            gl,
            LIM,
            {
                "thread": f"t-lmc-{mc}",
                "kind": "start",
                "expect": None,
                "input": pack({"messages": [HumanMessage(content="go")]}),
            },
        )
        dtc = rl1["tool_calls"][0]
        dt_res = run_tool(
            gl,
            {
                "thread": f"t-lmc-{mc}",
                "ckpt": rl1["ckpt"],
                "node": rl1["node"],
                "call_id": dtc["call_id"],
                "index": 0,
                "call": dtc["call"],
            },
        )
        run_step(
            gl,
            LIM,
            {
                "thread": f"t-lmc-{mc}",
                "kind": "tools",
                "expect": rl1["ckpt"],
                "node": "tools",
                "max_concurrency": mc,
                "outputs": {"0": dt_res["output"]},
            },
        )
        if mc == 1:
            assert order == [
                "start:0",
                "end:0",
                "start:1",
                "end:1",
                "start:2",
                "end:2",
                "start:3",
                "end:3",
            ]
        assert active["peak"] <= expected_peak


def test_toolnode_handle_tool_errors_retries_transient_then_formats_exhausted():
    """Default ToolNode(handle_tool_errors=True) raises during run_tool so DTS retries.

    transient
    failures, and converts exhausted failures into ToolMessage(status='error') via
    run_step(kind='tools').
    """

    class S(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    attempts = {"flaky": 0, " doomed": 0}

    @durable_tool(retries=3)
    @tool
    def flaky_api(x: int) -> str:
        """Fails twice, succeeds on 3rd attempt."""
        attempts["flaky"] += 1
        if attempts["flaky"] < 3:
            raise RuntimeError(f"transient network error #{attempts['flaky']}")
        return f"success:{x} on attempt {attempts['flaky']}"

    @durable_tool(retries=2)
    @tool
    def doomed_api(x: int) -> str:
        """Always fails."""
        attempts[" doomed"] += 1
        raise ValueError(f"permanent failure #{attempts[' doomed']}")

    def agent(s: S):
        if any(isinstance(m, ToolMessage) for m in s["messages"]):
            return {"messages": [AIMessage(content="handled")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "flaky_api",
                            "args": {"x": 42},
                            "id": "tc-flaky",
                            "type": "tool_call",
                        },
                        {
                            "name": "doomed_api",
                            "args": {"x": 99},
                            "id": "tc-doomed",
                            "type": "tool_call",
                        },
                    ],
                )
            ]
        }

    b = StateGraph(S)
    b.add_node("agent", agent)
    b.add_node("tools", ToolNode([flaky_api, doomed_api], handle_tool_errors=True))
    b.add_edge(START, "agent")
    b.add_conditional_edges(
        "agent",
        lambda s: "tools" if s["messages"][-1].tool_calls else END,
        {"tools": "tools", END: END},
    )
    b.add_edge("tools", "agent")
    g = b.compile(checkpointer=InMemorySaver())
    attach_durability(g, name="b1_err")

    r1 = run_step(
        g,
        LIM,
        {
            "thread": "t-err",
            "kind": "start",
            "expect": None,
            "input": pack({"messages": [HumanMessage(content="go")]}),
        },
    )
    assert r1["status"] == "tools" and all(
        tc["handle_errors"] is True for tc in r1["tool_calls"]
    )

    # Tool 0 (flaky_api): raises on attempts 1 and 2, succeeds on attempt 3
    tc0 = r1["tool_calls"][0]
    req0 = {
        "thread": "t-err",
        "ckpt": r1["ckpt"],
        "node": "tools",
        "call_id": tc0["call_id"],
        "index": 0,
        "call": tc0["call"],
    }
    with pytest.raises(RuntimeError, match="transient network error #1"):
        run_tool(g, req0)
    with pytest.raises(RuntimeError, match="transient network error #2"):
        run_tool(g, req0)
    res0 = run_tool(g, req0)
    assert res0["status"] == "ok" and attempts["flaky"] == 3

    # Tool 1 (doomed_api): fails all 2 attempts; orchestrator passes
    # __durable_tool_error__ to run_step(kind='tools')
    tc1 = r1["tool_calls"][1]
    req1 = {
        "thread": "t-err",
        "ckpt": r1["ckpt"],
        "node": "tools",
        "call_id": tc1["call_id"],
        "index": 1,
        "call": tc1["call"],
    }
    for _ in range(2):
        with pytest.raises(ValueError, match="permanent failure"):
            run_tool(g, req1)
    assert attempts[" doomed"] == 2
    err_out1 = pack(
        {
            "__durable_tool_error__": {
                "error_type": "ValueError",
                "message": "permanent failure #2",
            }
        }
    )

    r2 = run_step(
        g,
        LIM,
        {
            "thread": "t-err",
            "kind": "tools",
            "expect": r1["ckpt"],
            "node": "tools",
            "outputs": [res0["output"], err_out1],
        },
    )
    r3 = run_step(g, LIM, {"thread": "t-err", "kind": "step", "expect": r2["ckpt"]})
    assert r3["status"] == "done"
    tool_msgs = [
        m for m in read_output(g, "t-err")["messages"] if isinstance(m, ToolMessage)
    ]
    assert len(tool_msgs) == 2
    assert (
        tool_msgs[0].content == "success:42 on attempt 3"
        and tool_msgs[0].status != "error"
    )
    assert (
        tool_msgs[1].status == "error"
        and "ValueError('permanent failure #2')" in tool_msgs[1].content
    )


def test_run_config_and_context_propagation_to_nodes_and_tools():
    """Custom configurable keys, tags, metadata, and LangGraph 1.x Runtime[Context]
    propagate to both graph nodes and @durable_tool (via ToolRuntime and get_config()).
    """
    from dataclasses import dataclass

    from langchain_core.runnables import RunnableConfig
    from langgraph.config import get_config
    from langgraph.prebuilt.tool_node import ToolRuntime
    from langgraph.runtime import Runtime

    @dataclass
    class MyContext:
        tenant_id: str

    class S(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        node_info: str

    @durable_tool
    @tool
    def ctx_tool(x: int, runtime: ToolRuntime[Any, Any]) -> str:
        """Reads config and context inside durable tool."""
        cfg = get_config()
        uid = runtime.config.get("configurable", {}).get("user_id", "MISSING")
        uid_var = cfg.get("configurable", {}).get("user_id", "MISSING")
        tag0 = (runtime.config.get("tags") or ["NONE"])[0]
        env = (runtime.config.get("metadata") or {}).get("env", "NONE")
        tenant = runtime.context.tenant_id if runtime.context else "NONE"
        return f"tool_uid={uid}/{uid_var}|tag={tag0}|env={env}|tenant={tenant}|x={x}"

    def agent(s: S, config: RunnableConfig, runtime: Runtime[MyContext]):
        if any(isinstance(m, ToolMessage) for m in s["messages"]):
            return {"messages": [AIMessage(content="done")]}
        uid = config.get("configurable", {}).get("user_id", "MISSING")
        tenant = runtime.context.tenant_id if runtime.context else "NONE"
        return {
            "node_info": f"node_uid={uid}|tenant={tenant}",
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ctx_tool",
                            "args": {"x": 7},
                            "id": "c1",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
        }

    b = StateGraph(S, context_schema=MyContext)
    b.add_node("agent", agent)
    b.add_node("tools", ToolNode([ctx_tool]))
    b.add_edge(START, "agent")
    b.add_conditional_edges(
        "agent",
        lambda s: "tools" if s["messages"][-1].tool_calls else END,
        {"tools": "tools", END: END},
    )
    b.add_edge("tools", "agent")
    g = b.compile(checkpointer=InMemorySaver())
    reg = attach_durability(g, name="b1_ctx")

    scheduled = []

    class FakeSdk:
        def get_orchestration_state(self, iid, fetch_payloads=False):
            return None

        def schedule_new_orchestration(self, name, input=None, instance_id=None):
            scheduled.append(input)

    client = reg.client(FakeSdk())
    user_cfg = {
        "configurable": {"user_id": "alice_123"},
        "tags": ["tier1"],
        "metadata": {"env": "prod"},
    }
    client.start(
        {"messages": [HumanMessage(content="hi")], "node_info": ""},
        thread_id="t-ctx",
        config=user_cfg,
        context=MyContext(tenant_id="acme_corp"),
    )
    payload = scheduled[0]
    r1 = run_step(g, LIM, payload["req"])
    assert r1["status"] == "tools"
    tc = r1["tool_calls"][0]
    t_res = run_tool(
        g,
        {
            "thread": "t-ctx",
            "ckpt": r1["ckpt"],
            "node": "tools",
            "call_id": tc["call_id"],
            "index": 0,
            "call": tc["call"],
            "run_config": payload["run_config"],
            "context": payload["context"],
        },
    )
    r2 = run_step(
        g,
        LIM,
        {
            "thread": "t-ctx",
            "kind": "tools",
            "expect": r1["ckpt"],
            "node": "tools",
            "outputs": [t_res["output"]],
            "run_config": payload["run_config"],
            "context": payload["context"],
        },
    )
    r3 = run_step(
        g,
        LIM,
        {
            "thread": "t-ctx",
            "kind": "step",
            "expect": r2["ckpt"],
            "run_config": payload["run_config"],
            "context": payload["context"],
        },
    )
    assert r3["status"] == "done"
    out = read_output(g, "t-ctx")
    assert out["node_info"] == "node_uid=alice_123|tenant=acme_corp"
    tmsg = [m.content for m in out["messages"] if isinstance(m, ToolMessage)][0]
    assert (
        tmsg == "tool_uid=alice_123/alice_123|tag=tier1|env=prod|tenant=acme_corp|x=7"
    )


def test_remaining_steps_is_last_step_and_recursion_limit_countdown():
    """RemainingSteps counts down [4, 3, 2, 1] and IsLastStep becomes True at the exact
    same step as native LangGraph; exceeding recursion_limit raises GraphRecursionError.
    """
    from langgraph.errors import GraphRecursionError
    from langgraph.managed import IsLastStep, RemainingSteps

    class SLoop(TypedDict):
        count: int
        rem_log: Annotated[list[int], add]
        last_log: Annotated[list[bool], add]
        remaining: RemainingSteps
        is_last: IsLastStep

    def step_node(s: SLoop):
        return {
            "count": s["count"] + 1,
            "rem_log": [s["remaining"]],
            "last_log": [s["is_last"]],
        }

    def route_stop_on_last(s: SLoop):
        if s["is_last"] or s["count"] >= 4:
            return END
        return "step_node"

    b = StateGraph(SLoop)
    b.add_node("step_node", step_node)
    b.add_edge(START, "step_node")
    b.add_conditional_edges(
        "step_node", route_stop_on_last, {"step_node": "step_node", END: END}
    )

    native_out = b.compile(checkpointer=InMemorySaver()).invoke(
        {"count": 0, "rem_log": [], "last_log": []},
        {"configurable": {"thread_id": "n-rem"}, "recursion_limit": 5},
    )
    assert native_out["rem_log"] == [4, 3, 2, 1] and native_out["last_log"] == [
        False,
        False,
        False,
        True,
    ]

    g = b.compile(checkpointer=InMemorySaver())
    attach_durability(g, name="b1_rem")
    rcfg = pack({"recursion_limit": 5})
    rep = run_step(
        g,
        LIM,
        {
            "thread": "t-rem",
            "kind": "start",
            "expect": None,
            "input": pack({"count": 0, "rem_log": [], "last_log": []}),
            "run_config": rcfg,
        },
    )
    while rep["status"] != "done":
        rep = run_step(
            g,
            LIM,
            {
                "thread": "t-rem",
                "kind": "step",
                "expect": rep["ckpt"],
                "run_config": rcfg,
            },
        )
    out = read_output(g, "t-rem")
    assert out["rem_log"] == [4, 3, 2, 1] and out["last_log"] == [
        False,
        False,
        False,
        True,
    ]

    # Exceeding recursion_limit=3 when route does not stop at is_last raises
    # GraphRecursionError
    b2 = StateGraph(SLoop)
    b2.add_node("step_node", step_node)
    b2.add_edge(START, "step_node")
    b2.add_conditional_edges(
        "step_node",
        lambda s: END if s["count"] >= 10 else "step_node",
        {"step_node": "step_node", END: END},
    )
    g2 = b2.compile(checkpointer=InMemorySaver())
    attach_durability(g2, name="b1_rec_err")
    rcfg3 = pack({"recursion_limit": 3})
    with pytest.raises(GraphRecursionError, match="Recursion limit of 3 reached"):
        r = run_step(
            g2,
            LIM,
            {
                "thread": "t-rerr",
                "kind": "start",
                "expect": None,
                "input": pack({"count": 0, "rem_log": [], "last_log": []}),
                "run_config": rcfg3,
            },
        )
        while r["status"] != "done":
            r = run_step(
                g2,
                LIM,
                {
                    "thread": "t-rerr",
                    "kind": "step",
                    "expect": r["ckpt"],
                    "run_config": rcfg3,
                },
            )


def test_pydantic_and_dataclass_state_schemas_in_tools():
    """Pydantic BaseModel and @dataclass state schemas are hydrated into schema.

    instances
    for InjectedState and ToolRuntime.state inside @durable_tool."""
    from dataclasses import dataclass, field
    from typing import Any

    from langgraph.prebuilt import InjectedState
    from langgraph.prebuilt.tool_node import ToolRuntime
    from pydantic import BaseModel

    class PydState(BaseModel):
        messages: Annotated[list[AnyMessage], add_messages] = []
        count: int = 0

    @dataclass
    class DcState:
        messages: Annotated[list[AnyMessage], add_messages] = field(
            default_factory=list
        )
        count: int = 0

    @durable_tool
    @tool
    def inspect_state(
        x: int,
        state: Annotated[Any, InjectedState],
        count_field: Annotated[int, InjectedState("count")],
        runtime: ToolRuntime[Any, Any],
    ) -> str:
        """Access state.count via attribute on Pydantic or dataclass state."""
        return (
            f"type={type(state).__name__}|rt={type(runtime.state).__name__}|"
            f"attr={state.count}|field={count_field}|sum={state.count + x}"
        )

    for schema_cls in (PydState, DcState):

        def agent(s):
            if any(isinstance(m, ToolMessage) for m in s.messages):
                return {"messages": [AIMessage(content="done")], "count": s.count + 1}
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "inspect_state",
                                "args": {"x": 5},
                                "id": "c1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ],
                "count": s.count + 10,
            }

        b = StateGraph(schema_cls)
        b.add_node("agent", agent)
        b.add_node("tools", ToolNode([inspect_state]))
        b.add_edge(START, "agent")
        b.add_conditional_edges(
            "agent",
            lambda s: "tools" if s.messages[-1].tool_calls else END,
            {"tools": "tools", END: END},
        )
        b.add_edge("tools", "agent")
        g = b.compile(checkpointer=InMemorySaver())
        run_steps_until_pause(
            g,
            f"t-{schema_cls.__name__}",
            {"messages": [HumanMessage(content="go")], "count": 0},
        )
        out = read_output(g, f"t-{schema_cls.__name__}")
        tmsg = [m.content for m in out["messages"] if isinstance(m, ToolMessage)][0]
        assert tmsg == (
            f"type={schema_cls.__name__}|rt={schema_cls.__name__}|"
            "attr=10|field=10|sum=15"
        )
        assert out["count"] == 11


def test_parallel_tool_nodes_and_conditional_edge_after_parallel_tool_node():
    """Two parallel ToolNodes in the same superstep ('tools_a', 'tools_b') both run as
    durable tool activities and join at 'join'; conditional edge after a parallel
    ToolNode works.
    """

    # Part A: Two parallel ToolNodes joining at 'join'
    class SParTools(TypedDict):
        msgs_a: Annotated[list[AnyMessage], add_messages]
        msgs_b: Annotated[list[AnyMessage], add_messages]
        joined: str

    @durable_tool
    @tool
    def tool_a(x: int) -> str:
        """Tool A."""
        return f"alpha:{x}"

    @durable_tool
    @tool
    def tool_b(y: int) -> str:
        """Tool B."""
        return f"beta:{y}"

    def prep(s: SParTools):
        return {
            "msgs_a": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "tool_a",
                            "args": {"x": 10},
                            "id": "ca",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            "msgs_b": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "tool_b",
                            "args": {"y": 20},
                            "id": "cb",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
        }

    def join_node(s: SParTools):
        ta = [m.content for m in s["msgs_a"] if isinstance(m, ToolMessage)]
        tb = [m.content for m in s["msgs_b"] if isinstance(m, ToolMessage)]
        return {"joined": f"{ta[0]}+{tb[0]}"}

    b = StateGraph(SParTools)
    b.add_node("prep", prep)
    b.add_node("tools_a", ToolNode([tool_a], messages_key="msgs_a"))
    b.add_node("tools_b", ToolNode([tool_b], messages_key="msgs_b"))
    b.add_node("join", join_node)
    b.add_edge(START, "prep")
    b.add_edge("prep", "tools_a")
    b.add_edge("prep", "tools_b")
    b.add_edge(["tools_a", "tools_b"], "join")
    b.add_edge("join", END)
    g = b.compile(checkpointer=InMemorySaver())
    reports = run_steps_until_pause(
        g, "t-par-tn", {"msgs_a": [], "msgs_b": [], "joined": ""}
    )
    assert sum(r["status"] == "tools" for r in reports) == 2
    assert read_output(g, "t-par-tn")["joined"] == "alpha:10+beta:20"

    # Part B: Conditional edge out of a parallel ToolNode (ToolNode + sibling branch)
    class SCondPar(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        log: Annotated[list[str], add]
        route: str

    @durable_tool
    @tool
    def check_status(x: int) -> str:
        """Check status."""
        return "FAIL_STATUS" if x < 0 else "OK_STATUS"

    def start_both(s: SCondPar):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "check_status",
                            "args": {"x": -1},
                            "id": "c1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    def route_after_tools(s: SCondPar):
        last_tool = [m for m in s["messages"] if isinstance(m, ToolMessage)][-1]
        return "error_handler" if "FAIL" in last_tool.content else "normal_finish"

    bc = StateGraph(SCondPar)
    bc.add_node("start_both", start_both)
    bc.add_node("tools", ToolNode([check_status]))
    bc.add_node("sibling", lambda s: {"log": ["sibling_ran"]})
    bc.add_node(
        "error_handler", lambda s: {"route": "error_handler", "log": ["err_handled"]}
    )
    bc.add_node(
        "normal_finish", lambda s: {"route": "normal_finish", "log": ["ok_finished"]}
    )
    bc.add_edge(START, "start_both")
    bc.add_edge("start_both", "tools")
    bc.add_edge("start_both", "sibling")
    bc.add_conditional_edges(
        "tools",
        route_after_tools,
        {"error_handler": "error_handler", "normal_finish": "normal_finish"},
    )
    bc.add_edge("sibling", END)
    bc.add_edge("error_handler", END)
    bc.add_edge("normal_finish", END)
    gc = bc.compile(checkpointer=InMemorySaver())
    run_steps_until_pause(
        gc,
        "t-cond-par",
        {"messages": [HumanMessage(content="go")], "log": [], "route": ""},
    )
    out_c = read_output(gc, "t-cond-par")
    assert out_c["route"] == "error_handler" and sorted(out_c["log"]) == [
        "err_handled",
        "sibling_ran",
    ]


def test_durable_node_static_parallel_fan_out_and_fault_isolation():
    """Verify @durable_node on static parallel fan-out (START -> [branch_a, branch_b.

    branch_c] -> join)
    executes each branch as an isolated run_node activity so a failure in branch_b
    retries ONLY branch_b.
    """

    class ParState(TypedDict):
        seed: int
        results: Annotated[list[str], add]

    calls = {"branch_a": 0, "branch_b": 0, "branch_c": 0}

    @durable_node
    def branch_a(s: ParState):
        calls["branch_a"] += 1
        return {"results": [f"A:{s['seed'] + 1}"]}

    @durable_node(retries=4)
    def branch_b(s: ParState):
        calls["branch_b"] += 1
        if calls["branch_b"] == 1:
            raise RuntimeError("transient branch_b failure")
        return {"results": [f"B:{s['seed'] + 2}"]}

    @durable_node
    def branch_c(s: ParState):
        calls["branch_c"] += 1
        return {"results": [f"C:{s['seed'] + 3}"]}

    def join_node(s: ParState):
        return {"results": ["JOIN:" + ",".join(sorted(s["results"]))]}

    b = StateGraph(ParState)
    b.add_node("branch_a", branch_a)
    b.add_node("branch_b", branch_b)
    b.add_node("branch_c", branch_c)
    b.add_node("join", join_node)
    b.add_edge(START, "branch_a")
    b.add_edge(START, "branch_b")
    b.add_edge(START, "branch_c")
    b.add_edge(["branch_a", "branch_b", "branch_c"], "join")
    b.add_edge("join", END)
    g = b.compile(checkpointer=InMemorySaver())
    attach_durability(g, name="par_nodes")

    rep0 = run_step(
        g,
        LIM,
        {
            "thread": "t-par-node",
            "kind": "start",
            "expect": None,
            "input": pack({"seed": 10, "results": []}),
        },
    )
    assert rep0["status"] == "nodes"
    assert [nt["name"] for nt in rep0["node_tasks"]] == [
        "branch_a",
        "branch_b",
        "branch_c",
    ]
    assert rep0["node_tasks"][1]["retries"] == 4
    # PULL nodes omit "input" from node_tasks so DTS history stays small even when graph
    # state is large
    assert all("input" not in nt for nt in rep0["node_tasks"])

    outputs = {}
    for nt in rep0["node_tasks"]:
        req_n = {
            "thread": "t-par-node",
            "ckpt": rep0["ckpt"],
            "call_id": nt["call_id"],
            "index": nt["index"],
            "task_id": nt["task_id"],
            "name": nt["name"],
        }
        try:
            res_n = run_node(g, req_n)
        except RuntimeError:
            # Retry ONLY failed branch_b!
            res_n = run_node(g, req_n)
        outputs[str(nt["index"])] = res_n["output"]

    assert calls == {"branch_a": 1, "branch_b": 2, "branch_c": 1}

    rep1 = run_step(
        g,
        LIM,
        {
            "thread": "t-par-node",
            "kind": "nodes",
            "expect": rep0["ckpt"],
            "outputs": outputs,
        },
    )
    assert rep1["status"] == "running"
    rep2 = run_step(
        g, LIM, {"thread": "t-par-node", "kind": "step", "expect": rep1["ckpt"]}
    )
    assert rep2["status"] == "done"
    assert read_output(g, "t-par-node")["results"] == [
        "A:11",
        "B:12",
        "C:13",
        "JOIN:A:11,B:12,C:13",
    ]


def test_durable_node_send_map_reduce_interrupt_and_runtime_context():
    """Verify @durable_node with Send() map-reduce fan-out, Command(goto=[Send(...)]),
    interrupt() inside a durable node, and Runtime[Context] + RunnableConfig
    propagation.
    """
    from dataclasses import dataclass

    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime
    from langgraph.types import Send

    @dataclass
    class ResearchCtx:
        org: str

    class MapState(TypedDict):
        topics: list[str]
        findings: Annotated[list[str], add]
        summary: str

    calls = {"A": 0, "B": 0, "C": 0}

    def planner(s: MapState):
        return Command(
            update={"findings": ["planned"]},
            goto=[Send("research_subagent", {"topic": t}) for t in s["topics"]],
        )

    @durable_node(retries=3)
    def research_subagent(
        arg: dict, config: RunnableConfig, runtime: Runtime[ResearchCtx]
    ):
        topic = arg["topic"]
        calls[topic] += 1
        tag = (config.get("configurable") or {}).get("env_tag", "prod")
        if topic == "B" and calls["B"] == 1:
            raise RuntimeError("transient API timeout on topic B")
        if topic == "C":
            human_note = interrupt(f"approve-topic-{topic}")
            return {"findings": [f"{runtime.context.org}:{tag}:{topic}:{human_note}"]}
        return {"findings": [f"{runtime.context.org}:{tag}:{topic}"]}

    def synthesize(s: MapState):
        return {"summary": " | ".join(s["findings"])}

    b = StateGraph(MapState, context_schema=ResearchCtx)
    b.add_node("planner", planner)
    b.add_node("research_subagent", research_subagent)
    b.add_node("synthesize", synthesize)
    b.add_edge(START, "planner")
    b.add_edge("research_subagent", "synthesize")
    b.add_edge("synthesize", END)
    g = b.compile(checkpointer=InMemorySaver())
    attach_durability(g, name="map_reduce")

    run_cfg = pack({"configurable": {"env_tag": "stage"}})
    ctx_blob = pack({"org": "ACME"})
    rep0 = run_step(
        g,
        LIM,
        {
            "thread": "t-send-mr",
            "kind": "start",
            "expect": None,
            "input": pack({"topics": ["A", "B", "C"], "findings": [], "summary": ""}),
            "run_config": run_cfg,
            "context": ctx_blob,
        },
    )
    assert rep0["status"] == "nodes"
    assert len(rep0["node_tasks"]) == 3
    assert [unpack(nt["input"]) for nt in rep0["node_tasks"]] == [
        {"topic": "A"},
        {"topic": "B"},
        {"topic": "C"},
    ]

    outputs = {}
    for nt in rep0["node_tasks"]:
        req_n = {
            "thread": "t-send-mr",
            "ckpt": rep0["ckpt"],
            "call_id": nt["call_id"],
            "index": nt["index"],
            "task_id": nt["task_id"],
            "name": nt["name"],
            "input": nt["input"],
            "run_config": run_cfg,
            "context": ctx_blob,
        }
        try:
            res_n = run_node(g, req_n)
        except RuntimeError:
            res_n = run_node(g, req_n)
        if res_n["status"] == "interrupted":
            assert unpack(res_n["interrupt"]["value"]) == "approve-topic-C"
            res_n = run_node(g, {**req_n, "resumes": [pack("confirmed-C")]})
        outputs[str(nt["index"])] = res_n["output"]

    assert calls == {"A": 1, "B": 2, "C": 2}

    rep1 = run_step(
        g,
        LIM,
        {
            "thread": "t-send-mr",
            "kind": "nodes",
            "expect": rep0["ckpt"],
            "outputs": outputs,
            "run_config": run_cfg,
            "context": ctx_blob,
        },
    )
    assert rep1["status"] == "running"
    rep2 = run_step(
        g,
        LIM,
        {
            "thread": "t-send-mr",
            "kind": "step",
            "expect": rep1["ckpt"],
            "run_config": run_cfg,
            "context": ctx_blob,
        },
    )
    assert rep2["status"] == "done"
    final = read_output(g, "t-send-mr")
    assert (
        final["summary"]
        == "planned | ACME:stage:A | ACME:stage:B | ACME:stage:C:confirmed-C"
    )


def test_graph_versioning_and_live_migration_in_run_step():
    """Verify Registration with version='v2', previous_versions={'v1': g1}, and.

    migrations={('v1', 'v2'): ...}
    executes v1 steps on g1 and migrates checkpoints to g2."""

    class VState(TypedDict):
        val: int
        log: list[str]

    b1 = StateGraph(VState)
    b1.add_node(
        "step1", lambda s: {"val": s["val"] + 1, "log": s.get("log", []) + ["v1:step1"]}
    )
    b1.add_node(
        "old_gate", lambda s: {"log": s["log"] + [f"v1:gate:{interrupt('approve-v1')}"]}
    )
    b1.add_node(
        "step2", lambda s: {"val": s["val"] * 2, "log": s["log"] + ["v1:step2"]}
    )
    b1.add_edge(START, "step1")
    b1.add_edge("step1", "old_gate")
    b1.add_edge("old_gate", "step2")
    b1.add_edge("step2", END)
    g1 = b1.compile()

    b2 = StateGraph(VState)
    b2.add_node(
        "step1", lambda s: {"val": s["val"] + 1, "log": s.get("log", []) + ["v2:step1"]}
    )
    b2.add_node(
        "new_gate", lambda s: {"log": s["log"] + [f"v2:gate:{interrupt('approve-v2')}"]}
    )
    b2.add_node(
        "step2", lambda s: {"val": s["val"] * 10, "log": s["log"] + ["v2:step2"]}
    )
    b2.add_node("audit_v2", lambda s: {"log": s["log"] + ["v2:audit"]})
    b2.add_edge(START, "step1")
    b2.add_edge("step1", "new_gate")
    b2.add_edge("new_gate", "step2")
    b2.add_edge("step2", "audit_v2")
    b2.add_edge("audit_v2", END)
    g2 = b2.compile()

    reg = attach_durability(
        g2,
        name="ver_app",
        version="v2",
        previous_versions={"v1": g1},
        migrations={
            ("v1", "v2"): lambda state, snap, resume_val: {
                "as_node": "new_gate",
                "values": {
                    "val": state["val"] + 10,
                    "log": state["log"] + [f"migrated:{resume_val}"],
                },
            }
        },
        upgrade_on_continue_as_new=True,
    )

    # Start and step to interrupt on v1
    r0 = run_step(
        reg.graph,
        LIM,
        {
            "thread": "t-ver",
            "kind": "start",
            "expect": None,
            "input": pack({"val": 5, "log": []}),
            "version": "v1",
        },
    )
    assert r0["status"] == "running" and r0.get("target_version") == "v2"
    r1 = run_step(
        reg.graph,
        LIM,
        {"thread": "t-ver", "kind": "step", "expect": r0["ckpt"], "version": "v1"},
    )
    assert (
        r1["status"] == "interrupted"
        and unpack(r1["interrupt"]["value"]) == "approve-v1"
    )

    # Resume and migrate from v1 -> v2
    r2 = run_step(
        reg.graph,
        LIM,
        {
            "thread": "t-ver",
            "kind": "resume",
            "expect": r1["ckpt"],
            "resume": pack("ok-v2"),
            "migrate_from": "v1",
            "version": "v2",
        },
    )
    assert r2["status"] == "running"
    r3 = run_step(
        reg.graph,
        LIM,
        {"thread": "t-ver", "kind": "step", "expect": r2["ckpt"], "version": "v2"},
    )
    assert r3["status"] == "done"
    final = read_output(reg.graph, "t-ver")
    assert final == {
        "val": 160,
        "log": ["v1:step1", "migrated:ok-v2", "v2:step2", "v2:audit"],
    }


def test_no_attach_alias_unknown_version_and_reattach():
    """Verify legacy attach alias is removed, unknown graph version raises RunError,
    and re-attaching a graph with new options or custom previous_versions checkpointer
    works.
    """
    import langchain_azure_compute.graph as pkg

    assert not hasattr(pkg, "attach")
    assert "attach" not in pkg.__all__

    class S(TypedDict):
        x: int

    b1 = StateGraph(S)
    b1.add_node("a", lambda s: {"x": s["x"] + 1})
    b1.add_edge(START, "a")
    b1.add_edge("a", END)
    cp_v1 = InMemorySaver()
    g1 = b1.compile(checkpointer=cp_v1)

    b2 = StateGraph(S)
    b2.add_node("a", lambda s: {"x": s["x"] + 2})
    b2.add_node("b", lambda s: {"x": s["x"] * 10})
    b2.add_edge(START, "a")
    b2.add_edge("a", "b")
    b2.add_edge("b", END)
    cp_v2 = InMemorySaver()
    g2 = b2.compile(checkpointer=cp_v2, interrupt_before=["b"])

    reg = attach_durability(
        g2,
        name="ver_err",
        version="v2",
        previous_versions={"v1": g1},
        durable_nodes=False,
    )
    # g1's existing checkpointer is preserved (not overwritten by g2's checkpointer)
    assert g1.checkpointer is cp_v1
    assert g2.checkpointer is cp_v2

    with pytest.raises(
        pkg.RunError, match="unknown graph version 'v99' for graph 'ver_err'"
    ):
        reg._resolve_graph("v99")

    with pytest.raises(pkg.RunError, match="unknown graph version 'v99'"):
        run_step(
            reg.graph,
            LIM,
            {
                "thread": "t-unk",
                "kind": "start",
                "expect": None,
                "input": pack({"x": 1}),
                "version": "v99",
            },
        )

    # Re-attaching g2 with durable_nodes=True applies the new node config and preserves
    # original interrupt_before=['b']
    reg2 = attach_durability(g2, name="ver_err_2", durable_nodes=True)
    assert reg2.node_config.default_durable is True
    assert getattr(g2, "_durable_node_config").default_durable is True
    assert getattr(g2, "_durable_user_interrupt_before") == ("b",)


def test_migration_no_reducer_duplication_and_idempotency():
    """Verify _apply_version_migration never doubles Annotated[list, operator.add].

    channels when unchanged
    or when as_node remap is used with no values, and that kind='migrate' is idempotent
    on redelivery.
    """

    class MigState(TypedDict):
        x: int
        log: Annotated[list[str], add]

    b1 = StateGraph(MigState)
    b1.add_node("prep", lambda s: {"x": s["x"] + 1, "log": ["v1:prep"]})
    b1.add_node("old_gate", lambda s: {"log": [f"v1:gate:{interrupt('wait')}"]})
    b1.add_node("finish", lambda s: {"x": s["x"] * 2, "log": ["v1:finish"]})
    b1.add_edge(START, "prep")
    b1.add_edge("prep", "old_gate")
    b1.add_edge("old_gate", "finish")
    b1.add_edge("finish", END)

    b2 = StateGraph(MigState)
    b2.add_node("prep", lambda s: {"x": s["x"] + 1, "log": ["v2:prep"]})
    b2.add_node("new_gate", lambda s: {"log": [f"v2:gate:{interrupt('wait2')}"]})
    b2.add_node("finish", lambda s: {"x": s["x"] * 10, "log": ["v2:finish"]})
    b2.add_edge(START, "prep")
    b2.add_edge("prep", "new_gate")
    b2.add_edge("new_gate", "finish")
    b2.add_edge("finish", END)

    # Case 1: as_node upgrade with NO values and NO migration function -> log must NOT
    # be doubled!
    saver1 = InMemorySaver()
    g1_a, g2_a = b1.compile(checkpointer=saver1), b2.compile(checkpointer=saver1)
    reg_a = attach_durability(
        g2_a, name="mig_nodup_a", version="v2", previous_versions={"v1": g1_a}
    )
    r0 = run_step(
        reg_a.graph,
        LIM,
        {
            "thread": "t-m1",
            "kind": "start",
            "expect": None,
            "input": pack({"x": 4, "log": ["init"]}),
            "version": "v1",
        },
    )
    r1 = run_step(
        reg_a.graph,
        LIM,
        {"thread": "t-m1", "kind": "step", "expect": r0["ckpt"], "version": "v1"},
    )
    assert r1["status"] == "interrupted"
    r2 = run_step(
        reg_a.graph,
        LIM,
        {
            "thread": "t-m1",
            "kind": "resume",
            "expect": r1["ckpt"],
            "resume": pack(None),
            "migrate_from": "v1",
            "version": "v2",
            "as_node": "new_gate",
        },
    )
    assert r2["status"] == "done"
    assert read_output(reg_a.graph, "t-m1") == {
        "x": 50,
        "log": ["init", "v1:prep", "v2:finish"],
    }

    # Case 2: migration function that returns mutated full state dict + kind='migrate'
    # idempotency on redelivery
    mig_calls = 0

    def mig_fn(state: MigState):
        nonlocal mig_calls
        mig_calls += 1
        state["x"] = state["x"] + 100
        state["log"] = state["log"] + ["migrated"]
        return state

    saver2 = InMemorySaver()
    g1_b, g2_b = b1.compile(checkpointer=saver2), b2.compile(checkpointer=saver2)
    reg_b = attach_durability(
        g2_b,
        name="mig_idem",
        version="v2",
        previous_versions={"v1": g1_b},
        migrations={("v1", "v2"): mig_fn},
    )
    rb0 = run_step(
        reg_b.graph,
        LIM,
        {
            "thread": "t-m2",
            "kind": "start",
            "expect": None,
            "input": pack({"x": 5, "log": ["seed"]}),
            "version": "v1",
        },
    )
    mig_req = {
        "thread": "t-m2",
        "kind": "migrate",
        "expect": rb0["ckpt"],
        "migrate_from": "v1",
        "version": "v2",
    }
    m1 = run_step(reg_b.graph, LIM, mig_req)
    assert m1["replayed"] is False and mig_calls == 1 and m1["ckpt"] != rb0["ckpt"]
    # Redelivering the exact same kind='migrate' request returns replayed=True without
    # calling mig_fn again!
    m2 = run_step(reg_b.graph, LIM, mig_req)
    assert m2["replayed"] is True and mig_calls == 1 and m2["ckpt"] == m1["ckpt"]
    assert read_output(reg_b.graph, "t-m2") == {
        "x": 106,
        "log": ["seed", "v1:prep", "migrated"],
    }


def test_lastvalue_list_reducer_and_static_breakpoint_with_durable_node():
    """Verify _put_task_writes._read uses LastValue[list[int]] (replacing rather than.

    concatenating)
    when evaluating conditional edges after a @durable_node, and static breakpoint
    resume intercepts @durable_node.
    """

    class LVState(TypedDict):
        nums: list[int]
        route: str

    @durable_node
    def replace_nums(s: LVState):
        return {"nums": [10, 20]}

    def router(s: LVState):
        # If nums was wrongly concatenated ([1, 2, 10, 20]), len would be 4 instead of
        # 2!
        return "exact_two" if s["nums"] == [10, 20] else f"wrong_{len(s['nums'])}"

    b = StateGraph(LVState)
    b.add_node("prep", lambda s: {"nums": [1, 2]})
    b.add_node("replace_nums", replace_nums)
    b.add_node("exact_two", lambda s: {"route": "replaced_cleanly"})
    b.add_node("wrong_4", lambda s: {"route": "concatenated_bug"})
    b.add_edge(START, "prep")
    b.add_edge("prep", "replace_nums")
    b.add_conditional_edges(
        "replace_nums", router, {"exact_two": "exact_two", "wrong_4": "wrong_4"}
    )
    b.add_edge("exact_two", END)
    b.add_edge("wrong_4", END)

    g = b.compile(checkpointer=InMemorySaver(), interrupt_before=["prep"])
    attach_durability(g, name="lv_static_dn")

    # 1. Start hits static interrupt_before=['prep']
    r0 = run_step(
        g,
        LIM,
        {
            "thread": "t-lv",
            "kind": "start",
            "expect": None,
            "input": pack({"nums": [0], "route": ""}),
        },
    )
    assert r0["status"] == "interrupted" and r0["interrupt"].get("static") is True

    # 2. Resume from static breakpoint runs 'prep' and MUST intercept @durable_node
    # 'replace_nums' as status='nodes'!
    r1 = run_step(
        g,
        LIM,
        {
            "thread": "t-lv",
            "kind": "resume",
            "expect": r0["ckpt"],
            "resume": pack(None),
        },
    )
    assert r1["status"] == "nodes" and r1["node_tasks"][0]["name"] == "replace_nums"

    nt = r1["node_tasks"][0]
    n_res = run_node(
        g,
        {
            "thread": "t-lv",
            "ckpt": r1["ckpt"],
            "call_id": nt["call_id"],
            "index": nt["index"],
            "task_id": nt["task_id"],
            "name": nt["name"],
        },
    )
    r2 = run_step(
        g,
        LIM,
        {
            "thread": "t-lv",
            "kind": "nodes",
            "expect": r1["ckpt"],
            "outputs": [n_res["output"]],
        },
    )
    assert r2["status"] == "running"
    r3 = run_step(g, LIM, {"thread": "t-lv", "kind": "step", "expect": r2["ckpt"]})
    assert r3["status"] == "done"
    assert read_output(g, "t-lv") == {"nums": [10, 20], "route": "replaced_cleanly"}


def test_inline_bundle_cap_and_single_arg_run_resume():
    """Verify _INLINE_THREAD_BUNDLES is bounded at MAX_INLINE_THREAD_BUNDLES (256) and.

    Run.resume(val)
    looks up the active pending interrupt automatically when called with 1 positional
    argument.
    """
    import json
    from types import SimpleNamespace

    from langchain_azure_compute.graph.runner import (
        _INLINE_THREAD_BUNDLES,
        MAX_INLINE_THREAD_BUNDLES,
        _export_checkpoint_bundle,
    )

    class S(TypedDict):
        x: int

    b = StateGraph(S)
    b.add_node("a", lambda s: {"x": s["x"] + 1})
    b.add_edge(START, "a")
    b.add_edge("a", END)
    g = b.compile(checkpointer=InMemorySaver())
    reg = attach_durability(g, name="cap_test")

    _INLINE_THREAD_BUNDLES.clear()
    g.invoke({"x": 1}, {"configurable": {"thread_id": "t-0"}})
    for i in range(MAX_INLINE_THREAD_BUNDLES + 20):
        tid = f"t-{i}"
        g.checkpointer.storage[tid] = g.checkpointer.storage["t-0"]
        _export_checkpoint_bundle(g, tid)
    assert len(_INLINE_THREAD_BUNDLES) == MAX_INLINE_THREAD_BUNDLES
    assert "t-0" not in _INLINE_THREAD_BUNDLES

    events_raised = []

    class FakeSdk:
        def get_orchestration_state(self, iid, fetch_payloads=True):
            cs = json.dumps(
                {
                    "pending": {"id": "int-abc", "value": pack("ask?")},
                    "interrupts": [{"id": "int-abc", "value": pack("ask?")}],
                    "step": 1,
                    "ckpt": "c1",
                }
            )
            return SimpleNamespace(serialized_custom_status=cs)

        def raise_orchestration_event(self, iid, event_name, data=None):
            events_raised.append((iid, event_name, data))

    run_handle = reg.client(FakeSdk()).attach_run("inst-1", "thread-1")
    run_handle.resume("approved-single-arg")
    assert len(events_raised) == 1
    assert events_raised[0][1] == "resume:int-abc"
    assert unpack(events_raised[0][2]["value"]) == "approved-single-arg"


def test_sdk_options_emulator_hosted_azure_dts_and_connection_strings(monkeypatch):
    """Verify sdk_options() supports local emulator, hosted Azure DTS with.

    DefaultAzureCredential / token_credential,
    and DTS connection strings (Endpoint=...;TaskHub=...;Authentication=...)."""
    from types import SimpleNamespace

    from langchain_azure_compute.graph import sdk_options

    monkeypatch.delenv("DURABLE_TASK_SCHEDULER_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("DTS_ENDPOINT", raising=False)
    monkeypatch.delenv("DTS_EMULATOR", raising=False)
    monkeypatch.delenv("DTS_TASKHUB", raising=False)

    # 1. Local emulator ("127.0.0.1:8080" and "http://localhost:8080")
    opts_emu = sdk_options("127.0.0.1:8080")
    assert opts_emu["host_address"] == "127.0.0.1:8080"
    assert opts_emu["taskhub"] == "default"
    assert opts_emu["token_credential"] is None
    assert opts_emu["secure_channel"] is False

    opts_http_local = sdk_options("http://localhost:8080/")
    assert opts_http_local["host_address"] == "localhost:8080"
    assert opts_http_local["secure_channel"] is False
    assert opts_http_local["token_credential"] is None

    # 2. Hosted Azure DTS endpoint + token_credential + taskhub
    fake_cred = SimpleNamespace(
        get_token=lambda *a, **k: SimpleNamespace(token="tok", expires_on=9999999999)
    )
    opts_hosted = sdk_options(
        "https://my-dts.westus2.durabletask.io/",
        taskhub="prod-hub",
        token_credential=fake_cred,
    )
    assert opts_hosted["host_address"] == "my-dts.westus2.durabletask.io"
    assert opts_hosted["taskhub"] == "prod-hub"
    assert opts_hosted["token_credential"] is fake_cred
    assert opts_hosted["secure_channel"] is True

    # 3. DTS Connection String (Authentication=None, with explicit token_credential, and
    # Authentication=DefaultAzure)
    conn_none = "Endpoint=https://my-dts.westus2.durabletask.io;TaskHub=myhub;Authentication=None"
    opts_cs_none = sdk_options(connection_string=conn_none)
    assert opts_cs_none["host_address"] == "my-dts.westus2.durabletask.io"
    assert opts_cs_none["taskhub"] == "myhub"
    assert opts_cs_none["token_credential"] is None
    assert opts_cs_none["secure_channel"] is True

    # Connection string passed as first positional argument + explicit token_credential
    # override
    opts_cs_pos = sdk_options(conn_none, token_credential=fake_cred)
    assert opts_cs_pos["host_address"] == "my-dts.westus2.durabletask.io"
    assert opts_cs_pos["taskhub"] == "myhub"
    assert opts_cs_pos["token_credential"] is fake_cred
    assert opts_cs_pos["secure_channel"] is True

    # Emulator connection string with http://localhost:8080 and Authentication=None
    opts_cs_emu = sdk_options(
        "Endpoint=http://localhost:8080;TaskHub=localhub;Authentication=None"
    )
    assert opts_cs_emu["host_address"] == "localhost:8080"
    assert opts_cs_emu["taskhub"] == "localhub"
    assert opts_cs_emu["token_credential"] is None
    assert opts_cs_emu["secure_channel"] is False

    # Connection string with Authentication=DefaultAzure instantiates
    # DefaultAzureCredential
    opts_cs_aad = sdk_options(
        connection_string="Endpoint=https://my-dts.westus2.durabletask.io;TaskHub=aadhub;Authentication=DefaultAzure"
    )
    assert opts_cs_aad["host_address"] == "my-dts.westus2.durabletask.io"
    assert opts_cs_aad["taskhub"] == "aadhub"
    assert opts_cs_aad["secure_channel"] is True
    assert opts_cs_aad["token_credential"] is not None
    assert type(opts_cs_aad["token_credential"]).__name__ == "DefaultAzureCredential"

    # 4. Non-localhost emulator hostnames (Docker Compose / k8s) default
    # secure_channel=False,
    # while *.durabletask.io hostnames default secure_channel=True
    assert sdk_options("dts-emulator:8080")["secure_channel"] is False
    assert sdk_options("host.docker.internal:8080")["secure_channel"] is False
    assert sdk_options("my-sched.centralus.durabletask.io")["secure_channel"] is True

    # 5. Strict Authentication= validation raises ValueError on unknown modes
    with pytest.raises(ValueError, match="unsupported Authentication mode 'Bogus'"):
        sdk_options(
            connection_string="Endpoint=https://x.durabletask.io;TaskHub=h;Authentication=Bogus"
        )

    # 6. Precedence: DTS_EMULATOR takes precedence over
    # DURABLE_TASK_SCHEDULER_CONNECTION_STRING in env
    monkeypatch.setenv("DTS_EMULATOR", "127.0.0.1:8080")
    monkeypatch.setenv(
        "DURABLE_TASK_SCHEDULER_CONNECTION_STRING",
        "Endpoint=https://my-dts.westus2.durabletask.io;TaskHub=envhub;Authentication=DefaultAzure",
    )
    opts_prec = sdk_options()
    assert opts_prec["host_address"] == "127.0.0.1:8080"
    assert opts_prec["secure_channel"] is False
    assert opts_prec["token_credential"] is None


def test_migration_shrinking_list_reducer_and_resume_crash_window():
    """Verify:
    1. A migration that filters/rewrites/shrinks an Annotated[list[str], operator.add]
       channel
       (['old1', 'old2'] -> ['filtered']) overwrites the channel to ['filtered'] instead
       of appending.
    2. Resume + Version Upgrade crash window: if a worker crashes after
       _apply_version_migration
       but before _invoke(graph, cmd) finishes during run_step(kind='resume'), retrying
       the step
       completes the resume instead of returning replayed=True.
    """

    class FilterState(TypedDict):
        x: int
        log: Annotated[list[str], add]

    b1 = StateGraph(FilterState)
    b1.add_node("prep", lambda s: {"x": s["x"] + 1, "log": ["old1", "old2"]})
    b1.add_node("gate", lambda s: {"log": [f"gate:{interrupt('wait?')}"]})
    b1.add_node("finish", lambda s: {"x": s["x"] * 10, "log": ["done"]})
    b1.add_edge(START, "prep")
    b1.add_edge("prep", "gate")
    b1.add_edge("gate", "finish")
    b1.add_edge("finish", END)

    b2 = StateGraph(FilterState)
    b2.add_node("prep", lambda s: {"x": s["x"] + 1, "log": ["old1", "old2"]})
    b2.add_node("gate", lambda s: {"log": [f"gate:{interrupt('wait?')}"]})
    b2.add_node("finish", lambda s: {"x": s["x"] * 10, "log": ["done"]})
    b2.add_edge(START, "prep")
    b2.add_edge("prep", "gate")
    b2.add_edge("gate", "finish")
    b2.add_edge("finish", END)

    saver = InMemorySaver()
    g1, g2 = b1.compile(checkpointer=saver), b2.compile(checkpointer=saver)
    reg = attach_durability(
        g2,
        name="mig_filter",
        version="v2",
        previous_versions={"v1": g1},
        migrations={("v1", "v2"): lambda state: {"log": ["filtered"]}},
    )

    r0 = run_step(
        reg.graph,
        LIM,
        {
            "thread": "t-filt",
            "kind": "start",
            "expect": None,
            "input": pack({"x": 1, "log": []}),
            "version": "v1",
        },
    )
    r1 = run_step(
        reg.graph,
        LIM,
        {"thread": "t-filt", "kind": "step", "expect": r0["ckpt"], "version": "v1"},
    )
    assert r1["status"] == "interrupted"
    assert read_output(reg.graph, "t-filt")["log"] == ["old1", "old2"]

    # Simulate crash window: migration runs first, then worker crashes before
    # _invoke(graph, cmd) runs!
    from langchain_azure_compute.graph.runner import _apply_version_migration

    resume_req = {
        "thread": "t-filt",
        "kind": "resume",
        "expect": r1["ckpt"],
        "resume": pack("ok"),
        "migrate_from": "v1",
        "version": "v2",
    }
    _apply_version_migration(reg.graph, g2, "t-filt", "v1", "v2", request=resume_req)
    # Verify log was overwritten to ['filtered'], NOT ['old1', 'old2', 'filtered']!
    assert read_output(reg.graph, "t-filt")["log"] == ["filtered"]

    # Now retry run_step(kind='resume') with original expect=r1['ckpt']:
    # Must NOT return replayed=True; must execute the resume and finish the node!
    r2 = run_step(reg.graph, LIM, resume_req)
    assert r2["replayed"] is False
    r3 = run_step(
        reg.graph,
        LIM,
        {"thread": "t-filt", "kind": "step", "expect": r2["ckpt"], "version": "v2"},
    )
    assert r3["status"] == "done"
    assert read_output(reg.graph, "t-filt") == {
        "x": 20,
        "log": ["filtered", "gate:ok", "done"],
    }


def test_strict_run_resume_and_wait_pending_terminal_detection():
    """Verify strict run.resume() validation (no pending interrupt -> RunError.

    multiple pending with
    single value -> ValueError, unknown interrupt_id -> ValueError) and wait_pending()
    terminal return.
    """
    import json
    from types import SimpleNamespace

    from langchain_azure_compute.graph import PendingInterrupt, RunError

    class S(TypedDict):
        x: int

    b = StateGraph(S)
    b.add_node("a", lambda s: {"x": s["x"] + 1})
    b.add_edge(START, "a")
    b.add_edge("a", END)
    reg = attach_durability(b.compile(checkpointer=InMemorySaver()), name="strict_run")

    state_holder = {
        "status": "RUNNING",
        "cs": None,
        "out": json.dumps({"gen": 0, "steps": 1, "output": pack({"x": 2})}),
    }
    events = []

    class FakeSdk:
        def get_orchestration_state(self, iid, fetch_payloads=True):
            return SimpleNamespace(
                runtime_status=SimpleNamespace(name=state_holder["status"]),
                serialized_custom_status=state_holder["cs"],
                serialized_output=state_holder["out"],
                failure_details=SimpleNamespace(message="boom")
                if state_holder["status"] == "FAILED"
                else None,
            )

        def wait_for_orchestration_completion(
            self, iid, timeout=60, fetch_payloads=True
        ):
            return self.get_orchestration_state(iid, fetch_payloads=fetch_payloads)

        def raise_orchestration_event(self, iid, name, data=None):
            events.append((iid, name, data))

    run_h = reg.client(FakeSdk()).attach_run("i-1", "t-1")

    # 1. No pending interrupt -> RunError
    with pytest.raises(RunError, match="no pending interrupt to resume"):
        run_h.resume("val")
    with pytest.raises(RunError, match="no pending interrupt to resume"):
        run_h.resume()

    # 1b. Single pending interrupt with dict payload: run.resume({"approved": True}) and
    # run.resume({})
    # treat the dict as the resume value itself unless keys match pending interrupt_ids
    state_holder["cs"] = json.dumps(
        {
            "pending": {"id": "int-1", "value": pack("q1")},
            "interrupts": [{"id": "int-1", "value": pack("q1")}],
            "step": 1,
            "ckpt": "c1",
        }
    )
    run_h.resume({"approved": True})
    assert events[-1][1] == "resume:int-1" and unpack(events[-1][2]["value"]) == {
        "approved": True
    }
    run_h.resume({})
    assert events[-1][1] == "resume:int-1" and unpack(events[-1][2]["value"]) == {}
    run_h.resume(Command(resume={"approved": False}))
    assert events[-1][1] == "resume:int-1" and unpack(events[-1][2]["value"]) == {
        "approved": False
    }
    run_h.resume({"int-1": "explicit-map-val"})
    assert (
        events[-1][1] == "resume:int-1"
        and unpack(events[-1][2]["value"]) == "explicit-map-val"
    )
    events.clear()

    # 2. Multiple parallel interrupts pending -> single scalar or non-matching dict
    # raises ValueError; unknown ID raises ValueError
    state_holder["cs"] = json.dumps(
        {
            "pending": {"id": "int-1", "value": pack("q1")},
            "interrupts": [
                {"id": "int-1", "value": pack("q1")},
                {"id": "int-2", "value": pack("q2")},
            ],
            "step": 1,
            "ckpt": "c1",
        }
    )
    with pytest.raises(ValueError, match="multiple interrupts are pending"):
        run_h.resume("ambiguous-scalar")
    with pytest.raises(ValueError, match="multiple interrupts are pending"):
        run_h.resume({"bad_id": 1})
    with pytest.raises(ValueError, match="multiple interrupts are pending"):
        run_h.resume({})
    with pytest.raises(ValueError, match="unknown interrupt_id 'wrong-id'"):
        run_h.resume({PendingInterrupt("wrong-id", "q"): "v1"})
    with pytest.raises(ValueError, match="unknown interrupt_id 'wrong-id'"):
        run_h.resume(PendingInterrupt("wrong-id", "q"), "v1")
    with pytest.raises(ValueError, match="unknown interrupt_id 'wrong-id'"):
        run_h.resume("wrong-id", "v1")

    # Valid parallel map resume succeeds
    run_h.resume({"int-1": "a1", "int-2": "a2"})
    assert [e[1] for e in events] == ["resume:int-1", "resume:int-2"]

    # 3. wait_pending() returns None immediately when orchestration is COMPLETED without
    # an interrupt
    state_holder["cs"] = None
    state_holder["status"] = "COMPLETED"
    assert run_h.wait_pending(timeout=0.5) is None

    # And raises RunError immediately when FAILED
    state_holder["status"] = "FAILED"
    with pytest.raises(RunError, match="boom"):
        run_h.wait_pending(timeout=0.5)


def test_azure_functions_http_routes_and_server_with_real_http_requests():
    """Unit test Azure Functions v2 HTTP routes (`POST /start`, `GET /status/{id}`,
    `POST /respond/{id}/{interrupt_id}`, `POST /respond/{id}`, `GET /health`) with real
    `azure.functions.HttpRequest` objects and `FunctionAppHttpServer`."""
    import json
    import urllib.error
    import urllib.request
    from types import SimpleNamespace

    import azure.functions as func

    from langchain_azure_compute.graph import (
        FunctionAppHttpServer,
        create_function_app,
        dispatch_function_request,
    )

    class S(TypedDict):
        expense_id: str
        approved: bool

    b = StateGraph(S)
    b.add_node(
        "review",
        lambda s: {"approved": bool(interrupt({"expense_id": s["expense_id"]}))},
    )
    b.add_edge(START, "review")
    b.add_edge("review", END)
    reg = attach_durability(
        b.compile(checkpointer=InMemorySaver()), name="expense_approval"
    )

    scheduled: dict[str, dict[str, Any]] = {}
    raised_events: list[tuple[str, str, dict[str, Any]]] = []
    orch_states: dict[str, Any] = {}

    class FakeDtsSdk:
        def schedule_new_orchestration(self, name, input=None, instance_id=None):
            scheduled[instance_id] = {"name": name, "input": input}
            orch_states[instance_id] = SimpleNamespace(
                runtime_status=SimpleNamespace(name="RUNNING"),
                serialized_input=json.dumps(input),
                serialized_custom_status=json.dumps(
                    {
                        "pending": {
                            "id": "intr-exp-501",
                            "value": pack(
                                {"expense_id": "EXP-501", "prompt": "Approve?"}
                            ),
                        },
                        "interrupts": [
                            {
                                "id": "intr-exp-501",
                                "value": pack(
                                    {"expense_id": "EXP-501", "prompt": "Approve?"}
                                ),
                            }
                        ],
                        "step": 2,
                        "ckpt": "ckpt-501",
                    }
                ),
                serialized_output=None,
                failure_details=None,
            )
            return instance_id

        def get_orchestration_state(self, instance_id, fetch_payloads=False):
            return orch_states.get(instance_id)

        def raise_orchestration_event(self, instance_id, event_name, data=None):
            raised_events.append((instance_id, event_name, data))

        def wait_for_orchestration_completion(
            self, instance_id, timeout=60, fetch_payloads=True
        ):
            return orch_states.get(instance_id)

    fake_sdk = FakeDtsSdk()
    app = create_function_app(reg, fake_sdk, start_worker=False)

    # 1. GET /api/health with real azure.functions.HttpRequest
    health_req = func.HttpRequest(
        method="GET",
        url="http://localhost:7071/api/health",
        headers={},
        params={},
        route_params={},
        body=b"",
    )
    health_resp = dispatch_function_request(app, health_req)
    assert health_resp.status_code == 200
    assert json.loads(health_resp.get_body()) == {
        "status": "healthy",
        "graphs": ["expense_approval"],
    }

    # 2. POST /api/graphs/expense_approval/start (direct input body)
    start_req = func.HttpRequest(
        method="POST",
        url="http://localhost:7071/api/graphs/expense_approval/start?instance_id=inst-501&thread_id=thr-501",
        headers={"Content-Type": "application/json"},
        params={},
        route_params={},
        body=json.dumps({"expense_id": "EXP-501", "approved": False}).encode("utf-8"),
    )
    start_resp = dispatch_function_request(app, start_req)
    assert start_resp.status_code == 202
    start_data = json.loads(start_resp.get_body())
    assert start_data == {
        "instanceId": "inst-501",
        "threadId": "thr-501",
        "statusQueryGetUri": "http://localhost:7071/api/graphs/expense_approval/status/inst-501",
        "respondUri": "http://localhost:7071/api/graphs/expense_approval/respond/inst-501/{interruptId}",
    }

    # 3. GET /api/graphs/expense_approval/status/inst-501 while paused at interrupt()
    status_req = func.HttpRequest(
        method="GET",
        url="http://localhost:7071/api/graphs/expense_approval/status/inst-501",
        headers={},
        params={},
        route_params={},
        body=b"",
    )
    status_resp = dispatch_function_request(app, status_req)
    assert status_resp.status_code == 200
    status_data = json.loads(status_resp.get_body())
    assert status_data["instanceId"] == "inst-501"
    assert status_data["threadId"] == "thr-501"
    assert status_data["runtimeStatus"] == "RUNNING"
    assert status_data["output"] is None
    assert status_data["pendingInterrupts"] == [
        {
            "interruptId": "intr-exp-501",
            "value": {"expense_id": "EXP-501", "prompt": "Approve?"},
            "step": 2,
            "ckpt": "ckpt-501",
            "respondUrl": "http://localhost:7071/api/graphs/expense_approval/respond/inst-501/intr-exp-501",
        }
    ]

    # 4. POST /api/graphs/expense_approval/respond/inst-501/intr-exp-501 with human
    # response
    respond_req = func.HttpRequest(
        method="POST",
        url="http://localhost:7071/api/graphs/expense_approval/respond/inst-501/intr-exp-501",
        headers={"Content-Type": "application/json"},
        params={},
        route_params={},
        body=json.dumps({"approved": True, "reviewer": "alice"}).encode("utf-8"),
    )
    respond_resp = dispatch_function_request(app, respond_req)
    assert respond_resp.status_code == 200
    respond_data = json.loads(respond_resp.get_body())
    assert respond_data["instanceId"] == "inst-501"
    assert respond_data["interruptId"] == "intr-exp-501"
    assert respond_data["resumed"] is True
    assert len(raised_events) == 1
    assert raised_events[0][0] == "inst-501"
    assert raised_events[0][1] == "resume:intr-exp-501"
    assert unpack(raised_events[0][2]["value"]) == {
        "approved": True,
        "reviewer": "alice",
    }

    # 5. Transition to COMPLETED and verify GET /status returns final output and POST
    # /respond returns 409
    orch_states["inst-501"] = SimpleNamespace(
        runtime_status=SimpleNamespace(name="COMPLETED"),
        serialized_input=orch_states["inst-501"].serialized_input,
        serialized_custom_status=None,
        serialized_output=json.dumps(
            {
                "status": "done",
                "thread": "thr-501",
                "gen": 0,
                "steps": 3,
                "ckpt": "ckpt-502",
                "output": pack({"expense_id": "EXP-501", "approved": True}),
            }
        ),
        failure_details=None,
    )
    done_resp = dispatch_function_request(app, status_req)
    assert done_resp.status_code == 200
    done_data = json.loads(done_resp.get_body())
    assert done_data["runtimeStatus"] == "COMPLETED"
    assert done_data["pendingInterrupts"] == []
    assert done_data["output"] == {"expense_id": "EXP-501", "approved": True}

    # Calling POST /respond when no interrupt is pending returns 409 Conflict
    conflict_resp = dispatch_function_request(app, respond_req)
    assert conflict_resp.status_code == 409

    # Calling GET /status on unknown instance returns 404
    missing_req = func.HttpRequest(
        method="GET",
        url="http://localhost:7071/api/graphs/expense_approval/status/does-not-exist",
        headers={},
        params={},
        route_params={},
        body=b"",
    )
    assert dispatch_function_request(app, missing_req).status_code == 404

    # 6. Verify FunctionAppHttpServer over real TCP HTTP +
    # Registration.register_function_routes
    app2 = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
    reg.register_function_routes(app2, fake_sdk)
    with FunctionAppHttpServer(app2) as srv:
        with urllib.request.urlopen(f"{srv.base_url}/api/health", timeout=5) as r:
            assert r.status == 200
            assert json.loads(r.read().decode("utf-8")) == {
                "status": "healthy",
                "graphs": ["expense_approval"],
            }


def test_push_toolnode_parallel_command_and_toolmessage_commit():
    """Verify multi-tool `Send("tools", ...)` (`__pregel_push`) batches mixing.

    `Command(update=...)`
    and `ToolMessage` outputs (the pattern emitted by `langchain.agents.create_agent`,
    `deepagents.create_deep_agent`, and `create_react_agent(version='v2')`) commit all
    `Send` tasks
    in one step without leaving sibling `Send` tasks pending."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
    from langchain_core.tools import InjectedToolCallId, tool
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, create_react_agent
    from langgraph.types import Command

    class AgentFilesState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        files: dict[str, str]
        remaining_steps: int

    class FakeToolCapableModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    @durable_tool(retries=3)
    @tool
    def write_virtual_file(
        file_path: str,
        content: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Write a file into state['files'] via Command(update=...)."""
        return Command(
            update={
                "files": {file_path: content},
                "messages": [
                    ToolMessage(
                        content=f"Wrote {file_path}",
                        name="write_virtual_file",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    @durable_tool(retries=3)
    @tool
    def lookup_kb(query: str) -> str:
        """Search KB."""
        return f"KB:{query}"

    model = FakeToolCapableModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_virtual_file",
                            "args": {"file_path": "/plan.md", "content": "step 1"},
                            "id": "call_wf",
                            "type": "tool_call",
                        },
                        {
                            "name": "lookup_kb",
                            "args": {"query": "durable task"},
                            "id": "call_kb",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="Completed both write_virtual_file and lookup_kb!"),
            ]
        )
    )

    graph = create_react_agent(
        model,
        tools=ToolNode([write_virtual_file, lookup_kb]),
        state_schema=AgentFilesState,
        version="v2",
    )
    reg = attach_durability(graph, name="push_toolnode_v2")

    inp = pack(
        {"messages": [HumanMessage(content="Write plan and search KB")], "files": {}}
    )
    r1 = run_step(
        reg.graph,
        LIM,
        {"thread": "t-push-v2", "kind": "start", "expect": None, "input": inp},
        tool_config=reg.tool_config,
        node_config=reg.node_config,
    )
    assert r1["status"] == "tools"
    assert [tc["name"] for tc in r1["tool_calls"]] == [
        "write_virtual_file",
        "lookup_kb",
    ]

    outs = [
        run_tool(
            reg.graph,
            {
                "thread": "t-push-v2",
                "ckpt": r1["ckpt"],
                "node": r1["node"],
                "call_id": tc["call_id"],
                "index": tc["index"],
                "call": tc["call"],
                "checkpoint_bundle": r1.get("checkpoint_bundle"),
            },
        )["output"]
        for tc in r1["tool_calls"]
    ]

    r2 = run_step(
        reg.graph,
        LIM,
        {
            "thread": "t-push-v2",
            "kind": "tools",
            "expect": r1["ckpt"],
            "node": r1["node"],
            "outputs": outs,
            "checkpoint_bundle": r1.get("checkpoint_bundle"),
        },
        tool_config=reg.tool_config,
        node_config=reg.node_config,
    )
    assert r2["status"] == "running"

    r3 = run_step(
        reg.graph,
        LIM,
        {
            "thread": "t-push-v2",
            "kind": "step",
            "expect": r2["ckpt"],
            "checkpoint_bundle": r2.get("checkpoint_bundle"),
        },
        tool_config=reg.tool_config,
        node_config=reg.node_config,
    )
    assert r3["status"] == "done"
    final_state = unpack(r3["output"])
    assert final_state["files"] == {"/plan.md": "step 1"}
    assert (
        final_state["messages"][-1].content
        == "Completed both write_virtual_file and lookup_kb!"
    )
