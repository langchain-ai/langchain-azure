# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Mock LangGraph graphs used by the human-in-the-loop host tests.

Every builder here is the *user-authored* side of the contract: whatever a
customer writes with ``langgraph.types.interrupt``, the Responses host must
surface and resume faithfully. Keeping them in one module makes it easy to
see which graph shapes are covered, and lets the test modules stay focused
on assertions.

Shapes are grouped by the section of
https://docs.langchain.com/oss/python/langgraph/interrupts they exercise.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from typing_extensions import TypedDict


class MessagesState(TypedDict):
    """The default message-list state shared by most graphs below."""

    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Scripted agent plumbing
# ---------------------------------------------------------------------------


class AskHuman(BaseModel):
    """Argument schema for the pseudo-tool that hands control to a human."""

    question: str


@tool
def get_weather(location: str) -> str:
    """Fake weather tool."""
    return f"It's sunny in {location}."


class ScriptedModel:
    """Tiny chat model that yields preset assistant messages on each call.

    The graph calls ``model.invoke(state["messages"])`` once per ``agent``
    node visit. We hand back successive scripted ``AIMessage`` payloads so
    the test fully controls when the graph decides to ``AskHuman`` and
    when it produces the final answer.

    Scripts are registered under a key by the ``script`` fixture in
    ``conftest.py``, which also clears them again after each test.

    Every invocation records the message list it was handed under the same
    key in :attr:`seen`, so a test can assert on what actually reached the
    model's context.
    """

    script: ClassVar[dict[str, list[AIMessage]]] = {}
    seen: ClassVar[dict[str, list[list[BaseMessage]]]] = {}

    def __init__(self, key: str) -> None:
        self._key = key

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.seen.setdefault(self._key, []).append(list(messages))
        queue = self.script[self._key]
        if not queue:
            raise AssertionError("scripted model exhausted")
        return queue.pop(0)


def build_ask_human_graph(key: str) -> CompiledStateGraph:
    """Agent that pauses in a dedicated ``ask_human`` node.

    This is the canonical HITL agent shape: the model emits an ``AskHuman``
    tool call, a node turns that into ``interrupt(question)``, and the
    answer comes back as a ``ToolMessage``.
    """
    model = ScriptedModel(key)
    tool_node = ToolNode([get_weather])

    def call_model(state: MessagesState) -> dict[str, Any]:
        return {"messages": [model.invoke(state["messages"])]}

    def ask_human(state: MessagesState) -> dict[str, Any]:
        last = state["messages"][-1]
        tool_call = last.tool_calls[0]  # type: ignore[attr-defined]
        question = AskHuman.model_validate(tool_call["args"]).question
        answer = interrupt(question)
        return {
            "messages": [ToolMessage(content=str(answer), tool_call_id=tool_call["id"])]
        }

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None)
        if not calls:
            return END
        if calls[0]["name"] == "AskHuman":
            return "ask_human"
        return "action"

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("action", tool_node)
    builder.add_node("ask_human", ask_human)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", should_continue, path_map=["ask_human", "action", END]
    )
    builder.add_edge("action", "agent")
    builder.add_edge("ask_human", "agent")
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Minimal single-pause graphs
# ---------------------------------------------------------------------------


def build_simple_interrupt_graph() -> CompiledStateGraph:
    """Minimal single-pause graph, reused by the transport-level tests."""

    def ask(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=f"ok:{interrupt('name?')}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


def build_uncheckpointed_interrupt_graph() -> CompiledStateGraph:
    """The same single pause, but compiled without a checkpointer.

    https://docs.langchain.com/oss/python/langgraph/interrupts#pause-using-interrupt
    """

    def ask(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=f"ok:{interrupt('name?')}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile()


def build_static_breakpoint_graph() -> CompiledStateGraph:
    """Pauses via ``interrupt_before`` — a debugger breakpoint, not an ``Interrupt``.

    https://docs.langchain.com/oss/python/langgraph/interrupts#debugging-with-interrupts
    """

    def node(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="node-ran")]}

    builder = StateGraph(MessagesState)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    return builder.compile(checkpointer=InMemorySaver(), interrupt_before=["n"])


# ---------------------------------------------------------------------------
# Handling multiple interrupts
# https://docs.langchain.com/oss/python/langgraph/interrupts#handling-multiple-interrupts
# ---------------------------------------------------------------------------


def build_parallel_interrupt_graph() -> CompiledStateGraph:
    """Fan out to two nodes that each pause in the same superstep."""

    def ask_a(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=f"a={interrupt('question_a')}")]}

    def ask_b(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=f"b={interrupt('question_b')}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("a", ask_a)
    builder.add_node("b", ask_b)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Do not reorder interrupt calls within a node
# https://docs.langchain.com/oss/python/langgraph/interrupts#do-not-reorder-interrupt-calls-within-a-node
# ---------------------------------------------------------------------------


def build_sequential_interrupt_graph() -> CompiledStateGraph:
    """One node, two interrupts, always issued in the same order (✅)."""

    def ask(state: MessagesState) -> dict[str, Any]:
        name = interrupt("name?")
        city = interrupt("city?")
        return {"messages": [AIMessage(content=f"{name}@{city}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


def build_skipping_interrupt_graph(flags: dict[str, bool]) -> CompiledStateGraph:
    """The 🔴 pattern: a conditional ``interrupt()`` shifts the call order.

    Flip ``flags["ask_age"]`` between turns to make the node skip its
    second question on replay.
    """

    def ask(state: MessagesState) -> dict[str, Any]:
        name = interrupt("name?")
        if flags["ask_age"]:
            interrupt("age?")
        city = interrupt("city?")
        return {"messages": [AIMessage(content=f"{name}@{city}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


def build_reordering_interrupt_graph(flags: dict[str, bool]) -> CompiledStateGraph:
    """The 🔴 pattern the rule is actually named after: same calls, new order.

    Flip ``flags["reversed"]`` between turns to make the node ask the same
    two questions in the opposite order on replay.
    """

    def ask(state: MessagesState) -> dict[str, Any]:
        if flags["reversed"]:
            city = interrupt("city?")
            name = interrupt("name?")
        else:
            name = interrupt("name?")
            city = interrupt("city?")
        return {"messages": [AIMessage(content=f"{name}@{city}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Interrupts in tools
# https://docs.langchain.com/oss/python/langgraph/interrupts#interrupts-in-tools
# ---------------------------------------------------------------------------


@tool
def send_email(to: str, subject: str) -> str:
    """Send an email to a recipient."""
    response = interrupt({"action": "send_email", "to": to, "subject": subject})
    if isinstance(response, dict) and response.get("action") == "approve":
        return f"Email sent to {response.get('to', to)}"
    return "Email cancelled by user"


def build_tool_interrupt_graph(key: str) -> CompiledStateGraph:
    """Agent whose tool pauses for approval from inside ``ToolNode``."""
    model = ScriptedModel(key)

    def call_model(state: MessagesState) -> dict[str, Any]:
        return {"messages": [model.invoke(state["messages"])]}

    def should_continue(state: MessagesState) -> str:
        return "action" if getattr(state["messages"][-1], "tool_calls", None) else END

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("action", ToolNode([send_email]))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, path_map=["action", END])
    builder.add_edge("action", "agent")
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Interrupts inside subgraphs called as functions
# https://docs.langchain.com/oss/python/langgraph/interrupts#using-with-subgraphs-called-as-functions
# ---------------------------------------------------------------------------


def build_subgraph_interrupt_graph() -> CompiledStateGraph:
    """Parent node invokes a subgraph that pauses."""

    def sub_node(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=f"sub:{interrupt('sub-question')}")]}

    sub_builder = StateGraph(MessagesState)
    sub_builder.add_node("inner", sub_node)
    sub_builder.add_edge(START, "inner")
    sub_builder.add_edge("inner", END)
    subgraph = sub_builder.compile()

    def parent_node(state: MessagesState) -> dict[str, Any]:
        result = subgraph.invoke({"messages": state["messages"]})
        return {"messages": result["messages"][-1:]}

    builder = StateGraph(MessagesState)
    builder.add_node("parent", parent_node)
    builder.add_edge(START, "parent")
    builder.add_edge("parent", END)
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Do not wrap interrupt calls in try/except
# https://docs.langchain.com/oss/python/langgraph/interrupts#do-not-wrap-interrupt-calls-in-try%2Fexcept
# ---------------------------------------------------------------------------


def build_swallowed_interrupt_graph() -> CompiledStateGraph:
    """The documented anti-pattern: a bare ``except`` eats ``GraphInterrupt``."""

    def ask(state: MessagesState) -> dict[str, Any]:
        try:
            answer = interrupt("What's your name?")
        except Exception as exc:  # noqa: BLE001 - deliberately wrong
            return {"messages": [AIMessage(content=f"swallowed:{type(exc).__name__}")]}
        return {"messages": [AIMessage(content=f"ok:{answer}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Side effects called before interrupt must be idempotent
# https://docs.langchain.com/oss/python/langgraph/interrupts#side-effects-called-before-interrupt-must-be-idempotent
# ---------------------------------------------------------------------------


def build_side_effect_graph(trace: list[str]) -> CompiledStateGraph:
    """Record what runs before and after the pause, to expose node replay."""

    def ask(state: MessagesState) -> dict[str, Any]:
        trace.append("before")
        answer = interrupt("confirm?")
        trace.append("after")
        return {"messages": [AIMessage(content=f"done:{answer}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Do not return complex values in interrupt calls
# https://docs.langchain.com/oss/python/langgraph/interrupts#do-not-return-complex-values-in-interrupt-calls
# ---------------------------------------------------------------------------


def build_unserializable_value_graph() -> CompiledStateGraph:
    """Pause on a payload the checkpointer accepts but ``json`` rejects.

    A ``set`` survives LangGraph's msgpack serializer, so the pause is
    checkpointed normally and only the *wire* encoding has a problem.
    """

    def ask(state: MessagesState) -> dict[str, Any]:
        answer = interrupt({"q": "pick one", "choices": {"a", "b"}})
        return {"messages": [AIMessage(content=f"picked:{answer}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Validating human input (re-prompt loop)
# https://docs.langchain.com/oss/python/langgraph/interrupts#validating-human-input
# ---------------------------------------------------------------------------


class FormState(TypedDict):
    """State for the age-collection form used by the re-prompt graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    age: Optional[int]
    pending_question: Optional[str]


def build_reprompt_graph() -> CompiledStateGraph:
    """Collect an age, re-prompting through a conditional edge until valid.

    ``interrupt()`` is called exactly once per node invocation — the
    documented alternative to a ``while True`` loop inside the node.
    """

    def collect_age(state: FormState) -> dict[str, Any]:
        question = state.get("pending_question") or "What is your age?"
        answer = interrupt(question)
        if isinstance(answer, int) and answer > 0:
            return {
                "age": answer,
                "pending_question": None,
                "messages": [AIMessage(content=f"age={answer}")],
            }
        return {"pending_question": f"'{answer}' is not a valid age."}

    def route(state: FormState) -> str:
        return END if state.get("age") is not None else "collect_age"

    builder = StateGraph(FormState)
    builder.add_node("collect_age", collect_age)
    builder.add_edge(START, "collect_age")
    builder.add_conditional_edges("collect_age", route, path_map=["collect_age", END])
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Review and edit state
# https://docs.langchain.com/oss/python/langgraph/interrupts#review-and-edit-state
# ---------------------------------------------------------------------------


def build_review_graph() -> CompiledStateGraph:
    """Draft, then pause so a human can rewrite the draft."""

    def draft(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="Initial draft")]}

    def review(state: MessagesState) -> dict[str, Any]:
        edited = interrupt(
            {
                "instruction": "Review and edit this content",
                "content": state["messages"][-1].content,
            }
        )
        return {"messages": [AIMessage(content=str(edited))]}

    builder = StateGraph(MessagesState)
    builder.add_node("draft", draft)
    builder.add_node("review", review)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "review")
    builder.add_edge("review", END)
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Approve or reject
# https://docs.langchain.com/oss/python/langgraph/interrupts#approve-or-reject
# ---------------------------------------------------------------------------


def build_approval_routing_graph() -> CompiledStateGraph:
    """Route to a different node depending on the resume value."""

    def approval(state: MessagesState) -> Command:
        decision = interrupt({"question": "Do you want to proceed?"})
        return Command(goto="proceed" if decision else "cancel")

    def proceed(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="status:approved")]}

    def cancel(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="status:rejected")]}

    builder = StateGraph(MessagesState)
    builder.add_node("approval", approval)
    builder.add_node("proceed", proceed)
    builder.add_node("cancel", cancel)
    builder.add_edge(START, "approval")
    builder.add_edge("proceed", END)
    builder.add_edge("cancel", END)
    return builder.compile(checkpointer=InMemorySaver())
