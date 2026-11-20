# Copyright (c) Microsoft. All rights reserved.

"""Shared test graph: the conversation_approval shape (model -> approval interrupt ->.

ToolNode -> remember -> loop),
written against langgraph only, with an in-process scripted chat model (no network, no
LLM).
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    turn: int
    approved: bool


class ScriptedChat(BaseChatModel):
    """Always asks for the `double` tool with n = turn + 2."""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        turn = sum(isinstance(m, HumanMessage) for m in messages) - 1
        call = {
            "name": "double",
            "args": {"n": turn + 2},
            "id": f"call-{turn}",
            "type": "tool_call",
        }
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="", tool_calls=[call]))
            ]
        )


def conversation_graph(
    checkpointer: Any, turns: int, effects: list[int] | None = None, pad: int = 0
):
    effects = effects if effects is not None else []
    model = ScriptedChat()

    @tool
    def double(n: int) -> str:
        """Double an approved number."""
        effects.append(n)
        return str(n * 2) + "." * pad

    def ask(state: State):
        return {"messages": [model.invoke(state["messages"])]}

    def approve(state: State):
        return {
            "approved": interrupt(
                {"calls": getattr(state["messages"][-1], "tool_calls", None)}
            )
        }

    def remember(state: State):
        turn = state["turn"] + 1
        return {
            "turn": turn,
            "messages": [HumanMessage(content="another number", id=f"turn-{turn}")],
        }

    builder = StateGraph(State)
    builder.add_node("model", ask)
    builder.add_node("approval", approve)
    builder.add_node("tools", ToolNode([double]))
    builder.add_node("remember", remember)
    builder.add_edge(START, "model")
    builder.add_edge("model", "approval")
    builder.add_conditional_edges(
        "approval",
        lambda s: "tools" if s["approved"] else END,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "remember")
    builder.add_conditional_edges(
        "remember",
        lambda s: "model" if s["turn"] < turns else END,
        {"model": "model", END: END},
    )
    return builder.compile(checkpointer=checkpointer)


def initial_state() -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content="double", id="question")],
        "turn": 0,
        "approved": False,
    }
