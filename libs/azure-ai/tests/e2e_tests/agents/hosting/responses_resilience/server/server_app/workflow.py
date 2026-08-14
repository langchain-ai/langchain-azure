# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Clean deterministic workflow for Responses resilience E2E tests."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    message_chunk_to_message,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from .crash_points import INPUT_INSTRUCTION, parse_crash_point

PLAN_OUTPUT = "recovery plan created"
RESEARCH_OUTPUT = "checkpoint evidence gathered"
EXECUTE_OUTPUT = "recovery action executed"


class _WordStreamingChatModel(BaseChatModel):
    """Deterministic chat model that streams one whitespace-preserving word."""

    response_text: str
    response_id: str

    @property
    def _llm_type(self) -> str:
        return "responses-resilience-word-stream"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=self.response_text, id=self.response_id)
                )
            ]
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        del messages, stop, run_manager, kwargs
        chunks = re.findall(r"\S+\s*|\s+", self.response_text)
        for index, text in enumerate(chunks):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=text,
                    id=self.response_id,
                    chunk_position="last" if index == len(chunks) - 1 else None,
                )
            )


async def _stream_message(
    messages: list[BaseMessage],
    *,
    text: str,
    message_id: str,
) -> AIMessage:
    model = _WordStreamingChatModel(response_text=text, response_id=message_id)
    chunks = [chunk async for chunk in model.astream(messages)]
    combined = chunks[0]
    for chunk in chunks[1:]:
        combined += chunk
    message = message_chunk_to_message(combined)
    assert isinstance(message, AIMessage)
    return message


class TestState(TypedDict):
    """State for the fixed three-node recovery workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    stage: str
    request_valid: bool
    plan_writes: int
    research_writes: int
    execute_writes: int
    summarize_writes: int


def state_root() -> Path:
    """Return the durable workflow-state directory."""

    configured = os.environ.get("RESILIENCE_E2E_STATE_ROOT")
    root = Path(configured) if configured else Path.home() / ".responses-resilience-e2e"
    root.mkdir(parents=True, exist_ok=True)
    return root


def thread_id(config: RunnableConfig) -> str:
    """Read the hosting thread ID from a node configuration."""

    value = config.get("configurable", {}).get("thread_id")
    if not isinstance(value, str) or not value:
        raise RuntimeError("The E2E workflow requires a LangGraph thread_id")
    return value


def thread_key(value: str) -> str:
    """Return a filesystem-safe key for a thread ID."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _node_runs_path(value: str) -> Path:
    return state_root() / f"{thread_key(value)}.node-runs.jsonl"


def _record_node_run(node: str, config: RunnableConfig) -> None:
    record = {
        "node": node,
        "input_checkpoint_id": config.get("configurable", {}).get("checkpoint_id"),
    }
    with _node_runs_path(thread_id(config)).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_node_runs(value: str) -> list[dict[str, Any]]:
    path = _node_runs_path(value)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if not isinstance(message.content, list):
        return ""
    return "".join(
        part
        if isinstance(part, str)
        else part.get("text", "")
        if isinstance(part, dict) and isinstance(part.get("text"), str)
        else ""
        for part in message.content
    )


def _selected_crash_point(state: TestState) -> str | None:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return parse_crash_point(_message_text(message))
    return None


def build_graph(checkpointer: AsyncSqliteSaver) -> CompiledStateGraph:
    """Build the model-free workflow without any crash behavior."""

    async def node_1plan(state: TestState, config: RunnableConfig) -> dict[str, Any]:
        value = thread_id(config)
        if _selected_crash_point(state) is None:
            return {
                "stage": "input_rejected",
                "request_valid": False,
                "messages": [
                    AIMessage(
                        content=INPUT_INSTRUCTION,
                        id=f"input-instruction-{thread_key(value)[:16]}",
                    )
                ],
            }

        _record_node_run("1plan", config)
        write_count = state.get("plan_writes", 0) + 1
        message = await _stream_message(
            state["messages"],
            text=f"{PLAN_OUTPUT}\n",
            message_id=f"plan-{thread_key(value)[:16]}-{write_count}",
        )
        return {
            "stage": "planned",
            "request_valid": True,
            "plan_writes": write_count,
            "messages": [message],
        }

    async def node_2research(
        state: TestState, config: RunnableConfig
    ) -> dict[str, Any]:
        _record_node_run("2research", config)
        value = thread_id(config)
        write_count = state.get("research_writes", 0) + 1
        message = await _stream_message(
            state["messages"],
            text=f"{RESEARCH_OUTPUT}\n",
            message_id=f"research-{thread_key(value)[:16]}-{write_count}",
        )
        return {
            "stage": "researched",
            "research_writes": write_count,
            "messages": [message],
        }

    async def node_3execute(
        state: TestState, config: RunnableConfig
    ) -> dict[str, Any]:
        _record_node_run("3execute", config)
        value = thread_id(config)
        write_count = state.get("execute_writes", 0) + 1
        message = await _stream_message(
            state["messages"],
            text=f"{EXECUTE_OUTPUT}\n",
            message_id=f"execute-{thread_key(value)[:16]}-{write_count}",
        )
        return {
            "stage": "executed",
            "execute_writes": write_count,
            "messages": [message],
        }

    async def node_4summarize(
        state: TestState, config: RunnableConfig
    ) -> dict[str, Any]:
        _record_node_run("4summarize", config)
        value = thread_id(config)
        write_count = state.get("summarize_writes", 0) + 1
        node_runs = _read_node_runs(value)
        result = {
            "node_runs": {
                node: sum(item["node"] == node for item in node_runs)
                for node in ("1plan", "2research", "3execute", "4summarize")
            }
        }
        result["checkpoint_writes"] = {
            "1plan": state.get("plan_writes", 0),
            "2research": state.get("research_writes", 0),
            "3execute": state.get("execute_writes", 0),
            "4summarize": write_count,
        }
        message = await _stream_message(
            state["messages"],
            text=json.dumps(result, sort_keys=True),
            message_id=f"summary-{thread_key(value)[:16]}-{write_count}",
        )
        return {
            "summarize_writes": write_count,
            "messages": [message],
        }

    def route_after_1plan(state: TestState) -> str:
        return "2research" if state["request_valid"] else END

    builder = StateGraph(TestState)
    builder.add_node("1plan", node_1plan)
    builder.add_node("2research", node_2research)
    builder.add_node("3execute", node_3execute)
    builder.add_node("4summarize", node_4summarize)
    builder.add_edge(START, "1plan")
    builder.add_conditional_edges("1plan", route_after_1plan)
    builder.add_edge("2research", "3execute")
    builder.add_edge("3execute", "4summarize")
    builder.add_edge("4summarize", END)
    return builder.compile(checkpointer=checkpointer)