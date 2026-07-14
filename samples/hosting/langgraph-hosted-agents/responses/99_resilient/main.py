"""Sample 99 - Resilient background Responses with a linear LangGraph workflow.

This sample demonstrates the **resilient background responses** feature (see the
developer guide under
``azure-sdk-for-python/sdk/agentserver/azure-ai-agentserver-responses/docs``)
and how it composes with LangGraph's own native checkpointer.

The workload is deliberately the **least interesting** part: a **linear chain of
N phases**. Each phase is one LangGraph node that does a bit of deterministic
work and appends exactly one output item, so every phase boundary is its own
checkpoint -- the "one output item per phase" shape the resilience contract
recovers from. There is no LLM, no tools, and no routing, so the focus stays on
crash/recovery rather than on how an agent works.

Each phase stamps a per-process ``_LIFETIME`` id onto its output, so after a
crash you can SEE which phases ran before the crash and which re-ran on
recovery, e.g. ``[ab12:plan, ab12:research, cd34:summarize]`` -- the first two
ran in lifetime ``ab12`` and survived; the last re-ran after restart (``cd34``).

Graph shape::

    START -> plan -> research -> summarize -> END

Optional environment variables:

    PORT               optional, defaults to 8088
    CHECKPOINT_DB      optional path to the LangGraph checkpoint SQLite file.
                       Defaults to ``checkpoints.sqlite`` in the working
                       directory locally, or ``$HOME/checkpoints.sqlite`` when
                       hosted on Foundry, since only ``$HOME`` persists across a
                       hosted restart.
    TOKEN_DELAY_SECONDS optional default per-token sleep (default 0.05). A
                        request can override it through ``metadata.token_delay``.
    STEERABLE_CONVERSATIONS optional boolean (default false) controlling
                            whether newer turns can steer active conversations.

Run::

    python main.py

Then in another terminal (``background`` + ``stream`` engages the resilient
path)::

    curl -N -X POST http://127.0.0.1:8088/responses \
        -H 'Content-Type: application/json' \
        -d '{"input":"go","background":true,"stream":true}'

Set ``STEP_DELAY_SECONDS`` to widen the crash window, then kill the process
mid-run and restart it to watch recovery resume from the last checkpoint.
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys
from typing import Annotated
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict

from azure.ai.agentserver.core import AgentConfig
from azure.ai.agentserver.responses import ResponsesServerOptions

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from model import FakeChatModel


def _resolve_checkpoint_db() -> str:
    if AgentConfig.from_env().is_hosted:
        return os.path.join(os.path.expanduser("~"), "checkpoints.sqlite")
    return "checkpoints.sqlite"

_CHECKPOINT_DB = _resolve_checkpoint_db()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


class TodoItem(TypedDict):
    id: str
    label: str
    checked: bool


class State(TypedDict):
    # Chat history
    messages: Annotated[list[BaseMessage], add_messages]
    todos: NotRequired[list[TodoItem]]
    crash_requested: NotRequired[bool]
    crash_lifetime: NotRequired[str]
    num_turns: NotRequired[int]
    has_recovered: NotRequired[bool]


def latest_user_input(state: State) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def contains_crash_request(state: State) -> bool:
    return "crash" in latest_user_input(state).lower()


def initial_todos() -> list[TodoItem]:
    return [
        {"id": "plan", "label": "plan", "checked": False},
        {"id": "research", "label": "research", "checked": False},
        {"id": "summarize", "label": "summarize", "checked": False},
    ]


def check_todo(state: State, todo_id: str) -> list[TodoItem]:
    todos = state.get("todos") or initial_todos()
    return [
        {**todo, "checked": todo["checked"] or todo["id"] == todo_id}
        for todo in todos
    ]


def format_todos(todos: list[TodoItem]) -> str:
    return ", ".join(
        f"{todo['label']}[{'✓' if todo['checked'] else ' '}]" for todo in todos
    )


def token_delay_seconds(config: RunnableConfig) -> float:
    responses_context = config.get("configurable", {}).get("responses_context")
    request = getattr(responses_context, "request", None)
    metadata = getattr(request, "metadata", None)
    raw_delay = metadata.get("token_delay") if metadata is not None else None
    if raw_delay is None:
        return float(os.environ.get("TOKEN_DELAY_SECONDS", "0.05"))
    delay = float(raw_delay)
    if delay < 0:
        raise ValueError("metadata.token_delay must be greater than or equal to 0")
    return delay


async def stream_phase_message(
    state: State, config: RunnableConfig, text: str
) -> BaseMessage:
    model = FakeChatModel(
        reply=text,
        token_delay_seconds=token_delay_seconds(config),
    )
    return await model.ainvoke(state["messages"], config=config)


async def _sigkill_current_process() -> None:
    print("Crash trigger received; sending SIGKILL to current process.", flush=True)
    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    os.kill(os.getpid(), kill_signal)
    await asyncio.sleep(60 * 60 * 24)


def build_graph(checkpointer):
    # Three stages, run one after another. Each stage appends one output item
    # that shows the TODO state after that graph checkpoint.
    async def plan(state: State, config: RunnableConfig) -> dict:
        if contains_crash_request(state):
            print("Preparing crash now.", flush=True)
        todos: list[TodoItem] = [
            {**todo, "checked": todo["id"] == "plan"}
            for todo in initial_todos()
        ]
        num_turns = state.get("num_turns", 0) + 1
        text = f"Plan done. {format_todos(todos)}\n"
        return {
            "messages": [await stream_phase_message(state, config, text)],
            "todos": todos,
            "num_turns": num_turns,
        }

    async def research(state: State, config: RunnableConfig) -> dict:
        responses_context = config.get("configurable", {}).get("responses_context")
        is_recovery = bool(getattr(responses_context, "is_recovery", False))
        # Only trigger the crash when requested and not crash when recovering from a crash which causes infinite loop.
        crash_requested = contains_crash_request(state)
        if crash_requested and not is_recovery:
            await _sigkill_current_process()

        todos = check_todo(state, "research")
        text = f"Research done. {format_todos(todos)}\n"
        have_recovered = state.get("has_recovered", False) or is_recovery
        return {
            "messages": [await stream_phase_message(state, config, text)],
            "todos": todos,
            "has_recovered": have_recovered,
        }

    async def summarize(state: State, config: RunnableConfig) -> dict:
        num_turns = state.get("num_turns", 1)
        todos = check_todo(state, "summarize")
        user_input = latest_user_input(state)
        echo = f"{user_input[:20]}{'...' if len(user_input) > 20 else ''}"
        text = (
            f"Summarize done. {format_todos(todos)}\n"
            f'Echo: "{echo}"\n'
            f"Number of turns: {num_turns}\n"
        )
        return {
            "messages": [await stream_phase_message(state, config, text)],
            "todos": todos,
            "num_turns": num_turns,
        }

    builder = StateGraph(State)
    builder.add_node("plan", plan)
    builder.add_node("research", research)
    builder.add_node("summarize", summarize)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "summarize")
    builder.add_edge("summarize", END)

    return builder.compile(checkpointer=checkpointer)


async def amain() -> None:
    options = ResponsesServerOptions(
        resilient_background=True,
        steerable_conversations=env_bool("STEERABLE_CONVERSATIONS"),
    )
    async with AsyncSqliteSaver.from_conn_string(_CHECKPOINT_DB) as checkpointer:
        graph = build_graph(checkpointer)
        await ResponsesHostServer(graph, options=options).run_async(
            port=int(os.environ.get("PORT", "8088"))
        )


if __name__ == "__main__":
    asyncio.run(amain())
