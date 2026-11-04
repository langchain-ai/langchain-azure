"""Compose the official Deep Research workflow independently of its host."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, NotRequired, Required

from deepagents import SubAgent, create_deep_agent
from deepagents.backends.protocol import FileData
from deepagents.graph import DeepAgentState
from deepagents.middleware.filesystem import _file_data_reducer
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from research_agent.prompts import RESEARCHER_INSTRUCTIONS, WORKFLOW_INSTRUCTIONS


class FullCheckpointState(DeepAgentState):
    """Persist full messages and files because State Store has no batch-get API."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    # shortcut: pinned private reducer, replace when a public delta opt-out exists.
    files: NotRequired[Annotated[dict[str, FileData], _file_data_reducer]]


def build_agent(
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    tools: list[BaseTool],
    report_export_tool: BaseTool,
) -> CompiledStateGraph:
    """Build a research coordinator with a focused research subagent.

    Args:
        model: A LangChain chat model supporting tool calls.
        checkpointer: Store for conversation state, todos, and virtual files.
        tools: The web search and research-progress tools shared with researchers.
        report_export_tool: Publish the coordinator's saved report to Blob Storage.

    Returns:
        A native Deep Agents graph supporting invocation and streaming.
    """
    researcher: SubAgent = {
        "name": "research-agent",
        "description": "Research one topic and return findings with source URLs.",
        "system_prompt": RESEARCHER_INSTRUCTIONS.format(
            date=datetime.now(timezone.utc).date().isoformat()
        ),
        "tools": tools,
    }
    return create_deep_agent(
        model=model,
        tools=[*tools, report_export_tool],
        system_prompt=WORKFLOW_INSTRUCTIONS,
        subagents=[researcher],
        middleware=[TodoListMiddleware()],
        state_schema=FullCheckpointState,
        checkpointer=checkpointer,
        name="foundry-deep-research-agent",
    )
