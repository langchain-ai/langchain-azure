"""Compose the official Deep Research workflow independently of its host."""

from __future__ import annotations

from datetime import datetime, timezone

from deepagents import SubAgent, create_deep_agent
from deepagents.backends import StateBackend
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from research_agent.prompts import RESEARCHER_INSTRUCTIONS, WORKFLOW_INSTRUCTIONS


def build_agent(
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    tools: list[BaseTool],
) -> CompiledStateGraph:
    """Build a research coordinator with a focused research subagent.

    Args:
        model: A LangChain chat model supporting tool calls.
        checkpointer: Store for conversation state, todos, and virtual files.
        tools: The web search and research-progress tools shared with researchers.

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
        tools=tools,
        system_prompt=WORKFLOW_INSTRUCTIONS,
        subagents=[researcher],
        middleware=[TodoListMiddleware()],
        backend=StateBackend(),
        checkpointer=checkpointer,
        name="foundry-deep-research-agent",
    )
