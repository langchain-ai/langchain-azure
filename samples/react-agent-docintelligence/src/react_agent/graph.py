"""Document intake and analysis pipeline using multiple agents.

This sample demonstrates a multi-agent workflow built with LangGraph and Azure AI
Foundry Agent Service. Two specialized agents are composed in a custom StateGraph:

1. **Document Parser** — Extracts content from documents using Azure AI Document
   Intelligence.
2. **Analyst** — Receives the extracted content and produces a structured analysis
   including document classification, key entities, summary, and action items.

A bridging node (`prepare_analysis`) converts the parser's output into input for
the analyst, showing how to wire agents together in a pipeline.
"""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt.tool_node import ToolNode

from langchain_azure_ai.agents import AgentServiceFactory
from langchain_azure_ai.agents.prebuilt import AgentServiceAgentState
from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer
from langchain_azure_ai.tools import AzureAIDocumentIntelligenceTool

from react_agent.prompts import ANALYST_PROMPT, PARSER_PROMPT

service = AgentServiceFactory()

# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------
tools = [AzureAIDocumentIntelligenceTool()]

parser_node = service.create_prompt_agent_node(
    name="document-parser",
    description="Extracts content from documents using Azure AI Document Intelligence.",
    model="gpt-4.1",
    instructions=PARSER_PROMPT,
    tools=tools,
)

analyst_node = service.create_prompt_agent_node(
    name="document-analyst",
    description="Analyzes extracted document content and produces a structured report.",
    model="gpt-4.1",
    instructions=ANALYST_PROMPT,
)


# ---------------------------------------------------------------------------
# Routing & bridging helpers
# ---------------------------------------------------------------------------
def route_parser_output(
    state: AgentServiceAgentState,
) -> Literal["tools", "prepare_analysis"]:
    """Route parser output: call tools if needed, otherwise move to analysis."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "prepare_analysis"


def prepare_analysis(state: AgentServiceAgentState) -> AgentServiceAgentState:
    """Bridge between parser and analyst.

    Takes the parser's AI response and wraps it in a HumanMessage so the
    analyst agent can process it on its own thread. Resets the V2 agent
    state fields so the analyst starts a fresh conversation.
    """
    parser_output = state["messages"][-1].content
    handoff_message = HumanMessage(
        content=(
            "Analyze the following extracted document content and produce a "
            "structured report:\n\n"
            f"{parser_output}"
        )
    )
    return AgentServiceAgentState(messages=[handoff_message])


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
builder = StateGraph(AgentServiceAgentState)

# Nodes
builder.add_node("parser", parser_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("prepare_analysis", prepare_analysis)
builder.add_node("analyst", analyst_node)

# Edges
builder.add_edge(START, "parser")
builder.add_conditional_edges("parser", route_parser_output)
builder.add_edge("tools", "parser")
builder.add_edge("prepare_analysis", "analyst")
builder.add_edge("analyst", END)

# Compile with tracing
graph = builder.compile(name="document-intake-agent").with_config(
    {
        "callbacks": [
            AzureAIOpenTelemetryTracer(
                agent_id="document-intake-agent",
            )
        ]
    }
)