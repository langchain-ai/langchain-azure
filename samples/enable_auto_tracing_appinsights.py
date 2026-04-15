"""Minimal LangGraph sample for auto tracing to Azure Application Insights.

Requirements:
    pip install -U "langchain-azure-ai[opentelemetry]" langchain-openai \
        langgraph python-dotenv

Required environment variables:
    APPLICATION_INSIGHTS_CONNECTION_STRING
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME

Optional environment variables:
    AZURE_OPENAI_API_VERSION (defaults to ``2024-12-01-preview``)
    ENV_FILE (path to a dotenv file; defaults to ``.env``)

Usage:
    cd libs/azure-ai
    uv run python ../../samples/enable_auto_tracing_appinsights.py

Notes:
    - This example runs outside a hosted Azure AI Foundry agent, so it sets
      ``auto_configure_azure_monitor=True`` to configure Azure Monitor locally.
    - In hosted agents, keep ``auto_configure_azure_monitor=False`` because the
      host already configures the ``TracerProvider``.
"""

from __future__ import annotations

import os
from typing import Any

import dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from langchain_azure_ai.callbacks.tracers import enable_auto_tracing

dotenv.load_dotenv(os.environ.get("ENV_FILE", ".env"))


def build_graph(model: AzureChatOpenAI) -> Any:
    """Build a tiny graph so auto tracing can attach graph and model spans."""

    def call_model(state: MessagesState) -> dict[str, list]:
        response = model.invoke(
            [
                SystemMessage(
                    content="You are a concise assistant. Reply in one short sentence."
                ),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("chat", call_model, metadata={"langgraph_node": "chat"})
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(name="auto-tracing-appinsights")


def main() -> None:
    """Enable tracing, run one graph invocation, and print the final response."""
    enable_auto_tracing(
        connection_string=os.environ["APPLICATION_INSIGHTS_CONNECTION_STRING"],
        auto_configure_azure_monitor=True,
        enable_content_recording=False,
        provider_name="azure.ai.openai",
        trace_all_langgraph_nodes=True,
    )

    model = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0,
    )
    graph = build_graph(model)
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Explain why Application Insights tracing is useful."
                )
            ]
        }
    )

    print(result["messages"][-1].content)
    print("Check Application Insights for the emitted LangGraph and model spans.")


if __name__ == "__main__":
    main()
