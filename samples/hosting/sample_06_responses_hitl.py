"""Sample 06 - Human-in-the-loop over the Responses API.

The graph uses ``langgraph.types.interrupt`` to pause inside the
``ask_human`` node when the LLM decides it needs information only the
user can provide (modelled here as the ``AskHuman`` "tool"). The pause
is surfaced to the client as a ``function_call`` output item named
``__hosted_agent_adapter_interrupt__``; the client resumes by posting a
matching ``function_call_output`` with a JSON-encoded ``{"resume": ...}``
payload.

State is persisted by an ``InMemorySaver`` checkpointer keyed by the
``conversation`` id, so the second request continues the paused run.

Required environment variables (set in ``.env`` or your shell):

    AZURE_AI_PROJECT_ENDPOINT       e.g. https://<acct>.services.ai.azure.com/api/projects/<proj>
    AZURE_AI_MODEL_DEPLOYMENT_NAME  e.g. gpt-4o   (defaults to "gpt-4o")
    PORT                            optional, defaults to 8088

Run::

    az login
    cp .env.example .env  # then edit the values
    python sample_06_responses_hitl.py

Then in another terminal — first ask the agent to do something that
requires it to know where you are::

    curl -X POST http://127.0.0.1:8088/responses \\
      -H 'Content-Type: application/json' \\
      -d '{
        "input": "Ask me where I am, then look up the weather there.",
        "conversation": {"id": "demo-hitl-1"}
      }'

The response ``output`` array will contain:

    - one or more ``function_call`` items for any tool the model picked
    - a final ``function_call`` item with
      ``name == "__hosted_agent_adapter_interrupt__"`` and an ``arguments`` payload
      such as ``"Where are you located?"``.

Copy that item's ``call_id`` into the resume request::

    curl -X POST http://127.0.0.1:8088/responses \\
      -H 'Content-Type: application/json' \\
      -d '{
        "conversation": {"id": "demo-hitl-1"},
        "input": [{
          "type": "function_call_output",
          "call_id": "<call_id from the previous response>",
          "output": "{\\"resume\\": \\"Seattle\\"}"
        }]
      }'

The agent will resume from the ``ask_human`` node with the location you
provided, finish the weather lookup, and return a final assistant
``message`` item.
"""
from __future__ import annotations

import os
from typing import Annotated

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pydantic import BaseModel

from langchain_azure_ai.agents.hosting import AzureAIResponsesAgentHost

load_dotenv()

_AAD_SCOPE = "https://ai.azure.com/.default"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def get_weather(
    location: Annotated[str, "City and country, e.g. 'Seattle, US'."],
) -> str:
    """Return a fake weather snapshot for the given location."""
    return f"It's sunny and 22C in {location}."


class AskHuman(BaseModel):
    """Schema for asking the user a question.

    Bound as a "tool" so the LLM can request human input as part of its
    normal tool-calling flow. The dedicated ``ask_human`` node intercepts
    these calls and turns them into a LangGraph ``interrupt``.
    """

    question: str


# ---------------------------------------------------------------------------
# Chat model
# ---------------------------------------------------------------------------


def _build_chat_model() -> ChatOpenAI:
    project_endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"].rstrip("/")
    deployment = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")
    credential = DefaultAzureCredential()
    token = credential.get_token(_AAD_SCOPE).token
    return ChatOpenAI(
        model=deployment,
        api_key=token,  # type: ignore[arg-type]
        base_url=f"{project_endpoint}/openai/v1",
    )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def _build_graph() -> "object":
    llm = _build_chat_model()
    tools = [get_weather]
    tool_node = ToolNode(tools)
    model = llm.bind_tools(tools + [AskHuman])

    def call_model(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    def ask_human(state: MessagesState) -> dict:
        last = state["messages"][-1]
        tool_call = last.tool_calls[0]
        question = AskHuman.model_validate(tool_call["args"]).question
        # interrupt() pauses the graph. On resume, the value the client
        # sent in {"resume": ...} is returned here.
        answer = interrupt(question)
        return {
            "messages": [
                ToolMessage(content=str(answer), tool_call_id=tool_call["id"])
            ]
        }

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END
        if last.tool_calls[0]["name"] == "AskHuman":
            return "ask_human"
        return "action"

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.add_node("ask_human", ask_human)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        path_map=["ask_human", "action", END],
    )
    workflow.add_edge("action", "agent")
    workflow.add_edge("ask_human", "agent")
    return workflow.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = _build_graph()
    port = int(os.environ.get("PORT", "8088"))
    AzureAIResponsesAgentHost(graph).run(host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
