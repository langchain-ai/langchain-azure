"""Deterministic citation transport probe; no cloud model call."""
import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_azure_ai.agents.hosting import ResponsesHostServer

BUILD = {"test": "local Inspector citation transport"}
CITATION = {"type": "url_citation", "url": "https://docs.langchain.com/oss/python/langgraph/persistence",
            "title": "LangGraph persistence", "start_index": 0, "end_index": 43}
FILE = {"type": "file_citation", "file_id": "file-demo-report", "filename": "report.pdf", "index": 0}
CONTENT = [
    {"type": "text", "text": "LangGraph can persist agent execution state.", "annotations": [CITATION]},
    {"type": "text", "text": "The report is ready.", "annotations": [FILE]},
]


def answer(state: MessagesState):
    user = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    if user.text == "inspect-history":
        previous = [m.content for m in state["messages"] if isinstance(m, AIMessage)]
        return {"messages": [AIMessage(content=json.dumps({"build": BUILD, "previous": previous}))]}
    return {"messages": [AIMessage(content=CONTENT)]}


builder = StateGraph(MessagesState)
builder.add_node("answer", answer)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)
if __name__ == "__main__":
    import os
    from types import SimpleNamespace
    from agentdev import configure_agent_server
    server = ResponsesHostServer(builder.compile())
    server.app._agent = SimpleNamespace(id="citation-probe", name="Citation transport probe", description="Fixed URL and file citations; no model call.")
    configure_agent_server(server.app)
    server.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8088")))
