"""Real LangGraph tool execution through the official Responses host; no model."""
import base64
import os
from types import SimpleNamespace
from uuid import uuid4

from agentdev import configure_agent_server
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_azure_ai.agents.hosting import ResponsesHostServer


@tool
def make_report() -> list[dict]:
    """Return a public image and report file alongside their description."""
    return [
        {"type": "text", "text": "Created chart.png and report.txt"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1cAAAAASUVORK5CYII="}},
        {"type": "file", "file": {"filename": "report.txt", "file_data": "data:text/plain;base64," + base64.b64encode(b"Report: attachment preservation verified.").decode()}},
    ]


def plan(state: MessagesState):
    return {"messages": [AIMessage(content="", tool_calls=[{"name": "make_report", "args": {}, "id": "call_" + uuid4().hex}])]}


def answer(state: MessagesState):
    return {"messages": [AIMessage(content="The tool generated one image and one file. Open make_report below to inspect the actual returned output. This is a local deterministic agent; no model call.")]}


graph = StateGraph(MessagesState)
graph.add_node("plan", plan)
graph.add_node("tools", ToolNode([make_report]))
graph.add_node("answer", answer)
graph.add_edge(START, "plan")
graph.add_edge("plan", "tools")
graph.add_edge("tools", "answer")
graph.add_edge("answer", END)
server = ResponsesHostServer(graph.compile())
server.app._agent = SimpleNamespace(id="tool-output-probe", name="Tool attachment probe", description="Same tool returns an image and a file. No cloud model.")
configure_agent_server(server.app)
server.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8088")))
