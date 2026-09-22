"""Deterministic attachment probe; the official host handles the HTTP request."""
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_azure_ai.agents.hosting import ResponsesHostServer
from agentdev import configure_agent_server
from types import SimpleNamespace

def inspect(state: MessagesState):
    message = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    parts = message.content if isinstance(message.content, list) else []
    images = sum(isinstance(p, dict) and p.get("type") == "input_image" for p in parts)
    files = sum(isinstance(p, dict) and p.get("type") == "input_file" for p in parts)
    return {"messages": [AIMessage(content=f"Agent received: {images} image(s), {files} file(s). Total attachments: {images + files}. Local probe; no model call.")]}

graph = StateGraph(MessagesState)
graph.add_node("inspect_attachments", inspect)
graph.add_edge(START, "inspect_attachments")
graph.add_edge("inspect_attachments", END)
server = ResponsesHostServer(graph.compile())
server.app._agent = SimpleNamespace(id="local-multimodal-probe", name="Attachment input probe", description="Local attachment-count probe; no model calls.")
configure_agent_server(server.app)
server.run(host="127.0.0.1", port=8088)
