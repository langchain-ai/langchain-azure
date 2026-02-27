"""Integration tests for a REACT-style graph with Azure AI Foundry V2 agents.

These tests validate that a LangGraph REACT graph can use an Azure AI Foundry
V2 agent as the reasoning node, transitioning to a local ``ToolNode`` when the
agent decides to call a tool implemented in Python, and then returning the
tool result back to the agent for a final answer.

The graph follows the standard REACT pattern::

    START → foundryAgent → [external_tools_condition]
                             ├─→ tools → foundryAgent (loop back)
                             └─→ __end__

Environment variables required:

* ``AZURE_AI_PROJECT_ENDPOINT`` – endpoint of an existing Foundry project.
* ``MODEL_DEPLOYMENT_NAME`` – (optional) model deployment, defaults to
  ``gpt-4.1``.
"""

import os
import uuid

import pytest

try:
    from azure.identity import DefaultAzureCredential
except ImportError:
    pytest.skip("Azure dependencies not available", allow_module_level=True)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_azure_ai.agents.v2 import AgentServiceFactory

# ---------------------------------------------------------------------------
# Tool definitions – these run locally inside LangGraph's ToolNode
# ---------------------------------------------------------------------------


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together and return the result."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together and return the result."""
    return a * b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.requires("azure-ai-projects")
class TestReactGraphV2:
    """Integration tests for a REACT graph powered by a V2 Foundry agent."""

    service: AgentServiceFactory
    model: str

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test environment.

        Skips if ``AZURE_AI_PROJECT_ENDPOINT`` is not set.
        """
        endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")

        if not endpoint:
            pytest.skip("AZURE_AI_PROJECT_ENDPOINT environment variable not set")

        self.service = AgentServiceFactory(
            project_endpoint=endpoint,
            credential=DefaultAzureCredential(),
        )
        self.model = os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4.1")

    # ------------------------------------------------------------------
    # Test: tool invocation in a REACT loop
    # ------------------------------------------------------------------

    def test_react_agent_calls_tool_and_returns_result(self) -> None:
        """The agent should call the *add* tool and use its output."""
        tools = [add, multiply]

        agent = self.service.create_prompt_agent(
            name=f"test-react-tools-v2-{uuid.uuid4().hex[:8]}",
            model=self.model,
            instructions=(
                "You are a helpful assistant that performs arithmetic. "
                "Always use the provided tools to compute results. "
                "Do not try to compute the answer yourself."
            ),
            tools=tools,
        )

        try:
            state = agent.invoke({"messages": [HumanMessage(content="What is 3 + 4?")]})

            messages = state["messages"]

            # We expect at least:
            #  1. The original HumanMessage
            #  2. An AIMessage with tool_calls (agent decided to call `add`)
            #  3. A ToolMessage with the tool result
            #  4. A final AIMessage with the answer
            assert len(messages) >= 4, (
                f"Expected at least 4 messages in the REACT loop, got "
                f"{len(messages)}: {messages}"
            )

            # The final message should be from the agent (AIMessage)
            final_message = messages[-1]
            assert isinstance(final_message, AIMessage), (
                f"Expected final message to be AIMessage, got " f"{type(final_message)}"
            )

            # The answer should contain "7"
            assert (
                "7" in final_message.content
            ), f"Expected '7' in final answer, got: {final_message.content}"

            # Verify we went through the tool loop: there should be at least
            # one ToolMessage in the conversation
            tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
            assert (
                len(tool_messages) >= 1
            ), "Expected at least one ToolMessage in the conversation"

            # Verify there was an AIMessage with tool_calls
            ai_with_tools = [
                m
                for m in messages
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
            ]
            assert (
                len(ai_with_tools) >= 1
            ), "Expected at least one AIMessage with tool_calls"

        finally:
            self.service.delete_agent(agent)

    # ------------------------------------------------------------------
    # Test: multiplication tool
    # ------------------------------------------------------------------

    def test_react_agent_calls_multiply_tool(self) -> None:
        """The agent should call the *multiply* tool and use its output."""
        tools = [add, multiply]

        agent = self.service.create_prompt_agent(
            name=f"test-react-multiply-v2-{uuid.uuid4().hex[:8]}",
            model=self.model,
            instructions=(
                "You are a helpful assistant that performs arithmetic. "
                "Always use the provided tools to compute results. "
                "Do not try to compute the answer yourself."
            ),
            tools=tools,
        )

        try:
            state = agent.invoke(
                {"messages": [HumanMessage(content="What is 6 times 7?")]}
            )

            messages = state["messages"]

            # Verify the REACT loop completed
            assert (
                len(messages) >= 4
            ), f"Expected at least 4 messages, got {len(messages)}"

            # The final answer should contain "42"
            final_message = messages[-1]
            assert isinstance(final_message, AIMessage)
            assert (
                "42" in final_message.content
            ), f"Expected '42' in final answer, got: {final_message.content}"

            # Verify a ToolMessage exists
            tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
            assert len(tool_messages) >= 1

        finally:
            self.service.delete_agent(agent)

    # ------------------------------------------------------------------
    # Test: multi-step tool usage
    # ------------------------------------------------------------------

    def test_react_agent_multi_step_tool_usage(self) -> None:
        """The agent should use tools across multiple steps."""
        tools = [add, multiply]

        agent = self.service.create_prompt_agent(
            name=f"test-react-multistep-v2-{uuid.uuid4().hex[:8]}",
            model=self.model,
            instructions=(
                "You are a helpful assistant that performs arithmetic. "
                "Always use the provided tools to compute results. "
                "Do not try to compute the answer yourself. "
                "Break complex operations into individual tool calls."
            ),
            tools=tools,
        )

        try:
            state = agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=(
                                "What is (3 + 4) * 2? "
                                "First add 3 and 4, then multiply the result by 2."
                            )
                        )
                    ]
                }
            )

            messages = state["messages"]

            # The final answer should contain "14"
            final_message = messages[-1]
            assert isinstance(final_message, AIMessage)
            assert (
                "14" in final_message.content
            ), f"Expected '14' in final answer, got: {final_message.content}"

            # At least two tool invocations should have happened
            tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
            assert len(tool_messages) >= 2, (
                f"Expected at least 2 ToolMessages for multi-step, got "
                f"{len(tool_messages)}"
            )

        finally:
            self.service.delete_agent(agent)

    # ------------------------------------------------------------------
    # Test: agent node can be used directly in a custom graph
    # ------------------------------------------------------------------

    def test_react_agent_node_in_custom_graph(self) -> None:
        """Build a REACT graph manually using create_prompt_agent_node."""
        from langgraph.graph import START, MessagesState, StateGraph
        from langgraph.prebuilt.tool_node import ToolNode

        from langchain_azure_ai.agents.v2 import (
            external_tools_condition,
        )

        tools = [add, multiply]

        agent_node = self.service.create_prompt_agent_node(
            name=f"test-react-custom-graph-v2-{uuid.uuid4().hex[:8]}",
            model=self.model,
            instructions=(
                "You are a helpful assistant that performs arithmetic. "
                "Always use the provided tools to compute results."
            ),
            tools=tools,
        )

        try:
            # Build the REACT graph manually
            builder = StateGraph(MessagesState)
            builder.add_node("agent", agent_node)
            builder.add_node("tools", ToolNode(tools))
            builder.add_edge(START, "agent")
            builder.add_conditional_edges("agent", external_tools_condition)
            builder.add_edge("tools", "agent")

            graph = builder.compile(name="test-react-custom-graph-v2")

            state = graph.invoke({"messages": [HumanMessage(content="What is 5 + 8?")]})

            messages = state["messages"]

            # Verify tool was called
            tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
            assert (
                len(tool_messages) >= 1
            ), "Expected at least one ToolMessage in custom graph"

            # Verify final answer
            final_message = messages[-1]
            assert isinstance(final_message, AIMessage)
            assert (
                "13" in final_message.content
            ), f"Expected '13' in final answer, got: {final_message.content}"

        finally:
            agent_node.delete_agent_from_node()
