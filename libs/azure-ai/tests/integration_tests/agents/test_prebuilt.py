"""Integration tests for Azure AI Agents."""

import os

import pytest

try:
    from azure.identity import DefaultAzureCredential
except ImportError:
    pytest.skip("Azure dependencies not available", allow_module_level=True)

from langchain_core.messages import HumanMessage

from langchain_azure_ai.agents import AgentServiceFactory


@pytest.mark.requires("azure-ai-agents")
class TestAgentServiceFactoryIntegration:
    """Integration tests for Azure AI Agents service."""

    service: AgentServiceFactory
    model: str

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test environment."""
        # These environment variables need to be set for integration tests
        endpoint = os.environ.get("PROJECT_ENDPOINT")

        if not endpoint:
            pytest.skip("PROJECT_ENDPOINT environment variable not set")

        self.service = AgentServiceFactory(
            project_endpoint=endpoint,
            credential=DefaultAzureCredential(),
        )
        self.model = os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4.1")

    def test_basic_agent_creation_and_interaction(self) -> None:
        """Test basic agent creation and interaction."""
        agent = self.service.create_prompt_agent(
            name="test-integration-agent",
            model=self.model,
            instructions="You are a helpful test assistant. Keep responses brief.",
        )

        try:
            # Test basic generation
            input = {"messages": [HumanMessage(content="What is 2+2?")]}
            response = agent.invoke(input)
            assert response is not None
            assert "messages" in response
            assert len(response["messages"]) > 0

            # Test that the agent was created
            agent_id = self.service.get_agents_id_from_graph(agent)
            assert agent_id is not None

        finally:
            # Clean up
            self.service.delete_agent(agent)

    def test_multi_turn(self) -> None:
        """Test handling multiple turns."""
        agent = self.service.create_prompt_agent(
            model=self.model,
            name="test-multi-turn-agent",
            instructions="You are a helpful test assistant. Keep responses brief.",
        )

        try:
            state = {"messages": [HumanMessage(content="My name is John.")]}
            state = agent.invoke(state)
            state["messages"].append(
                HumanMessage(content="Did I mention my name? Answer yes or no.")
            )
            state = agent.invoke(state)

            assert state is not None
            assert "messages" in state
            assert len(state["messages"]) == 4
            assert isinstance(state["messages"][-1].content, str)
            assert state["messages"][-1].content == "yes"
        finally:
            self.service.delete_agent(agent)

    def test_agent_with_temperature(self) -> None:
        """Test agent creation with temperature parameter."""
        agent = self.service.create_prompt_agent(
            model=self.model,
            name="test-temperature-agent",
            instructions="You are a helpful test assistant.",
            temperature=0.1,  # Low temperature for deterministic responses
        )

        try:
            input = {"messages": [HumanMessage(content="Say exactly: 'Hello World'")]}
            response = agent.invoke(input)
            assert response is not None

            # With low temperature, should be more deterministic
            assert "messages" in response
            assert response["messages"][0].content == "Hello World"

        finally:
            self.service.delete_agent(agent)

    def test_delete_agent(self) -> None:
        """Test agent deletion."""
        agent = self.service.create_prompt_agent(
            model=self.model,
            name="test-temperature-agent",
            instructions="You are a helpful test assistant.",
        )

        assert self.service.get_agents_id_from_graph(agent) is not None

        self.service.delete_agent(agent)

        assert self.service.get_agents_id_from_graph(agent) is None

    @pytest.mark.asyncio
    async def test_async_operations(self) -> None:
        """Test async operations."""
        agent = self.service.create_prompt_agent(
            model=self.model,
            name="test-temperature-agent",
            instructions="You are a helpful test assistant.",
        )

        try:
            # Test async generation
            input = {"messages": [HumanMessage(content="What is async programming?")]}
            response = await agent.ainvoke(input)
            assert response is not None
            assert len(response["messages"]) > 0
        finally:
            self.service.delete_agent(agent)

    def test_error_handling_invalid_model(self) -> None:
        """Test error handling with invalid model."""
        agent = self.service.create_prompt_agent(
            model="non-existent-model",
            name="test-error-agent",
            instructions="You are a test assistant.",
        )

        # This should raise an error when trying to create the agent
        with pytest.raises(RuntimeError):
            input = {"messages": [HumanMessage(content="Hello")]}
            agent.invoke(input)

        self.service.delete_agent(agent)
