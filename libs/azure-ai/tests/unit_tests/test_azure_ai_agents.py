"""Unit tests for Azure AI Agents integration."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

from langchain_azure_ai.azure_ai_agents import AzureAIAgentsService


class TestAzureAIAgentsService:
    """Test cases for AzureAIAgentsService."""

    def test_init_with_endpoint_and_credential(self):
        """Test initialization with endpoint and credential."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential="test-key",
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        assert service.endpoint == "https://test.azure.com"
        assert isinstance(service.credential, str)
        assert service.model == "gpt-4"
        assert service.agent_name == "test-agent"
        assert service.instructions == "Test instructions"

    def test_init_with_project_connection_string(self):
        """Test initialization with project connection string."""
        service = AzureAIAgentsService(
            project_connection_string="test-connection-string",
            credential=DefaultAzureCredential(),
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        assert service.project_connection_string == "test-connection-string"
        assert isinstance(service.credential, DefaultAzureCredential)

    def test_init_validation_error(self):
        """Test that initialization fails without endpoint or connection string."""
        with pytest.raises(
            ValueError,
            match="Either 'endpoint' or 'project_connection_string' must be provided",
        ):
            AzureAIAgentsService(
                model="gpt-4", agent_name="test-agent", instructions="Test instructions"
            )

    def test_llm_type(self):
        """Test the _llm_type property."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential="test-key",
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        assert service._llm_type == "azure_ai_agents"

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AgentsClient")
    def test_create_client(self, mock_agents_client):
        """Test client creation."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential="test-key",
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        mock_client = Mock()
        mock_agents_client.return_value = mock_client

        client = service._create_client()

        assert client == mock_client
        mock_agents_client.assert_called_once()

        # Test that credential is converted to AzureKeyCredential
        args, kwargs = mock_agents_client.call_args
        assert kwargs["endpoint"] == "https://test.azure.com"
        assert isinstance(kwargs["credential"], AzureKeyCredential)

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AsyncAgentsClient")
    def test_create_async_client(self, mock_async_agents_client):
        """Test async client creation."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential="test-key",
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        mock_client = Mock()
        mock_async_agents_client.return_value = mock_client

        client = service._create_async_client()

        assert client == mock_client
        mock_async_agents_client.assert_called_once()

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AgentsClient")
    def test_get_or_create_agent(self, mock_agents_client):
        """Test agent creation."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential="test-key",
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
            temperature=0.7,
            max_completion_tokens=500,
        )

        mock_client = Mock()
        mock_agent = Mock()
        mock_agent.id = "agent-123"
        mock_client.create_agent.return_value = mock_agent
        mock_agents_client.return_value = mock_client

        agent = service._get_or_create_agent()

        assert agent == mock_agent
        mock_client.create_agent.assert_called_once()

        # Check that agent parameters were passed correctly
        args, kwargs = mock_client.create_agent.call_args
        assert kwargs["model"] == "gpt-4"
        assert kwargs["name"] == "test-agent"
        assert kwargs["instructions"] == "Test instructions"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_completion_tokens"] == 500

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AgentsClient")
    def test_generate_single(self, mock_agents_client):
        """Test single generation."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential="test-key",
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        # Mock the client and its methods
        mock_client = Mock()
        mock_agent = Mock()
        mock_agent.id = "agent-123"
        mock_thread = Mock()
        mock_thread.id = "thread-123"
        mock_message = Mock()
        mock_run = Mock()

        # Mock the response message
        mock_response_message = Mock()
        mock_response_message.role = "assistant"
        mock_text_message = Mock()
        mock_text_message.text.value = "This is the response"
        mock_response_message.text_messages = [mock_text_message]

        mock_messages = Mock()
        mock_messages.data = [mock_response_message]

        mock_client.create_agent.return_value = mock_agent
        mock_client.threads.create.return_value = mock_thread
        mock_client.messages.create.return_value = mock_message
        mock_client.runs.create_and_process.return_value = mock_run
        mock_client.messages.list.return_value = mock_messages
        mock_client.threads.delete.return_value = None

        mock_agents_client.return_value = mock_client

        generation = service._generate_single("Test prompt")

        assert generation.text == "This is the response"
        mock_client.threads.create.assert_called_once()
        mock_client.messages.create.assert_called_once()
        mock_client.runs.create_and_process.assert_called_once()
        mock_client.messages.list.assert_called_once()
        mock_client.threads.delete.assert_called_once()

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AgentsClient")
    def test_generate_multiple_prompts(self, mock_agents_client):
        """Test generation with multiple prompts."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential="test-key",
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        # Mock similar to test_generate_single but for multiple calls
        mock_client = Mock()
        mock_agent = Mock()
        mock_agent.id = "agent-123"

        mock_client.create_agent.return_value = mock_agent

        # Mock thread creation to return different threads
        mock_client.threads.create.side_effect = [
            Mock(id="thread-1"),
            Mock(id="thread-2"),
        ]

        # Mock responses
        mock_response_1 = Mock()
        mock_response_1.role = "assistant"
        mock_text_1 = Mock()
        mock_text_1.text.value = "Response 1"
        mock_response_1.text_messages = [mock_text_1]

        mock_response_2 = Mock()
        mock_response_2.role = "assistant"
        mock_text_2 = Mock()
        mock_text_2.text.value = "Response 2"
        mock_response_2.text_messages = [mock_text_2]

        mock_client.messages.list.side_effect = [
            Mock(data=[mock_response_1]),
            Mock(data=[mock_response_2]),
        ]

        mock_agents_client.return_value = mock_client

        result = service._generate(["Prompt 1", "Prompt 2"])

        assert len(result.generations) == 2
        assert result.generations[0][0].text == "Response 1"
        assert result.generations[1][0].text == "Response 2"

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AgentsClient")
    def test_delete_agent(self, mock_agents_client):
        """Test agent deletion."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential="test-key",
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        mock_client = Mock()
        mock_agent = Mock()
        mock_agent.id = "agent-123"

        mock_client.create_agent.return_value = mock_agent
        mock_agents_client.return_value = mock_client

        # Create agent first
        service._get_or_create_agent()

        # Delete agent
        service.delete_agent()

        mock_client.delete_agent.assert_called_once_with("agent-123")
        assert service._agent is None

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AgentsClient")
    def test_delete_specific_agent(self, mock_agents_client):
        """Test deletion of specific agent by ID."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential="test-key",
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        mock_client = Mock()
        mock_agents_client.return_value = mock_client

        service.delete_agent("specific-agent-id")

        mock_client.delete_agent.assert_called_once_with("specific-agent-id")

    def test_delete_agent_without_creating(self):
        """Test that deleting agent without creating it raises error."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential="test-key",
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        with pytest.raises(ValueError, match="No agent to delete"):
            service.delete_agent()

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AgentsClient")
    def test_get_client(self, mock_agents_client):
        """Test getting the client."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential="test-key",
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        mock_client = Mock()
        mock_agents_client.return_value = mock_client

        client = service.get_client()
        assert client == mock_client

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AsyncAgentsClient")
    def test_get_async_client(self, mock_async_agents_client):
        """Test getting the async client."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential="test-key",
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        mock_client = Mock()
        mock_async_agents_client.return_value = mock_client

        client = service.get_async_client()
        assert client == mock_client

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AgentsClient")
    def test_get_agent(self, mock_agents_client):
        """Test getting the agent."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential="test-key",
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        # Initially no agent
        assert service.get_agent() is None

        # Create agent
        mock_client = Mock()
        mock_agent = Mock()
        mock_client.create_agent.return_value = mock_agent
        mock_agents_client.return_value = mock_client

        service._get_or_create_agent()

        # Now agent should be available
        assert service.get_agent() == mock_agent

    @pytest.mark.asyncio
    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AsyncAgentsClient")
    async def test_async_generate_single(self, mock_async_agents_client):
        """Test async single generation."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential="test-key",
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        # Mock the async client and its methods
        mock_client = AsyncMock()
        mock_agent = Mock()
        mock_agent.id = "agent-123"
        mock_thread = Mock()
        mock_thread.id = "thread-123"

        # Mock the response message
        mock_response_message = Mock()
        mock_response_message.role = "assistant"
        mock_text_message = Mock()
        mock_text_message.text.value = "Async response"
        mock_response_message.text_messages = [mock_text_message]

        mock_messages = Mock()
        mock_messages.data = [mock_response_message]

        mock_client.create_agent.return_value = mock_agent
        mock_client.threads.create.return_value = mock_thread
        mock_client.messages.create.return_value = Mock()
        mock_client.runs.create_and_process.return_value = Mock()
        mock_client.messages.list.return_value = mock_messages
        mock_client.threads.delete.return_value = None

        mock_async_agents_client.return_value = mock_client

        generation = await service._agenerate_single("Test async prompt")

        assert generation.text == "Async response"
