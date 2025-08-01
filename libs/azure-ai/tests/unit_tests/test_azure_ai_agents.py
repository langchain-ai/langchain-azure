"""Unit tests for Azure AI Agents integration."""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

try:
    from azure.core.credentials import TokenCredential
except ImportError:
    pytest.skip("Azure dependencies not available", allow_module_level=True)

from langchain_azure_ai.azure_ai_agents import AzureAIAgentsService


class TestAzureAIAgentsService:
    """Test cases for AzureAIAgentsService."""

    @pytest.fixture
    def mock_credential(self) -> Mock:
        """Create a mock TokenCredential for testing."""
        mock_cred = Mock(spec=TokenCredential)
        return mock_cred

    def test_init_with_endpoint_and_credential(self, mock_credential: Mock) -> None:
        """Test initialization with endpoint and credential."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential=mock_credential,
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        assert service.endpoint == "https://test.azure.com"
        assert service.credential == mock_credential
        assert service.model == "gpt-4"
        assert service.agent_name == "test-agent"
        assert service.instructions == "Test instructions"

    def test_init_validation_error(self) -> None:
        """Test that initialization fails without endpoint."""
        with pytest.raises(ValidationError, match="Field required"):
            AzureAIAgentsService(  # type: ignore[call-arg]
                model="gpt-4", agent_name="test-agent", instructions="Test instructions"
            )

    def test_llm_type(self, mock_credential: Mock) -> None:
        """Test the _llm_type property."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential=mock_credential,
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        assert service._llm_type == "azure_ai_agents"

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AIProjectClient")
    def test_create_client(
        self, mock_ai_project_client: Any, mock_credential: Mock
    ) -> None:
        """Test client creation."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential=mock_credential,
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        mock_client = Mock()
        mock_ai_project_client.return_value = mock_client

        client = service._create_client()

        assert client == mock_client
        mock_ai_project_client.assert_called_once()

        # Test that credential is passed through as TokenCredential
        args, kwargs = mock_ai_project_client.call_args
        assert kwargs["endpoint"] == "https://test.azure.com"
        assert kwargs["credential"] == mock_credential

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AIProjectClient")
    def test_get_async_client(
        self, mock_ai_project_client: Any, mock_credential: Mock
    ) -> None:
        """Test getting the async client."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential=mock_credential,
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        mock_client = Mock()
        mock_ai_project_client.return_value = mock_client

        client = service.get_async_client()

        assert client == mock_client

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AIProjectClient")
    def test_get_or_create_agent(
        self, mock_ai_project_client: Any, mock_credential: Mock
    ) -> None:
        """Test agent creation."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential=mock_credential,
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
            temperature=0.7,
        )

        mock_client = Mock()
        mock_agent = Mock()
        mock_agent.id = "agent-123"

        # Mock the agents property and create_agent method
        mock_agents = Mock()
        mock_agents.create_agent.return_value = mock_agent
        mock_client.agents = mock_agents

        mock_ai_project_client.return_value = mock_client

        agent = service._get_or_create_agent()

        assert agent == mock_agent
        mock_agents.create_agent.assert_called_once()

        # Check that agent parameters were passed correctly
        args, kwargs = mock_agents.create_agent.call_args
        assert kwargs["model"] == "gpt-4"
        assert kwargs["name"] == "test-agent"
        assert kwargs["instructions"] == "Test instructions"
        assert kwargs["temperature"] == 0.7

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AIProjectClient")
    def test_generate_single(
        self, mock_ai_project_client: Any, mock_credential: Mock
    ) -> None:
        """Test single generation."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential=mock_credential,
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
        mock_content_item = Mock()
        mock_content_item.type = "text"
        mock_content_item.text.value = "This is the response"
        mock_response_message.content = [mock_content_item]

        # Make messages list directly iterable
        mock_messages_list = [mock_response_message]

        # Mock the agents property and all its sub-properties
        mock_agents = Mock()
        mock_agents.create_agent.return_value = mock_agent

        # Mock the threads property and its methods
        mock_threads = Mock()
        mock_threads.create.return_value = mock_thread
        mock_threads.delete.return_value = None
        mock_agents.threads = mock_threads

        # Mock the messages property and its methods
        mock_messages_client = Mock()
        mock_messages_client.create.return_value = mock_message
        mock_messages_client.list.return_value = mock_messages_list
        mock_agents.messages = mock_messages_client

        # Mock the runs property and its methods
        mock_runs = Mock()
        mock_runs.create_and_process.return_value = mock_run
        mock_agents.runs = mock_runs

        # Set the agents property on the client
        mock_client.agents = mock_agents

        mock_ai_project_client.return_value = mock_client

        generation = service._generate_single("Test prompt")

        assert generation.text == "This is the response"
        mock_threads.create.assert_called_once()
        mock_messages_client.create.assert_called_once()
        mock_runs.create_and_process.assert_called_once()
        mock_messages_client.list.assert_called_once()
        mock_threads.delete.assert_called_once()

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AIProjectClient")
    def test_generate_multiple_prompts(
        self, mock_ai_project_client: Any, mock_credential: Mock
    ) -> None:
        """Test generation with multiple prompts."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential=mock_credential,
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        # Mock similar to test_generate_single but for multiple calls
        mock_client = Mock()
        mock_agent = Mock()
        mock_agent.id = "agent-123"

        # Mock the agents property and create_agent method
        mock_agents = Mock()
        mock_agents.create_agent.return_value = mock_agent
        mock_client.agents = mock_agents

        # Mock thread creation to return different threads
        mock_threads = Mock()
        mock_threads.create.side_effect = [
            Mock(id="thread-1"),
            Mock(id="thread-2"),
        ]
        mock_agents.threads = mock_threads

        # Mock responses
        mock_response_1 = Mock()
        mock_response_1.role = "assistant"
        mock_content_1 = Mock()
        mock_content_1.type = "text"
        mock_content_1.text.value = "Response 1"
        mock_response_1.content = [mock_content_1]

        mock_response_2 = Mock()
        mock_response_2.role = "assistant"
        mock_content_2 = Mock()
        mock_content_2.type = "text"
        mock_content_2.text.value = "Response 2"
        mock_response_2.content = [mock_content_2]

        mock_messages_client = Mock()
        mock_messages_client.list.side_effect = [
            [mock_response_1],  # Direct list instead of Mock(data=...)
            [mock_response_2],
        ]
        mock_agents.messages = mock_messages_client

        mock_ai_project_client.return_value = mock_client

        result = service._generate(["Prompt 1", "Prompt 2"])

        assert len(result.generations) == 2
        assert result.generations[0][0].text == "Response 1"
        assert result.generations[1][0].text == "Response 2"

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AIProjectClient")
    def test_delete_agent(
        self, mock_ai_project_client: Any, mock_credential: Mock
    ) -> None:
        """Test agent deletion."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential=mock_credential,
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        mock_client = Mock()
        mock_agent = Mock()
        mock_agent.id = "agent-123"

        # Mock the agents property and create_agent method
        mock_agents = Mock()
        mock_agents.create_agent.return_value = mock_agent
        mock_agents.delete_agent.return_value = None
        mock_client.agents = mock_agents

        mock_ai_project_client.return_value = mock_client

        # Create agent first
        service._get_or_create_agent()

        # Delete agent
        service.delete_agent()

        mock_agents.delete_agent.assert_called_once_with("agent-123")
        assert service._agent is None

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AIProjectClient")
    def test_delete_specific_agent(
        self, mock_ai_project_client: Any, mock_credential: Mock
    ) -> None:
        """Test deletion of specific agent by ID."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential=mock_credential,
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        mock_client = Mock()
        mock_agents = Mock()
        mock_agents.delete_agent.return_value = None
        mock_client.agents = mock_agents

        mock_ai_project_client.return_value = mock_client

        service.delete_agent("specific-agent-id")

        mock_agents.delete_agent.assert_called_once_with("specific-agent-id")

    def test_delete_agent_without_creating(self, mock_credential: Mock) -> None:
        """Test that deleting agent without creating it raises error."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential=mock_credential,
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        with pytest.raises(ValueError, match="No agent to delete"):
            service.delete_agent()

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AIProjectClient")
    def test_get_client(
        self, mock_ai_project_client: Any, mock_credential: Mock
    ) -> None:
        """Test getting the client."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential=mock_credential,
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        mock_client = Mock()
        mock_ai_project_client.return_value = mock_client

        client = service.get_client()
        assert client == mock_client

    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AIProjectClient")
    def test_get_agent(
        self, mock_ai_project_client: Any, mock_credential: Mock
    ) -> None:
        """Test getting the agent."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential=mock_credential,
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        # Initially no agent
        assert service.get_agent() is None

        # Create agent
        mock_client = Mock()
        mock_agent = Mock()

        mock_agents = Mock()
        mock_agents.create_agent.return_value = mock_agent
        mock_client.agents = mock_agents

        mock_ai_project_client.return_value = mock_client

        service._get_or_create_agent()

        # Now agent should be available
        assert service.get_agent() == mock_agent

    @pytest.mark.asyncio
    @patch("langchain_azure_ai.azure_ai_agents.agent_service.AIProjectClient")
    async def test_async_generate_single(
        self, mock_ai_project_client: Any, mock_credential: Mock
    ) -> None:
        """Test async single generation."""
        service = AzureAIAgentsService(
            endpoint="https://test.azure.com",
            credential=mock_credential,
            model="gpt-4",
            agent_name="test-agent",
            instructions="Test instructions",
        )

        # Mock the client and its methods (same as sync since it uses to_thread)
        mock_client = Mock()
        mock_agent = Mock()
        mock_agent.id = "agent-123"
        mock_thread = Mock()
        mock_thread.id = "thread-123"

        # Mock the response message
        mock_response_message = Mock()
        mock_response_message.role = "assistant"
        mock_content_item = Mock()
        mock_content_item.type = "text"
        mock_content_item.text.value = "Async response"
        mock_response_message.content = [mock_content_item]

        mock_messages_list = [mock_response_message]

        # Mock the agents property and all its sub-properties
        mock_agents = Mock()
        mock_agents.create_agent.return_value = mock_agent

        # Mock the threads property and its methods
        mock_threads = Mock()
        mock_threads.create.return_value = mock_thread
        mock_threads.delete.return_value = None
        mock_agents.threads = mock_threads

        # Mock the messages property and its methods
        mock_messages_client = Mock()
        mock_messages_client.create.return_value = Mock()
        mock_messages_client.list.return_value = mock_messages_list
        mock_agents.messages = mock_messages_client

        # Mock the runs property and its methods
        mock_runs = Mock()
        mock_runs.create_and_process.return_value = Mock()
        mock_agents.runs = mock_runs

        # Set the agents property on the client
        mock_client.agents = mock_agents

        mock_ai_project_client.return_value = mock_client

        generation = await service._agenerate_single("Test async prompt")

        assert generation.text == "Async response"
