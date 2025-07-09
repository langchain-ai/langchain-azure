"""Integration tests for Azure AI Agents."""

import os
import pytest
from azure.identity import DefaultAzureCredential

from langchain_azure_ai.azure_ai_agents import AzureAIAgentsService


@pytest.mark.requires("azure-ai-agents")
class TestAzureAIAgentsIntegration:
    """Integration tests for Azure AI Agents service."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        # These environment variables need to be set for integration tests
        self.endpoint = os.environ.get("PROJECT_ENDPOINT")
        self.model = os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4")
        
        if not self.endpoint:
            pytest.skip("PROJECT_ENDPOINT environment variable not set")

    def test_basic_agent_creation_and_interaction(self):
        """Test basic agent creation and interaction."""
        service = AzureAIAgentsService(
            endpoint=self.endpoint,
            credential=DefaultAzureCredential(),
            model=self.model,
            agent_name="test-integration-agent",
            instructions="You are a helpful test assistant. Keep responses brief.",
        )
        
        try:
            # Test basic generation
            response = service.invoke("What is 2+2?")
            assert response is not None
            assert len(response) > 0
            
            # Test that the agent was created
            agent = service.get_agent()
            assert agent is not None
            assert agent.id is not None
            
        finally:
            # Clean up
            service.delete_agent()

    def test_multiple_prompts(self):
        """Test handling multiple prompts."""
        service = AzureAIAgentsService(
            endpoint=self.endpoint,
            credential=DefaultAzureCredential(),
            model=self.model,
            agent_name="test-multi-prompt-agent",
            instructions="You are a helpful test assistant. Keep responses brief.",
        )
        
        try:
            prompts = [
                "What is the capital of France?",
                "What is 5 + 5?",
                "Name one color."
            ]
            
            result = service.generate(prompts)
            
            assert len(result.generations) == 3
            for generation_list in result.generations:
                assert len(generation_list) == 1
                assert len(generation_list[0].text) > 0
                
        finally:
            service.delete_agent()

    def test_agent_with_temperature(self):
        """Test agent creation with temperature parameter."""
        service = AzureAIAgentsService(
            endpoint=self.endpoint,
            credential=DefaultAzureCredential(),
            model=self.model,
            agent_name="test-temperature-agent",
            instructions="You are a helpful test assistant.",
            temperature=0.1,  # Low temperature for deterministic responses
        )
        
        try:
            response = service.invoke("Say exactly: 'Hello World'")
            assert response is not None
            # With low temperature, should be more deterministic
            
        finally:
            service.delete_agent()

    def test_direct_client_access(self):
        """Test accessing the underlying client directly."""
        service = AzureAIAgentsService(
            endpoint=self.endpoint,
            credential=DefaultAzureCredential(),
            model=self.model,
            agent_name="test-direct-client-agent",
            instructions="You are a helpful test assistant.",
        )
        
        try:
            # Get the underlying client
            client = service.get_client()
            assert client is not None
            
            # Create an agent manually using the client
            manual_agent = client.create_agent(
                model=self.model,
                name="manual-test-agent",
                instructions="You are a manually created test agent."
            )
            
            assert manual_agent.id is not None
            
            # Clean up the manual agent
            client.delete_agent(manual_agent.id)
            
        finally:
            service.delete_agent()

    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test async operations."""
        service = AzureAIAgentsService(
            endpoint=self.endpoint,
            credential=DefaultAzureCredential(),
            model=self.model,
            agent_name="test-async-agent",
            instructions="You are a helpful async test assistant.",
        )
        
        try:
            # Test async generation
            response = await service.ainvoke("What is async programming?")
            assert response is not None
            assert len(response) > 0
            
            # Test async multiple prompts
            prompts = ["Hello", "How are you?"]
            result = await service.agenerate(prompts)
            
            assert len(result.generations) == 2
            for generation_list in result.generations:
                assert len(generation_list) == 1
                assert len(generation_list[0].text) > 0
                
        finally:
            await service.adelete_agent()

    def test_error_handling_invalid_model(self):
        """Test error handling with invalid model."""
        service = AzureAIAgentsService(
            endpoint=self.endpoint,
            credential=DefaultAzureCredential(),
            model="non-existent-model",
            agent_name="test-error-agent",
            instructions="You are a test assistant.",
        )
        
        # This should raise an error when trying to create the agent
        with pytest.raises(Exception):
            service.invoke("Test message")

    def test_langchain_compatibility(self):
        """Test compatibility with LangChain patterns."""
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        service = AzureAIAgentsService(
            endpoint=self.endpoint,
            credential=DefaultAzureCredential(),
            model=self.model,
            agent_name="test-langchain-agent",
            instructions="You are a helpful assistant that answers questions clearly.",
        )
        
        try:
            # Create a simple chain
            prompt = PromptTemplate(
                input_variables=["question"],
                template="Please answer this question briefly: {question}"
            )
            
            chain = prompt | service | StrOutputParser()
            
            result = chain.invoke({"question": "What is AI?"})
            assert isinstance(result, str)
            assert len(result) > 0
            
        finally:
            service.delete_agent()

    def test_with_project_connection_string(self):
        """Test using project connection string instead of endpoint."""
        connection_string = os.environ.get("PROJECT_CONNECTION_STRING")
        
        if not connection_string:
            pytest.skip("PROJECT_CONNECTION_STRING environment variable not set")
            
        service = AzureAIAgentsService(
            project_connection_string=connection_string,
            credential=DefaultAzureCredential(),
            model=self.model,
            agent_name="test-connection-string-agent",
            instructions="You are a helpful test assistant.",
        )
        
        try:
            response = service.invoke("Hello from connection string!")
            assert response is not None
            assert len(response) > 0
            
        finally:
            service.delete_agent()
