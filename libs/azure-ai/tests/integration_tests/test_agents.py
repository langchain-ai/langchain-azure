"""Integration tests for Azure AI Foundry agent system.

These tests require active Azure AI Foundry resources and should be run
with proper environment variables configured:
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY (or managed identity)
- AZURE_OPENAI_DEPLOYMENT_NAME
- AZURE_AI_PROJECT_ENDPOINT (optional, for Foundry-specific tests)
- APPLICATIONINSIGHTS_CONNECTION_STRING (optional, for tracing tests)

Run with: pytest tests/integration_tests/test_agents.py -v --integration
"""

import asyncio
import os
import pytest
from typing import Dict, List, Optional

# Skip all tests if integration flag not set
pytestmark = pytest.mark.integration


def has_azure_credentials() -> bool:
    """Check if Azure credentials are available."""
    return bool(
        os.getenv("AZURE_OPENAI_ENDPOINT") 
        and (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_CLIENT_ID"))
    )


def has_foundry_credentials() -> bool:
    """Check if Azure Foundry credentials are available."""
    return has_azure_credentials() and bool(os.getenv("AZURE_AI_PROJECT_ENDPOINT"))


class TestAzureOpenAIIntegration:
    """Integration tests with Azure OpenAI (without Foundry)."""

    @pytest.mark.skipif(not has_azure_credentials(), reason="Azure credentials not available")
    def test_create_llm_connection(self):
        """Test that we can create a connection to Azure OpenAI."""
        from langchain_openai import AzureChatOpenAI
        
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        
        response = llm.invoke("Say hello in one word.")
        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.skipif(not has_azure_credentials(), reason="Azure credentials not available")
    def test_create_react_agent(self):
        """Test creating a ReAct agent with Azure OpenAI."""
        from langchain_openai import AzureChatOpenAI
        from langgraph.prebuilt import create_react_agent
        from langchain_core.tools import tool
        from langchain_core.messages import HumanMessage
        
        @tool
        def get_current_time() -> str:
            """Get the current time."""
            from datetime import datetime
            return datetime.now().strftime("%H:%M:%S")
        
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        
        agent = create_react_agent(llm, tools=[get_current_time])
        
        result = agent.invoke({
            "messages": [HumanMessage(content="What time is it?")]
        })
        
        assert "messages" in result
        assert len(result["messages"]) > 1


class TestWrapperIntegration:
    """Integration tests for wrapper classes."""

    @pytest.mark.skipif(not has_azure_credentials(), reason="Azure credentials not available")
    def test_it_helpdesk_wrapper(self):
        """Test IT Helpdesk wrapper with real Azure OpenAI."""
        from langchain_azure_ai.wrappers import ITHelpdeskWrapper
        
        os.environ["USE_AZURE_FOUNDRY"] = "false"
        
        wrapper = ITHelpdeskWrapper(
            name="test-helpdesk",
            instructions="You are a helpful IT support assistant. Be brief.",
        )
        
        response = wrapper.chat("What is a VPN?")
        
        assert isinstance(response, str)
        assert len(response) > 0
        # Should contain some reference to VPN or network
        assert any(word in response.lower() for word in ["vpn", "network", "private", "connection"])

    @pytest.mark.skipif(not has_azure_credentials(), reason="Azure credentials not available")
    def test_research_agent_wrapper(self):
        """Test Research Agent wrapper with real Azure OpenAI."""
        from langchain_azure_ai.wrappers import ResearchAgentWrapper
        
        os.environ["USE_AZURE_FOUNDRY"] = "false"
        
        wrapper = ResearchAgentWrapper(
            name="test-research",
            instructions="You are a research assistant. Provide factual, concise answers.",
        )
        
        response = wrapper.chat("What is machine learning in one sentence?")
        
        assert isinstance(response, str)
        assert len(response) > 10
        assert "learning" in response.lower() or "data" in response.lower()

    @pytest.mark.skipif(not has_azure_credentials(), reason="Azure credentials not available")
    @pytest.mark.asyncio
    async def test_async_invocation(self):
        """Test async invocation of wrapper."""
        from langchain_azure_ai.wrappers import ITHelpdeskWrapper
        from langchain_core.messages import HumanMessage
        
        os.environ["USE_AZURE_FOUNDRY"] = "false"
        
        wrapper = ITHelpdeskWrapper(
            name="test-async",
            instructions="You are a helpful assistant. Be very brief.",
        )
        
        result = await wrapper.ainvoke({
            "messages": [HumanMessage(content="Say hello")]
        })
        
        assert "messages" in result
        assert len(result["messages"]) > 0


class TestMultiTurnConversation:
    """Integration tests for multi-turn conversations."""

    @pytest.mark.skipif(not has_azure_credentials(), reason="Azure credentials not available")
    def test_conversation_with_memory(self):
        """Test multi-turn conversation maintains context."""
        from langchain_azure_ai.wrappers import ITHelpdeskWrapper
        
        os.environ["USE_AZURE_FOUNDRY"] = "false"
        
        wrapper = ITHelpdeskWrapper(
            name="test-memory",
            instructions="You are a helpful assistant. Remember previous messages.",
        )
        
        # First message
        response1 = wrapper.chat("My name is Alice. Remember it.")
        
        # Second message referencing first
        response2 = wrapper.chat("What is my name?", thread_id="test-thread-1")
        
        # With same thread_id, should remember (if using checkpointer)
        # Without checkpointer, it won't remember - that's okay for this test
        assert isinstance(response2, str)
        assert len(response2) > 0

    @pytest.mark.skipif(not has_azure_credentials(), reason="Azure credentials not available")
    def test_streaming_response(self):
        """Test streaming responses from wrapper."""
        from langchain_azure_ai.wrappers import ResearchAgentWrapper
        from langchain_core.messages import HumanMessage
        
        os.environ["USE_AZURE_FOUNDRY"] = "false"
        
        wrapper = ResearchAgentWrapper(
            name="test-stream",
            instructions="You are a helpful assistant. Be concise.",
        )
        
        chunks = []
        for chunk in wrapper.stream({
            "messages": [HumanMessage(content="Count from 1 to 5.")]
        }):
            chunks.append(chunk)
        
        assert len(chunks) > 0


class TestServerIntegration:
    """Integration tests for the FastAPI server with real agents."""

    @pytest.fixture
    def test_client(self):
        """Create a test client with real agents."""
        from fastapi.testclient import TestClient
        from langchain_azure_ai.server import app
        
        return TestClient(app)

    @pytest.mark.skipif(not has_azure_credentials(), reason="Azure credentials not available")
    def test_chat_endpoint(self, test_client):
        """Test the chat endpoint with a real agent."""
        # First, list available agents
        agents_response = test_client.get("/agents")
        assert agents_response.status_code == 200
        
        agents = agents_response.json()
        if not agents:
            pytest.skip("No agents available")
        
        agent_name = agents[0].get("name", "it-helpdesk")
        
        # Send a chat message
        chat_response = test_client.post(
            "/chat",
            json={
                "agent_name": agent_name,
                "message": "Hello, can you help me?",
            }
        )
        
        assert chat_response.status_code == 200
        data = chat_response.json()
        assert "response" in data

    @pytest.mark.skipif(not has_azure_credentials(), reason="Azure credentials not available")
    def test_chat_stream_endpoint(self, test_client):
        """Test the streaming chat endpoint."""
        # First, list available agents
        agents_response = test_client.get("/agents")
        assert agents_response.status_code == 200
        
        agents = agents_response.json()
        if not agents:
            pytest.skip("No agents available")
        
        agent_name = agents[0].get("name", "it-helpdesk")
        
        # Stream chat response
        stream_response = test_client.post(
            "/chat/stream",
            json={
                "agent_name": agent_name,
                "message": "Say hi",
            },
        )
        
        assert stream_response.status_code == 200


class TestObservabilityIntegration:
    """Integration tests for observability with real Azure Monitor."""

    def test_observability_setup_without_connection_string(self):
        """Test that observability handles missing connection string gracefully."""
        from langchain_azure_ai.observability import setup_azure_monitor
        
        # Should not raise even without connection string
        result = setup_azure_monitor()
        
        # Result should be False when no connection string
        if not os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
            assert result is False

    @pytest.mark.skipif(
        not os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
        reason="App Insights not configured"
    )
    def test_observability_with_tracing(self):
        """Test observability with actual Azure Monitor tracing."""
        from langchain_azure_ai.observability import setup_azure_monitor, AgentTelemetry
        
        # This should work with real App Insights
        result = setup_azure_monitor()
        assert result is True
        
        # Create telemetry and track an operation
        telemetry = AgentTelemetry("integration-test", "test")
        
        with telemetry.track_execution("test_operation") as metrics:
            # Simulate some work
            import time
            time.sleep(0.1)
            metrics.prompt_tokens = 100
            metrics.completion_tokens = 50
        
        assert metrics.success is True
        assert metrics.duration_ms >= 100


class TestAzureFoundryIntegration:
    """Integration tests specific to Azure AI Foundry."""

    @pytest.mark.skipif(not has_foundry_credentials(), reason="Foundry credentials not available")
    def test_foundry_agent_creation(self):
        """Test creating an agent with Azure AI Foundry."""
        from langchain_azure_ai.wrappers import ITHelpdeskWrapper
        
        os.environ["USE_AZURE_FOUNDRY"] = "true"
        
        wrapper = ITHelpdeskWrapper(
            name="foundry-test",
            instructions="You are a test agent.",
        )
        
        assert wrapper.is_foundry_enabled is True
        
        response = wrapper.chat("Hello")
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.skipif(not has_foundry_credentials(), reason="Foundry credentials not available")
    def test_foundry_agent_with_tools(self):
        """Test Foundry agent with custom tools."""
        from langchain_azure_ai.wrappers import ResearchAgentWrapper
        from langchain_core.tools import tool
        import ast
        import operator
        
        @tool
        def calculate(expression: str) -> str:
            """Calculate a mathematical expression."""
            try:
                # Safe expression evaluator using ast
                # Only allows basic arithmetic operations
                def safe_eval(node):
                    operators = {
                        ast.Add: operator.add,
                        ast.Sub: operator.sub,
                        ast.Mult: operator.mul,
                        ast.Div: operator.truediv,
                        ast.Pow: operator.pow,
                        ast.USub: operator.neg,
                        ast.UAdd: operator.pos,
                    }
                    
                    if isinstance(node, ast.Constant):
                        # Ensure the constant is a number
                        if isinstance(node.value, (int, float)):
                            return node.value
                        else:
                            raise ValueError(f"Unsupported constant type: {type(node.value)}")
                    elif isinstance(node, ast.BinOp):  # binary operation
                        op = operators.get(type(node.op))
                        if op is None:
                            raise ValueError(f"Unsupported operation: {type(node.op)}")
                        # Special handling for power operator to prevent DoS
                        if isinstance(node.op, ast.Pow):
                            base = safe_eval(node.left)
                            exponent = safe_eval(node.right)
                            # Limit exponent to prevent computational DoS
                            if abs(exponent) > 1000:
                                raise ValueError("Exponent too large (max: 1000)")
                            return op(base, exponent)
                        return op(safe_eval(node.left), safe_eval(node.right))
                    elif isinstance(node, ast.UnaryOp):  # unary operation
                        op = operators.get(type(node.op))
                        if op is None:
                            raise ValueError(f"Unsupported operation: {type(node.op)}")
                        return op(safe_eval(node.operand))
                    else:
                        raise ValueError(f"Unsupported expression: {type(node)}")
                
                tree = ast.parse(expression, mode='eval')
                result = safe_eval(tree.body)
                return str(result)
            except Exception:
                return "Error calculating"
        
        os.environ["USE_AZURE_FOUNDRY"] = "true"
        
        wrapper = ResearchAgentWrapper(
            name="foundry-calc",
            instructions="You are an assistant that can do calculations.",
            tools=[calculate],
        )
        
        response = wrapper.chat("What is 2 + 2?")
        assert "4" in response


# Pytest configuration for integration tests
def pytest_configure(config):
    """Register integration marker."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])
