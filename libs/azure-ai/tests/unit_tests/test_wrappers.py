"""Comprehensive unit tests for the Azure AI Foundry wrapper system.

This module tests:
- Base wrapper class functionality
- IT agent wrappers
- Enterprise agent wrappers
- DeepAgent wrappers
- Server endpoints with mocked agents
- Observability components
"""

import asyncio
import os
import pytest
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Test fixtures and mocks
@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-12-01-preview")
    monkeypatch.setenv("USE_AZURE_FOUNDRY", "false")
    monkeypatch.setenv("ENABLE_TRACING", "false")


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    from langchain_core.messages import AIMessage
    
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content="Test response")
    return mock


@pytest.fixture
def mock_agent():
    """Create a mock compiled agent graph."""
    from langchain_core.messages import AIMessage, HumanMessage
    
    mock = MagicMock()
    mock.invoke.return_value = {
        "messages": [
            HumanMessage(content="Test input"),
            AIMessage(content="Test response from agent"),
        ]
    }
    mock.ainvoke = AsyncMock(return_value={
        "messages": [
            HumanMessage(content="Test input"),
            AIMessage(content="Async test response"),
        ]
    })
    return mock


class TestWrapperConfig:
    """Tests for WrapperConfig dataclass."""

    def test_from_env_defaults(self, mock_env_vars):
        """Test loading config from environment with defaults."""
        from langchain_azure_ai.wrappers.base import WrapperConfig
        
        config = WrapperConfig.from_env()
        
        assert config.use_azure_foundry is False
        assert config.enable_tracing is False
        assert config.langsmith_enabled is True

    def test_from_env_azure_foundry_enabled(self, mock_env_vars, monkeypatch):
        """Test loading config with Azure Foundry enabled."""
        from langchain_azure_ai.wrappers.base import WrapperConfig
        
        monkeypatch.setenv("USE_AZURE_FOUNDRY", "true")
        monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", "https://test.ai.azure.com")
        
        config = WrapperConfig.from_env()
        
        assert config.use_azure_foundry is True
        assert config.project_endpoint == "https://test.ai.azure.com"

    def test_validate_missing_endpoint(self, mock_env_vars, monkeypatch):
        """Test validation catches missing endpoint when Foundry enabled."""
        from langchain_azure_ai.wrappers.base import WrapperConfig
        
        monkeypatch.setenv("USE_AZURE_FOUNDRY", "true")
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        
        config = WrapperConfig.from_env()
        issues = config.validate()
        
        assert len(issues) == 1
        assert "AZURE_AI_PROJECT_ENDPOINT not set" in issues[0]


class TestFoundryAgentWrapper:
    """Tests for the base FoundryAgentWrapper class."""

    def test_init_with_existing_agent(self, mock_env_vars, mock_agent):
        """Test initializing wrapper with an existing agent."""
        from langchain_azure_ai.wrappers.base import FoundryAgentWrapper, AgentType
        
        # Create a concrete subclass for testing
        class TestWrapper(FoundryAgentWrapper):
            def _create_agent_impl(self, llm, tools):
                return mock_agent
        
        wrapper = TestWrapper(
            name="test-agent",
            instructions="Test instructions",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-agent"
        assert wrapper._agent == mock_agent
        assert wrapper.agent_type == AgentType.CUSTOM

    def test_invoke_returns_response(self, mock_env_vars, mock_agent):
        """Test that invoke returns agent response."""
        from langchain_azure_ai.wrappers.base import FoundryAgentWrapper
        from langchain_core.messages import HumanMessage
        
        class TestWrapper(FoundryAgentWrapper):
            def _create_agent_impl(self, llm, tools):
                return mock_agent
        
        wrapper = TestWrapper(
            name="test-agent",
            instructions="Test instructions",
            existing_agent=mock_agent,
        )
        
        result = wrapper.invoke({"messages": [HumanMessage(content="Hello")]})
        
        assert "messages" in result
        assert len(result["messages"]) == 2

    def test_chat_returns_string(self, mock_env_vars, mock_agent):
        """Test that chat returns a string response."""
        from langchain_azure_ai.wrappers.base import FoundryAgentWrapper
        
        class TestWrapper(FoundryAgentWrapper):
            def _create_agent_impl(self, llm, tools):
                return mock_agent
        
        wrapper = TestWrapper(
            name="test-agent",
            instructions="Test instructions",
            existing_agent=mock_agent,
        )
        
        response = wrapper.chat("Hello, agent!")
        
        assert isinstance(response, str)
        assert response == "Test response from agent"

    @pytest.mark.asyncio
    async def test_ainvoke_async(self, mock_env_vars, mock_agent):
        """Test async invocation."""
        from langchain_azure_ai.wrappers.base import FoundryAgentWrapper
        from langchain_core.messages import HumanMessage
        
        class TestWrapper(FoundryAgentWrapper):
            def _create_agent_impl(self, llm, tools):
                return mock_agent
        
        wrapper = TestWrapper(
            name="test-agent",
            instructions="Test instructions",
            existing_agent=mock_agent,
        )
        
        result = await wrapper.ainvoke({"messages": [HumanMessage(content="Hello")]})
        
        assert "messages" in result

    def test_is_foundry_enabled_false_by_default(self, mock_env_vars, mock_agent):
        """Test that Foundry is disabled by default."""
        from langchain_azure_ai.wrappers.base import FoundryAgentWrapper
        
        class TestWrapper(FoundryAgentWrapper):
            def _create_agent_impl(self, llm, tools):
                return mock_agent
        
        wrapper = TestWrapper(
            name="test-agent",
            instructions="Test instructions",
            existing_agent=mock_agent,
        )
        
        assert wrapper.is_foundry_enabled is False

    def test_context_manager(self, mock_env_vars, mock_agent):
        """Test using wrapper as context manager."""
        from langchain_azure_ai.wrappers.base import FoundryAgentWrapper
        
        class TestWrapper(FoundryAgentWrapper):
            def _create_agent_impl(self, llm, tools):
                return mock_agent
        
        with TestWrapper(
            name="test-agent",
            instructions="Test instructions",
            existing_agent=mock_agent,
        ) as wrapper:
            assert wrapper.name == "test-agent"


class TestITAgentWrappers:
    """Tests for IT agent wrappers."""

    def test_helpdesk_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating IT Helpdesk wrapper."""
        from langchain_azure_ai.wrappers import ITHelpdeskWrapper
        
        wrapper = ITHelpdeskWrapper(
            name="test-helpdesk",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-helpdesk"
        assert wrapper.agent_subtype == "helpdesk"

    def test_servicenow_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating ServiceNow wrapper."""
        from langchain_azure_ai.wrappers import ServiceNowWrapper
        
        wrapper = ServiceNowWrapper(
            name="test-servicenow",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-servicenow"
        assert wrapper.agent_subtype == "servicenow"

    def test_hitl_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating HITL Support wrapper."""
        from langchain_azure_ai.wrappers import HITLSupportWrapper
        
        wrapper = HITLSupportWrapper(
            name="test-hitl",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-hitl"
        assert wrapper.agent_subtype == "hitl"


class TestEnterpriseAgentWrappers:
    """Tests for Enterprise agent wrappers."""

    def test_research_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating Research Agent wrapper."""
        from langchain_azure_ai.wrappers import ResearchAgentWrapper
        
        wrapper = ResearchAgentWrapper(
            name="test-research",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-research"
        assert wrapper.agent_subtype == "research"

    def test_content_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating Content Agent wrapper."""
        from langchain_azure_ai.wrappers import ContentAgentWrapper
        
        wrapper = ContentAgentWrapper(
            name="test-content",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-content"
        assert wrapper.agent_subtype == "content"

    def test_data_analyst_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating Data Analyst wrapper."""
        from langchain_azure_ai.wrappers import DataAnalystWrapper
        
        wrapper = DataAnalystWrapper(
            name="test-data-analyst",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-data-analyst"
        assert wrapper.agent_subtype == "data_analyst"

    def test_code_assistant_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating Code Assistant wrapper."""
        from langchain_azure_ai.wrappers import CodeAssistantWrapper
        
        wrapper = CodeAssistantWrapper(
            name="test-code-assistant",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-code-assistant"
        assert wrapper.agent_subtype == "code_assistant"


class TestDeepAgentWrappers:
    """Tests for DeepAgent wrappers."""

    def test_it_operations_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating IT Operations wrapper."""
        from langchain_azure_ai.wrappers import ITOperationsWrapper
        
        wrapper = ITOperationsWrapper(
            name="test-it-ops",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-it-ops"
        assert wrapper.agent_subtype == "it_operations"

    def test_sales_intelligence_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating Sales Intelligence wrapper."""
        from langchain_azure_ai.wrappers import SalesIntelligenceWrapper
        
        wrapper = SalesIntelligenceWrapper(
            name="test-sales",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-sales"
        assert wrapper.agent_subtype == "sales_intelligence"

    def test_recruitment_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating Recruitment wrapper."""
        from langchain_azure_ai.wrappers import RecruitmentWrapper
        
        wrapper = RecruitmentWrapper(
            name="test-recruitment",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-recruitment"
        assert wrapper.agent_subtype == "recruitment"


class TestObservability:
    """Tests for observability module."""

    def test_telemetry_config_from_env(self, mock_env_vars, monkeypatch):
        """Test TelemetryConfig creation from environment."""
        from langchain_azure_ai.observability import TelemetryConfig
        
        monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "test-connection")
        monkeypatch.setenv("ENABLE_AZURE_MONITOR", "true")
        
        config = TelemetryConfig.from_env()
        
        assert config.app_insights_connection == "test-connection"
        assert config.enable_azure_monitor is True

    def test_execution_metrics_finalize(self):
        """Test ExecutionMetrics finalization."""
        from langchain_azure_ai.observability import ExecutionMetrics
        import time
        
        metrics = ExecutionMetrics(agent_name="test", agent_type="custom")
        metrics.prompt_tokens = 100
        metrics.completion_tokens = 50
        
        time.sleep(0.01)  # Small delay to ensure measurable duration
        metrics.finalize()
        
        assert metrics.end_time is not None
        assert metrics.duration_ms > 0
        assert metrics.total_tokens == 150

    def test_agent_telemetry_track_execution(self, mock_env_vars):
        """Test AgentTelemetry context manager."""
        from langchain_azure_ai.observability import AgentTelemetry
        
        telemetry = AgentTelemetry("test-agent", "enterprise")
        
        with telemetry.track_execution("invoke") as metrics:
            metrics.prompt_tokens = 50
            metrics.completion_tokens = 25
        
        assert metrics.success is True
        assert metrics.duration_ms > 0
        assert metrics.total_tokens == 75

    def test_agent_telemetry_error_tracking(self, mock_env_vars):
        """Test that errors are tracked correctly."""
        from langchain_azure_ai.observability import AgentTelemetry
        
        telemetry = AgentTelemetry("test-agent", "enterprise")
        
        # Capture metrics reference for validation after exception
        captured_metrics = None
        try:
            with telemetry.track_execution("invoke") as metrics:
                captured_metrics = metrics
                raise ValueError("Test error")
        except ValueError:
            pass
        
        assert captured_metrics is not None
        assert captured_metrics.success is False
        assert "Test error" in captured_metrics.error


class TestServerEndpoints:
    """Tests for FastAPI server endpoints."""

    @pytest.fixture
    def test_client(self):
        """Create a test client for the FastAPI app."""
        from fastapi.testclient import TestClient
        from langchain_azure_ai.server import app
        
        return TestClient(app)

    def test_health_endpoint(self, test_client):
        """Test the health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "agents_loaded" in data

    def test_list_agents_endpoint(self, test_client):
        """Test the agents listing endpoint."""
        response = test_client.get("/agents")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_chat_ui_endpoint(self, test_client):
        """Test the chat UI endpoint returns HTML."""
        response = test_client.get("/chat")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestMiddleware:
    """Tests for observability middleware."""

    def test_request_logging_middleware_init(self):
        """Test RequestLoggingMiddleware initialization."""
        from langchain_azure_ai.observability.middleware import RequestLoggingMiddleware
        from starlette.applications import Starlette
        
        app = Starlette()
        middleware = RequestLoggingMiddleware(
            app,
            log_request_body=True,
            exclude_paths=["/health"],
        )
        
        assert middleware.log_request_body is True
        assert "/health" in middleware.exclude_paths

    def test_tracing_middleware_init(self):
        """Test TracingMiddleware initialization."""
        from langchain_azure_ai.observability.middleware import TracingMiddleware
        from starlette.applications import Starlette
        
        app = Starlette()
        middleware = TracingMiddleware(
            app,
            service_name="test-service",
        )
        
        assert middleware.service_name == "test-service"

    def test_metrics_middleware_init(self):
        """Test MetricsMiddleware initialization."""
        from langchain_azure_ai.observability.middleware import MetricsMiddleware
        from starlette.applications import Starlette
        
        app = Starlette()
        middleware = MetricsMiddleware(
            app,
            service_name="test-service",
        )
        
        assert middleware.service_name == "test-service"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
