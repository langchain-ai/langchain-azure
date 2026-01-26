"""Pytest configuration and shared fixtures for Azure AI tests.

This module provides:
- Shared fixtures for mocking Azure services
- Environment variable setup for different test modes
- Custom pytest markers for categorizing tests
"""

import os
import pytest
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, AsyncMock, patch


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires Azure credentials)"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "foundry: mark test as requiring Azure AI Foundry"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --integration flag is passed."""
    if not config.getoption("--integration", default=False):
        skip_integration = pytest.mark.skip(reason="need --integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that require Azure credentials"
    )


# ============================================================================
# Environment Fixtures
# ============================================================================

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
def mock_foundry_env_vars(monkeypatch, mock_env_vars):
    """Set up environment for Azure AI Foundry testing."""
    monkeypatch.setenv("USE_AZURE_FOUNDRY", "true")
    monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", "https://test.ai.azure.com/projects/test-project")


@pytest.fixture
def mock_observability_env(monkeypatch, mock_env_vars):
    """Set up environment for observability testing."""
    monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "InstrumentationKey=test-key")
    monkeypatch.setenv("ENABLE_AZURE_MONITOR", "true")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "test-service")


# ============================================================================
# Mock Object Fixtures
# ============================================================================

@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    from langchain_core.messages import AIMessage
    
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content="Test response from mock LLM")
    mock.ainvoke = AsyncMock(return_value=AIMessage(content="Async test response from mock LLM"))
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
            AIMessage(content="Async test response from agent"),
        ]
    })
    mock.stream = MagicMock(return_value=iter([
        {"messages": [AIMessage(content="Streaming response chunk 1")]},
        {"messages": [AIMessage(content="Streaming response chunk 2")]},
    ]))
    mock.astream = AsyncMock(return_value=mock_async_generator([
        {"messages": [AIMessage(content="Async streaming chunk")]},
    ]))
    return mock


async def mock_async_generator(items):
    """Helper to create async generator from list."""
    for item in items:
        yield item


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    from langchain_core.tools import tool
    
    @tool
    def mock_search(query: str) -> str:
        """Mock search tool for testing."""
        return f"Search results for: {query}"
    
    return mock_search


# ============================================================================
# Wrapper Fixtures
# ============================================================================

@pytest.fixture
def it_helpdesk_wrapper(mock_env_vars, mock_agent):
    """Create an IT Helpdesk wrapper with mock agent."""
    from langchain_azure_ai.wrappers import ITHelpdeskWrapper
    
    return ITHelpdeskWrapper(
        name="test-helpdesk",
        instructions="Test IT helpdesk agent",
        existing_agent=mock_agent,
    )


@pytest.fixture
def research_wrapper(mock_env_vars, mock_agent):
    """Create a Research Agent wrapper with mock agent."""
    from langchain_azure_ai.wrappers import ResearchAgentWrapper
    
    return ResearchAgentWrapper(
        name="test-research",
        instructions="Test research agent",
        existing_agent=mock_agent,
    )


@pytest.fixture
def code_assistant_wrapper(mock_env_vars, mock_agent):
    """Create a Code Assistant wrapper with mock agent."""
    from langchain_azure_ai.wrappers import CodeAssistantWrapper
    
    return CodeAssistantWrapper(
        name="test-code-assistant",
        instructions="Test code assistant agent",
        existing_agent=mock_agent,
    )


# ============================================================================
# Server Fixtures
# ============================================================================

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    from langchain_azure_ai.server import app
    
    return TestClient(app)


@pytest.fixture
def mock_server_agents(monkeypatch, mock_agent):
    """Mock the server's agent registry."""
    agents = {
        "test-helpdesk": mock_agent,
        "test-research": mock_agent,
        "test-code-assistant": mock_agent,
    }
    
    # Patch the server's agent registry
    monkeypatch.setattr("langchain_azure_ai.server.AGENTS", agents)
    return agents


# ============================================================================
# Observability Fixtures
# ============================================================================

@pytest.fixture
def mock_tracer():
    """Create a mock OpenTelemetry tracer."""
    tracer = MagicMock()
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=None)
    tracer.start_as_current_span.return_value = span
    return tracer


@pytest.fixture
def mock_meter():
    """Create a mock OpenTelemetry meter."""
    meter = MagicMock()
    meter.create_counter.return_value = MagicMock()
    meter.create_histogram.return_value = MagicMock()
    meter.create_up_down_counter.return_value = MagicMock()
    return meter


@pytest.fixture
def agent_telemetry(mock_env_vars):
    """Create an AgentTelemetry instance for testing."""
    from langchain_azure_ai.observability import AgentTelemetry
    
    return AgentTelemetry("test-agent", "enterprise")


# ============================================================================
# Async Fixtures
# ============================================================================

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    
    return [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello, how are you?"),
        AIMessage(content="I'm doing well, thank you!"),
        HumanMessage(content="Can you help me with a task?"),
    ]


@pytest.fixture
def sample_tool_result():
    """Create a sample tool result for testing."""
    return {
        "tool_name": "search",
        "tool_input": {"query": "test query"},
        "tool_output": "Search results for: test query",
    }
