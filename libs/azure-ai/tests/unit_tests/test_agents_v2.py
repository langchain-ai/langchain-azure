"""Unit tests for Azure AI Foundry V2 agent classes."""

import json
from typing import Any, Dict
from unittest import mock
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

from langchain_azure_ai.agents.prebuilt.tools_v2 import (
    AgentServiceBaseToolV2,
    _get_v2_tool_definitions,
)


# ---------------------------------------------------------------------------
# Tests for tools_v2.py
# ---------------------------------------------------------------------------


class TestAgentServiceBaseToolV2:
    """Tests for AgentServiceBaseToolV2 wrapper."""

    def test_wraps_tool(self) -> None:
        """Test that a V2 tool can be wrapped."""
        from azure.ai.projects.models import CodeInterpreterTool

        tool = CodeInterpreterTool()
        wrapper = AgentServiceBaseToolV2(tool=tool)
        assert wrapper.tool is tool


class TestGetV2ToolDefinitions:
    """Tests for _get_v2_tool_definitions."""

    def test_callable_tool(self) -> None:
        """Test converting a callable to a V2 FunctionTool definition."""

        def my_func(x: int) -> int:
            """Add one to x."""
            return x + 1

        with patch(
            "langchain_core.utils.function_calling.convert_to_openai_function"
        ) as mock_convert:
            mock_convert.return_value = {
                "name": "my_func",
                "description": "Add one to x.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                },
            }
            defs = _get_v2_tool_definitions([my_func])
            assert len(defs) == 1
            assert defs[0]["name"] == "my_func"

    def test_agent_service_base_tool_v2(self) -> None:
        """Test that AgentServiceBaseToolV2 is passed through."""
        from azure.ai.projects.models import CodeInterpreterTool

        tool = CodeInterpreterTool()
        wrapper = AgentServiceBaseToolV2(tool=tool)
        defs = _get_v2_tool_definitions([wrapper])
        assert len(defs) == 1
        assert defs[0] is tool

    def test_invalid_tool_raises(self) -> None:
        """Test that invalid tool types raise ValueError."""
        with pytest.raises(ValueError, match="Each tool must be"):
            _get_v2_tool_definitions([42])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Tests for declarative_v2.py helper functions
# ---------------------------------------------------------------------------


class TestDeclarativeV2Helpers:
    """Tests for helper functions in declarative_v2."""

    def test_function_call_to_ai_message(self) -> None:
        """Test converting a FunctionToolCallItemResource to AIMessage."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _function_call_to_ai_message,
        )

        mock_fc = MagicMock()
        mock_fc.call_id = "call_123"
        mock_fc.name = "my_func"
        mock_fc.arguments = '{"x": 42}'

        msg = _function_call_to_ai_message(mock_fc)
        assert isinstance(msg, AIMessage)
        assert msg.content == ""
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["id"] == "call_123"
        assert msg.tool_calls[0]["name"] == "my_func"
        assert msg.tool_calls[0]["args"] == {"x": 42}

    def test_tool_message_to_output(self) -> None:
        """Test converting a ToolMessage to a FunctionToolCallOutputItemParam."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _tool_message_to_output,
        )

        tool_msg = ToolMessage(
            content="result_value", tool_call_id="call_123"
        )
        output = _tool_message_to_output(tool_msg)
        assert output.call_id == "call_123"
        assert output.output == "result_value"

    def test_get_thread_input_from_state_dict(self) -> None:
        """Test extracting message from dict state."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _get_thread_input_from_state,
        )

        msg = HumanMessage(content="hello")
        state: Dict[str, Any] = {"messages": [msg]}
        result = _get_thread_input_from_state(state)
        assert result is msg

    def test_get_thread_input_from_state_missing_raises(self) -> None:
        """Test that missing messages key raises ValueError."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _get_thread_input_from_state,
        )

        with pytest.raises(ValueError, match="messages"):
            _get_thread_input_from_state({})

    def test_content_from_human_message_string(self) -> None:
        """Test converting a string HumanMessage."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _content_from_human_message,
        )

        msg = HumanMessage(content="hello world")
        result = _content_from_human_message(msg)
        assert result == "hello world"

    def test_content_from_human_message_list_with_text(self) -> None:
        """Test converting a HumanMessage with text blocks."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _content_from_human_message,
        )

        msg = HumanMessage(content=[{"type": "text", "text": "hello"}])
        result = _content_from_human_message(msg)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_content_from_human_message_unsupported_block(self) -> None:
        """Test that unsupported block types raise ValueError."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _content_from_human_message,
        )

        msg = HumanMessage(content=[{"type": "video"}])
        with pytest.raises(ValueError, match="Unsupported block type"):
            _content_from_human_message(msg)


# ---------------------------------------------------------------------------
# Tests for _PromptBasedAgentModelV2
# ---------------------------------------------------------------------------


class TestPromptBasedAgentModelV2:
    """Tests for _PromptBasedAgentModelV2."""

    def test_completed_response_with_text(self) -> None:
        """Test that a completed response yields AIMessage with text."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output = []
        mock_response.output_text = "Hello from the agent"
        mock_response.usage = None

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            agent_name="test-agent",
            model_name="gpt-4.1",
        )
        result = model.invoke([HumanMessage(content="hi")])
        assert isinstance(result, AIMessage)
        assert result.content == "Hello from the agent"

    def test_failed_response_raises(self) -> None:
        """Test that a failed response raises RuntimeError."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        mock_response = MagicMock()
        mock_response.status = "failed"
        mock_response.error = "Something went wrong"

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            agent_name="test-agent",
            model_name="gpt-4.1",
        )
        with pytest.raises(RuntimeError, match="failed"):
            model.invoke([HumanMessage(content="hi")])

    def test_function_call_response(self) -> None:
        """Test that function calls produce AIMessage with tool_calls."""
        from azure.ai.projects.models import ItemType

        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        mock_fc = MagicMock()
        mock_fc.type = ItemType.FUNCTION_CALL
        mock_fc.call_id = "call_abc"
        mock_fc.name = "calculator"
        mock_fc.arguments = '{"expr": "2+2"}'

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output = [mock_fc]
        mock_response.output_text = None
        mock_response.usage = None

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            agent_name="test-agent",
            model_name="gpt-4.1",
        )
        result = model.invoke([HumanMessage(content="compute 2+2")])
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "calculator"


# ---------------------------------------------------------------------------
# Tests for AgentServiceFactoryV2
# ---------------------------------------------------------------------------


class TestAgentServiceFactoryV2:
    """Tests for AgentServiceFactoryV2."""

    def test_validate_environment_from_env(self) -> None:
        """Test environment variable validation."""
        from langchain_azure_ai.agents.agent_service_v2 import (
            AgentServiceFactoryV2,
        )

        with mock.patch.dict(
            "os.environ",
            {"AZURE_AI_PROJECT_ENDPOINT": "https://test.endpoint.com"},
        ):
            factory = AgentServiceFactoryV2()
            assert factory.project_endpoint == "https://test.endpoint.com"

    def test_validate_environment_from_param(self) -> None:
        """Test explicit parameter takes priority."""
        from langchain_azure_ai.agents.agent_service_v2 import (
            AgentServiceFactoryV2,
        )

        factory = AgentServiceFactoryV2(
            project_endpoint="https://explicit.endpoint.com"
        )
        assert factory.project_endpoint == "https://explicit.endpoint.com"

    def test_get_agents_id_from_graph(self) -> None:
        """Test extraction of agent IDs from graph metadata."""
        from langchain_azure_ai.agents.agent_service_v2 import (
            AgentServiceFactoryV2,
        )

        factory = AgentServiceFactoryV2(
            project_endpoint="https://test.endpoint.com"
        )

        mock_graph = MagicMock(spec_set=["nodes"])
        mock_node = MagicMock()
        mock_node.metadata = {"agent_id": "my-agent:v1"}
        mock_graph.nodes = {"foundryAgent": mock_node}

        ids = factory.get_agents_id_from_graph(mock_graph)
        assert ids == {"my-agent:v1"}

    def test_create_prompt_agent_node_non_string_instructions_raises(
        self,
    ) -> None:
        """Test that non-string instructions raise ValueError."""
        from langchain_azure_ai.agents.agent_service_v2 import (
            AgentServiceFactoryV2,
        )

        factory = AgentServiceFactoryV2(
            project_endpoint="https://test.endpoint.com"
        )

        with pytest.raises(ValueError, match="Only string instructions"):
            factory.create_prompt_agent_node(
                name="test",
                model="gpt-4.1",
                instructions=None,
            )
