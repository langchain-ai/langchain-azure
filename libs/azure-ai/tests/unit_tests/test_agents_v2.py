"""Unit tests for Azure AI Foundry V2 agent classes."""

from typing import Any, Dict
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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

    def test_mcp_approval_to_ai_message(self) -> None:
        """Test converting an MCPApprovalRequestItemResource to AIMessage."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _mcp_approval_to_ai_message,
        )

        mock_ar = MagicMock()
        mock_ar.id = "approval_req_123"
        mock_ar.server_label = "api-specs"
        mock_ar.name = "read_file"
        mock_ar.arguments = '{"path": "/README.md"}'

        msg = _mcp_approval_to_ai_message(mock_ar)
        assert isinstance(msg, AIMessage)
        assert msg.content == ""
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["id"] == "approval_req_123"
        assert msg.tool_calls[0]["name"] == "mcp_approval_request"
        assert msg.tool_calls[0]["args"]["server_label"] == "api-specs"
        assert msg.tool_calls[0]["args"]["name"] == "read_file"
        assert msg.tool_calls[0]["args"]["arguments"] == '{"path": "/README.md"}'

    def test_approval_message_to_output_json_approve(self) -> None:
        """Test converting a ToolMessage with JSON approve=true."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _approval_message_to_output,
        )

        tool_msg = ToolMessage(
            content='{"approve": true}', tool_call_id="approval_req_123"
        )
        output = _approval_message_to_output(tool_msg)
        assert output.approval_request_id == "approval_req_123"
        assert output.approve is True

    def test_approval_message_to_output_json_deny_with_reason(self) -> None:
        """Test converting a ToolMessage with JSON approve=false and reason."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _approval_message_to_output,
        )

        tool_msg = ToolMessage(
            content='{"approve": false, "reason": "not allowed"}',
            tool_call_id="approval_req_456",
        )
        output = _approval_message_to_output(tool_msg)
        assert output.approval_request_id == "approval_req_456"
        assert output.approve is False
        assert output.reason == "not allowed"

    def test_approval_message_to_output_string_true(self) -> None:
        """Test converting a plain string 'true' ToolMessage."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _approval_message_to_output,
        )

        tool_msg = ToolMessage(
            content="true", tool_call_id="approval_req_789"
        )
        output = _approval_message_to_output(tool_msg)
        assert output.approve is True

    def test_approval_message_to_output_string_false(self) -> None:
        """Test converting a plain string 'false' ToolMessage."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _approval_message_to_output,
        )

        tool_msg = ToolMessage(
            content="false", tool_call_id="approval_req_000"
        )
        output = _approval_message_to_output(tool_msg)
        assert output.approve is False

    def test_approval_message_to_output_string_deny(self) -> None:
        """Test converting a plain string 'deny' ToolMessage."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _approval_message_to_output,
        )

        tool_msg = ToolMessage(
            content="deny", tool_call_id="approval_req_111"
        )
        output = _approval_message_to_output(tool_msg)
        assert output.approve is False


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

    def test_mcp_approval_request_response(self) -> None:
        """Test that MCP approval requests produce AIMessage with tool_calls."""
        from azure.ai.projects.models import ItemType

        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        mock_ar = MagicMock()
        mock_ar.type = ItemType.MCP_APPROVAL_REQUEST
        mock_ar.id = "approval_req_xyz"
        mock_ar.server_label = "api-specs"
        mock_ar.name = "read_file"
        mock_ar.arguments = '{"path": "/README.md"}'

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output = [mock_ar]
        mock_response.output_text = None
        mock_response.usage = None

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            agent_name="test-agent",
            model_name="gpt-4.1",
        )
        result = model.invoke([HumanMessage(content="summarize specs")])
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "mcp_approval_request"
        assert result.tool_calls[0]["id"] == "approval_req_xyz"
        assert result.tool_calls[0]["args"]["server_label"] == "api-specs"
        assert result.tool_calls[0]["args"]["name"] == "read_file"

        # Verify the model tracks pending approvals
        assert len(model.pending_mcp_approvals) == 1
        assert len(model.pending_function_calls) == 0


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


# ---------------------------------------------------------------------------
# Additional coverage for declarative_v2.py helper functions
# ---------------------------------------------------------------------------


class TestDeclarativeV2HelpersAdditional:
    """Additional tests for helper functions in declarative_v2."""

    def test_tool_message_to_output_non_string_content(self) -> None:
        """Test converting a ToolMessage with non-string content (JSON)."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _tool_message_to_output,
        )

        # ToolMessage serializes dict content to its str() representation,
        # but _tool_message_to_output should json.dumps() it.
        tool_msg = ToolMessage(
            content=[{"type": "text", "text": "result value"}],
            tool_call_id="call_456",
        )
        output = _tool_message_to_output(tool_msg)
        assert output.call_id == "call_456"
        # Non-string content gets json.dumps'd
        assert "result value" in output.output

    def test_get_thread_input_from_state_object(self) -> None:
        """Test extracting message from an object state with messages attr."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _get_thread_input_from_state,
        )

        msg = HumanMessage(content="hello")

        class FakeState:
            messages = [msg]

        result = _get_thread_input_from_state(FakeState())  # type: ignore[arg-type]
        assert result is msg

    def test_content_from_human_message_list_with_plain_string(self) -> None:
        """Test converting a HumanMessage with a plain string in list."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _content_from_human_message,
        )

        msg = HumanMessage(content=["hello world"])
        result = _content_from_human_message(msg)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_content_from_human_message_image_url_block(self) -> None:
        """Test converting a HumanMessage with an image_url block."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _content_from_human_message,
        )

        msg = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                }
            ]
        )
        result = _content_from_human_message(msg)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_content_from_human_message_image_base64_block(self) -> None:
        """Test converting a HumanMessage with a base64 image block."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _content_from_human_message,
        )

        msg = HumanMessage(
            content=[
                {
                    "type": "image",
                    "source_type": "base64",
                    "mime_type": "image/png",
                    "data": "iVBORw0KGgo=",
                }
            ]
        )
        result = _content_from_human_message(msg)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_content_from_human_message_image_url_source_block(self) -> None:
        """Test converting a HumanMessage with an image url source block."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _content_from_human_message,
        )

        msg = HumanMessage(
            content=[
                {
                    "type": "image",
                    "source_type": "url",
                    "url": "https://example.com/photo.jpg",
                }
            ]
        )
        result = _content_from_human_message(msg)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_content_from_human_message_image_unsupported_source(self) -> None:
        """Test that unsupported image source types raise ValueError."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _content_from_human_message,
        )

        msg = HumanMessage(
            content=[{"type": "image", "source_type": "file"}]
        )
        with pytest.raises(ValueError, match="base64.*url"):
            _content_from_human_message(msg)

    def test_content_from_human_message_unexpected_block_type(self) -> None:
        """Test that unexpected block types in list raise ValueError."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _content_from_human_message,
        )

        # HumanMessage validates content, so we use a mock to bypass Pydantic
        mock_msg = MagicMock(spec=HumanMessage)
        mock_msg.content = [123]
        with pytest.raises(ValueError, match="Unexpected block type"):
            _content_from_human_message(mock_msg)

    def test_content_from_human_message_non_string_non_list(self) -> None:
        """Test that non-string, non-list content raises ValueError."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _content_from_human_message,
        )

        # HumanMessage validates content, so we use a mock to bypass Pydantic
        mock_msg = MagicMock(spec=HumanMessage)
        mock_msg.content = 42
        with pytest.raises(ValueError, match="string or a list"):
            _content_from_human_message(mock_msg)

    def test_approval_message_to_output_dict_content(self) -> None:
        """Test converting a ToolMessage with dict content via dict branch."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _approval_message_to_output,
        )

        # ToolMessage may serialize dict content to string, so we use a mock
        # to test the dict-content branch directly
        mock_msg = MagicMock(spec=ToolMessage)
        mock_msg.content = {"approve": False, "reason": "risky"}
        mock_msg.tool_call_id = "approval_req_dict"

        output = _approval_message_to_output(mock_msg)
        assert output.approval_request_id == "approval_req_dict"
        assert output.approve is False
        assert output.reason == "risky"

    def test_approval_message_to_output_list_content(self) -> None:
        """Test converting a ToolMessage with list content."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _approval_message_to_output,
        )

        tool_msg = ToolMessage(
            content=[{"type": "text", "text": "false"}],  # type: ignore[arg-type]
            tool_call_id="approval_req_list",
        )
        output = _approval_message_to_output(tool_msg)
        assert output.approval_request_id == "approval_req_list"
        assert output.approve is False

    def test_approval_message_to_output_list_approve(self) -> None:
        """Test converting a ToolMessage with list content approving."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _approval_message_to_output,
        )

        tool_msg = ToolMessage(
            content=[{"type": "text", "text": "yes please"}],  # type: ignore[arg-type]
            tool_call_id="approval_req_list2",
        )
        output = _approval_message_to_output(tool_msg)
        assert output.approve is True


# ---------------------------------------------------------------------------
# Additional coverage for _PromptBasedAgentModelV2
# ---------------------------------------------------------------------------


class TestPromptBasedAgentModelV2Additional:
    """Additional tests for _PromptBasedAgentModelV2."""

    def test_fallback_message_items(self) -> None:
        """Test fallback to MESSAGE items when output_text is None."""
        from azure.ai.projects.models import ItemType

        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        mock_content_part = MagicMock()
        mock_content_part.text = "Fallback text response"

        mock_item = MagicMock()
        mock_item.type = ItemType.MESSAGE
        mock_item.content = [mock_content_part]

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output = [mock_item]
        mock_response.output_text = None
        mock_response.usage = None

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            agent_name="test-agent",
            model_name="gpt-4.1",
        )
        result = model.invoke([HumanMessage(content="hi")])
        assert isinstance(result, AIMessage)
        assert result.content == "Fallback text response"
        assert result.name == "test-agent"

    def test_usage_tracking(self) -> None:
        """Test that token usage is tracked in llm_output."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        mock_usage = MagicMock()
        mock_usage.total_tokens = 150

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output = []
        mock_response.output_text = "Some text"
        mock_response.usage = mock_usage

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            agent_name="test-agent",
            model_name="gpt-4.1",
        )
        result = model._generate([HumanMessage(content="hi")])
        assert result.llm_output is not None
        assert result.llm_output["token_usage"] == 150
        assert result.llm_output["model"] == "gpt-4.1"

    def test_empty_output_no_text(self) -> None:
        """Test that empty output with no text produces no generations."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output = []
        mock_response.output_text = None
        mock_response.usage = None

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            agent_name="test-agent",
            model_name="gpt-4.1",
        )
        result = model._generate([HumanMessage(content="hi")])
        assert len(result.generations) == 0

    def test_response_without_status(self) -> None:
        """Test response object without status attribute."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        mock_response = MagicMock(spec=[])
        mock_response.output = []
        mock_response.output_text = "Works without status"
        mock_response.usage = None

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            agent_name="test-agent",
            model_name="gpt-4.1",
        )
        result = model.invoke([HumanMessage(content="hi")])
        assert isinstance(result, AIMessage)
        assert result.content == "Works without status"


# ---------------------------------------------------------------------------
# Tests for external_tools_condition
# ---------------------------------------------------------------------------


class TestExternalToolsCondition:
    """Tests for external_tools_condition routing function."""

    def test_routes_to_tools_with_tool_calls(self) -> None:
        """Test that messages with tool_calls route to 'tools'."""
        from langchain_azure_ai.agents.agent_service_v2 import (
            external_tools_condition,
        )

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "add", "args": {"a": 1}}],
        )
        state = {"messages": [HumanMessage(content="hi"), ai_msg]}
        assert external_tools_condition(state) == "tools"

    def test_routes_to_end_without_tool_calls(self) -> None:
        """Test that messages without tool_calls route to '__end__'."""
        from langchain_azure_ai.agents.agent_service_v2 import (
            external_tools_condition,
        )

        ai_msg = AIMessage(content="The answer is 42")
        state = {"messages": [HumanMessage(content="hi"), ai_msg]}
        assert external_tools_condition(state) == "__end__"

    def test_routes_to_end_with_empty_tool_calls(self) -> None:
        """Test that messages with empty tool_calls route to '__end__'."""
        from langchain_azure_ai.agents.agent_service_v2 import (
            external_tools_condition,
        )

        ai_msg = AIMessage(content="Done", tool_calls=[])
        state = {"messages": [HumanMessage(content="hi"), ai_msg]}
        assert external_tools_condition(state) == "__end__"


# ---------------------------------------------------------------------------
# Tests for PromptBasedAgentNodeV2 (_func, delete, properties)
# ---------------------------------------------------------------------------


class TestPromptBasedAgentNodeV2:
    """Tests for PromptBasedAgentNodeV2 core execution logic."""

    def _make_node(
        self,
        agent_name: str = "test-agent",
        agent_version: str = "v1",
    ) -> Any:
        """Create a PromptBasedAgentNodeV2 bypassing real client calls."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            PromptBasedAgentNodeV2,
        )

        # We'll build the object manually, avoiding __init__ which calls
        # the real client.agents.create_version().
        node = object.__new__(PromptBasedAgentNodeV2)
        # RunnableCallable fields
        node.name = "PromptAgentV2"
        node.tags = None
        node.func = node._func
        node.afunc = node._afunc
        node.trace = True
        node.recurse = True

        mock_client = MagicMock()
        node._client = mock_client

        mock_agent = MagicMock()
        mock_agent.name = agent_name
        mock_agent.version = agent_version
        mock_agent.definition = {"model": "gpt-4.1"}
        node._agent = mock_agent
        node._agent_name = agent_name
        node._agent_version = agent_version
        node._conversation_id = None
        node._previous_response_id = None
        node._pending_function_calls = []
        node._pending_mcp_approvals = []

        return node

    def test_agent_id_property(self) -> None:
        """Test that _agent_id returns name:version."""
        node = self._make_node()
        assert node._agent_id == "test-agent:v1"

    def test_agent_id_property_none(self) -> None:
        """Test that _agent_id returns None when name or version is None."""
        node = self._make_node()
        node._agent_name = None
        assert node._agent_id is None

    def test_delete_agent_from_node(self) -> None:
        """Test successful agent deletion."""
        node = self._make_node()
        node.delete_agent_from_node()

        node._client.agents.delete_version.assert_called_once_with(
            agent_name="test-agent",
            agent_version="v1",
        )
        assert node._agent is None
        assert node._agent_name is None
        assert node._agent_version is None

    def test_delete_agent_from_node_no_agent_raises(self) -> None:
        """Test that deleting without an agent raises ValueError."""
        node = self._make_node()
        node._agent_name = None
        node._agent_version = None

        with pytest.raises(ValueError, match="does not have an associated agent"):
            node.delete_agent_from_node()

    def test_func_raises_when_agent_deleted(self) -> None:
        """Test that _func raises RuntimeError when agent is deleted."""
        node = self._make_node()
        node._agent = None

        state = {"messages": [HumanMessage(content="hi")]}
        config: Dict[str, Any] = {}

        with pytest.raises(RuntimeError, match="not been initialized"):
            node._func(state, config, store=None)

    def test_func_raises_on_unsupported_message(self) -> None:
        """Test that _func raises RuntimeError for unsupported message types."""
        node = self._make_node()

        # Use a BaseMessage that is not HumanMessage or ToolMessage
        from langchain_core.messages import SystemMessage

        state = {"messages": [SystemMessage(content="system prompt")]}
        config: Dict[str, Any] = {"callbacks": None, "metadata": None, "tags": None}

        # Mock the openai_client
        mock_openai = MagicMock()
        node._client.get_openai_client.return_value = mock_openai

        with pytest.raises(RuntimeError, match="Unsupported message type"):
            node._func(state, config, store=None)

    def test_func_human_message_new_conversation(self) -> None:
        """Test _func with a HumanMessage creates a new conversation."""
        node = self._make_node()
        config: Dict[str, Any] = {"callbacks": None, "metadata": None, "tags": None}

        mock_openai = MagicMock()
        node._client.get_openai_client.return_value = mock_openai

        # Mock conversation creation
        mock_conversation = MagicMock()
        mock_conversation.id = "conv_123"
        mock_openai.conversations.create.return_value = mock_conversation

        # Mock response
        mock_response = MagicMock()
        mock_response.id = "resp_456"
        mock_response.status = "completed"
        mock_response.output = []
        mock_response.output_text = "Hello back!"
        mock_response.usage = None
        mock_openai.responses.create.return_value = mock_response

        state = {"messages": [HumanMessage(content="Hello!")]}
        result = node._func(state, config, store=None)

        assert "messages" in result
        assert node._conversation_id == "conv_123"
        assert node._previous_response_id == "resp_456"
        mock_openai.close.assert_called_once()

    def test_func_human_message_existing_conversation(self) -> None:
        """Test _func with HumanMessage adds to existing conversation."""
        node = self._make_node()
        node._conversation_id = "conv_existing"
        config: Dict[str, Any] = {"callbacks": None, "metadata": None, "tags": None}

        mock_openai = MagicMock()
        node._client.get_openai_client.return_value = mock_openai

        mock_response = MagicMock()
        mock_response.id = "resp_789"
        mock_response.status = "completed"
        mock_response.output = []
        mock_response.output_text = "I see"
        mock_response.usage = None
        mock_openai.responses.create.return_value = mock_response

        state = {"messages": [HumanMessage(content="Follow up")]}
        result = node._func(state, config, store=None)

        assert "messages" in result
        # Should add to existing conversation
        mock_openai.conversations.items.create.assert_called_once()
        call_kwargs = mock_openai.conversations.items.create.call_args
        assert call_kwargs.kwargs["conversation_id"] == "conv_existing"
        # Should not create new conversation
        mock_openai.conversations.create.assert_not_called()

    def test_func_tool_message_function_call(self) -> None:
        """Test _func with a ToolMessage for pending function calls."""
        node = self._make_node()
        node._conversation_id = "conv_123"
        node._previous_response_id = "resp_prev"

        mock_fc = MagicMock()
        mock_fc.call_id = "call_abc"
        mock_fc.name = "add"
        mock_fc.arguments = '{"a": 1, "b": 2}'
        node._pending_function_calls = [mock_fc]

        config: Dict[str, Any] = {"callbacks": None, "metadata": None, "tags": None}

        mock_openai = MagicMock()
        node._client.get_openai_client.return_value = mock_openai

        mock_response = MagicMock()
        mock_response.id = "resp_tool"
        mock_response.status = "completed"
        mock_response.output = []
        mock_response.output_text = "The sum is 3"
        mock_response.usage = None
        mock_openai.responses.create.return_value = mock_response

        tool_msg = ToolMessage(content="3", tool_call_id="call_abc")
        state = {"messages": [tool_msg]}
        result = node._func(state, config, store=None)

        assert "messages" in result
        # Verify responses.create was called with function call items
        call_kwargs = mock_openai.responses.create.call_args.kwargs
        input_items = call_kwargs["input"]
        types = [item["type"] for item in input_items]
        assert "function_call" in types
        assert "function_call_output" in types
        assert call_kwargs["extra_body"]["agent_reference"]["name"] == "test-agent"

    def test_func_tool_message_mcp_approval(self) -> None:
        """Test _func with a ToolMessage for MCP approval response."""
        node = self._make_node()
        node._conversation_id = "conv_mcp"
        node._previous_response_id = "resp_mcp_prev"

        mock_ar = MagicMock()
        mock_ar.id = "approval_req_1"
        node._pending_mcp_approvals = [mock_ar]

        config: Dict[str, Any] = {"callbacks": None, "metadata": None, "tags": None}

        mock_openai = MagicMock()
        node._client.get_openai_client.return_value = mock_openai

        mock_response = MagicMock()
        mock_response.id = "resp_approval"
        mock_response.status = "completed"
        mock_response.output = []
        mock_response.output_text = "MCP tool ran successfully"
        mock_response.usage = None
        mock_openai.responses.create.return_value = mock_response

        tool_msg = ToolMessage(
            content='{"approve": true}', tool_call_id="approval_req_1"
        )
        state = {"messages": [tool_msg]}
        result = node._func(state, config, store=None)

        assert "messages" in result
        call_kwargs = mock_openai.responses.create.call_args.kwargs
        input_items = call_kwargs["input"]
        assert input_items[0]["type"] == "mcp_approval_response"
        assert input_items[0]["approve"] is True
        assert input_items[0]["approval_request_id"] == "approval_req_1"

    def test_func_tool_message_no_pending_raises(self) -> None:
        """Test that ToolMessage without pending calls raises RuntimeError."""
        node = self._make_node()

        config: Dict[str, Any] = {"callbacks": None, "metadata": None, "tags": None}
        mock_openai = MagicMock()
        node._client.get_openai_client.return_value = mock_openai

        tool_msg = ToolMessage(content="result", tool_call_id="call_orphan")
        state = {"messages": [tool_msg]}

        with pytest.raises(RuntimeError, match="No pending function calls"):
            node._func(state, config, store=None)

    def test_func_tracks_pending_after_function_call_response(self) -> None:
        """Test that pending function calls are tracked from response."""
        from azure.ai.projects.models import ItemType

        node = self._make_node()
        config: Dict[str, Any] = {"callbacks": None, "metadata": None, "tags": None}

        mock_openai = MagicMock()
        node._client.get_openai_client.return_value = mock_openai

        mock_conversation = MagicMock()
        mock_conversation.id = "conv_fc"
        mock_openai.conversations.create.return_value = mock_conversation

        # Response with a function call
        mock_fc = MagicMock()
        mock_fc.type = ItemType.FUNCTION_CALL
        mock_fc.call_id = "call_new"
        mock_fc.name = "multiply"
        mock_fc.arguments = '{"a": 3, "b": 4}'

        mock_response = MagicMock()
        mock_response.id = "resp_fc"
        mock_response.status = "completed"
        mock_response.output = [mock_fc]
        mock_response.output_text = None
        mock_response.usage = None
        mock_openai.responses.create.return_value = mock_response

        state = {"messages": [HumanMessage(content="multiply 3 by 4")]}
        node._func(state, config, store=None)

        # The node should now have pending function calls
        assert len(node._pending_function_calls) == 1
        assert len(node._pending_mcp_approvals) == 0


# ---------------------------------------------------------------------------
# Additional coverage for AgentServiceFactoryV2
# ---------------------------------------------------------------------------


class TestAgentServiceFactoryV2Additional:
    """Additional tests for AgentServiceFactoryV2."""

    def test_delete_agent_with_node(self) -> None:
        """Test deleting an agent via PromptBasedAgentNodeV2."""
        from langchain_azure_ai.agents.agent_service_v2 import (
            AgentServiceFactoryV2,
        )
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            PromptBasedAgentNodeV2,
        )

        factory = AgentServiceFactoryV2(
            project_endpoint="https://test.endpoint.com"
        )

        mock_node = MagicMock(spec=PromptBasedAgentNodeV2)
        factory.delete_agent(mock_node)
        mock_node.delete_agent_from_node.assert_called_once()

    def test_delete_agent_with_graph(self) -> None:
        """Test deleting an agent from a compiled state graph."""
        from langgraph.graph.state import CompiledStateGraph

        from langchain_azure_ai.agents.agent_service_v2 import (
            AgentServiceFactoryV2,
        )

        factory = AgentServiceFactoryV2(
            project_endpoint="https://test.endpoint.com"
        )

        mock_graph = MagicMock(spec=CompiledStateGraph)
        mock_node = MagicMock()
        mock_node.metadata = {"agent_id": "my-agent:v2"}
        mock_graph.nodes = {"foundryAgent": mock_node}

        mock_client = MagicMock()
        with patch.object(factory, "_initialize_client", return_value=mock_client):
            factory.delete_agent(mock_graph)

        mock_client.agents.delete_version.assert_called_once_with(
            agent_name="my-agent",
            agent_version="v2",
        )

    def test_delete_agent_invalid_type_raises(self) -> None:
        """Test that invalid agent type raises ValueError."""
        from langchain_azure_ai.agents.agent_service_v2 import (
            AgentServiceFactoryV2,
        )

        factory = AgentServiceFactoryV2(
            project_endpoint="https://test.endpoint.com"
        )

        with pytest.raises(ValueError, match="CompiledStateGraph"):
            factory.delete_agent("not_an_agent")  # type: ignore[arg-type]

    def test_delete_agent_no_ids_in_graph(self) -> None:
        """Test deleting when no agent IDs found in graph metadata."""
        from langgraph.graph.state import CompiledStateGraph

        from langchain_azure_ai.agents.agent_service_v2 import (
            AgentServiceFactoryV2,
        )

        factory = AgentServiceFactoryV2(
            project_endpoint="https://test.endpoint.com"
        )

        mock_graph = MagicMock(spec=CompiledStateGraph)
        mock_node = MagicMock()
        mock_node.metadata = {}  # No agent_id
        mock_graph.nodes = {"foundryAgent": mock_node}

        mock_client = MagicMock()
        with patch.object(factory, "_initialize_client", return_value=mock_client):
            # Should not raise, just log a warning
            factory.delete_agent(mock_graph)

        mock_client.agents.delete_version.assert_not_called()

    def test_external_tools_condition_with_tool_calls(self) -> None:
        """Test external_tools_condition routes to tools."""
        from langchain_azure_ai.agents.agent_service_v2 import (
            external_tools_condition,
        )

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "f1", "args": {}}],
        )
        result = external_tools_condition({"messages": [ai_msg]})
        assert result == "tools"

    def test_external_tools_condition_without_tool_calls(self) -> None:
        """Test external_tools_condition routes to end."""
        from langchain_azure_ai.agents.agent_service_v2 import (
            external_tools_condition,
        )

        ai_msg = AIMessage(content="Done")
        result = external_tools_condition({"messages": [ai_msg]})
        assert result == "__end__"


# ---------------------------------------------------------------------------
# Tests for BaseTool in _get_v2_tool_definitions
# ---------------------------------------------------------------------------


class TestGetV2ToolDefinitionsBaseTool:
    """Tests for _get_v2_tool_definitions with BaseTool instances."""

    def test_base_tool(self) -> None:
        """Test converting a BaseTool to a V2 FunctionTool definition."""
        from langchain_core.tools import BaseTool

        mock_tool = MagicMock(spec=BaseTool)

        with patch(
            "langchain_core.utils.function_calling.convert_to_openai_function"
        ) as mock_convert:
            mock_convert.return_value = {
                "name": "my_tool",
                "description": "A test tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            }
            defs = _get_v2_tool_definitions([mock_tool])
            assert len(defs) == 1
            assert defs[0]["name"] == "my_tool"


# ---------------------------------------------------------------------------
# Tests for file upload helpers (V2)
# ---------------------------------------------------------------------------


class TestAgentHasCodeInterpreterV2:
    """Tests for _agent_has_code_interpreter_v2."""

    def test_with_code_interpreter(self) -> None:
        """Test detection of CodeInterpreterTool in agent definition."""
        from azure.ai.projects.models import CodeInterpreterTool

        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _agent_has_code_interpreter_v2,
        )

        mock_agent = MagicMock()
        # Use a dict-like definition (as returned by the API)
        mock_agent.definition = {"tools": [CodeInterpreterTool()]}

        assert _agent_has_code_interpreter_v2(mock_agent) is True

    def test_without_code_interpreter(self) -> None:
        """Test that non-CodeInterpreter tools return False."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _agent_has_code_interpreter_v2,
        )

        mock_agent = MagicMock()
        mock_agent.definition = {"tools": [MagicMock()]}

        assert _agent_has_code_interpreter_v2(mock_agent) is False

    def test_no_definition(self) -> None:
        """Test agent with no definition returns False."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _agent_has_code_interpreter_v2,
        )

        mock_agent = MagicMock()
        mock_agent.definition = None

        assert _agent_has_code_interpreter_v2(mock_agent) is False

    def test_no_tools(self) -> None:
        """Test agent with no tools returns False."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _agent_has_code_interpreter_v2,
        )

        mock_agent = MagicMock()
        mock_agent.definition = {"tools": None}

        assert _agent_has_code_interpreter_v2(mock_agent) is False


class TestUploadFileBlocksV2:
    """Tests for _upload_file_blocks_v2."""

    def test_string_content_passthrough(self) -> None:
        """Test that string content is returned unchanged."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _upload_file_blocks_v2,
        )

        msg = HumanMessage(content="hello")
        mock_client = MagicMock()

        result_msg, file_ids = _upload_file_blocks_v2(msg, mock_client)
        assert result_msg is msg
        assert file_ids == []
        mock_client.files.create.assert_not_called()

    def test_no_file_blocks(self) -> None:
        """Test that non-file blocks are returned unchanged."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _upload_file_blocks_v2,
        )

        msg = HumanMessage(content=[{"type": "text", "text": "hello"}])
        mock_client = MagicMock()

        result_msg, file_ids = _upload_file_blocks_v2(msg, mock_client)
        assert result_msg is msg
        assert file_ids == []

    def test_file_block_uploaded(self) -> None:
        """Test that file blocks are uploaded and removed from content."""
        import base64

        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _upload_file_blocks_v2,
        )

        raw_data = b"test file content"
        b64_data = base64.b64encode(raw_data).decode("utf-8")

        msg = HumanMessage(
            content=[
                {"type": "text", "text": "analyze this"},
                {
                    "type": "file",
                    "source_type": "base64",
                    "mime_type": "text/csv",
                    "base64": b64_data,
                },
            ]
        )

        mock_file_info = MagicMock()
        mock_file_info.id = "file_abc123"
        mock_client = MagicMock()
        mock_client.files.create.return_value = mock_file_info

        result_msg, file_ids = _upload_file_blocks_v2(msg, mock_client)

        assert len(file_ids) == 1
        assert file_ids[0] == "file_abc123"
        # The text block should remain
        assert len(result_msg.content) == 1
        assert result_msg.content[0]["type"] == "text"
        # Verify files.create was called with purpose="assistants"
        mock_client.files.create.assert_called_once()
        call_kwargs = mock_client.files.create.call_args.kwargs
        assert call_kwargs["purpose"] == "assistants"

    def test_invalid_base64_raises(self) -> None:
        """Test that invalid base64 data raises ValueError."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _upload_file_blocks_v2,
        )

        msg = HumanMessage(
            content=[
                {
                    "type": "file",
                    "source_type": "base64",
                    "mime_type": "text/csv",
                    "base64": "!!!invalid!!!",
                },
            ]
        )
        mock_client = MagicMock()

        with pytest.raises(ValueError, match="Failed to decode base64"):
            _upload_file_blocks_v2(msg, mock_client)

    def test_upload_failure_raises_runtime_error(self) -> None:
        """Test that upload failure raises RuntimeError."""
        import base64

        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _upload_file_blocks_v2,
        )

        raw_data = b"test file content"
        b64_data = base64.b64encode(raw_data).decode("utf-8")

        msg = HumanMessage(
            content=[
                {
                    "type": "file",
                    "source_type": "base64",
                    "mime_type": "text/csv",
                    "base64": b64_data,
                },
            ]
        )

        mock_client = MagicMock()
        mock_client.files.create.side_effect = Exception("upload failed")

        with pytest.raises(RuntimeError, match="Failed to upload file block"):
            _upload_file_blocks_v2(msg, mock_client)
