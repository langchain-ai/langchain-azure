"""Unit tests for Azure AI Foundry V2 agent classes."""

from typing import Any, Dict
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from azure.ai.projects.models import ItemType
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

        tool_msg = ToolMessage(content="result_value", tool_call_id="call_123")
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

        tool_msg = ToolMessage(content="true", tool_call_id="approval_req_789")
        output = _approval_message_to_output(tool_msg)
        assert output.approve is True

    def test_approval_message_to_output_string_false(self) -> None:
        """Test converting a plain string 'false' ToolMessage."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _approval_message_to_output,
        )

        tool_msg = ToolMessage(content="false", tool_call_id="approval_req_000")
        output = _approval_message_to_output(tool_msg)
        assert output.approve is False

    def test_approval_message_to_output_string_deny(self) -> None:
        """Test converting a plain string 'deny' ToolMessage."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _approval_message_to_output,
        )

        tool_msg = ToolMessage(content="deny", tool_call_id="approval_req_111")
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

        factory = AgentServiceFactoryV2(project_endpoint="https://test.endpoint.com")

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

        factory = AgentServiceFactoryV2(project_endpoint="https://test.endpoint.com")

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

        msg = HumanMessage(content=[{"type": "image", "source_type": "file"}])
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


class TestCodeInterpreterFileDownload:
    """Tests for downloading code-interpreter generated files."""

    @staticmethod
    def _make_annotation(container_id: str, file_id: str, filename: str) -> MagicMock:
        """Create a mock ``container_file_citation`` annotation."""
        ann = MagicMock()
        ann.type = "container_file_citation"
        ann.container_id = container_id
        ann.file_id = file_id
        ann.filename = filename
        ann.start_index = 0
        ann.end_index = 10
        return ann

    @staticmethod
    def _make_message_item(annotations: list, text: str = "some text") -> MagicMock:
        """Create a mock MESSAGE output item with annotations."""
        text_part = MagicMock()
        text_part.type = "output_text"
        text_part.text = text
        text_part.annotations = annotations

        msg_item = MagicMock()
        msg_item.type = ItemType.MESSAGE
        msg_item.content = [text_part]
        return msg_item

    def test_image_via_annotation(self) -> None:
        """An image annotation produces an image content block."""
        import base64

        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        ann = self._make_annotation("cntr_a", "fid_img", "chart.png")
        msg_item = self._make_message_item([ann])

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output = [msg_item]
        mock_response.output_text = "Here is the chart."
        mock_response.usage = None

        raw_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_openai = MagicMock()
        mock_binary = MagicMock()
        mock_binary.read.return_value = raw_image
        mock_openai.containers.files.content.retrieve.return_value = mock_binary

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            openai_client=mock_openai,
            agent_name="test",
            model_name="gpt-4.1",
        )
        result = model.invoke([HumanMessage(content="chart")])

        assert isinstance(result.content, list)
        assert len(result.content) == 2
        assert result.content[0] == "Here is the chart."

        img = result.content[1]
        assert img["type"] == "image"
        assert img["mime_type"] == "image/png"
        assert img["base64"] == base64.b64encode(raw_image).decode("utf-8")

        # Download uses file_id directly — no container listing needed.
        mock_openai.containers.files.content.retrieve.assert_called_once_with(
            file_id="fid_img", container_id="cntr_a"
        )
        mock_openai.containers.files.list.assert_not_called()

    def test_non_image_file_via_annotation(self) -> None:
        """A CSV annotation produces a file content block."""
        import base64

        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        ann = self._make_annotation("cntr_csv", "fid_csv", "report.csv")
        msg_item = self._make_message_item([ann])

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output = [msg_item]
        mock_response.output_text = "Here is the export."
        mock_response.usage = None

        csv_bytes = b"col1,col2\n1,2\n"
        mock_openai = MagicMock()
        mock_binary = MagicMock()
        mock_binary.read.return_value = csv_bytes
        mock_openai.containers.files.content.retrieve.return_value = mock_binary

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            openai_client=mock_openai,
            agent_name="test",
            model_name="gpt-4.1",
        )
        result = model.invoke([HumanMessage(content="export")])

        assert isinstance(result.content, list)
        assert len(result.content) == 2

        block = result.content[1]
        assert block["type"] == "file"
        assert block["mime_type"] == "text/csv"
        assert block["filename"] == "report.csv"
        assert block["data"] == base64.b64encode(csv_bytes).decode("utf-8")
        mock_openai.containers.files.list.assert_not_called()

    def test_multiple_annotations_different_types(self) -> None:
        """Image + file annotations from the same message both download."""
        import base64

        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        ann_img = self._make_annotation("cntr_m", "fid_img", "plot.png")
        ann_csv = self._make_annotation("cntr_m", "fid_csv", "data.xlsx")
        msg_item = self._make_message_item([ann_img, ann_csv])

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output = [msg_item]
        mock_response.output_text = "Chart and data."
        mock_response.usage = None

        img_bytes = b"\x89PNG" + b"\x00" * 50
        xlsx_bytes = b"PK\x03\x04" + b"\x00" * 50

        mock_openai = MagicMock()

        def _retrieve(file_id: str, container_id: str) -> MagicMock:
            resp = MagicMock()
            resp.read.return_value = img_bytes if file_id == "fid_img" else xlsx_bytes
            return resp

        mock_openai.containers.files.content.retrieve.side_effect = _retrieve

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            openai_client=mock_openai,
            agent_name="test",
            model_name="gpt-4.1",
        )
        result = model.invoke([HumanMessage(content="go")])

        assert isinstance(result.content, list)
        assert len(result.content) == 3
        types = {b["type"] for b in result.content[1:]}
        assert types == {"image", "file"}
        mock_openai.containers.files.list.assert_not_called()

    def test_duplicate_annotation_downloaded_once(self) -> None:
        """The same file_id appearing twice only downloads once."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        ann1 = self._make_annotation("cntr_d", "fid_dup", "img.png")
        ann2 = self._make_annotation("cntr_d", "fid_dup", "img.png")
        msg_item = self._make_message_item([ann1, ann2])

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output = [msg_item]
        mock_response.output_text = "Two refs same file."
        mock_response.usage = None

        mock_openai = MagicMock()
        mock_binary = MagicMock()
        mock_binary.read.return_value = b"\x89PNG" + b"\x00" * 10
        mock_openai.containers.files.content.retrieve.return_value = mock_binary

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            openai_client=mock_openai,
            agent_name="test",
            model_name="gpt-4.1",
        )
        result = model.invoke([HumanMessage(content="hi")])

        assert isinstance(result.content, list)
        # text + 1 image (not 2)
        assert len(result.content) == 2
        mock_openai.containers.files.content.retrieve.assert_called_once()

    def test_fallback_output_image_without_annotation(self) -> None:
        """OutputImage without a matching annotation falls back to
        listing container files."""
        import base64

        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        ci_output = MagicMock()
        ci_output.type = "image"
        ci_output.url = "/mnt/data/chart.png"

        ci_item = MagicMock()
        ci_item.type = ItemType.CODE_INTERPRETER_CALL
        ci_item.container_id = "cntr_fb"
        ci_item.outputs = [ci_output]

        # No annotations on the message item.
        msg_item = self._make_message_item([], text="Here is the chart.")

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output = [ci_item, msg_item]
        mock_response.output_text = "Here is the chart."
        mock_response.usage = None

        mock_openai = MagicMock()
        cf = MagicMock()
        cf.id = "fid_fallback"
        cf.path = "/mnt/data/chart.png"
        mock_openai.containers.files.list.return_value = [cf]

        raw = b"\x89PNG" + b"\x00" * 20
        mock_binary = MagicMock()
        mock_binary.read.return_value = raw
        mock_openai.containers.files.content.retrieve.return_value = mock_binary

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            openai_client=mock_openai,
            agent_name="test",
            model_name="gpt-4.1",
        )
        result = model.invoke([HumanMessage(content="chart")])

        assert isinstance(result.content, list)
        assert len(result.content) == 2
        assert result.content[1]["type"] == "image"
        assert result.content[1]["base64"] == base64.b64encode(raw).decode("utf-8")
        # Fallback path does list container files.
        mock_openai.containers.files.list.assert_called_once()

    def test_no_files_returns_plain_text(self) -> None:
        """When no annotations/images exist, output is a plain string."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output = []
        mock_response.output_text = "No files here"
        mock_response.usage = None

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            openai_client=MagicMock(),
            agent_name="test",
            model_name="gpt-4.1",
        )
        result = model.invoke([HumanMessage(content="hi")])
        assert isinstance(result, AIMessage)
        assert result.content == "No files here"

    def test_no_openai_client_skips_download(self) -> None:
        """When openai_client is None, files are not downloaded."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        ann = self._make_annotation("cntr_x", "fid_x", "chart.png")
        msg_item = self._make_message_item([ann])

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output = [msg_item]
        mock_response.output_text = "Chart rendered"
        mock_response.usage = None

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            openai_client=None,
            agent_name="test",
            model_name="gpt-4.1",
        )
        result = model.invoke([HumanMessage(content="hi")])
        assert isinstance(result, AIMessage)
        assert result.content == "Chart rendered"

    def test_unmatched_image_url_becomes_image_url_block(self) -> None:
        """OutputImage that can't be resolved via listing falls back to
        an image_url block."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _PromptBasedAgentModelV2,
        )

        ci_output = MagicMock()
        ci_output.type = "image"
        ci_output.url = "/mnt/data/missing.png"

        ci_item = MagicMock()
        ci_item.type = ItemType.CODE_INTERPRETER_CALL
        ci_item.container_id = "cntr_miss"
        ci_item.outputs = [ci_output]

        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output = [ci_item]
        mock_response.output_text = "Chart"
        mock_response.usage = None

        mock_openai = MagicMock()
        mock_openai.containers.files.list.return_value = []

        model = _PromptBasedAgentModelV2(
            response=mock_response,
            openai_client=mock_openai,
            agent_name="test",
            model_name="gpt-4.1",
        )
        result = model.invoke([HumanMessage(content="hi")])
        assert isinstance(result.content, list)
        assert len(result.content) == 2
        assert result.content[1]["type"] == "image_url"
        assert result.content[1]["image_url"]["url"] == "/mnt/data/missing.png"


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
        node._uses_container_template = False

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

        # Mock conversation creation (V2: empty conversation)
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
        # V2 pattern: conversation created empty, input passed
        # directly to responses.create
        mock_openai.conversations.create.assert_called_once_with()
        call_kwargs = mock_openai.responses.create.call_args.kwargs
        assert call_kwargs["input"] == "Hello!"
        assert call_kwargs["conversation"] == "conv_123"
        mock_openai.close.assert_called_once()

    def test_func_human_message_existing_conversation(self) -> None:
        """Test _func with HumanMessage uses existing conversation."""
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
        # V2 pattern: input goes directly to responses.create,
        # not as conversation items
        call_kwargs = mock_openai.responses.create.call_args.kwargs
        assert call_kwargs["input"] == "Follow up"
        assert call_kwargs["conversation"] == "conv_existing"
        # Should not create new conversation or add items directly
        mock_openai.conversations.create.assert_not_called()
        mock_openai.conversations.items.create.assert_not_called()

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

    def test_func_human_message_with_file_uploads(self) -> None:
        """Test _func with a HumanMessage containing file blocks for code interpreter.

        When the agent uses the ``{{container_id}}`` template (indicated by
        ``_uses_container_template = True``), file blocks should be uploaded
        to a new container and the container ID passed via
        ``structured_inputs`` in extra_body.
        """
        import base64

        node = self._make_node()
        # Enable the container-template pattern.
        node._uses_container_template = True
        config: Dict[str, Any] = {"callbacks": None, "metadata": None, "tags": None}

        mock_openai = MagicMock()
        node._client.get_openai_client.return_value = mock_openai

        # Mock container creation
        mock_container = MagicMock()
        mock_container.id = "container_abc123"
        mock_openai.containers.create.return_value = mock_container

        # Mock conversation creation
        mock_conversation = MagicMock()
        mock_conversation.id = "conv_files"
        mock_openai.conversations.create.return_value = mock_conversation

        # Mock response
        mock_response = MagicMock()
        mock_response.id = "resp_files"
        mock_response.status = "completed"
        mock_response.output = []
        mock_response.output_text = "Here is your chart."
        mock_response.usage = None
        mock_openai.responses.create.return_value = mock_response

        raw_data = b"month,sales\nJan,100"
        b64_data = base64.b64encode(raw_data).decode("utf-8")
        state = {
            "messages": [
                HumanMessage(
                    content=[
                        {
                            "type": "file",
                            "source_type": "base64",
                            "mime_type": "text/csv",
                            "base64": b64_data,
                        },
                        {"type": "text", "text": "make a chart"},
                    ]
                )
            ]
        }
        result = node._func(state, config, store=None)

        assert "messages" in result
        # Conversation created empty
        mock_openai.conversations.create.assert_called_once_with()
        # No conversation items created
        mock_openai.conversations.items.create.assert_not_called()
        # Container created for file uploads
        mock_openai.containers.create.assert_called_once()
        # File uploaded to the container
        mock_openai.containers.files.create.assert_called_once()
        container_call = mock_openai.containers.files.create.call_args.kwargs
        assert container_call["container_id"] == "container_abc123"
        # Text goes to responses.create as input (list form after file
        # block removal, wrapped as a user-role message).
        resp_call = mock_openai.responses.create.call_args.kwargs
        resp_input = resp_call["input"]
        assert isinstance(resp_input, list)
        assert len(resp_input) == 1
        assert resp_input[0]["role"] == "user"
        # The remaining text content block
        assert any(
            part.get("text") == "make a chart" for part in resp_input[0]["content"]
        )
        assert resp_call["conversation"] == "conv_files"
        # No tools parameter — file access is via structured_inputs
        assert "tools" not in resp_call
        # structured_inputs passed in extra_body with container_id
        extra_body = resp_call["extra_body"]
        assert extra_body["structured_inputs"] == {
            "container_id": "container_abc123",
        }


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

        factory = AgentServiceFactoryV2(project_endpoint="https://test.endpoint.com")

        mock_node = MagicMock(spec=PromptBasedAgentNodeV2)
        factory.delete_agent(mock_node)
        mock_node.delete_agent_from_node.assert_called_once()

    def test_delete_agent_with_graph(self) -> None:
        """Test deleting an agent from a compiled state graph."""
        from langgraph.graph.state import CompiledStateGraph

        from langchain_azure_ai.agents.agent_service_v2 import (
            AgentServiceFactoryV2,
        )

        factory = AgentServiceFactoryV2(project_endpoint="https://test.endpoint.com")

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

        factory = AgentServiceFactoryV2(project_endpoint="https://test.endpoint.com")

        with pytest.raises(ValueError, match="CompiledStateGraph"):
            factory.delete_agent("not_an_agent")  # type: ignore[arg-type]

    def test_delete_agent_no_ids_in_graph(self) -> None:
        """Test deleting when no agent IDs found in graph metadata."""
        from langgraph.graph.state import CompiledStateGraph

        from langchain_azure_ai.agents.agent_service_v2 import (
            AgentServiceFactoryV2,
        )

        factory = AgentServiceFactoryV2(project_endpoint="https://test.endpoint.com")

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


class TestContentFromHumanMessageFileBlocks:
    """Tests for _content_from_human_message handling of file blocks."""

    def test_file_blocks_are_skipped(self) -> None:
        """File blocks are handled by _upload_file_blocks_v2, not here."""
        import base64

        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _content_from_human_message,
        )

        raw_data = b"test file content"
        b64_data = base64.b64encode(raw_data).decode("utf-8")

        msg = HumanMessage(
            content=[
                {
                    "type": "file",
                    "mime_type": "text/csv",
                    "base64": b64_data,
                },
                {"type": "text", "text": "analyze this"},
            ]
        )

        result = _content_from_human_message(msg)
        # file block is skipped; only text block remains
        assert len(result) == 1

    def test_only_file_blocks_returns_empty_list(self) -> None:
        """When all blocks are file blocks, result is an empty list."""
        import base64

        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _content_from_human_message,
        )

        raw_data = b"test file content"
        b64_data = base64.b64encode(raw_data).decode("utf-8")

        msg = HumanMessage(
            content=[
                {
                    "type": "file",
                    "mime_type": "text/csv",
                    "base64": b64_data,
                },
            ]
        )

        result = _content_from_human_message(msg)
        assert result == []


class TestAgentHasCodeInterpreterV2Dict:
    """Tests for _agent_has_code_interpreter_v2 with dict-based tools."""

    def test_dict_code_interpreter_tool(self) -> None:
        """Test detection of code_interpreter in dict tool definitions."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _agent_has_code_interpreter_v2,
        )

        mock_agent = MagicMock()
        mock_agent.definition = {"tools": [{"type": "code_interpreter"}]}

        assert _agent_has_code_interpreter_v2(mock_agent) is True

    def test_dict_non_code_interpreter_tool(self) -> None:
        """Test that dict tools without code_interpreter return False."""
        from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
            _agent_has_code_interpreter_v2,
        )

        mock_agent = MagicMock()
        mock_agent.definition = {"tools": [{"type": "function"}]}

        assert _agent_has_code_interpreter_v2(mock_agent) is False
