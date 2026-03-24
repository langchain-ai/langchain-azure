"""Unit tests for AzureAIOpenAIApiChatModel."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import ChatMessage, HumanMessage, SystemMessage

from langchain_azure_ai.chat_models.openai import AzureAIOpenAIApiChatModel

# Suppress ExperimentalWarning in this file so tool-binding tests are clean.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_azure_ai._api.base.ExperimentalWarning"
)


@pytest.fixture()
def model() -> AzureAIOpenAIApiChatModel:
    """Create an AzureAIOpenAIApiChatModel with mocked clients."""
    with patch(
        "langchain_azure_ai.chat_models.openai._configure_openai_credential_values"
    ) as mock_configure:
        sync_client = MagicMock()
        async_client = MagicMock()
        mock_configure.return_value = (
            {"endpoint": "https://test.openai.azure.com", "model": "gpt-4o"},
            (sync_client, async_client),
        )
        m = AzureAIOpenAIApiChatModel(
            endpoint="https://test.openai.azure.com",
            credential="fake-key",
            model="gpt-4o",
        )
    return m


class TestResponsesApiInputTypeField:
    """Verify that _get_request_payload adds type: 'message' for Responses API."""

    def test_input_items_get_type_message(
        self, model: AzureAIOpenAIApiChatModel
    ) -> None:
        """Each input dict with a role should get type='message' added."""
        messages = [
            SystemMessage(content="You are a translator."),
            HumanMessage(content="hi"),
        ]
        payload = model._get_request_payload(messages)

        assert "input" in payload
        for item in payload["input"]:
            if isinstance(item, dict) and "role" in item:
                assert item["type"] == "message"

    def test_developer_role_gets_type_message(
        self, model: AzureAIOpenAIApiChatModel
    ) -> None:
        """ChatMessage(role='developer') should also get type='message'."""
        messages = [
            ChatMessage(role="developer", content="Translate into Italian."),
            HumanMessage(content="hi"),
        ]
        payload = model._get_request_payload(messages)

        assert "input" in payload
        assert len(payload["input"]) == 2
        for item in payload["input"]:
            assert item["type"] == "message"
        assert payload["input"][0]["role"] == "developer"
        assert payload["input"][1]["role"] == "user"

    def test_no_type_added_when_responses_api_disabled(
        self, model: AzureAIOpenAIApiChatModel
    ) -> None:
        """When use_responses_api is False, messages should not be modified."""
        model.use_responses_api = False
        messages = [HumanMessage(content="hi")]
        payload = model._get_request_payload(messages)

        assert "messages" in payload
        assert "input" not in payload
        for msg in payload["messages"]:
            assert "type" not in msg

    def test_existing_type_not_overwritten(
        self, model: AzureAIOpenAIApiChatModel
    ) -> None:
        """If an input item already has a type, don't overwrite it."""
        messages = [HumanMessage(content="hi")]
        payload = model._get_request_payload(messages)

        # Manually set a different type to simulate pre-existing type
        for item in payload["input"]:
            item["type"] = "custom"

        # Re-running should not overwrite
        payload2 = model._get_request_payload(messages)
        for item in payload2["input"]:
            assert item["type"] == "message"


# ---------------------------------------------------------------------------
# bind_tools: automatic request-header injection from BuiltinTool
# ---------------------------------------------------------------------------


class TestBindToolsHeaderInjection:
    """Verify bind_tools collects BuiltinTool.request_headers as extra_headers."""

    def test_no_builtin_tools_no_extra_headers(
        self, model: AzureAIOpenAIApiChatModel
    ) -> None:
        """Non-BuiltinTool tools don't add extra_headers."""
        from langchain_azure_ai.tools.builtin import WebSearchTool

        bound = model.bind_tools([WebSearchTool()])
        # WebSearchTool has no request_headers, so extra_headers must not appear
        assert (
            bound.kwargs.get("extra_headers") is None
            or bound.kwargs.get("extra_headers") == {}
        )

    def test_image_generation_tool_injects_header(
        self, model: AzureAIOpenAIApiChatModel
    ) -> None:
        """ImageGenerationTool with model_deployment injects the correct header."""
        from langchain_azure_ai.tools.builtin import ImageGenerationTool

        tool = ImageGenerationTool(model_deployment="my-img-deploy")
        bound = model.bind_tools([tool])
        assert bound.kwargs["extra_headers"] == {
            "x-ms-oai-image-generation-deployment": "my-img-deploy"
        }

    def test_caller_extra_headers_take_precedence(
        self, model: AzureAIOpenAIApiChatModel
    ) -> None:
        """Explicitly passed extra_headers override tool-defined ones."""
        from langchain_azure_ai.tools.builtin import ImageGenerationTool

        tool = ImageGenerationTool(model_deployment="tool-deploy")
        bound = model.bind_tools(
            [tool],
            extra_headers={"x-ms-oai-image-generation-deployment": "override-deploy"},
        )
        assert bound.kwargs["extra_headers"][
            "x-ms-oai-image-generation-deployment"
        ] == ("override-deploy")

    def test_headers_merged_from_multiple_tools(
        self, model: AzureAIOpenAIApiChatModel
    ) -> None:
        """Headers from multiple BuiltinTools are merged together."""
        from langchain_azure_ai.tools.builtin import CodeInterpreterTool, WebSearchTool

        class CodeToolWithHeader(CodeInterpreterTool):
            def __init__(self) -> None:
                super().__init__()
                self._request_headers = {"X-Tool-A": "a"}

        class WebSearchToolWithHeader(WebSearchTool):
            def __init__(self) -> None:
                super().__init__()
                self._request_headers = {"X-Tool-B": "b"}

        bound = model.bind_tools([CodeToolWithHeader(), WebSearchToolWithHeader()])
        headers = bound.kwargs["extra_headers"]
        assert headers["X-Tool-A"] == "a"
        assert headers["X-Tool-B"] == "b"

    def test_no_model_deployment_headers(
        self, model: AzureAIOpenAIApiChatModel
    ) -> None:
        """ImageGenerationTool without model_deployment uses model param."""
        from langchain_azure_ai.tools.builtin import ImageGenerationTool

        tool = ImageGenerationTool(model="gpt-image-1", quality="high")
        bound = model.bind_tools([tool])
        assert bound.kwargs.get("extra_headers")
        assert bound.kwargs["extra_headers"][
            "x-ms-oai-image-generation-deployment"
        ] == ("gpt-image-1")
