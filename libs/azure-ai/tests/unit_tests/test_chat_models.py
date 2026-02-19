import asyncio
import json
import logging
import os
from typing import Any, Generator
from unittest import mock

# import aiohttp to force Pants to include it in the required dependencies
import aiohttp  # noqa
import pytest
from azure.ai.inference.models import (
    ChatChoice,
    ChatCompletions,
    ChatCompletionsToolCall,
    ChatResponseMessage,
    CompletionsFinishReason,
    ModelInfo,
)
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_azure_ai.chat_models.inference import (
    _convert_message_content,
    _format_tool_call_for_azure_inference,
    to_inference_message,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_params() -> dict:
    return {
        "input": [
            SystemMessage(
                content="You are a helpful assistant. When you are asked about if this "
                "is a test, you always reply 'Yes, this is a test.'",
            ),
            HumanMessage(role="user", content="Is this a test?"),
        ],
    }


@pytest.fixture(scope="session")
def test_llm() -> AzureAIChatCompletionsModel:
    with mock.patch(
        "langchain_azure_ai.chat_models.inference.ChatCompletionsClient", autospec=True
    ):
        with mock.patch(
            "langchain_azure_ai.chat_models.inference.ChatCompletionsClientAsync",
            autospec=True,
        ):
            llm = AzureAIChatCompletionsModel(
                endpoint="https://my-endpoint.inference.ai.azure.com",
                credential="my-api-key",
            )
    llm._client.complete.return_value = ChatCompletions(  # type: ignore
        choices=[
            ChatChoice(
                index=0,
                finish_reason=CompletionsFinishReason.STOPPED,
                message=ChatResponseMessage(
                    content="Yes, this is a test.", role="assistant"
                ),
            ),
        ]
    )
    llm._client.get_model_info.return_value = ModelInfo(  # type: ignore
        model_name="my_model_name",
        model_provider_name="my_provider_name",
        model_type="chat-completions",
    )
    llm._async_client.complete = mock.AsyncMock(  # type: ignore
        return_value=ChatCompletions(  # type: ignore
            choices=[
                ChatChoice(
                    index=0,
                    finish_reason=CompletionsFinishReason.STOPPED,
                    message=ChatResponseMessage(
                        content="Yes, this is a test.", role="assistant"
                    ),
                ),
            ]
        )
    )
    return llm


@pytest.fixture()
def test_llm_json() -> AzureAIChatCompletionsModel:
    with mock.patch(
        "langchain_azure_ai.chat_models.inference.ChatCompletionsClient", autospec=True
    ):
        llm = AzureAIChatCompletionsModel(
            endpoint="https://my-endpoint.inference.ai.azure.com",
            credential="my-api-key",
        )
    llm._client.complete.return_value = ChatCompletions(  # type: ignore
        choices=[
            ChatChoice(
                index=0,
                finish_reason=CompletionsFinishReason.STOPPED,
                message=ChatResponseMessage(
                    content='{ "message": "Yes, this is a test." }', role="assistant"
                ),
            ),
        ]
    )
    return llm


@pytest.fixture()
def test_llm_tools() -> AzureAIChatCompletionsModel:
    with mock.patch(
        "langchain_azure_ai.chat_models.inference.ChatCompletionsClient", autospec=True
    ):
        llm = AzureAIChatCompletionsModel(
            endpoint="https://my-endpoint.inference.ai.azure.com",
            credential="my-api-key",
        )
    llm._client.complete.return_value = ChatCompletions(  # type: ignore
        choices=[
            ChatChoice(
                index=0,
                finish_reason=CompletionsFinishReason.TOOL_CALLS,
                message=ChatResponseMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ChatCompletionsToolCall(
                            {
                                "id": "abc0dF1gh",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": '{ "message": "Is this a test?" }',
                                    "call_id": None,
                                },
                            }
                        )
                    ],
                ),
            )
        ]
    )
    return llm


def test_chat_completion(
    test_llm: AzureAIChatCompletionsModel, test_params: dict
) -> None:
    """Tests the basic chat completion functionality."""
    response = test_llm.invoke(**test_params)

    assert isinstance(response, AIMessage)
    if isinstance(response.content, str):
        assert response.content.strip() == "Yes, this is a test."


def test_achat_completion(
    test_llm: AzureAIChatCompletionsModel,
    loop: asyncio.AbstractEventLoop,
    test_params: dict,
) -> None:
    """Tests the basic chat completion functionality asynchronously."""
    response = loop.run_until_complete(test_llm.ainvoke(**test_params))

    assert isinstance(response, AIMessage)
    if isinstance(response.content, str):
        assert response.content.strip() == "Yes, this is a test."


@pytest.mark.skipif(
    not {
        "AZURE_INFERENCE_ENDPOINT",
        "AZURE_INFERENCE_CREDENTIAL",
    }.issubset(set(os.environ)),
    reason="Azure AI endpoint and/or credential are not set.",
)
def test_stream_chat_completion(test_params: dict) -> None:
    """Tests the basic chat completion functionality with streaming."""
    model_name = os.environ.get("AZURE_INFERENCE_MODEL", None)

    llm = AzureAIChatCompletionsModel(model=model_name)

    response_stream = llm.stream(**test_params)

    buffer = ""
    for chunk in response_stream:
        buffer += chunk.content  # type: ignore

    assert buffer.strip() == "Yes, this is a test."


@pytest.mark.skipif(
    not {
        "AZURE_INFERENCE_ENDPOINT",
        "AZURE_INFERENCE_CREDENTIAL",
    }.issubset(set(os.environ)),
    reason="Azure AI endpoint and/or credential are not set.",
)
def test_astream_chat_completion(
    test_params: dict, loop: asyncio.AbstractEventLoop
) -> None:
    """Tests the basic chat completion functionality with streaming."""
    model_name = os.environ.get("AZURE_INFERENCE_MODEL", None)

    llm = AzureAIChatCompletionsModel(model=model_name)

    async def iterate() -> str:
        stream = llm.astream(**test_params)
        buffer = ""
        async for chunk in stream:
            buffer += chunk.content  # type: ignore

        return buffer

    response = loop.run_until_complete(iterate())
    assert response.strip() == "Yes, this is a test."


def test_chat_completion_kwargs(
    test_llm_json: AzureAIChatCompletionsModel,
) -> None:
    """Tests chat completions using extra parameters."""
    test_llm_json.model_kwargs.update({"response_format": {"type": "json_object"}})
    response = test_llm_json.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant. When you are asked about if "
                "this is a test, you always reply 'Yes, this is a test.' in a JSON "
                "object with key 'message'.",
            ),
            HumanMessage(content="Is this a test?"),
        ],
        temperature=0.0,
        top_p=1.0,
    )

    assert isinstance(response, AIMessage)
    if isinstance(response.content, str):
        assert (
            json.loads(response.content.strip()).get("message")
            == "Yes, this is a test."
        )


def test_chat_completion_with_tools(
    test_llm_tools: AzureAIChatCompletionsModel,
) -> None:
    """Tests the chat completion functionality with the help of tools."""

    def echo(message: str) -> str:
        """Echoes the user's message.

        Args:
            message: The message to echo
        """
        print("Echo: " + message)
        return message

    model_with_tools = test_llm_tools.bind_tools([echo])

    response = model_with_tools.invoke(
        [
            SystemMessage(
                content="You are an assistant that always echoes the user's message. "
                "To echo a message, use the 'Echo' tool.",
            ),
            HumanMessage(content="Is this a test?"),
        ]
    )

    assert isinstance(response, AIMessage)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "echo"


def test_with_structured_output_json_mode(
    test_llm_json: AzureAIChatCompletionsModel,
) -> None:
    """Tests with_structured_output using method='json_mode'."""
    # The schema is not actually used by the model in json_mode, but for
    # completeness, pass a dict.
    schema = {"type": "object", "properties": {"message": {"type": "string"}}}

    runnable = test_llm_json.with_structured_output(schema, method="json_mode")

    messages = [
        SystemMessage(
            content="You are a helpful assistant. When you are asked if this is "
            "a test, reply with a JSON object with key 'message'."
        ),
        HumanMessage(content="Is this a test?"),
    ]

    response = runnable.invoke(messages)
    # The output should be a dict after parsing
    assert isinstance(response, dict)
    assert response.get("message") == "Yes, this is a test."


@pytest.mark.skipif(
    not {
        "AZURE_INFERENCE_ENDPOINT",
        "AZURE_INFERENCE_CREDENTIAL",
    }.issubset(set(os.environ)),
    reason="Azure AI endpoint and/or credential are not set.",
)
def test_chat_completion_gpt4o_api_version(test_params: dict) -> None:
    """Test chat completions endpoint with api_version indicated for a GPT model."""
    # In case the endpoint being tested serves more than one model
    model_name = os.environ.get("AZURE_INFERENCE_MODEL", "gpt-4o")

    llm = AzureAIChatCompletionsModel(
        model=model_name, api_version="2024-05-01-preview"
    )

    response = llm.invoke(**test_params)

    assert isinstance(response, AIMessage)
    if isinstance(response.content, str):
        assert response.content.strip() == "Yes, this is a test."


def test_get_metadata(test_llm: AzureAIChatCompletionsModel, caplog: Any) -> None:
    """Tests if we can get model metadata back from the endpoint. If so,
    `_model_name` should not be 'unknown'. Some endpoints may not support this
    and in those cases a warning should be logged.
    """
    assert (
        test_llm._model_name != "unknown"
        or "does not support model metadata retrieval" in caplog.text
    )


def test_format_tool_call_has_function_type() -> None:
    tool_call = ToolCall(
        id="test-id-123",
        name="echo",
        args=json.loads('{"message": "Is this a test?"}'),
    )
    result = _format_tool_call_for_azure_inference(tool_call)
    assert result.get("type") == "function"
    assert result.get("function", {}).get("name") == "echo"


# ---------------------------------------------------------------------------
# _convert_message_content
# ---------------------------------------------------------------------------


def test_convert_message_content_string_passthrough() -> None:
    """Plain strings are returned unchanged."""
    assert _convert_message_content("hello") == "hello"


def test_convert_message_content_list_of_strings() -> None:
    """String items in a list are wrapped with type='text'."""
    result = _convert_message_content(["hello", "world"])
    assert result == [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
    ]


def test_convert_message_content_list_with_typed_dicts() -> None:
    """Dicts that already have a 'type' key are passed through unchanged."""
    content = [{"type": "text", "text": "hello"}]
    assert _convert_message_content(content) == [{"type": "text", "text": "hello"}]


def test_convert_message_content_list_dict_missing_type() -> None:
    """Dicts without 'type' get 'type': 'text' injected."""
    content = [{"text": "hello"}]
    result = _convert_message_content(content)
    assert result == [{"type": "text", "text": "hello"}]


def test_convert_message_content_mixed_list() -> None:
    """Mixed lists of strings and dicts are all normalised."""
    content = [
        "plain text",
        {"type": "text", "text": "already typed"},
        {"text": "no type"},
    ]
    result = _convert_message_content(content)
    assert result == [
        {"type": "text", "text": "plain text"},
        {"type": "text", "text": "already typed"},
        {"type": "text", "text": "no type"},
    ]


# ---------------------------------------------------------------------------
# to_inference_message – content normalisation
# ---------------------------------------------------------------------------


def test_to_inference_message_human_string_content() -> None:
    """String content for HumanMessage is left as a string."""
    msgs = to_inference_message([HumanMessage(content="hi")])
    assert msgs[0]["content"] == "hi"


def test_to_inference_message_human_list_content() -> None:
    """List content for HumanMessage has type fields ensured."""
    msgs = to_inference_message(
        [HumanMessage(content=[{"type": "text", "text": "hi"}])]
    )
    assert msgs[0]["content"] == [{"type": "text", "text": "hi"}]


def test_to_inference_message_human_list_string_content() -> None:
    """String items inside a list are wrapped with type='text'."""
    msgs = to_inference_message([HumanMessage(content=["hi"])])
    assert msgs[0]["content"] == [{"type": "text", "text": "hi"}]


def test_to_inference_message_system_list_content() -> None:
    """SystemMessage list content is normalised."""
    msgs = to_inference_message([SystemMessage(content=["be helpful"])])
    assert msgs[0]["content"] == [{"type": "text", "text": "be helpful"}]


def test_to_inference_message_ai_list_content() -> None:
    """AIMessage list content is normalised."""
    msgs = to_inference_message([AIMessage(content=["sure"])])
    assert msgs[0]["content"] == [{"type": "text", "text": "sure"}]


def test_to_inference_message_tool_list_content() -> None:
    """ToolMessage list content is normalised."""
    msgs = to_inference_message(
        [
            ToolMessage(
                content=["result"],
                tool_call_id="call-1",
                name="my_tool",
            )
        ]
    )
    assert msgs[0]["content"] == [{"type": "text", "text": "result"}]


def test_to_inference_message_multiple_messages_list_content() -> None:
    """Multiple messages with list content are all normalised (the original bug)."""
    messages = [
        SystemMessage(content="system prompt"),
        HumanMessage(content="user question"),
    ]
    result = to_inference_message(messages)
    # Both messages should have string content unchanged
    assert result[0]["content"] == "system prompt"
    assert result[1]["content"] == "user question"


def test_to_inference_message_chat_message_list_content() -> None:
    """ChatMessage list content is normalised."""
    msgs = to_inference_message([ChatMessage(role="user", content=["hello"])])
    assert msgs[0]["content"] == [{"type": "text", "text": "hello"}]
