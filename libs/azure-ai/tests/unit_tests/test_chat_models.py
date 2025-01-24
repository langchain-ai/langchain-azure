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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

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

    llm = AzureAIChatCompletionsModel(model_name=model_name)

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

    llm = AzureAIChatCompletionsModel(model_name=model_name)

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
    assert len(response.additional_kwargs["tool_calls"]) == 1
    assert response.additional_kwargs["tool_calls"][0]["name"] == "echo"


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
        model_name=model_name, api_version="2024-05-01-preview"
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
