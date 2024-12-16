"""Azure AI Inference Chat Models API."""

import json
import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.aio import ChatCompletionsClient as ChatCompletionsClientAsync
from azure.ai.inference.models import (
    ChatCompletions,
    ChatRequestMessage,
    ChatResponseMessage,
)
from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.core.exceptions import HttpResponseError
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel, ChatGeneration
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env, pre_init
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr, model_validator

logger = logging.getLogger(__name__)


def to_inference_message(
    messages: List[BaseMessage],
) -> List[ChatRequestMessage]:
    """Converts a sequence of `BaseMessage` to `ChatRequestMessage`.

    Args:
        messages (Sequence[BaseMessage]): The messages to convert.

    Returns:
        List[ChatRequestMessage]: The converted messages.
    """
    new_messages = []
    for m in messages:
        message_dict: Dict[str, Any] = {}
        if isinstance(m, ChatMessage):
            message_dict = {
                "role": m.type,
                "content": m.content,
            }
        elif isinstance(m, HumanMessage):
            message_dict = {
                "role": "user",
                "content": m.content,
            }
        elif isinstance(m, AIMessage):
            message_dict = {
                "role": "assistant",
                "content": m.content,
            }
            tool_calls = []
            if m.tool_calls:
                for tool_call in m.tool_calls:
                    tool_calls.append(_format_tool_call_for_azure_inference(tool_call))
            elif "tool_calls" in m.additional_kwargs:
                for tc in m.additional_kwargs["tool_calls"]:
                    chunk = {
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        }
                    }
                    if _id := tc.get("id"):
                        chunk["id"] = _id
                    tool_calls.append(chunk)
            else:
                pass
            if tool_calls:
                message_dict["tool_calls"] = tool_calls

        elif isinstance(m, SystemMessage):
            message_dict = {
                "role": "system",
                "content": m.content,
            }
        elif isinstance(m, ToolMessage):
            message_dict = {
                "role": "tool",
                "content": m.content,
                "name": m.name,
                "tool_call_id": m.tool_call_id,
            }
        new_messages.append(ChatRequestMessage(message_dict))
    return new_messages


def from_inference_message(message: ChatResponseMessage) -> BaseMessage:
    """Convert an inference message dict to generic message."""
    if message.role == "user":
        return HumanMessage(content=message.content)
    elif message.role == "assistant":
        additional_kwargs: Dict = {}
        if message.tool_calls is not None:
            tool_calls: List[ToolCall] = []
            for tool_call in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tool_call["function"]["name"],
                        args=tool_call["function"]["arguments"],
                        id=tool_call.get("id"),
                    )
                )
            additional_kwargs.update(tool_calls=tool_calls)
        return AIMessage(content=message.content, additional_kwargs=additional_kwargs)
    elif message.role == "system":
        return SystemMessage(content=message.content)
    else:
        return ChatMessage(content=message.content, role=message.role)


def _convert_delta_to_message_chunk(
    _dict: Any, default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Convert a delta response to a message chunk."""
    role = _dict.role
    content = _dict.content or ""
    additional_kwargs: Dict = {}

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict.name)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)  # type: ignore[call-arg]


def _format_tool_call_for_azure_inference(tool_call: ToolCall) -> dict:
    """Format Langchain ToolCall to dict expected by Azure AI Inference."""
    result: Dict[str, Any] = {
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        }
    }
    if _id := tool_call.get("id"):
        result["id"] = _id

    return result


class AzureAIChatCompletionsModel(BaseChatModel):
    """Azure AI Chat Completions Model.

    The Azure AI model inference API (https://aka.ms/azureai/modelinference)
    provides a common layer to talk with most models deployed to Azure AI. This class
    providers inference for chat completions models supporting it. See documentation
    for the list of models supporting the API.

    Examples:
        .. code-block:: python
            from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
            from langchain_core.messages import HumanMessage, SystemMessage

            model = AzureAIChatCompletionsModel(
                endpoint="https://[your-service].services.ai.azure.com/models",
                credential="your-api-key",
                model_name="mistral-large-2407",
            )

            messages = [
                SystemMessage(
                    content="Translate the following from English into Italian"
                ),
                HumanMessage(content="hi!"),
            ]

            model.invoke(messages)

        For serverless endpoints running a single model, the `model_name` parameter
        can be omitted:

        .. code-block:: python
            from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
            from langchain_core.messages import HumanMessage, SystemMessage

            model = AzureAIChatCompletionsModel(
                endpoint="https://[your-service].inference.ai.azure.com",
                credential="your-api-key",
            )

            messages = [
                SystemMessage(
                    content="Translate the following from English into Italian"
                ),
                HumanMessage(content="hi!"),
            ]

            model.invoke(messages)

        You can pass additional properties to the underlying model, including
        `temperature`, `top_p`, `presence_penalty`, etc.

        .. code-block:: python
            model = AzureAIChatCompletionsModel(
                endpoint="https://[your-service].services.ai.azure.com/models",
                credential="your-api-key",
                model_name="mistral-large-2407",
                temperature=0.5,
                top_p=0.9,
            )

        Certain models may require to pass the `api_version` parameter. When
        not indicate, the default version of the Azure AI Inference SDK is used.
        Check the model documentation to know which api version to use.

        .. code-block:: python
            model = AzureAIChatCompletionsModel(
                endpoint="https://[your-service].services.ai.azure.com/models",
                credential="your-api-key",
                model_name="gpt-4o",
                api_version="2024-05-01-preview",
            )

    Troubleshooting:
        To diagnostic issues with the model, you can enable debug logging:

        .. code-block:: python
            import sys
            import logging
            from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

            logger = logging.getLogger("azure")

            # Set the desired logging level. logging.
            logger.setLevel(logging.DEBUG)

            handler = logging.StreamHandler(stream=sys.stdout)
            logger.addHandler(handler)

            model = AzureAIChatCompletionsModel(
                endpoint="https://[your-service].services.ai.azure.com/models",
                credential="your-api-key",
                model_name="mistral-large-2407",
                client_kwargs={ "logging_enable": True }
            )
    """

    endpoint: Optional[str] = None
    """The endpoint URI where the model is deployed."""

    credential: Optional[Union[str, AzureKeyCredential, TokenCredential]] = None
    """The API key or credential to use for the Azure AI model inference service."""

    api_version: Optional[str] = None
    """The API version to use for the Azure AI model inference API. If None, the 
    default version is used."""

    model_name: Optional[str] = None
    """The name of the model to use for inference, if the endpoint is running more 
    than one model. If
    not, this parameter is ignored."""

    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate in the response. If None, the 
    default maximum tokens is used."""

    temperature: Optional[float] = None
    """The temperature to use for sampling from the model. If None, the default 
    temperature is used."""

    top_p: Optional[float] = None
    """The top-p value to use for sampling from the model. If None, the default 
    top-p value is used."""

    presence_penalty: Optional[float] = None
    """The presence penalty to use for sampling from the model. If None, the 
    default presence penalty is used."""

    frequency_penalty: Optional[float] = None
    """The frequency penalty to use for sampling from the model. If None, the 
    default frequency penalty is used."""

    stop: Optional[str] = None
    """The stop token to use for stopping generation. If None, the default stop 
    token is used."""

    seed: Optional[int] = None
    """The seed to use for random number generation. If None, the default seed 
    is used."""

    model_kwargs: Dict[str, Any] = {}
    """Additional kwargs model parameters."""

    client_kwargs: Dict[str, Any] = {}
    """Additional kwargs for the Azure AI client used."""

    _client: ChatCompletionsClient = PrivateAttr()
    _async_client: ChatCompletionsClientAsync = PrivateAttr()
    _model_name: str = PrivateAttr()

    @pre_init
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        values["endpoint"] = get_from_dict_or_env(
            values, "endpoint", "AZURE_INFERENCE_ENDPOINT"
        )
        values["credential"] = get_from_dict_or_env(
            values, "credential", "AZURE_INFERENCE_CREDENTIAL"
        )

        return values

    @model_validator(mode="after")
    def initialize_client(self) -> "AzureAIChatCompletionsModel":
        """Initialize the Azure AI model inference client."""
        credential = (
            AzureKeyCredential(self.credential)
            if isinstance(self.credential, str)
            else self.credential
        )

        if not self.endpoint:
            raise ValueError(
                "You must provide an endpoint to use the Azure AI model inference "
                "client. Pass the endpoint as a parameter or set the "
                "AZURE_INFERENCE_ENDPOINT environment variable."
            )

        if not self.credential:
            raise ValueError(
                "You must provide an credential to use the Azure AI model inference."
                "client. Pass the credential as a parameter or set the "
                "AZURE_INFERENCE_CREDENTIAL environment variable."
            )

        self._client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=credential,  # type: ignore[arg-type]
            model=self.model_name,
            user_agent="langchain-azure-inference",
            **self.client_kwargs,
        )

        self._async_client = ChatCompletionsClientAsync(
            endpoint=self.endpoint,
            credential=credential,  # type: ignore[arg-type]
            model=self.model_name,
            user_agent="langchain-azure-inference",
            **self.client_kwargs,
        )

        if not self.model_name:
            try:
                # Get model info from the endpoint. This method may not be supported
                # by all endpoints.
                model_info = self._client.get_model_info()
                self._model_name = model_info.get("model_name", None)
            except HttpResponseError:
                logger.warning(
                    f"Endpoint '{self.endpoint}' does not support model metadata "
                    "retrieval. Unable to populate model attributes. If this endpoint "
                    "supports multiple models, you may be forgetting to indicate "
                    "`model_name` parameter."
                )
        else:
            self._model_name = self.model_name

        return self

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "AzureAIChatCompletionsModel"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if self.temperature:
            params["temperature"] = self.temperature
        if self.top_p:
            params["top_p"] = self.top_p
        if self.presence_penalty:
            params["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty:
            params["frequency_penalty"] = self.frequency_penalty
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        if self.seed:
            params["seed"] = self.seed
        if self.model_kwargs:
            params["model_extras"] = self.model_kwargs
        return params

    def _create_chat_result(self, response: ChatCompletions) -> ChatResult:
        generations = []
        token_usage = response.get("usage", {})
        for res in response["choices"]:
            finish_reason = res.get("finish_reason")
            message = from_inference_message(res.message)
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = {
                    "input_tokens": token_usage.get("prompt_tokens", 0),
                    "output_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
            gen = ChatGeneration(
                message=message,
                generation_info={"finish_reason": finish_reason},
            )
            generations.append(gen)

        llm_output: Dict[str, Any] = {"model": self._model_name}
        if isinstance(message, AIMessage):
            llm_output["token_usage"] = message.usage_metadata
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        inference_messages = to_inference_message(messages)
        response = self._client.complete(
            messages=inference_messages,
            stop=stop or self.stop,
            **self._identifying_params,
            **kwargs,
        )
        return self._create_chat_result(response)  # type: ignore[arg-type]

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        inference_messages = to_inference_message(messages)
        response = await self._async_client.complete(
            messages=inference_messages,
            stop=stop or self.stop,
            **self._identifying_params,
            **kwargs,
        )
        return self._create_chat_result(response)  # type: ignore[arg-type]

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        inference_messages = to_inference_message(messages)
        default_chunk_class = AIMessageChunk

        response = self._client.complete(
            messages=inference_messages,
            stream=True,
            stop=stop or self.stop,
            **self._identifying_params,
            **kwargs,
        )

        for chunk in response:
            choice = chunk.choices[0]
            chunk = _convert_delta_to_message_chunk(choice.delta, default_chunk_class)
            finish_reason = choice.finish_reason
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__  # type: ignore[assignment]
            cg_chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info
            )
            if run_manager:
                run_manager.on_llm_new_token(cg_chunk.text, chunk=cg_chunk)
            yield cg_chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        inference_messages = to_inference_message(messages)
        default_chunk_class = AIMessageChunk

        response = await self._async_client.complete(
            messages=inference_messages,
            stream=True,
            stop=stop or self.stop,
            **self._identifying_params,
            **kwargs,
        )

        async for chunk in response:  # type: ignore[union-attr]
            choice = chunk.choices[0]
            chunk = _convert_delta_to_message_chunk(choice.delta, default_chunk_class)
            finish_reason = choice.finish_reason
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__  # type: ignore[assignment]
            cg_chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info
            )
            if run_manager:
                await run_manager.on_llm_new_token(token=chunk.content, chunk=cg_chunk)  # type: ignore[arg-type]
            yield cg_chunk

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any), or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            kwargs: Any additional parameters are passed directly to
                ``self.bind(**kwargs)``.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "azure_inference"]
