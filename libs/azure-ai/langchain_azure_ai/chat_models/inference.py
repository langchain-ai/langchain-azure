"""Azure AI Inference Chat Models API."""

import json
import logging
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.aio import ChatCompletionsClient as ChatCompletionsClientAsync
from azure.ai.inference.models import (
    ChatCompletions,
    ChatRequestMessage,
    ChatResponseMessage,
    JsonSchemaFormat,
    StreamingChatCompletionsUpdate,
)
from azure.core.credentials import AzureKeyCredential
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
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from langchain_azure_ai._resources import ModelInferenceService

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
        tool_calls: List[dict[str, Any]] = []
        invalid_tool_calls: List[InvalidToolCall] = []
        additional_kwargs: Dict = {}
        if message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    raw_tool_call = parse_tool_call(tool_call.as_dict(), return_id=True)
                    if raw_tool_call:
                        tool_calls.append(raw_tool_call)
                except json.JSONDecodeError as e:
                    invalid_tool_calls.append(
                        make_invalid_tool_call(tool_call.as_dict(), str(e))
                    )
            additional_kwargs.update(tool_calls=tool_calls)
        if audio := message.get("audio"):
            additional_kwargs.update(audio=audio)
        return AIMessage(
            id=message.get("id"),
            content=message.content or "",
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    elif message.role == "system":
        return SystemMessage(content=message.content)
    elif message == "tool":
        additional_kwargs = {}
        if tool_name := message.get("name"):
            additional_kwargs["name"] = tool_name
        return ToolMessage(
            content=message.content,
            tool_call_id=cast(str, message.get("tool_call_id")),
            additional_kwargs=additional_kwargs,
            name=tool_name,
            id=message.get("id"),
        )
    else:
        return ChatMessage(content=message.content, role=message.role)


def _convert_streaming_result_to_message_chunk(
    chunk: StreamingChatCompletionsUpdate,
    default_class: Type[BaseMessageChunk],
) -> Iterable[ChatGenerationChunk]:
    token_usage = chunk.get("usage", {})
    for res in chunk["choices"]:
        finish_reason = res.get("finish_reason")
        message = _convert_delta_to_message_chunk(res.delta, default_class)
        if token_usage and isinstance(message, AIMessage):
            message.usage_metadata = {
                "input_tokens": token_usage.get("prompt_tokens", 0),
                "output_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
            }
        gen = ChatGenerationChunk(
            message=message,
            generation_info={"finish_reason": finish_reason},
        )
        yield gen


def _convert_delta_to_message_chunk(
    _dict: Any, default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Convert a delta response to a message chunk."""
    id = _dict.get("id", None)
    role = _dict.role
    content = _dict.content or ""
    additional_kwargs: Dict = {}

    tool_call_chunks: List[ToolCallChunk] = []
    if raw_tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        try:
            tool_call_chunks = [
                tool_call_chunk(
                    name=rtc["function"].get("name"),
                    args=rtc["function"].get("arguments"),
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                for rtc in raw_tool_calls
            ]
        except KeyError:
            pass

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            id=id,
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
        )
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict.name)
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content, tool_call_id=_dict["tool_call_id"], id=id
        )
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
        },
        "type": "function",
    }
    if _id := tool_call.get("id"):
        result["id"] = _id

    return result


class AzureAIChatCompletionsModel(BaseChatModel, ModelInferenceService):
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
                model="mistral-large-2407",
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
                model="mistral-large-2407",
                temperature=0.5,
                top_p=0.9,
            )

        Azure OpenAI models may require to pass the `api_version` parameter. When
        not indicate, the default version of the Azure AI Inference SDK is used.
        Check the model documentation to know which api version to use.

        .. code-block:: python
            model = AzureAIChatCompletionsModel(
                endpoint="https://[your-service].services.ai.azure.com/openai/deployments/gpt-4o",
                credential="your-api-key",
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
                model="mistral-large-2407",
                client_kwargs={ "logging_enable": True }
            )
    """

    model_name: Optional[str] = Field(default=None, alias="model")
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

    _client: ChatCompletionsClient = PrivateAttr()
    _async_client: ChatCompletionsClientAsync = PrivateAttr()
    _model_name: str = PrivateAttr()

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
            endpoint=self.endpoint,  # type: ignore[arg-type]
            credential=credential,  # type: ignore[arg-type]
            model=self.model_name,
            **self.client_kwargs,
        )

        self._async_client = ChatCompletionsClientAsync(
            endpoint=self.endpoint,  # type: ignore[arg-type]
            credential=credential,  # type: ignore[arg-type]
            model=self.model_name,
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
                self._model_name = ""
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

        llm_output: Dict[str, Any] = {"model": response.model or self._model_name}
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
        assert isinstance(response, Iterator)

        for chunk in response:
            cg_chunks = _convert_streaming_result_to_message_chunk(
                chunk, default_chunk_class
            )
            for cg_chunk in cg_chunks:
                default_chunk_class = cg_chunk.message.__class__  # type: ignore[assignment]
                if run_manager:
                    run_manager.on_llm_new_token(
                        cg_chunk.message.content,  # type: ignore[arg-type]
                        chunk=cg_chunk,
                    )
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
        assert isinstance(response, AsyncIterator)

        async for chunk in response:
            cg_chunks = _convert_streaming_result_to_message_chunk(
                chunk, default_chunk_class
            )
            for cg_chunk in cg_chunks:
                default_chunk_class = cg_chunk.message.__class__  # type: ignore[assignment]
                if run_manager:
                    await run_manager.on_llm_new_token(
                        cg_chunk.message.content,  # type: ignore[arg-type]
                        chunk=cg_chunk,
                    )
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
        if kwargs.get("tool_choice") == "any":
            kwargs["tool_choice"] = "required"

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Union[Dict, type],  # noqa: UP006
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        strict: Optional[bool] = None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:  # noqa: UP006
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The schema to use for the output. If a pydantic model is
                provided, it will be used as the output type. If a dict is
                provided, it will be used as the schema for the output.
            method: The method to use for structured output. Can be
                "function_calling", "json_mode", or "json_schema".
            strict: Whether to enforce strict mode for "json_schema".
            include_raw: Whether to include the raw response from the model
                in the output.
            kwargs: Any additional parameters are passed directly to
                ``self.with_structured_output(**kwargs)``.
        """
        if strict is not None and method == "json_mode":
            raise ValueError(
                "Argument `strict` is not supported with `method`='json_mode'"
            )
        if method == "json_schema" and schema is None:
            raise ValueError(
                "Argument `schema` must be specified when method is 'json_schema'. "
            )

        if method in ["json_mode", "json_schema"]:
            if method == "json_mode":
                llm = self.bind(response_format="json_object")
                output_parser = JsonOutputParser()
            elif method == "json_schema":
                if isinstance(schema, dict):
                    json_schema = schema.copy()
                    schema_name = json_schema.pop("name", None)
                    output_parser = JsonOutputParser()
                elif is_basemodel_subclass(schema):
                    json_schema = schema.model_json_schema()  # type: ignore[attr-defined]
                    schema_name = json_schema.pop("title", None)
                    output_parser = PydanticOutputParser(pydantic_object=schema)
                else:
                    raise ValueError("Invalid schema type. Must be dict or BaseModel.")
                llm = self.bind(
                    response_format=JsonSchemaFormat(
                        name=schema_name,
                        schema=json_schema,
                        description=json_schema.pop("description", None),
                        strict=strict,
                    )
                )

            if include_raw:
                parser_assign = RunnablePassthrough.assign(
                    parsed=itemgetter("raw") | output_parser,
                    parsing_error=lambda _: None,
                )
                parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
                parser_with_fallback = parser_assign.with_fallbacks(
                    [parser_none], exception_key="parsing_error"
                )
                return RunnableMap(raw=llm) | parser_with_fallback
            else:
                return llm | output_parser
        else:
            return super().with_structured_output(
                schema, include_raw=include_raw, **kwargs
            )

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "azure_inference"]

    async def aclose(self) -> None:
        """Close the async client to prevent unclosed session warnings.

        This method should be called to properly clean up HTTP connections
        when using async operations.
        """
        if hasattr(self, "_async_client") and self._async_client:
            await self._async_client.close()
