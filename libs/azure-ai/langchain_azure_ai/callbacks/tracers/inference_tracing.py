"""Provides a tracer for Azure AI using OpenTelemetry.

This module requires the extra `opentelemetry` to be installed. Install the package with
`pip install langchain-azure-ai[opentelemetry]`.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from langchain_azure_ai.callbacks.tracers import _semantic_conventions_gen_ai
from langchain_azure_ai.utils.utils import JSONObjectEncoder

try:
    from opentelemetry import context as context_api
    from opentelemetry import trace
    from opentelemetry.context.context import Context
    from opentelemetry.instrumentation.threading import ThreadingInstrumentor
    from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
    from opentelemetry.trace import SpanKind
    from opentelemetry.trace.propagation import set_span_in_context
    from opentelemetry.trace.span import Span
except ImportError:
    raise ImportError(
        "Using tracing capabilities requires the extra `opentelemetry`."
        "Install the package with `pip install langchain-azure-ai[opentelemetry]`"
    )

from azure.ai.inference.tracing import AIInferenceInstrumentor
from azure.core.settings import settings
from azure.monitor.opentelemetry import configure_azure_monitor

logger = logging.getLogger(__name__)


@dataclass
class SpanHolder:
    """Dataclass to hold the span and its related information."""

    span: Span
    token: Any
    context: Context
    children: list[UUID]
    agent_name: str
    entity_name: str
    entity_path: str


def _message_type_to_role(message_type: str) -> str:
    if message_type == "human":
        return "user"
    elif message_type == "system":
        return "system"
    elif message_type == "ai":
        return "assistant"
    else:
        return "unknown"


def _set_span_attribute(span: Span, name: str, value: Optional[str]) -> None:
    if value is not None:
        span.set_attribute(name, value)


def _set_request_params(span: Span, kwargs: Dict[str, Any]) -> None:
    model = kwargs.get("model", "unknown")
    span.set_attribute(_semantic_conventions_gen_ai.GEN_AI_REQUEST_MODEL, model)

    params = kwargs

    _set_span_attribute(
        span,
        _semantic_conventions_gen_ai.GEN_AI_REQUEST_MAX_INPUT_TOKENS,
        params.get("max_tokens") or params.get("max_new_tokens"),
    )
    _set_span_attribute(
        span,
        _semantic_conventions_gen_ai.GEN_AI_REQUEST_TEMPERATURE,
        params.get("temperature"),
    )
    _set_span_attribute(
        span, _semantic_conventions_gen_ai.GEN_AI_REQUEST_TOP_P, params.get("top_p")
    )


def _set_response_params(
    span: Span, response: LLMResult, should_send_prompts: bool
) -> None:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    model_name = None

    if should_send_prompts:
        span.set_attribute(
            _semantic_conventions_gen_ai.OUTPUTS,
            json.dumps(response.generations, cls=JSONObjectEncoder),
        )

    if response.llm_output is not None:
        model_name = (
            response.llm_output.get("model")
            or response.llm_output.get("model_id")
            or response.llm_output.get("model_name")
        )

        token_usage = (response.llm_output or {}).get("token_usage") or (
            response.llm_output or {}
        ).get("usage")
        if token_usage is not None:
            input_tokens = (
                token_usage.get("prompt_tokens")
                or token_usage.get("input_token_count")
                or token_usage.get("input_tokens")
            )
            output_tokens = (
                token_usage.get("completion_tokens")
                or token_usage.get("generated_token_count")
                or token_usage.get("output_tokens")
            )
            total_tokens = token_usage.get("total_tokens") or (
                input_tokens + output_tokens
            )

    if input_tokens > 0:
        span.set_attribute(
            _semantic_conventions_gen_ai.GEN_AI_USAGE_INPUT_TOKENS,
            input_tokens,
        )
    if output_tokens > 0:
        span.set_attribute(
            _semantic_conventions_gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS,
            output_tokens,
        )
    if total_tokens > 0:
        span.set_attribute(
            _semantic_conventions_gen_ai.GEN_AI_USAGE_TOTAL_TOKENS,
            total_tokens,
        )

    if model_name:
        span.set_attribute(
            _semantic_conventions_gen_ai.GEN_AI_RESPONSE_MODEL, model_name
        )


def _handle_event_error(e: Exception) -> None:
    """Handles errors in the callbacks."""
    logger.error(f"Error in tracing callback: {e}", exc_info=True)


class AzureAIInferenceTracer(BaseCallbackHandler):
    """Tracer for Azure AI Inference integrations inside of LangChain.

    This tracer uses OpenTelemetry to instrument LangChain callbacks and trace
    them in Azure Application Insights for monitoring and debugging purposes. Since
    LangChain uses `asyncio` for running callbacks, context is propagated to other
    threads using OpenTelemetry.

    For more information and tutorials about how to use langchain-azure-ai, including
    the tracing capabilities, see https://aka.ms/azureai/langchain.

    Example:
        .. code-block:: python
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
            from langchain_azure_ai.callbacks.tracers import AzureAIInferenceTracer

            model = AzureAIChatCompletionsModel(
                endpoint="https://[your-service].services.ai.azure.com/models",
                credential="your-api-key",
                model="mistral-large-2407",
            )

    Create the tracer. Use the `connection_string` to the Azure Application Insights
    you are using. When working on projects, you can get the connection string directly
    from the tab **Tracing** in the portal.

    Use `enable_content_recording=True` to record the inputs and outputs int he traces.
    This can be useful for debugging and monitoring purposes but can also capture
    sensitive information. By default, this is set to `False` or to the value of the
    environment variable `AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED`.

        .. code-block:: python
            tracer = AzureAIInferenceTracer(
                connection_string="InstrumentationKey=....",
                enable_content_recording=True,
            )

    Then, pass the tracer as a callback in your code:

        .. code-block:: python
            prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")
            chain = prompt | model
            chain.invoke({"number": "2"}, config={"callbacks": [tracer]})

    Alternately, use `with_config()` to pass the tracer as a callback in your runnable,
    instead of passing it on each invocation:

        .. code-block:: python
            instrumented_chain = chain.with_config({"callbacks": [tracer]})
            instrumented_chain.invoke({"number": "2"})

    Note: This module also works with other chat clients, like `OpenAI`. However,
    certain instrumentation in the `openai` library may not be available.
    """

    def __init__(
        self,
        connection_string: Optional[str],
        enable_content_recording: Optional[bool] = None,
        instrument_inference: Optional[bool] = True,
    ) -> None:
        """Initializes the tracer.

        Args:
            connection_string (str, optional): The connection string to the
                Azure Application Insights. If not indicated, the current
                context will be used.
            enable_content_recording (bool, optional): Whether to record the
                inputs and outputs in the traces. Defaults to None. If None,
                the value is taken from the environment variable
                `AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED` unless
                `instrument_inference` is set to False.
            instrument_inference (bool, optional): Whether to instrument the
                inference calls. Defaults to True.
        """
        super().__init__()
        self.spans: dict[UUID, SpanHolder] = {}
        self.run_inline = True

        if connection_string:
            settings.tracing_implementation = "opentelemetry"
            configure_azure_monitor(connection_string=connection_string)

        if instrument_inference:
            ThreadingInstrumentor().instrument()
            instrumentor = AIInferenceInstrumentor()
            instrumentor.instrument(enable_content_recording=enable_content_recording)
            self.should_send_prompts = instrumentor.is_content_recording_enabled()

        self.should_send_prompts = enable_content_recording or False
        self.tracer = trace.get_tracer(__name__)

    @staticmethod
    def _get_name_from_callback(
        serialized: dict[str, Any],
        _tags: Optional[list[str]] = None,
        _metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Get the name to be used for the span."""
        if serialized and "kwargs" in serialized and serialized["kwargs"].get("name"):
            return serialized["kwargs"]["name"]
        if kwargs.get("name"):
            return kwargs["name"]
        if serialized.get("name"):
            return serialized["name"]
        if "id" in serialized:
            return serialized["id"][-1]

        return "unknown"

    def _get_span(self, run_id: UUID) -> Span:
        """Gets the current span according to the run_id.

        This method doesn't actives the span in the current context.
        """
        return self.spans[run_id].span

    def _end_span(self, span: Span, run_id: UUID) -> None:
        """Ends the span and detaches from the current context."""
        if span:
            for child_id in self.spans[run_id].children:
                child_span = self.spans[child_id].span
                if child_span.is_recording():  # avoid warning on ended spans
                    child_span.end()
            if span.is_recording():  # avoid warning on ended spans
                span.end()

    def _create_span(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        span_name: str,
        type: str,
        agent_name: str = "",
        entity_name: str = "",
        entity_path: str = "",
        tags: Optional[list[str]] = [],
        metadata: Optional[dict[str, Any]] = None,
    ) -> Span:
        """Creates a new span and attaches it to the current context.

        Spans are created as child in the current span if there is already a parent
        span running.
        """
        if metadata is not None:
            current_association_properties: Dict[str, Any] = (
                context_api.get_value("association_properties") or {}  # type: ignore[assignment]
            )
            context_api.attach(
                context_api.set_value(
                    "association_properties",
                    {**current_association_properties, **metadata},
                )
            )

        if parent_run_id is not None and parent_run_id in self.spans:
            span = self.tracer.start_span(
                span_name,
                kind=SpanKind.INTERNAL,
                context=self.spans[parent_run_id].context,
            )
        else:
            span = self.tracer.start_span(span_name, kind=SpanKind.INTERNAL)

        span.set_attribute(_semantic_conventions_gen_ai.GEN_AI_AGENT_NAME, agent_name)
        span.set_attribute(_semantic_conventions_gen_ai.GEN_AI_OPERATION_NAME, type)
        if tags:
            span.set_attribute(_semantic_conventions_gen_ai.TAGS, tags)

        ctx = set_span_in_context(span)
        token = context_api.attach(ctx)

        self.spans[run_id] = SpanHolder(
            span, token, ctx, [], agent_name, entity_name, entity_path
        )

        if parent_run_id is not None and parent_run_id in self.spans:
            self.spans[parent_run_id].children.append(run_id)

        return span

    def get_parent_span(
        self, parent_run_id: Optional[UUID] = None
    ) -> Union[SpanHolder, None]:
        """Gets the parent span when nested spans are constructed."""
        if parent_run_id is None:
            return None
        return self.spans[parent_run_id]

    def get_agent_name(self, parent_run_id: Optional[UUID]) -> str:
        """Gets the agent name from the parent span when nested spans."""
        parent_span = self.get_parent_span(parent_run_id)

        if parent_span is None:
            return ""

        return parent_span.agent_name

    def get_entity_path(self, parent_run_id: Optional[UUID]) -> str:
        """Gets the entity path from the parent span when nested spans exist."""
        parent_span = self.get_parent_span(parent_run_id)

        if parent_span is None:
            return ""
        elif parent_span.entity_path == "":
            return f"{parent_span.entity_name}"
        else:
            return f"{parent_span.entity_path}.{parent_span.entity_name}"

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain starts running."""
        try:
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return

            object_name = self._get_name_from_callback(serialized, **kwargs)
            agent_name = self.get_agent_name(parent_run_id)
            entity_path = self.get_entity_path(parent_run_id)

            span = self._create_span(
                run_id=run_id,
                parent_run_id=parent_run_id,
                span_name=object_name,
                type=kwargs.get("run_type") or "chain",
                agent_name=agent_name,
                entity_name=object_name,
                entity_path=entity_path,
                metadata=metadata,
            )

            if self.should_send_prompts:
                span.set_attribute(
                    _semantic_conventions_gen_ai.INPUTS,
                    json.dumps(inputs, cls=JSONObjectEncoder),
                )
        except Exception as e:
            _handle_event_error(e)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""
        try:
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return

            span = self._get_span(run_id)
            if span and span.is_recording():
                if self.should_send_prompts:
                    span.set_attribute(
                        _semantic_conventions_gen_ai.OUTPUTS,
                        json.dumps(outputs, cls=JSONObjectEncoder),
                    )

        except Exception as e:
            _handle_event_error(e)
        finally:
            self._end_span(span, run_id)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when Chat Model starts running."""
        try:
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return

            object_name = self._get_name_from_callback(serialized, **kwargs)
            agent_name = self.get_agent_name(parent_run_id)
            entity_path = self.get_entity_path(parent_run_id)

            span = self._create_span(
                run_id=run_id,
                parent_run_id=parent_run_id,
                span_name=object_name,
                type=kwargs.get("run_type") or "chat_model",
                agent_name=agent_name,
                entity_name=object_name,
                entity_path=entity_path,
                metadata=metadata,
            )

            if span:
                _set_request_params(span, serialized.get("kwargs", {}))

                if self.should_send_prompts:
                    # TODO: Add tools tracing

                    inputs: Dict[str, Any] = {"messages": []}
                    for message in messages:
                        for msg in message:
                            if isinstance(msg.content, str):
                                inputs["messages"].append(
                                    {
                                        "role": _message_type_to_role(msg.type),
                                        "content": msg.content,
                                    }
                                )
                            else:
                                inputs["messages"].append(
                                    {
                                        "role": _message_type_to_role(msg.type),
                                        "content": json.dumps(msg.content),
                                    }
                                )

                    _set_span_attribute(
                        span,
                        _semantic_conventions_gen_ai.INPUTS,
                        json.dumps(inputs, cls=JSONObjectEncoder),
                    )
        except Exception as e:
            _handle_event_error(e)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when Chat Model starts running."""
        try:
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return

            object_name = self._get_name_from_callback(serialized, **kwargs)
            agent_name = self.get_agent_name(parent_run_id)
            entity_path = self.get_entity_path(parent_run_id)

            span = self._create_span(
                run_id=run_id,
                parent_run_id=parent_run_id,
                span_name=object_name,
                type=kwargs.get("run_type") or "LLM",
                agent_name=agent_name,
                entity_name=object_name,
                entity_path=entity_path,
                metadata=metadata,
            )

            if span:
                _set_request_params(span, serialized.get("kwargs", {}))

                if self.should_send_prompts:
                    span.set_attribute(
                        _semantic_conventions_gen_ai.INPUTS,
                        json.dumps({"prompts": prompts}, cls=JSONObjectEncoder),
                    )
        except Exception as e:
            _handle_event_error(e)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running."""
        try:
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return

            span = self._get_span(run_id)

            if span:
                _set_response_params(span, response, self.should_send_prompts)
        except Exception as e:
            _handle_event_error(e)
        finally:
            self._end_span(span, run_id)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""
        try:
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return

            name = self._get_name_from_callback(serialized, kwargs=kwargs)
            agent_name = self.get_agent_name(parent_run_id)
            entity_path = self.get_entity_path(parent_run_id)

            span = self._create_span(
                run_id,
                parent_run_id,
                name,
                kwargs.get("run_type") or "tool",
                agent_name,
                name,
                entity_path,
            )

            if span and self.should_send_prompts:
                span.set_attribute(
                    _semantic_conventions_gen_ai.INPUTS,
                    json.dumps(
                        {
                            "input_str": input_str,
                            "tags": tags,
                            "metadata": metadata,
                            "inputs": inputs,
                            "kwargs": kwargs,
                        },
                        cls=JSONObjectEncoder,
                    ),
                )
        except Exception as e:
            _handle_event_error(e)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        try:
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return

            span = self._get_span(run_id)

            if span and self.should_send_prompts:
                span.set_attribute(
                    _semantic_conventions_gen_ai.OUTPUTS,
                    json.dumps(
                        {"output": output, "kwargs": kwargs}, cls=JSONObjectEncoder
                    ),
                )
        except Exception as e:
            _handle_event_error(e)
        finally:
            self._end_span(span, run_id)
