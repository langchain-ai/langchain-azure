"""Azure OpenAI Agent Tracing Callback Handler.

OpenTelemetry tracing for LangChain/LangGraph apps using Azure OpenAI.
Aligns to Generative AI semantic conventions and exports to Azure
Application Insights (via azure-monitor-opentelemetry).

What it captures
- Chat/LLM calls as CLIENT spans named "chat <deployment>".
    - Request: model/deployment, temperature, server.address, and API
        version (as metadata).
    - Prompt/completion events when recording is enabled using:
        - gen_ai.input.messages (prompts)
        - gen_ai.output.messages (completions)
    - Response: gen_ai.response.model, gen_ai.response.id,
        gen_ai.response.finish_reasons
    - Token usage: gen_ai.usage.input_tokens, gen_ai.usage.output_tokens
- Chains and agents:
    - Agents are detected heuristically and emit "invoke_agent <name>"
        CLIENT spans.
    - Non-agent chains emit INTERNAL spans named "chain <name>" and are
        treated as app spans (no gen_ai.operation.name="chain").
    - Agent invocation input/output recorded when enabled via
        gen_ai.agent.invocation_input and gen_ai.agent.invocation_output.
    - metadata.* stores extras (tags, run_id, api_version, endpoint, etc).
- Tools:
    - "execute_tool <name>" INTERNAL spans with args/results when tool
        callbacks fire, using gen_ai.tool.* attributes.
    - When callbacks are missing, tool-intent spans are inferred from
        AIMessage.tool_calls.

Span hierarchy and context (attach/detach)
- Each on_* start creates a span and attaches to OTel context; each end
    or error ends the span and detaches. This yields proper nesting.
- Create a root span before agent/LLM calls for a cohesive hierarchy.

Attribute schema and compliance
- Uses GenAI conventions (registry.yaml, spans.yaml, gen_ai.md):
    gen_ai.provider.name, gen_ai.operation.name, gen_ai.request/response.*,
    gen_ai.usage.*, gen_ai.agent.*, gen_ai.tool.*, server.address.
- Only primitives or lists of primitives are used; complex values are
    JSON-serialized.
- Provider name for Azure AI Inference is "azure.ai.inference".

Content recording and privacy
- Controlled by enable_content_recording or the environment variable
    AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED.
- When disabled, message bodies are redacted as "[REDACTED]".
- Be mindful of PII in production.

Azure Monitor configuration
- On init, calls configure_azure_monitor with connection string or uses
    APPLICATIONINSIGHTS_CONNECTION_STRING if set.

Lifecycle, errors, and cleanup
- Active spans are tracked by run_id and cleaned up on end/error.
- Exceptions are recorded with StatusCode.ERROR and error.type.
- Token metrics and response metadata are captured where available.

Workarounds for unavailable callbacks
- Some agent executors don’t emit on_tool_start/on_tool_end.
- The tracer inspects AIMessage.tool_calls and creates child
    "execute_tool" spans (when recording enabled) so tool intent is
    visible. These inferred spans lack duration/results unless tool
    callbacks also fire.
- Parenting uses context attach/detach, not parent_run_id.
- Complex attributes are JSON-stringified to avoid type warnings.
- High-frequency token events are not emitted to reduce noise.

Quickstart
        from langchain_azure_ai.callbacks.tracers.azure_openai_agent_tracing \
                import AzureOpenAITracingCallback
        from langchain_openai import AzureChatOpenAI

        tracer = AzureOpenAITracingCallback(enable_content_recording=True)
        llm = AzureChatOpenAI(deployment_name="gpt-4o", callbacks=[tracer])
        llm.invoke("Hello!")

Advanced: root span + agents/tools
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import SpanKind
        from langchain.agents import AgentExecutor, create_openai_tools_agent

        tracer_cb = AzureOpenAITracingCallback(enable_content_recording=True)
        agent = create_openai_tools_agent(llm, tools=[])
        agent_exec = AgentExecutor(
            agent=agent, tools=[], callbacks=[tracer_cb]
        )

        ot = otel_trace.get_tracer(__name__)
        with ot.start_as_current_span(
                "invoke_agent travel_planner", kind=SpanKind.SERVER
        ):
                result = agent_exec.invoke(
                    {"input": "Plan a 3-day London trip"}
                )
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, LLMResult

try:
    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.context import attach, detach
except ImportError:
    raise ImportError(
        (
            "Using tracing requires Azure Monitor and OpenTelemetry "
            "packages."
            " Install with: pip install azure-monitor-opentelemetry"
        )
    )

# Configure logging - suppress Azure SDK HTTP logging
logging.getLogger(
    "azure.core.pipeline.policies.http_logging_policy"
).setLevel(logging.WARNING)
logging.getLogger(
    "azure.monitor.opentelemetry.exporter.export._base"
).setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define semantic convention constants based on the provided YAML specs


class GenAIConventions:
    GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    GEN_AI_REQUEST_SEED = "gen_ai.request.seed"
    GEN_AI_RESPONSE_ID = "gen_ai.response.id"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    # TODO: review - not in registry.yaml; prefer metadata.* (registry.yaml)
    GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    GEN_AI_AGENT_ID = "gen_ai.agent.id"
    GEN_AI_AGENT_NAME = "gen_ai.agent.name"
    GEN_AI_AGENT_DESCRIPTION = "gen_ai.agent.description"
    GEN_AI_AGENT_INVOCATION_INPUT = "gen_ai.agent.invocation_input"
    GEN_AI_AGENT_INVOCATION_OUTPUT = "gen_ai.agent.invocation_output"
    GEN_AI_AGENT_CHILD_AGENTS = "gen_ai.agent.child_agents"
    GEN_AI_TOOL_NAME = "gen_ai.tool.name"
    GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
    GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
    GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
    GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
    GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
    GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"
    GEN_AI_OUTPUT_TYPE = "gen_ai.output.type"
    GEN_AI_DATA_SOURCE_ID = "gen_ai.data_source.id"
    GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
    # Structured message attributes per gen_ai.md
    # TODO: review - per gen_ai.md, use for prompts
    GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
    # TODO: review - per gen_ai.md, use for completions
    GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
    SERVER_ADDRESS = "server.address"
    SERVER_PORT = "server.port"
    ERROR_TYPE = "error.type"
    # TODO: review - 'tags' not in registry.yaml; use metadata.*
    TAGS = "metadata.tags"


conventions = GenAIConventions


class AzureOpenAITracingCallback(BaseCallbackHandler):
    """Trace LangChain/LangGraph GenAI calls to Azure App Insights.

    Implements OpenTelemetry Generative AI semantic conventions and provides
    standardized telemetry for monitoring and debugging LLM and agent apps.

    The tracer captures:
    - LLM request/response details (model, parameters, token usage)
    - Agent invocations with input/output messages
    - Tool executions
    - Chain executions
    - Message content (when content recording is enabled)
    - Errors and exceptions
    - Custom metadata and tags
    - Proper parent-child span relationships for multi-agent scenarios

    Example:
        Basic usage with connection string:

        .. code-block:: python

            from azure_gen_ai.callbacks.tracers \
                import AzureGenAITracingCallback
            from langchain_openai import AzureChatOpenAI

            # Initialize the tracer
            tracer = AzureGenAITracingCallback(
                connection_string="InstrumentationKey=...",
                enable_content_recording=True
            )

            # Use with Azure OpenAI
            llm = AzureChatOpenAI(
                deployment_name="gpt-4",
                callbacks=[tracer]
            )

            response = llm.invoke("Hello, how are you?")

        For agents and chains:

        .. code-block:: python

            agent = create_react_agent(llm, tools)
            agent_executor = AgentExecutor(
                agent=agent, tools=tools, callbacks=[tracer]
            )
            result = agent_executor.invoke({"input": "What's the weather?"})

    Attributes:
        tracer: OpenTelemetry tracer instance
    active_spans: Dict tracking active spans and context tokens by run ID
        enable_content_recording: Whether to record message content
        instrument_inference: Whether inference instrumentation is enabled
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        enable_content_recording: Optional[bool] = None,
        instrument_inference: Optional[bool] = True,
    ) -> None:
        """Initialize the Azure GenAI tracing callback handler.

        Args:
            connection_string: Azure Application Insights connection string.
                If not provided, uses APPLICATIONINSIGHTS_CONNECTION_STRING
                environment variable.
            enable_content_recording: Whether to record message content and
                prompts in traces. If None, defaults to False unless
                AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED is 'true'.
                Recording can capture sensitive info.
            instrument_inference: Whether to enable inference instrumentation.
                When True, enables additional telemetry. Defaults to True.

        Raises:
            ImportError: If required OpenTelemetry packages are not installed.

        Note:
            Content recording should be used cautiously in production since it
            may capture sensitive user data or proprietary info.
        """
        super().__init__()

        # Configure Azure Monitor
        if connection_string:
            configure_azure_monitor(connection_string=connection_string)
        else:
            # Will use APPLICATIONINSIGHTS_CONNECTION_STRING if available
            configure_azure_monitor()

        self.tracer = otel_trace.get_tracer(__name__)
        self.active_spans: Dict[str, tuple[Any, Any]] = {}  # (span, token)
        self.instrument_inference = instrument_inference

        # Determine content recording setting
        if enable_content_recording is not None:
            self.enable_content_recording = enable_content_recording
        else:
            # Check environment variable
            env_value = os.getenv(
                "AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "false"
            )
            self.enable_content_recording = env_value.lower() == "true"

        logger.info(
            f"AzureGenAITracingCallback initialized - "
            f"content_recording: {self.enable_content_recording}, "
            f"instrument_inference: {self.instrument_inference}"
        )

    def _should_record_content(self) -> bool:
        """Check if content should be recorded based on settings.

        Returns:
            bool: True if content recording is enabled and inference
                instrumentation is on.
        """
        return self.enable_content_recording and self.instrument_inference

    def _extract_model_info(
        self, serialized: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract model information from serialized data.

        Args:
            serialized: Serialized model configuration

        Returns:
            Dictionary containing extracted model information
        """
        kwargs = serialized.get("kwargs", {})
        return {
            "deployment_name": kwargs.get("deployment_name", ""),
            "model_name": kwargs.get("model_name", ""),
            "temperature": kwargs.get("temperature", 0.0),
            "azure_endpoint": kwargs.get("azure_endpoint", ""),
            "api_version": kwargs.get("openai_api_version", ""),
        }

    def _format_message_to_invocation(
        self, msg: BaseMessage
    ) -> Dict[str, Any]:
        """Format a single LangChain message to invocation message format.

        Args:
            msg: LangChain BaseMessage instance

        Returns:
            Formatted message dictionary
        """
        role = msg.__class__.__name__.replace("Message", "").lower()
        content = (
            msg.content if self._should_record_content() else "[REDACTED]"
        )
        message_dict = {
            "role": role,
            "body": [
                {
                    "type": "text",
                    "content": content,
                }
            ],
        }

        # Add finish_reason if available
        if (
            hasattr(msg, "additional_kwargs")
            and "finish_reason" in msg.additional_kwargs
        ):
            finish_reason = msg.additional_kwargs["finish_reason"]
            message_dict["finish_reason"] = finish_reason

        return message_dict

    def _format_messages_to_invocation(
        self, messages: List[Any]
    ) -> List[Dict[str, Any]]:
        """Format list of messages to invocation format.

        Args:
            messages: List of messages (can be nested)

        Returns:
            List of formatted invocation message dictionaries
        """
        formatted = []
        for msg in messages:
            if isinstance(msg, list):
                formatted.extend(self._format_messages_to_invocation(msg))
            elif isinstance(msg, BaseMessage):
                formatted.append(self._format_message_to_invocation(msg))
        return formatted

    def _format_invocation_input(
        self, inputs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Format chain/agent inputs to invocation input format.

        Args:
            inputs: Input dictionary

        Returns:
            List of formatted messages
        """
        if "messages" in inputs and isinstance(inputs["messages"], list):
            return self._format_messages_to_invocation(inputs["messages"])
        elif "input" in inputs:
            content = inputs["input"]
            if isinstance(content, str):
                return [
                    {
                        "role": "user",
                        "body": [
                            {
                                "type": "text",
                                "content": (
                                    content
                                    if self._should_record_content()
                                    else "[REDACTED]"
                                ),
                            }
                        ],
                    }
                ]
            elif isinstance(content, list):
                return self._format_messages_to_invocation(content)
        return []

    def _format_invocation_output(
        self, outputs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Format chain/agent outputs to invocation output format.

        Args:
            outputs: Output dictionary

        Returns:
            List of formatted messages
        """
        if "output" in outputs:
            output = outputs["output"]
            if isinstance(output, str):
                return [
                    {
                        "role": "assistant",
                        "body": [
                            {
                                "type": "text",
                                "content": (
                                    output
                                    if self._should_record_content()
                                    else "[REDACTED]"
                                ),
                            }
                        ],
                        "finish_reason": "stop",
                    }
                ]
            elif isinstance(output, BaseMessage):
                return [self._format_message_to_invocation(output)]
            elif isinstance(output, list):
                return self._format_messages_to_invocation(output)
        elif "messages" in outputs and isinstance(outputs["messages"], list):
            return self._format_messages_to_invocation(outputs["messages"])
        return []

    def _safe_json_dumps(self, obj: Any) -> str:
        """Safely convert object to JSON string.

        Args:
            obj: Object to serialize

        Returns:
            JSON string representation or string fallback
        """
        try:
            return json.dumps(obj, default=str)
        except Exception as e:
            logger.warning(f"Failed to serialize object: {e}")
            try:
                return str(obj)
            except Exception:
                return "<unserializable>"

    def _handle_tool_calls(self, message: AIMessage) -> None:
        """Create execute_tool spans for tool calls in the message."""
        current_span = otel_trace.get_current_span()
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return

        for tool_call in message.tool_calls:
            try:
                tool_name = tool_call.get("name", "unknown_tool")
                tool_id = tool_call.get("id", "")
                tool_args = tool_call.get("args", {})

                # Create execute_tool span as child of current
                with self.tracer.start_as_current_span(
                    name=f"execute_tool {tool_name}",
                    kind=SpanKind.INTERNAL,
                    attributes={
                        conventions.GEN_AI_OPERATION_NAME: "execute_tool",
                        conventions.GEN_AI_TOOL_NAME: tool_name,
                        conventions.GEN_AI_TOOL_CALL_ID: tool_id,
                    }
                ) as tool_span:
                    # Add tool arguments if content recording is enabled
                    if self._should_record_content():
                        tool_span.set_attribute(
                            conventions.GEN_AI_TOOL_CALL_ARGUMENTS,
                            self._safe_json_dumps(tool_args)
                        )

                    # Add event for tool call
                    tool_span.add_event(
                        name="gen_ai.tool.call",
                        attributes={
                            "tool_name": tool_name,
                            "tool_id": tool_id,
                        }
                    )

            except Exception as e:
                logger.error(f"Error creating tool call span: {e}")
                if current_span:
                    current_span.record_exception(e)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle the start of a chat model invocation."""
        if not self.instrument_inference:
            return

        try:
            model_info = self._extract_model_info(serialized)

            # Create span name per specs
            operation_name = "chat"
            span_name = f"{operation_name} {model_info['deployment_name']}"

            # Prepare attributes
            attributes = {
                # TODO: review - spans.yaml span.azure.ai.inference.client
                conventions.GEN_AI_PROVIDER_NAME: "azure.ai.inference",
                conventions.GEN_AI_REQUEST_MODEL: (
                    model_info["deployment_name"]
                ),
                conventions.GEN_AI_REQUEST_TEMPERATURE: (
                    model_info["temperature"]
                ),
                conventions.GEN_AI_OPERATION_NAME: operation_name,
                # Azure specifics in metadata.*
                # TODO: review - api_version not in registry.yaml
                "metadata.api_version": model_info["api_version"],
                # TODO: review - endpoint not in registry.yaml; server.address
                # is set separately
                "metadata.endpoint": model_info["azure_endpoint"],
                conventions.SERVER_ADDRESS: (
                    urlparse(model_info["azure_endpoint"]).netloc
                    if model_info["azure_endpoint"]
                    else ""
                ),
                # TODO: review - run_id not in registry.yaml
                "metadata.run_id": str(run_id),
            }

            if tags:
                attributes[conventions.TAGS] = self._safe_json_dumps(tags)

            # Start span
            span = self.tracer.start_span(
                name=span_name,
                attributes=attributes,
                kind=SpanKind.CLIENT,
            )
            token = attach(otel_trace.set_span_in_context(span))

            # Store messages as events if content recording enabled
            if self._should_record_content():
                formatted_messages = self._format_messages_to_invocation(
                    messages[0] if messages else []
                )  # Usually single list
                for i, msg in enumerate(formatted_messages):
                    span.add_event(
                        # TODO: review - event name not standardized
                        name="gen_ai.content.prompt",
                        attributes={
                            conventions.GEN_AI_INPUT_MESSAGES: (
                                self._safe_json_dumps(msg)
                            ),
                            # TODO: review - helper attribute; not in registry
                            "metadata.message_index": i,
                        },
                    )

            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"metadata.{key}", value)
                    else:
                        span.set_attribute(
                            f"metadata.{key}",
                            self._safe_json_dumps(value),
                        )

            self.active_spans[str(run_id)] = (span, token)

        except Exception as e:
            logger.error(f"Error in on_chat_model_start: {e}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle the completion of an LLM invocation."""
        if not self.instrument_inference:
            return

        span_key = str(run_id)
        if span_key not in self.active_spans:
            logger.warning(f"No active span found for run_id: {run_id}")
            return

        span, token = self.active_spans[span_key]

        try:
            # Token usage
            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage", {})
            if token_usage:
                span.set_attribute(
                    conventions.GEN_AI_USAGE_INPUT_TOKENS,
                    token_usage.get("prompt_tokens", 0),
                )
                span.set_attribute(
                    conventions.GEN_AI_USAGE_OUTPUT_TOKENS,
                    token_usage.get("completion_tokens", 0),
                )
                span.set_attribute(
                    "metadata.gen_ai.usage.total_tokens",
                    token_usage.get("total_tokens", 0),
                )  # TODO: review - total_tokens not in registry.yaml

            # Process generations
            if self._should_record_content():
                for generation_list in response.generations:
                    for generation in generation_list:
                        if isinstance(generation, ChatGeneration):
                            message = generation.message
                            completion_attrs = {
                                conventions.GEN_AI_OUTPUT_MESSAGES: (
                                    self._safe_json_dumps(
                                        self._format_message_to_invocation(
                                            message
                                        )
                                    )
                                )
                            }
                            if (
                                hasattr(message, "tool_calls")
                                and message.tool_calls
                            ):
                                # TODO: review - no standard key; use
                                # metadata.*
                                completion_attrs["metadata.tool_calls"] = (
                                    self._safe_json_dumps(message.tool_calls)
                                )
                                self._handle_tool_calls(message)

                            span.add_event(
                                # TODO: review - event name not standardized
                                name="gen_ai.content.completion",
                                attributes=completion_attrs,
                            )

                            if hasattr(message, "response_metadata"):
                                resp_meta = message.response_metadata
                                span.set_attribute(
                                    conventions.GEN_AI_RESPONSE_MODEL,
                                    resp_meta.get("model_name", ""),
                                )
                                span.set_attribute(
                                    conventions.GEN_AI_RESPONSE_ID,
                                    resp_meta.get("id", ""),
                                )
                                finish_reason = resp_meta.get(
                                    "finish_reason", ""
                                )
                                if finish_reason:
                                    span.set_attribute(
                                        conventions.
                                        GEN_AI_RESPONSE_FINISH_REASONS,
                                        [finish_reason],
                                    )

            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            logger.error(f"Error processing LLM response: {e}")
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
        finally:
            span.end()
            detach(token)
            del self.active_spans[span_key]

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle errors during LLM invocation."""
        if not self.instrument_inference:
            return

        span_key = str(run_id)
        if span_key not in self.active_spans:
            logger.warning(f"No active span found for run_id: {run_id}")
            return

        span, token = self.active_spans[span_key]

        try:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.set_attribute(conventions.ERROR_TYPE, type(error).__name__)
        finally:
            span.end()
            detach(token)
            del self.active_spans[span_key]

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle the start of a chain or agent execution."""
        if not self.instrument_inference:
            return

        try:
            # Prefer metadata['agent_type'] if available; else kwargs['name']
            # or serialized['name']
            chain_name = (
                metadata.get("agent_type", kwargs.get("name", "Unknown"))
                if metadata
                else kwargs.get("name", "Unknown")
            )
            if serialized:
                chain_name = (
                    metadata.get(
                        "agent_type", serialized.get("name", chain_name)
                    )
                    if metadata
                    else serialized.get("name", chain_name)
                )

            chain_id = serialized.get("id", []) if serialized else []

            # Detect if this is an agent
            is_agent = (
                any("agent" in c.lower() for c in chain_id)
                or "agent" in chain_name.lower()
                or "AgentExecutor" in chain_name
            )

            if is_agent:
                operation_name = "invoke_agent"
                span_name = (
                    f"{operation_name} {chain_name}"
                    if chain_name != "Unknown"
                    else operation_name
                )
            else:
                # TODO: review - no gen_ai.operation.name for non-agent chains
                # (spans.yaml guidance)
                operation_name = None
                span_name = (
                    f"invoke_agent {chain_name}"
                    if chain_name != "Unknown"
                    else "invoke_agent"
                )

            # Prepare attributes
            attributes: Dict[str, Any] = {}
            if is_agent:
                attributes.update({
                    # TODO: review - spans.yaml span.azure.ai.inference.client
                    conventions.GEN_AI_PROVIDER_NAME: "azure.ai.inference",
                    conventions.GEN_AI_OPERATION_NAME: "invoke_agent",
                    # TODO: review - run_id not in registry; store under
                    # metadata
                    "metadata.run_id": str(run_id),
                })

            if is_agent:
                attributes[conventions.GEN_AI_AGENT_NAME] = chain_name
                # Add description if available
                if serialized and "description" in serialized:
                    attributes[conventions.GEN_AI_AGENT_DESCRIPTION] = (
                        serialized["description"]
                    )
                # Invocation input
                invocation_input = self._format_invocation_input(inputs)
                if invocation_input and self._should_record_content():
                    attributes[conventions.GEN_AI_AGENT_INVOCATION_INPUT] = (
                        self._safe_json_dumps(invocation_input)
                    )

            if tags:
                # TODO: review - tags under metadata.* (registry.yaml)
                attributes[conventions.TAGS] = self._safe_json_dumps(tags)

            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        attributes[f"metadata.{key}"] = value
                    else:
                        attributes[f"metadata.{key}"] = (
                            self._safe_json_dumps(value)
                        )

            # Start span
            span = self.tracer.start_span(
                name=span_name,
                attributes=attributes,
                kind=SpanKind.CLIENT if is_agent else SpanKind.INTERNAL,
            )
            token = attach(otel_trace.set_span_in_context(span))

            self.active_spans[str(run_id)] = (span, token)

        except Exception as e:
            logger.error(f"Error in on_chain_start: {e}")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle the completion of a chain or agent execution."""
        if not self.instrument_inference:
            return

        span_key = str(run_id)
        if span_key not in self.active_spans:
            return

        span, token = self.active_spans[span_key]

        try:
            # Check if agent
            if (
                span.attributes.get(conventions.GEN_AI_OPERATION_NAME)
                == "invoke_agent"
            ):
                invocation_output = self._format_invocation_output(outputs)
                if invocation_output and self._should_record_content():
                    span.set_attribute(
                        conventions.GEN_AI_AGENT_INVOCATION_OUTPUT,
                        self._safe_json_dumps(invocation_output)
                    )

            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            logger.error(f"Error in on_chain_end: {e}")
            span.set_status(Status(StatusCode.ERROR, str(e)))
        finally:
            span.end()
            detach(token)
            del self.active_spans[span_key]

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle errors during chain or agent execution."""
        if not self.instrument_inference:
            return

        span_key = str(run_id)
        if span_key not in self.active_spans:
            return

        span, token = self.active_spans[span_key]

        try:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.set_attribute(conventions.ERROR_TYPE, type(error).__name__)
        finally:
            span.end()
            detach(token)
            del self.active_spans[span_key]

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle the start of a tool execution."""
        if not self.instrument_inference:
            return

        try:
            tool_name = kwargs.get(
                "name", serialized.get("name", "unknown_tool")
            )
            if tool_name is None:
                tool_name = "unknown_tool"
            elif not isinstance(tool_name, (str, bytes)):
                tool_name = str(tool_name)
            tool_description = serialized.get("description", "")
            tool_args = input_str
            try:
                tool_args = json.loads(input_str)
            except json.JSONDecodeError:
                pass

            # Attributes
            attributes = {
                conventions.GEN_AI_OPERATION_NAME: "execute_tool",
                conventions.GEN_AI_TOOL_NAME: tool_name,
                # TODO: review - run_id not defined; store under metadata.*
                "metadata.run_id": str(run_id),
            }  # TODO: review - provider not required on execute_tool

            if tool_description:
                attributes[conventions.GEN_AI_TOOL_DESCRIPTION] = (
                    tool_description
                )

            if tags:
                attributes[conventions.TAGS] = self._safe_json_dumps(tags)

            if self._should_record_content():
                attributes[conventions.GEN_AI_TOOL_CALL_ARGUMENTS] = (
                    self._safe_json_dumps(tool_args)
                )

            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        attributes[f"metadata.{key}"] = value
                    else:
                        attributes[f"metadata.{key}"] = (
                            self._safe_json_dumps(value)
                        )

            # Start span
            span = self.tracer.start_span(
                name=f"execute_tool {tool_name}",
                attributes=attributes,
                kind=SpanKind.INTERNAL,
            )
            token = attach(otel_trace.set_span_in_context(span))

            self.active_spans[str(run_id)] = (span, token)

        except Exception as e:
            logger.error(f"Error in on_tool_start: {e}")

    def on_tool_end(
        self,
        output: Union[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle the completion of a tool execution."""
        if not self.instrument_inference:
            return

        span_key = str(run_id)
        if span_key not in self.active_spans:
            return

        span, token = self.active_spans[span_key]

        try:
            tool_result = output
            # Normalize ToolMessage or other message objects
            if isinstance(tool_result, BaseMessage):
                msg = tool_result
                role = (
                    msg.__class__.__name__.replace("Message", "").lower()
                )
                tool_result = {
                    "role": role,
                    "content": getattr(msg, "content", None),
                }
                # Include tool metadata when available (safe primitives only)
                tool_call_id = getattr(msg, "tool_call_id", None)
                if tool_call_id:
                    tool_result["tool_call_id"] = tool_call_id
                name = getattr(msg, "name", None)
                if name:
                    tool_result["name"] = name

            # Attempt to parse JSON strings, else keep as string
            if isinstance(tool_result, str):
                try:
                    tool_result = json.loads(tool_result)
                except Exception:
                    pass

            if self._should_record_content():
                span.set_attribute(
                    conventions.GEN_AI_TOOL_CALL_RESULT,
                    self._safe_json_dumps(tool_result)
                )

            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            logger.error(f"Error in on_tool_end: {e}")
            span.set_status(Status(StatusCode.ERROR, str(e)))
        finally:
            span.end()
            detach(token)
            del self.active_spans[span_key]

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle errors during tool execution."""
        if not self.instrument_inference:
            return

        span_key = str(run_id)
        if span_key not in self.active_spans:
            return

        span, token = self.active_spans[span_key]

        try:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.set_attribute(conventions.ERROR_TYPE, type(error).__name__)
        finally:
            span.end()
            detach(token)
            del self.active_spans[span_key]

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent action (e.g., deciding to call tools)."""
        if not self.instrument_inference:
            return

        span_key = str(run_id)
        if span_key not in self.active_spans:
            return

        span, _ = self.active_spans[span_key]

        try:
            tool_calls = (
                action.tool_calls if hasattr(action, "tool_calls") else []
            )
            span.add_event(
                # TODO: review - custom event name; no spec entry
                name="agent.action",
                attributes={
                    # TODO: review - custom metadata wrapper (registry.yaml)
                    "metadata.tool_call_count": len(tool_calls),
                    # TODO: review - custom metadata wrapper
                    "metadata.log": (
                        action.log if hasattr(action, "log") else ""
                    ),
                }
            )
        except Exception as e:
            logger.error(f"Error in on_agent_action: {e}")

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent finish. Additional logging hook if needed."""
        if not self.instrument_inference:
            return

        span_key = str(run_id)
        if span_key not in self.active_spans:
            return

        span, _ = self.active_spans[span_key]

        try:
            span.add_event(
                # TODO: review - custom event name; no spec entry
                name="agent.finish",
                attributes={
                    # TODO: review - custom metadata wrapper
                    "metadata.return_values": (
                        self._safe_json_dumps(finish.return_values)
                        if hasattr(finish, "return_values")
                        else "{}"
                    ),
                }
            )
        except Exception as e:
            logger.error(f"Error in on_agent_finish: {e}")
