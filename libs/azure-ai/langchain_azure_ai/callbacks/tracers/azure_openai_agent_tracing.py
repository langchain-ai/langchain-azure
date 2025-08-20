"""Azure OpenAI Agent Tracing Callback Handler.

OpenTelemetry tracing for LangChain/LangGraph apps using Azure OpenAI, aligned to
Generative AI semantic conventions and exporting to Azure Application Insights
(via azure-monitor-opentelemetry).

What it captures
- Chat/LLM calls as CLIENT spans named "chat <deployment>":
  - Request details: deployment/model, temperature, server.address, API version (as metadata).
  - Prompt/completion events recorded when content recording is enabled, using spec keys:
    - gen_ai.input.messages (events for prompts)
    - gen_ai.output.messages (events for completions)
  - Response metadata (gen_ai.response.model, gen_ai.response.id, gen_ai.response.finish_reasons)
  - Token usage (gen_ai.usage.input_tokens, gen_ai.usage.output_tokens)
- Chains and agents:
  - Agents are detected heuristically; emits "invoke_agent <name>" as CLIENT spans.
  - Non-agent chains emit INTERNAL spans named "chain <name>" and are treated as app spans
    (not GenAI spans) to stay within spec (no gen_ai.operation.name="chain").
  - Agent invocation input/output messages recorded when content recording is enabled:
    - gen_ai.agent.invocation_input / gen_ai.agent.invocation_output
  - metadata.* captures extra context (tags, run_id, api_version, endpoint, etc).
- Tools:
  - "execute_tool <name>" INTERNAL spans with arguments/results when on_tool_start/on_tool_end fire,
    using gen_ai.tool.* attributes per spec.
  - Also infers tool-intent spans from AIMessage.tool_calls when explicit tool callbacks aren’t available.

Span hierarchy and context (attach/detach)
- Each on_* start creates a span, attaches it to OTel context (attach); each on_* end/error ends the span
  and detaches the context (detach). This LIFO attach/detach yields proper parent-child nesting.
- For a single, well-formed trace, create a root span in your app and make it current before agent/LLM calls.

Attribute schema and compliance
- Uses GenAI conventions (see registry.yaml, spans.yaml, gen_ai.md): gen_ai.provider.name, gen_ai.operation.name,
  gen_ai.request/response.*, gen_ai.usage.*, gen_ai.agent.*, gen_ai.tool.*, server.address.
- OpenTelemetry only accepts primitives or lists of primitives as attributes; complex values are JSON-serialized.
- Provider name for Azure AI Inference is set to "azure.ai.inference" at span creation time.

Content recording and privacy
- Controlled by:
  - enable_content_recording=True/False (constructor), or
  - AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED=true (env).
- When disabled, message bodies are redacted as "[REDACTED]".
- Be mindful of PII and sensitive content in production.

Azure Monitor configuration
- On init, calls configure_azure_monitor:
  - If a connection string is provided, it is used;
  - Otherwise APPLICATIONINSIGHTS_CONNECTION_STRING is used if set.

Lifecycle, errors, and cleanup
- Active spans are tracked by run_id and cleaned up on end/error (span.end + detach).
- Exceptions are recorded with StatusCode.ERROR and error.type attributes.
- Token metrics and response metadata are captured where available.

Workarounds for unavailable callbacks
- Missing tool callbacks:
  - Some agent executors don’t emit on_tool_start/on_tool_end.
  - The tracer inspects AIMessage.tool_calls and creates child "execute_tool" spans with name/id/args (when recording is enabled)
    so tool intent is visible even without explicit callbacks. These spans won’t include duration or final results unless the
    tool callbacks also fire.
- Context-first parenting:
  - Parent-child relationships are formed via OTel context attach/detach rather than reconstructing from parent_run_id.
  - Ensure a root span is active in your app to enforce a cohesive hierarchy.
- Attribute type constraints:
  - Any dicts/objects are JSON-stringified to avoid OpenTelemetry “invalid type” warnings.
- Token streaming:
  - High-frequency token events aren’t emitted to avoid noisy traces.

Quickstart
    from langchain_azure_ai.callbacks.tracers.azure_openai_agent_tracing import AzureOpenAITracingCallback
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
    agent_exec = AgentExecutor(agent=agent, tools=[], callbacks=[tracer_cb])

    ot = otel_trace.get_tracer(__name__)
    with ot.start_as_current_span("invoke_agent travel_planner", kind=SpanKind.SERVER):
        result = agent_exec.invoke({"input": "Plan a 3-day London trip"})
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

try:
    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.context import attach, detach
except ImportError:
    raise ImportError(
        "Using tracing capabilities requires Azure Monitor and OpenTelemetry packages. "
        "Install them with: pip install azure-monitor-opentelemetry"
    )

# Configure logging - suppress Azure SDK HTTP logging
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.monitor.opentelemetry.exporter.export._base").setLevel(logging.WARNING)
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
    GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"  # Derived
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
    SERVER_ADDRESS = "server.address"
    SERVER_PORT = "server.port"
    ERROR_TYPE = "error.type"
    TAGS = "tags"

conventions = GenAIConventions


class AzureOpenAITracingCallback(BaseCallbackHandler):
    """Callback handler for tracing LangChain and LangGraph GenAI calls to Azure Application Insights.

    This tracer implements the OpenTelemetry Semantic Conventions for Generative AI systems,
    providing standardized telemetry data for monitoring and debugging LLM and agent applications.

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

            from azure_gen_ai.callbacks.tracers import AzureGenAITracingCallback
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
            agent_executor = AgentExecutor(agent=agent, tools=tools, callbacks=[tracer])
            result = agent_executor.invoke({"input": "What's the weather?"})

    Attributes:
        tracer: OpenTelemetry tracer instance
        active_spans: Dictionary tracking active spans and context tokens by run ID
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
            enable_content_recording: Whether to record message content and prompts
                in traces. If None, defaults to False unless AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED
                environment variable is set to 'true'. Recording content can be useful
                for debugging but may capture sensitive information.
            instrument_inference: Whether to enable inference instrumentation.
                When True, enables additional telemetry collection. Defaults to True.

        Raises:
            ImportError: If required OpenTelemetry packages are not installed.

        Note:
            Content recording should be used cautiously in production environments
            as it may capture sensitive user data or proprietary information.
        """
        super().__init__()

        # Configure Azure Monitor
        if connection_string:
            configure_azure_monitor(connection_string=connection_string)
        else:
            # Will use APPLICATIONINSIGHTS_CONNECTION_STRING env var if available
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
            bool: True if content recording is enabled and inference instrumentation is on.
        """
        return self.enable_content_recording and self.instrument_inference

    def _extract_model_info(self, serialized: Dict[str, Any]) -> Dict[str, Any]:
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

    def _format_message_to_invocation(self, msg: BaseMessage) -> Dict[str, Any]:
        """Format a single LangChain message to invocation message format.

        Args:
            msg: LangChain BaseMessage instance

        Returns:
            Formatted message dictionary
        """
        role = msg.__class__.__name__.replace("Message", "").lower()
        content = msg.content if self._should_record_content() else "[REDACTED]"
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
        if hasattr(msg, "additional_kwargs") and "finish_reason" in msg.additional_kwargs:
            message_dict["finish_reason"] = msg.additional_kwargs["finish_reason"]

        return message_dict

    def _format_messages_to_invocation(self, messages: List[Any]) -> List[Dict[str, Any]]:
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

    def _format_invocation_input(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
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
                                "content": content if self._should_record_content() else "[REDACTED]",
                            }
                        ],
                    }
                ]
            elif isinstance(content, list):
                return self._format_messages_to_invocation(content)
        return []

    def _format_invocation_output(self, outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
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
                                "content": output if self._should_record_content() else "[REDACTED]",
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
            return str(obj)

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
                conventions.GEN_AI_PROVIDER_NAME: "azure.ai.openai",
                conventions.GEN_AI_REQUEST_MODEL: model_info["deployment_name"],
                conventions.GEN_AI_REQUEST_TEMPERATURE: model_info["temperature"],
                conventions.GEN_AI_OPERATION_NAME: operation_name,
                # Azure specific
                "gen_ai.request.api_version": model_info["api_version"],
                "gen_ai.request.endpoint": model_info["azure_endpoint"],
                conventions.SERVER_ADDRESS: urlparse(model_info["azure_endpoint"]).netloc if model_info["azure_endpoint"] else "",
                "run_id": str(run_id),
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
                formatted_messages = self._format_messages_to_invocation(messages[0] if messages else [])  # Usually single list
                for i, msg in enumerate(formatted_messages):
                    span.add_event(
                        name="gen_ai.content.prompt",
                        attributes={
                            "gen_ai.prompt": self._safe_json_dumps(msg),
                            "message_index": i,
                        },
                    )

            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"metadata.{key}", value)
                    else:
                        span.set_attribute(f"metadata.{key}", self._safe_json_dumps(value))

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
                    conventions.GEN_AI_USAGE_TOTAL_TOKENS,
                    token_usage.get("total_tokens", 0),
                )

            # Process generations
            if self._should_record_content():
                for generation_list in response.generations:
                    for generation in generation_list:
                        if isinstance(generation, ChatGeneration):
                            message = generation.message
                            completion_attrs = {
                                "gen_ai.completion": self._safe_json_dumps(
                                    self._format_message_to_invocation(message)
                                )
                            }
                            if hasattr(message, "tool_calls") and message.tool_calls:
                                completion_attrs["tool_calls"] = self._safe_json_dumps(message.tool_calls)
                                self._handle_tool_calls(message)

                            span.add_event(
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
                                finish_reason = resp_meta.get("finish_reason", "")
                                if finish_reason:
                                    span.set_attribute(
                                        conventions.GEN_AI_RESPONSE_FINISH_REASONS,
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
            # Get chain name - prefer metadata['agent_type'] if available, else kwargs['name'] or serialized['name']
            chain_name = metadata.get("agent_type", kwargs.get("name", "Unknown")) if metadata else kwargs.get("name", "Unknown")
            if serialized:
                chain_name = metadata.get("agent_type", serialized.get("name", chain_name)) if metadata else serialized.get("name", chain_name)

            chain_id = serialized.get("id", []) if serialized else []

            # Detect if this is an agent
            is_agent = any("agent" in component.lower() for component in chain_id) or "agent" in chain_name.lower() or "AgentExecutor" in chain_name

            if is_agent:
                operation_name = "invoke_agent"
                span_name = f"{operation_name} {chain_name}" if chain_name != "Unknown" else operation_name
            else:
                operation_name = "invoke_agent"
                span_name = f"{operation_name} {chain_name}" if chain_name != "Unknown" else operation_name

            # Prepare attributes
            attributes = {
                conventions.GEN_AI_PROVIDER_NAME: "azure.ai.openai",
                conventions.GEN_AI_OPERATION_NAME: operation_name,
                "run_id": str(run_id),
            }

            if is_agent:
                attributes[conventions.GEN_AI_AGENT_NAME] = chain_name
                # Add description if available
                if serialized and "description" in serialized:
                    attributes[conventions.GEN_AI_AGENT_DESCRIPTION] = serialized["description"]
                # Invocation input
                invocation_input = self._format_invocation_input(inputs)
                if invocation_input and self._should_record_content():
                    attributes[conventions.GEN_AI_AGENT_INVOCATION_INPUT] = self._safe_json_dumps(invocation_input)

            if tags:
                attributes[conventions.TAGS] = self._safe_json_dumps(tags)

            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        attributes[f"metadata.{key}"] = value
                    else:
                        attributes[f"metadata.{key}"] = self._safe_json_dumps(value)

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
            if span.attributes.get(conventions.GEN_AI_OPERATION_NAME) == "invoke_agent":
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
            tool_name = kwargs.get("name", serialized.get("name", "unknown_tool"))
            tool_description = serialized.get("description", "")
            tool_args = input_str
            try:
                tool_args = json.loads(input_str)
            except json.JSONDecodeError:
                pass

            # Attributes
            attributes = {
                conventions.GEN_AI_PROVIDER_NAME: "azure.ai.openai",
                conventions.GEN_AI_OPERATION_NAME: "execute_tool",
                conventions.GEN_AI_TOOL_NAME: tool_name,
                "run_id": str(run_id),
            }

            if tool_description:
                attributes[conventions.GEN_AI_TOOL_DESCRIPTION] = tool_description

            if tags:
                attributes[conventions.TAGS] = self._safe_json_dumps(tags)

            if self._should_record_content():
                attributes[conventions.GEN_AI_TOOL_CALL_ARGUMENTS] = self._safe_json_dumps(tool_args)

            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        attributes[f"metadata.{key}"] = value
                    else:
                        attributes[f"metadata.{key}"] = self._safe_json_dumps(value)

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
        output: str,
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
            try:
                tool_result = json.loads(output)
            except json.JSONDecodeError:
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
            tool_calls = action.tool_calls if hasattr(action, "tool_calls") else []
            span.add_event(
                name="agent.action",
                attributes={
                    "tool_call_count": len(tool_calls),
                    "log": action.log if hasattr(action, "log") else "",
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
        """Handle agent finish. Can be used for additional logging if needed."""
        if not self.instrument_inference:
            return

        span_key = str(run_id)
        if span_key not in self.active_spans:
            return

        span, _ = self.active_spans[span_key]

        try:
            span.add_event(
                name="agent.finish",
                attributes={
                    "return_values": self._safe_json_dumps(finish.return_values) if hasattr(finish, "return_values") else "{}",
                }
            )
        except Exception as e:
            logger.error(f"Error in on_agent_finish: {e}")