"""Azure OpenAI Agent Tracing Callback Handler.

This module provides OpenTelemetry-based tracing for LangChain applications using
Azure OpenAI, following the OpenTelemetry Semantic Conventions for GenAI.

The tracer captures detailed information about LLM calls, including:
- Model parameters and configurations
- Input messages and prompts
- Output completions and token usage
- Tool calls and chain executions
- Error handling and exceptions

All traces are sent to Azure Application Insights for monitoring and analysis.
"""
# flake8: noqa: E501
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

try:
    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, SpanKind
except ImportError:
    raise ImportError(
        "Using tracing capabilities requires Azure Monitor and OpenTelemetry packages. "
        "Install them with: pip install azure-monitor-opentelemetry"
    )

# Import semantic conventions
from langchain_azure_ai.callbacks.tracers import _semantic_conventions_gen_ai as conventions

# Configure logging - suppress Azure SDK HTTP logging
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.monitor.opentelemetry.exporter.export._base").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureOpenAITracingCallback(BaseCallbackHandler):
    """Callback handler for tracing LangChain Azure OpenAI calls to Azure Application Insights.

    This tracer implements the OpenTelemetry Semantic Conventions for Generative AI systems,
    providing standardized telemetry data for monitoring and debugging LLM applications.

    The tracer captures:
    - LLM request/response details (model, parameters, token usage)
    - Message content (when content recording is enabled)
    - Chain and tool executions
    - Errors and exceptions
    - Custom metadata and tags

    Example:
        Basic usage with connection string:

        .. code-block:: python

            from langchain_azure_ai.callbacks.tracers import AzureOpenAITracingCallback
            from langchain_openai import AzureChatOpenAI

            # Initialize the tracer
            tracer = AzureOpenAITracingCallback(
                connection_string="InstrumentationKey=...",
                enable_content_recording=True
            )

            # Use with Azure OpenAI
            llm = AzureChatOpenAI(
                deployment_name="gpt-4",
                callbacks=[tracer]
            )

            response = llm.invoke("Hello, how are you?")

        Using environment variable for connection:

        .. code-block:: python

            # Set APPLICATIONINSIGHTS_CONNECTION_STRING environment variable
            tracer = AzureOpenAITracingCallback()

            # Use with chains
            chain = prompt | llm | parser
            result = chain.invoke({"input": "data"}, config={"callbacks": [tracer]})

    Attributes:
        tracer: OpenTelemetry tracer instance
        active_spans: Dictionary tracking active spans by run ID
        enable_content_recording: Whether to record message content
        instrument_inference: Whether inference instrumentation is enabled
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        enable_content_recording: Optional[bool] = None,
        instrument_inference: Optional[bool] = True,
    ) -> None:
        """Initialize the Azure OpenAI tracing callback handler.

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

        self.tracer = trace.get_tracer(__name__)
        self.active_spans: Dict[str, Any] = {}
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
            f"AzureOpenAITracingCallback initialized - "
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

    def _format_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Format messages for tracing.

        Args:
            messages: List of messages (can be nested)

        Returns:
            List of formatted message dictionaries
        """
        formatted = []
        for msg in messages:
            if isinstance(msg, list):
                # Handle nested message lists
                formatted.extend(self._format_messages(msg))
            elif isinstance(msg, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
                msg_dict = {
                    "role": msg.__class__.__name__.replace("Message", "").lower(),
                    "content": (
                        msg.content if self._should_record_content() else "[REDACTED]"
                    ),
                }
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    msg_dict["tool_calls"] = msg.tool_calls
                if isinstance(msg, ToolMessage):
                    msg_dict["tool_call_id"] = msg.tool_call_id
                    msg_dict["name"] = msg.name
                formatted.append(msg_dict)
        return formatted

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

    def _handle_tool_calls(self, message: AIMessage, parent_span: Any) -> None:
        """Create execute_tool spans for tool calls in the message.

        Args:
            message: AIMessage containing tool calls
            parent_span: Parent span for the tool execution
        """
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return

        for tool_call in message.tool_calls:
            try:
                tool_name = tool_call.get("name", "unknown_tool")
                tool_id = tool_call.get("id", "")
                tool_args = tool_call.get("args", {})

                # Create execute_tool span
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
                            "gen_ai.tool.call.arguments",
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
                # TODO: Replace print with logger once error handling is finalized
                print(f"Error creating tool call span: {e}")

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
        """Handle the start of a chat model invocation.

        Creates a new span for tracking the chat completion request.

        Args:
            serialized: Serialized model configuration
            messages: Input messages for the model
            run_id: Unique identifier for this run
            parent_run_id: Parent run ID if this is a nested call
            tags: Optional tags for categorization
            metadata: Optional metadata dictionary
            **kwargs: Additional keyword arguments
        """
        if not self.instrument_inference:
            return

        try:
            model_info = self._extract_model_info(serialized)

            # Create span name
            span_name = f"chat.completions {model_info['deployment_name']}"

            # Prepare attributes
            attributes = {
                # OpenTelemetry Semantic Conventions for GenAI
                conventions.GEN_AI_SYSTEM: "azure_openai",
                conventions.GEN_AI_REQUEST_MODEL: model_info["deployment_name"],
                conventions.GEN_AI_REQUEST_TEMPERATURE: model_info["temperature"],
                
                # Operation name
                conventions.GEN_AI_OPERATION_NAME: "chat.completions",
                
                # Azure specific attributes
                "gen_ai.request.api_version": model_info["api_version"],
                "gen_ai.request.endpoint": model_info["azure_endpoint"],
                conventions.SERVER_ADDRESS: (
                    urlparse(model_info["azure_endpoint"]).netloc
                    if model_info["azure_endpoint"]
                    else ""
                ),
                
                # Run information
                "run_id": str(run_id),
            }

            # Add parent_run_id only if it's not None
            if parent_run_id is not None:
                attributes["parent_run_id"] = str(parent_run_id)

            # Add tags only if they exist
            if tags:
                attributes[conventions.TAGS] = self._safe_json_dumps(tags)

            # Start span
            span = self.tracer.start_span(
                name=span_name,
                attributes=attributes,
            )

            # Store messages as events if content recording is enabled
            if self._should_record_content():
                formatted_messages = self._format_messages(messages)
                for i, msg in enumerate(formatted_messages):
                    span.add_event(
                        name="gen_ai.content.prompt",
                        attributes={
                            "gen_ai.prompt": self._safe_json_dumps(msg),
                            "message_index": i,
                        },
                    )

            # Add metadata as span attributes
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"metadata.{key}", value)
                    else:
                        span.set_attribute(
                            f"metadata.{key}", self._safe_json_dumps(value)
                        )

            # Store span for later use
            self.active_spans[str(run_id)] = span

        except Exception as e:
            # TODO: Replace print with logger once error handling is finalized
            print(f"Error in on_chat_model_start: {e}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle the completion of an LLM invocation.

        Updates the span with response data and token usage information.

        Args:
            response: The LLM response object
            run_id: Unique identifier for this run
            parent_run_id: Parent run ID if this is a nested call
            tags: Optional tags for categorization
            **kwargs: Additional keyword arguments
        """
        if not self.instrument_inference:
            return

        span_key = str(run_id)

        if span_key not in self.active_spans:
            logger.warning(f"No active span found for run_id: {run_id}")
            return

        span = self.active_spans[span_key]

        try:
            # Extract token usage
            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage", {})

            # Set token usage attributes
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
                    token_usage.get("total_tokens", 0)
                )

            # Process generations
            if self._should_record_content():
                for generation_list in response.generations:
                    for generation in generation_list:
                        if isinstance(generation, ChatGeneration):
                            message = generation.message

                            # Add completion event
                            completion_attrs = {
                                conventions.GEN_AI_EVENT_CONTENT: self._safe_json_dumps(
                                    {
                                        "role": message.__class__.__name__.replace(
                                            "Message", ""
                                        ).lower(),
                                        "content": message.content,
                                    }
                                )
                            }

                            # Add tool calls if present
                            if hasattr(message, "tool_calls") and message.tool_calls:
                                completion_attrs["tool_calls"] = self._safe_json_dumps(
                                    message.tool_calls
                                )
                                # Create execute_tool spans for each tool call
                                self._handle_tool_calls(message, span)

                            span.add_event(
                                name="gen_ai.content.completion",
                                attributes=completion_attrs,
                            )

                            # Set response attributes
                            if hasattr(message, "response_metadata"):
                                resp_meta = message.response_metadata
                                span.set_attribute(
                                    conventions.GEN_AI_RESPONSE_MODEL,
                                    resp_meta.get("model_name", ""),
                                )
                                span.set_attribute(
                                    "gen_ai.response.id", resp_meta.get("id", "")
                                )
                                span.set_attribute(
                                    "gen_ai.response.finish_reason",
                                    resp_meta.get("finish_reason", ""),
                                )

            # Set status to OK
            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            # TODO: Replace print with logger once error handling is finalized
            print(f"Error processing LLM response: {e}")
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
        finally:
            # End span and remove from active spans
            span.end()
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
        """Handle errors that occur during LLM invocation.

        Records the error in the span and sets appropriate status.

        Args:
            error: The exception that occurred
            run_id: Unique identifier for this run
            parent_run_id: Parent run ID if this is a nested call
            tags: Optional tags for categorization
            **kwargs: Additional keyword arguments
        """
        if not self.instrument_inference:
            return

        span_key = str(run_id)

        if span_key not in self.active_spans:
            logger.warning(f"No active span found for run_id: {run_id}")
            return

        span = self.active_spans[span_key]

        try:
            # Record exception
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))

            # Add error details
            span.set_attribute(conventions.ERROR_TYPE, type(error).__name__)
            span.set_attribute("error.message", str(error))

        finally:
            # End span and remove from active spans
            span.end()
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
        """Handle the start of a chain execution.

        Creates a span to track the entire chain execution.

        Args:
            serialized: Serialized chain configuration
            inputs: Input data for the chain
            run_id: Unique identifier for this run
            parent_run_id: Parent run ID if this is a nested call
            tags: Optional tags for categorization
            metadata: Optional metadata dictionary
            **kwargs: Additional keyword arguments
        """
        if not self.instrument_inference:
            return

        try:
            # Create span name
            chain_name = kwargs.get("name", "LangGraph")
            span_name = f"chain.{chain_name}"

            # Prepare attributes
            attributes = {
                "chain.name": chain_name,
                "run_id": str(run_id),
                conventions.INPUTS: list(inputs.keys()),
            }

            # Add parent_run_id only if it's not None
            if parent_run_id is not None:
                attributes["parent_run_id"] = str(parent_run_id)

            # Add tags only if they exist
            if tags:
                attributes[conventions.TAGS] = self._safe_json_dumps(tags)

            # Start span
            span = self.tracer.start_span(
                name=span_name,
                attributes=attributes,
            )

            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"metadata.{key}", value)
                    else:
                        span.set_attribute(
                            f"metadata.{key}", self._safe_json_dumps(value)
                        )
                
                # Add user request as input if available
                if metadata.get("user_request"):
                    span.set_attribute("user.request", metadata["user_request"])

            # Log inputs (be careful with sensitive data)
            if self._should_record_content() and "messages" in inputs:
                formatted_messages = self._format_messages(inputs["messages"])
                span.set_attribute("chain.input.message_count", len(formatted_messages))
                span.set_attribute(
                    "chain.input.messages", self._safe_json_dumps(formatted_messages)
                )

            # Store span
            self.active_spans[f"chain_{run_id}"] = span

        except Exception as e:
            # TODO: Replace print with logger once error handling is finalized
            print(f"Error in on_chain_start: {e}")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle the completion of a chain execution.

        Updates the span with output information.

        Args:
            outputs: Output data from the chain
            run_id: Unique identifier for this run
            parent_run_id: Parent run ID if this is a nested call
            tags: Optional tags for categorization
            **kwargs: Additional keyword arguments
        """
        if not self.instrument_inference:
            return

        span_key = f"chain_{run_id}"

        if span_key not in self.active_spans:
            return

        span = self.active_spans[span_key]

        try:
            span.set_attribute(conventions.OUTPUTS, list(outputs.keys()))

            if self._should_record_content():
                span.set_attribute("chain.outputs", self._safe_json_dumps(outputs))

            # Check if this is the final travel plan output
            if "messages" in outputs and outputs["messages"]:
                last_message = outputs["messages"][-1]
                if isinstance(last_message, AIMessage):
                    # Check if this contains a final travel plan
                    if "FINAL TRAVEL PLAN" in last_message.content or "comprehensive plan" in last_message.content.lower():
                        span.set_attribute("travel.plan.final", last_message.content)

            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            # TODO: Replace print with logger once error handling is finalized
            print(f"Error in on_chain_end: {e}")
            span.set_status(Status(StatusCode.ERROR, str(e)))
        finally:
            span.end()
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
        """Handle errors that occur during chain execution.

        Records the error in the span and sets appropriate status.

        Args:
            error: The exception that occurred
            run_id: Unique identifier for this run
            parent_run_id: Parent run ID if this is a nested call
            tags: Optional tags for categorization
            **kwargs: Additional keyword arguments
        """
        if not self.instrument_inference:
            return

        span_key = f"chain_{run_id}"

        if span_key not in self.active_spans:
            return

        span = self.active_spans[span_key]

        try:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.set_attribute(conventions.ERROR_TYPE, type(error).__name__)
            span.set_attribute("error.message", str(error))
        finally:
            span.end()
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
        """Handle the start of a tool execution.

        Creates a span to track tool usage following the execute_tool semantic convention.

        Args:
            serialized: Serialized tool configuration
            input_str: Input string for the tool
            run_id: Unique identifier for this run
            parent_run_id: Parent run ID if this is a nested call
            tags: Optional tags for categorization
            metadata: Optional metadata dictionary
            **kwargs: Additional keyword arguments
        """
        if not self.instrument_inference:
            return

        try:
            # Extract tool name and description
            tool_name = kwargs.get("name", serialized.get("name", "unknown_tool"))
            tool_description = serialized.get("description", "")
            
            # Parse input if it's JSON
            tool_args = input_str
            try:
                tool_args = json.loads(input_str)
            except:
                pass  # Keep as string if not JSON

            # Prepare attributes
            attributes = {
                conventions.GEN_AI_OPERATION_NAME: "execute_tool",
                conventions.GEN_AI_TOOL_NAME: tool_name,
                "run_id": str(run_id),
            }

            # Add parent_run_id only if it's not None
            if parent_run_id is not None:
                attributes["parent_run_id"] = str(parent_run_id)

            # Add tags only if they exist
            if tags:
                attributes[conventions.TAGS] = self._safe_json_dumps(tags)

            # Add tool description if available
            if tool_description:
                attributes["gen_ai.tool.description"] = tool_description

            # Create span following the execute_tool convention
            span = self.tracer.start_span(
                name=f"execute_tool {tool_name}",
                kind=SpanKind.INTERNAL,
                attributes=attributes,
            )

            # Record tool arguments if content recording is enabled
            if self._should_record_content():
                span.set_attribute(
                    "gen_ai.tool.call.arguments",
                    self._safe_json_dumps(tool_args) if isinstance(tool_args, dict) else tool_args
                )

            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"metadata.{key}", value)
                    else:
                        span.set_attribute(f"metadata.{key}", self._safe_json_dumps(value))

            self.active_spans[f"tool_{run_id}"] = span

        except Exception as e:
            # TODO: Replace print with logger once error handling is finalized
            print(f"Error in on_tool_start: {e}")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle the completion of a tool execution.

        Updates the span with output information following the execute_tool semantic convention.

        Args:
            output: Output from the tool
            run_id: Unique identifier for this run
            parent_run_id: Parent run ID if this is a nested call
            tags: Optional tags for categorization
            **kwargs: Additional keyword arguments
        """
        if not self.instrument_inference:
            return

        span_key = f"tool_{run_id}"

        if span_key not in self.active_spans:
            return

        span = self.active_spans[span_key]

        try:
            # Parse output if it's JSON
            tool_result = output
            try:
                tool_result = json.loads(output)
            except:
                pass  # Keep as string if not JSON

            # Record output if content recording is enabled
            if self._should_record_content():
                span.set_attribute(
                    "gen_ai.tool.call.result",
                    self._safe_json_dumps(tool_result) if isinstance(tool_result, (dict, list)) else tool_result
                )

            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            # TODO: Replace print with logger once error handling is finalized
            print(f"Error in on_tool_end: {e}")
            span.set_status(Status(StatusCode.ERROR, str(e)))
        finally:
            span.end()
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
        """Handle errors that occur during tool execution.

        Records the error in the span and sets appropriate status following the execute_tool semantic convention.

        Args:
            error: The exception that occurred
            run_id: Unique identifier for this run
            parent_run_id: Parent run ID if this is a nested call
            tags: Optional tags for categorization
            **kwargs: Additional keyword arguments
        """
        if not self.instrument_inference:
            return

        span_key = f"tool_{run_id}"

        if span_key not in self.active_spans:
            return

        span = self.active_spans[span_key]

        try:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.set_attribute(conventions.ERROR_TYPE, type(error).__name__)
            span.set_attribute("error.message", str(error))
        finally:
            span.end()
            del self.active_spans[span_key]