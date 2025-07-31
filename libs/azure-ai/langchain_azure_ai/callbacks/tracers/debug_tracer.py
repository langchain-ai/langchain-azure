"""Debug callback handler that implements all LangChain callback methods with OpenTelemetry tracing."""

import json
import logging
from typing import Any, Dict, List, Optional, Union, Sequence
from uuid import UUID
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult
from tenacity import RetryCallState

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
except ImportError:
    raise ImportError(
        "Using tracing capabilities requires the extra `opentelemetry`."
        "Install the package with `pip install opentelemetry-api opentelemetry-sdk opentelemetry-semantic-conventions`"
    )

logger = logging.getLogger(__name__)


class DebugCallbackHandler(BaseCallbackHandler):
    """Debug callback handler that logs all callback events and creates OpenTelemetry spans."""

    def __init__(
        self, 
        connection_string: Optional[str] = None,
        enable_content_recording: bool = True,
        debug: bool = False,
        **kwargs: Any
    ):
        """Initialize the debug callback handler.
        
        Args:
            connection_string: Azure Application Insights connection string
            enable_content_recording: Whether to record message content
            debug: Whether to enable debug logging
        """
        super().__init__()
        self.connection_string = connection_string
        self.enable_content_recording = enable_content_recording
        self.debug = debug
        self.run_inline = True
        self.tracer = trace.get_tracer(__name__)
        self.spans: Dict[UUID, trace.Span] = {}
        self.tool_call_spans: Dict[str, trace.Span] = {}
        self.llm_messages: Dict[UUID, List[Dict[str, Any]]] = {}
        self.root_span: Optional[trace.Span] = None
        self.root_run_id: Optional[UUID] = None
        self.agent_spans: Dict[UUID, trace.Span] = {}
        
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.debug(f"Initialized DebugCallbackHandler with connection_string: {connection_string[:50]}...")
    
    def _log_span_info(self, span: trace.Span, span_type: str):
        """Log span information for debugging and evaluation."""
        if span:
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x')
            span_id = format(span_context.span_id, '016x')
            
            print(f"\n{'='*60}")
            print(f"📍 {span_type} Span Created")
            print(f"{'='*60}")
            print(f"Span Name: {span.name}")
            print(f"Trace ID: {trace_id}")
            print(f"Span ID: {span_id}")
            print(f"\n💡 To evaluate this span, add to your .env:")
            print(f"LANGGRAPH_SPAN_ID={span_id}")
            if span_type == "Root Session":
                print(f"LANGGRAPH_ROOT_SPAN_ID={span_id}")
            print(f"{'='*60}\n")
        
    def _create_span(self, name: str, run_id: UUID, parent_run_id: Optional[UUID] = None, 
                     span_kind: SpanKind = SpanKind.INTERNAL) -> Optional[trace.Span]:
        """Create and start a new span."""
        if not self.tracer:
            return None
            
        # Get parent span if exists
        parent_span = None
        if parent_run_id and parent_run_id in self.spans:
            parent_span = self.spans[parent_run_id]
        elif self.root_span and not parent_run_id:
            # If no parent specified but we have a root span, use it
            parent_span = self.root_span
            
        # Create context with parent
        if parent_span:
            ctx = trace.set_span_in_context(parent_span)
            span = self.tracer.start_span(name, context=ctx, kind=span_kind)
        else:
            span = self.tracer.start_span(name, kind=span_kind)
            # If this is the first span, make it the root
            if not self.root_span:
                self.root_span = span
                self.root_run_id = run_id
            
        # Store span
        self.spans[run_id] = span
        
        if self.debug:
            logger.debug(f"Created span: {name} (run_id: {run_id}, parent_run_id: {parent_run_id})")
            
        return span
    
    def _end_span(self, run_id: UUID, status: Status = Status(StatusCode.OK)):
        """End a span."""
        span = self.spans.pop(run_id, None)
        if span:
            span.set_status(status)
            span.end()
            
            if self.debug:
                logger.debug(f"Ended span for run_id: {run_id}")
            
            # If this was the root span, clear it
            if run_id == self.root_run_id:
                self.root_span = None
                self.root_run_id = None
    
    def _add_event_to_span(self, span: trace.Span, event_name: str, content: Dict[str, Any]):
        """Add an event to a span following PR #2528 conventions."""
        if not span or not self.enable_content_recording:
            return
            
        # Format event body based on event type
        event_body = None
        attributes = {"event.name": event_name}
        
        if event_name in ["gen_ai.system.message", "gen_ai.user.message", "gen_ai.assistant.message"]:
            # Message events have role and content in body
            event_body = {
                "role": content.get("role", "unknown"),
                "content": content.get("content", "")
            }
            # Add tool_calls if present (for assistant messages)
            if content.get("tool_calls"):
                event_body["tool_calls"] = content["tool_calls"]
                
        elif event_name == "gen_ai.tool.message":
            # Tool message events
            event_body = {
                "role": "tool",
                "content": content.get("content", ""),
                "tool_call_id": content.get("tool_call_id", "")
            }
            
        elif event_name == "gen_ai.choice":
            # Choice events have different structure
            event_body = {
                "role": content.get("role", "assistant"),
                "content": content.get("content", ""),
                "tool_calls": content.get("tool_calls")
            }
            
        else:
            # Generic events
            event_body = content
            
        # Convert body to JSON string for OpenTelemetry
        span.add_event(event_name, attributes=attributes)
        
        if self.debug:
            logger.debug(f"Added event '{event_name}' to span: {event_body}")
    
    def _format_message_for_event(self, message: BaseMessage) -> Dict[str, Any]:
        """Format a message for event recording following PR #2528 conventions."""
        content = {"content": message.content}
        
        if isinstance(message, SystemMessage):
            content["role"] = "system"
        elif isinstance(message, HumanMessage):
            content["role"] = "user"
        elif isinstance(message, AIMessage):
            content["role"] = "assistant"
            if hasattr(message, 'tool_calls') and message.tool_calls:
                content["tool_calls"] = message.tool_calls
        elif isinstance(message, ToolMessage):
            content["role"] = "tool"
            if hasattr(message, 'tool_call_id'):
                content["tool_call_id"] = message.tool_call_id
                
        return content
    
    # Synchronous LLM callbacks
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""
        # Skip - we handle this in on_chat_model_start
        pass

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running."""
        # Get model info
        invocation_params = kwargs.get("invocation_params", {})
        model_name = invocation_params.get("azure_deployment", invocation_params.get("model", "unknown"))
        
        # Check if we should create a root span for the entire execution
        if not self.root_span and tags and "travel-planning-execution" in tags:
            # Create root span for the entire travel planning session
            session_id = metadata.get("session_id", str(UUID(int=0)))
            root_span = self.tracer.start_span("travel_planning_session", kind=SpanKind.SERVER)
            root_span.set_attribute("span_type", "Session")
            root_span.set_attribute("gen_ai.system", "langchain")
            root_span.set_attribute("gen_ai.operation.name", "travel_planning")
            root_span.set_attribute("session.id", session_id)
            if metadata:
                if metadata.get("user_request"):
                    root_span.set_attribute("user.request", metadata["user_request"])
                if metadata.get("timestamp"):
                    root_span.set_attribute("session.start_time", metadata["timestamp"])
            self.root_span = root_span
            self.root_run_id = UUID('00000000-0000-0000-0000-000000000000')  # Special UUID for root
            self.spans[self.root_run_id] = root_span
            
            # Log root span info
            self._log_span_info(root_span, "Root Session")
        
        # Create generation span following PR #2528 conventions
        span = self._create_span("chat.completions", run_id, parent_run_id, SpanKind.CLIENT)
        
        if span:
            # Set standard attributes per PR #2528
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.request.model", model_name)
            
            # Additional attributes for multi-agent systems
            if metadata and metadata.get("agent_name"):
                span.set_attribute("gen_ai.agent.name", metadata["agent_name"])
            if metadata and metadata.get("agent_id"):
                span.set_attribute("gen_ai.agent.id", metadata["agent_id"])
                
            # Log generation span info
            self._log_span_info(span, "Generation")
            
            # Store messages for later processing
            if messages and messages[0]:
                self.llm_messages[run_id] = messages[0]
                
                # Add message events per PR #2528
                for msg in messages[0]:
                    if isinstance(msg, SystemMessage):
                        self._add_event_to_span(span, "gen_ai.system.message", self._format_message_for_event(msg))
                    elif isinstance(msg, HumanMessage):
                        self._add_event_to_span(span, "gen_ai.user.message", self._format_message_for_event(msg))
                    elif isinstance(msg, AIMessage):
                        self._add_event_to_span(span, "gen_ai.assistant.message", self._format_message_for_event(msg))
                    elif isinstance(msg, ToolMessage):
                        self._add_event_to_span(span, "gen_ai.tool.message", self._format_message_for_event(msg))
                
                # Format messages for input attribute per PR #2528
                input_messages = []
                for msg in messages[0]:
                    parts = [{"type": "text", "content": msg.content}]
                    
                    if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            parts.append({
                                "type": "tool_call",
                                "id": tc.get('id'),
                                "name": tc.get('name'),
                                "arguments": json.dumps(tc.get('args', {})) if isinstance(tc.get('args'), dict) else tc.get('args', '{}')
                            })
                    
                    role = "system" if isinstance(msg, SystemMessage) else \
                           "user" if isinstance(msg, HumanMessage) else \
                           "assistant" if isinstance(msg, AIMessage) else \
                           "tool" if isinstance(msg, ToolMessage) else "unknown"
                    
                    input_messages.append({"role": role, "parts": parts})
                
                span.set_attribute("gen_ai.input.messages", json.dumps(input_messages))

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        # Skip individual tokens for now
        pass

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running."""
        span = self.spans.get(run_id)
        if span and response.llm_output:
            # Token usage per PR #2528
            token_usage = response.llm_output.get("token_usage", {})
            if token_usage:
                span.set_attribute("gen_ai.usage.input_tokens", token_usage.get("prompt_tokens", 0))
                span.set_attribute("gen_ai.usage.output_tokens", token_usage.get("completion_tokens", 0))
                # Note: PR #2528 specifies total_tokens as string
                span.set_attribute("gen_ai.usage.total_tokens", str(token_usage.get("total_tokens", 0)))
            
            # Model name in response
            model_name = response.llm_output.get("model_name", "")
            if model_name:
                span.set_attribute("gen_ai.response.model", model_name)
            
            # Output messages and choice events
            if response.generations and response.generations[0]:
                output_messages = []
                for gen in response.generations[0]:
                    text = gen.text
                    finish_reason = response.llm_output.get("finish_reason", "stop")
                    
                    # Check if there's a message with tool calls
                    tool_calls = None
                    if hasattr(gen, 'message'):
                        message = gen.message
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            tool_calls = message.tool_calls
                            
                            # Create tool execution spans per PR #2528
                            for tool_call in tool_calls:
                                tool_span = self.tracer.start_span(
                                    f"execute-tool", 
                                    context=trace.set_span_in_context(span),
                                    kind=SpanKind.INTERNAL
                                )
                                tool_span.set_attribute("gen_ai.operation.name", "execute-tool")
                                tool_span.set_attribute("gen_ai.tool.name", tool_call['name'])
                                tool_span.set_attribute("gen_ai.tool.id", tool_call['id'])
                                tool_span.set_attribute("gen_ai.tool.input", json.dumps(tool_call['args']))
                                self.tool_call_spans[tool_call['id']] = tool_span
                    
                    # Format output message
                    parts = [{"type": "text", "content": text}]
                    if tool_calls:
                        for tc in tool_calls:
                            parts.append({
                                "type": "tool_call",
                                "id": tc.get('id'),
                                "name": tc.get('name'),
                                "arguments": json.dumps(tc.get('args', {})) if isinstance(tc.get('args'), dict) else tc.get('args', '{}')
                            })
                    
                    output_messages.append({
                        "role": "assistant",
                        "parts": parts,
                        "finish_reason": finish_reason
                    })
                    
                    # Add choice event per PR #2528
                    self._add_event_to_span(span, "gen_ai.choice", {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": tool_calls
                    })
                
                span.set_attribute("gen_ai.output.messages", json.dumps(output_messages))
        
        # Clean up stored messages
        self.llm_messages.pop(run_id, None)
        
        # End the span
        self._end_span(run_id)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""
        span = self.spans.get(run_id)
        if span:
            span.set_attribute("error.type", type(error).__name__)
            span.set_attribute("error.message", str(error))
            span.record_exception(error)
        
        self._end_span(run_id, Status(StatusCode.ERROR, str(error)))

    # Chain-related callbacks
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
        """Run when a chain starts running."""
        # Check if this is an agent-specific chain
        chain_name = serialized.get("name", "")
        
        # Create agent span if this looks like an agent execution
        if "agent" in chain_name.lower() or (metadata and metadata.get("agent_name")):
            span = self._create_span("agent", run_id, parent_run_id, SpanKind.INTERNAL)
            if span:
                span.set_attribute("gen_ai.operation.name", "agent")
                if metadata and metadata.get("agent_name"):
                    span.set_attribute("gen_ai.agent.name", metadata["agent_name"])
                if metadata and metadata.get("agent_id"):
                    span.set_attribute("gen_ai.agent.id", metadata["agent_id"])
                self.agent_spans[run_id] = span

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when a chain ends running."""
        # End agent span if exists
        if run_id in self.agent_spans:
            agent_span = self.agent_spans.pop(run_id)
            agent_span.end()
            
        # Check if this is the final chain end (root level)
        if tags and "travel-planning-execution" in tags and not parent_run_id:
            # Add the final output to root span if available
            if self.root_span and outputs:
                # Extract the final message if available
                if "messages" in outputs and outputs["messages"]:
                    last_message = outputs["messages"][-1]
                    if isinstance(last_message, AIMessage) and "FINAL TRAVEL PLAN" in last_message.content:
                        self.root_span.set_attribute("gen_ai.final_output", last_message.content)
                        # Add final output event
                        self._add_event_to_span(self.root_span, "gen_ai.final_output", {
                            "role": "assistant",
                            "content": last_message.content
                        })
            
            # End the root span
            if self.root_run_id:
                self._end_span(self.root_run_id)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors."""
        # End agent span if exists
        if run_id in self.agent_spans:
            agent_span = self.agent_spans.pop(run_id)
            agent_span.set_attribute("error.type", type(error).__name__)
            agent_span.set_attribute("error.message", str(error))
            agent_span.record_exception(error)
            agent_span.set_status(Status(StatusCode.ERROR, str(error)))
            agent_span.end()
            
        # If this is the root chain error, end the root span with error
        if tags and "travel-planning-execution" in tags and not parent_run_id:
            if self.root_span:
                self.root_span.set_attribute("error.type", type(error).__name__)
                self.root_span.set_attribute("error.message", str(error))
                self.root_span.record_exception(error)
            if self.root_run_id:
                self._end_span(self.root_run_id, Status(StatusCode.ERROR, str(error)))

    # Tool-related callbacks
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
        """Run when the tool starts running."""
        # Tool spans are created in on_llm_end when we detect tool calls
        # This ensures proper parent-child relationship
        pass

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when the tool ends running."""
        # End the first available tool span
        for tool_id, span in list(self.tool_call_spans.items()):
            if span:
                span.set_attribute("gen_ai.tool.output", str(output)[:1000])
                span.end()
                del self.tool_call_spans[tool_id]
                break

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""
        for tool_id, span in list(self.tool_call_spans.items()):
            if span:
                span.set_attribute("error.type", type(error).__name__)
                span.set_attribute("error.message", str(error))
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.end()
                del self.tool_call_spans[tool_id]
                break

    # Agent-related callbacks
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action."""
        if self.debug:
            logger.debug(f"Agent action: {action.tool} with input: {action.tool_input}")

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on the agent end."""
        if self.debug:
            logger.debug(f"Agent finish: {finish.return_values}")

    # Retriever-related callbacks
    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on the retriever start."""
        pass

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on the retriever end."""
        pass

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on retriever error."""
        pass

    # Other callbacks
    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on an arbitrary text."""
        if self.debug:
            logger.debug(f"Text output: {text[:100]}...")

    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on a retry event."""
        pass

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Override to define a handler for a custom event."""
        if self.debug:
            logger.debug(f"Custom event '{name}': {data}")

    # Async versions of callbacks (delegate to sync versions)
    async def on_llm_start_async(self, *args, **kwargs):
        """Async version of on_llm_start."""
        self.on_llm_start(*args, **kwargs)
        
    async def on_chat_model_start_async(self, *args, **kwargs):
        """Async version of on_chat_model_start."""
        self.on_chat_model_start(*args, **kwargs)
        
    async def on_llm_end_async(self, *args, **kwargs):
        """Async version of on_llm_end."""
        self.on_llm_end(*args, **kwargs)
        
    async def on_llm_error_async(self, *args, **kwargs):
        """Async version of on_llm_error."""
        self.on_llm_error(*args, **kwargs)
        
    async def on_chain_start_async(self, *args, **kwargs):
        """Async version of on_chain_start."""
        self.on_chain_start(*args, **kwargs)
        
    async def on_chain_end_async(self, *args, **kwargs):
        """Async version of on_chain_end."""
        self.on_chain_end(*args, **kwargs)
        
    async def on_chain_error_async(self, *args, **kwargs):
        """Async version of on_chain_error."""
        self.on_chain_error(*args, **kwargs)
        
    async def on_tool_start_async(self, *args, **kwargs):
        """Async version of on_tool_start."""
        self.on_tool_start(*args, **kwargs)
        
    async def on_tool_end_async(self, *args, **kwargs):
        """Async version of on_tool_end."""
        self.on_tool_end(*args, **kwargs)
        
    async def on_tool_error_async(self, *args, **kwargs):
        """Async version of on_tool_error."""
        self.on_tool_error(*args, **kwargs)

    # Properties to control callback behavior
    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return False

    @property
    def ignore_retry(self) -> bool:
        """Whether to ignore retry callbacks."""
        return False

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return False

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return False

    @property
    def ignore_retriever(self) -> bool:
        """Whether to ignore retriever callbacks."""
        return False

    @property
    def ignore_chat_model(self) -> bool:
        """Whether to ignore chat model callbacks."""
        return False

    @property
    def ignore_custom_event(self) -> bool:
        """Ignore custom event."""
        return False