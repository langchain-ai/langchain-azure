"""Azure Monitor OpenTelemetry observability module.

This module provides comprehensive observability for Azure AI Foundry agents:
- Azure Monitor integration via Application Insights
- Request/response logging middleware
- Agent execution time tracking
- Token usage monitoring
- Custom metrics and spans

Usage:
    from langchain_azure_ai.observability import (
        setup_azure_monitor,
        TracingMiddleware,
        AgentTelemetry,
    )
    
    # Initialize Azure Monitor
    setup_azure_monitor()
    
    # Use telemetry in agents
    telemetry = AgentTelemetry("my-agent")
    with telemetry.track_execution() as span:
        result = agent.invoke(input)
        telemetry.record_tokens(prompt_tokens=100, completion_tokens=50)
"""

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# Type for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class TelemetryConfig:
    """Configuration for observability features.
    
    Attributes:
        app_insights_connection: Application Insights connection string.
        enable_azure_monitor: Whether to enable Azure Monitor tracing.
        enable_request_logging: Whether to log requests/responses.
        enable_token_tracking: Whether to track token usage.
        sample_rate: Sampling rate for telemetry (0.0 to 1.0).
        custom_dimensions: Additional dimensions to include in all telemetry.
    """
    app_insights_connection: Optional[str] = None
    enable_azure_monitor: bool = True
    enable_request_logging: bool = True
    enable_token_tracking: bool = True
    sample_rate: float = 1.0
    custom_dimensions: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> "TelemetryConfig":
        """Create configuration from environment variables."""
        sample_rate_str = os.getenv("TELEMETRY_SAMPLE_RATE", "1.0")
        try:
            sample_rate = float(sample_rate_str)
        except ValueError:
            logger.warning(
                "Invalid TELEMETRY_SAMPLE_RATE value %r; falling back to default 1.0",
                sample_rate_str,
            )
            sample_rate = 1.0
        else:
            if sample_rate < 0.0 or sample_rate > 1.0:
                logger.warning(
                    "TELEMETRY_SAMPLE_RATE %r is out of range [0.0, 1.0]; clamping.",
                    sample_rate,
                )
                sample_rate = max(0.0, min(1.0, sample_rate))
        return cls(
            app_insights_connection=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
            enable_azure_monitor=os.getenv("ENABLE_AZURE_MONITOR", "true").lower() == "true",
            enable_request_logging=os.getenv("ENABLE_REQUEST_LOGGING", "true").lower() == "true",
            enable_token_tracking=os.getenv("ENABLE_TOKEN_TRACKING", "true").lower() == "true",
            sample_rate=sample_rate,
        )


# Global telemetry state
_azure_monitor_initialized = False
_tracer = None
_meter = None


def setup_azure_monitor(
    connection_string: Optional[str] = None,
    config: Optional[TelemetryConfig] = None,
) -> bool:
    """Initialize Azure Monitor OpenTelemetry integration.
    
    This function sets up Azure Monitor for distributed tracing and metrics.
    It should be called once at application startup.
    
    Args:
        connection_string: Application Insights connection string.
            If not provided, reads from APPLICATIONINSIGHTS_CONNECTION_STRING.
        config: Optional TelemetryConfig for advanced settings.
        
    Returns:
        True if Azure Monitor was successfully initialized, False otherwise.
        
    Example:
        >>> from langchain_azure_ai.observability import setup_azure_monitor
        >>> setup_azure_monitor()
        True
    """
    global _azure_monitor_initialized, _tracer, _meter
    
    if _azure_monitor_initialized:
        logger.debug("Azure Monitor already initialized")
        return True
    
    config = config or TelemetryConfig.from_env()
    conn_string = connection_string or config.app_insights_connection
    
    if not conn_string:
        logger.warning(
            "APPLICATIONINSIGHTS_CONNECTION_STRING not set. "
            "Azure Monitor tracing will be disabled."
        )
        return False
    
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        from opentelemetry import trace, metrics
        from opentelemetry.instrumentation.threading import ThreadingInstrumentor
        
        # Configure Azure Monitor
        configure_azure_monitor(
            connection_string=conn_string,
            enable_live_metrics=True,
            instrumentation_options={
                "azure_sdk": {"enabled": True},
                "flask": {"enabled": False},
                "django": {"enabled": False},
                "fastapi": {"enabled": True},
                "requests": {"enabled": True},
            },
        )
        
        # Instrument threading for async operations
        ThreadingInstrumentor().instrument()
        
        # Get tracer and meter
        _tracer = trace.get_tracer("langchain_azure_ai.agents")
        _meter = metrics.get_meter("langchain_azure_ai.agents")
        
        _azure_monitor_initialized = True
        logger.info("Azure Monitor OpenTelemetry initialized successfully")
        return True
        
    except ImportError as e:
        logger.warning(
            f"OpenTelemetry packages not installed: {e}. "
            "Install with: pip install langchain-azure-ai[opentelemetry]"
        )
        return False
    except Exception as e:
        logger.error(f"Failed to initialize Azure Monitor: {e}")
        return False


def get_tracer():
    """Get the OpenTelemetry tracer instance."""
    global _tracer
    if _tracer is None:
        try:
            from opentelemetry import trace
            _tracer = trace.get_tracer("langchain_azure_ai.agents")
        except ImportError:
            return None
    return _tracer


def get_meter():
    """Get the OpenTelemetry meter instance."""
    global _meter
    if _meter is None:
        try:
            from opentelemetry import metrics
            _meter = metrics.get_meter("langchain_azure_ai.agents")
        except ImportError:
            return None
    return _meter


@dataclass
class ExecutionMetrics:
    """Metrics captured during agent execution.
    
    Attributes:
        start_time: Execution start timestamp.
        end_time: Execution end timestamp.
        duration_ms: Total execution time in milliseconds.
        prompt_tokens: Number of prompt tokens used.
        completion_tokens: Number of completion tokens generated.
        total_tokens: Total tokens used.
        success: Whether execution was successful.
        error: Error message if execution failed.
        agent_name: Name of the agent.
        agent_type: Type of the agent.
    """
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    success: bool = True
    error: Optional[str] = None
    agent_name: str = ""
    agent_type: str = ""
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def finalize(self) -> None:
        """Finalize metrics calculation."""
        self.end_time = datetime.utcnow()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.total_tokens = self.prompt_tokens + self.completion_tokens


class AgentTelemetry:
    """Telemetry collector for agent operations.
    
    This class provides convenient methods for tracking agent execution,
    recording metrics, and creating spans for distributed tracing.
    
    Example:
        >>> telemetry = AgentTelemetry("my-agent", "enterprise")
        >>> with telemetry.track_execution() as metrics:
        ...     result = agent.invoke(input)
        ...     metrics.prompt_tokens = 100
        ...     metrics.completion_tokens = 50
        >>> print(f"Execution took {metrics.duration_ms}ms")
    """
    
    def __init__(
        self,
        agent_name: str,
        agent_type: str = "custom",
        config: Optional[TelemetryConfig] = None,
    ):
        """Initialize agent telemetry.
        
        Args:
            agent_name: Name of the agent for identification.
            agent_type: Type of agent (it, enterprise, deep, custom).
            config: Optional telemetry configuration.
        """
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.config = config or TelemetryConfig.from_env()
        self._tracer = get_tracer()
        self._meter = get_meter()
        
        # Create metrics instruments if available
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Set up OpenTelemetry metrics instruments."""
        if self._meter is None:
            return
            
        try:
            # Execution duration histogram
            self._duration_histogram = self._meter.create_histogram(
                name="agent.execution.duration",
                description="Agent execution duration in milliseconds",
                unit="ms",
            )
            
            # Token usage counter
            self._token_counter = self._meter.create_counter(
                name="agent.tokens.total",
                description="Total tokens used by agent",
                unit="tokens",
            )
            
            # Request counter
            self._request_counter = self._meter.create_counter(
                name="agent.requests.total",
                description="Total agent requests",
                unit="requests",
            )
            
            # Error counter
            self._error_counter = self._meter.create_counter(
                name="agent.errors.total",
                description="Total agent errors",
                unit="errors",
            )
            
        except Exception as e:
            logger.warning(f"Failed to setup metrics instruments: {e}")
    
    @contextmanager
    def track_execution(
        self,
        operation: str = "invoke",
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracking agent execution.
        
        Args:
            operation: Name of the operation (invoke, stream, chat).
            attributes: Additional attributes to record.
            
        Yields:
            ExecutionMetrics instance for recording metrics.
            
        Example:
            >>> with telemetry.track_execution("chat") as metrics:
            ...     response = agent.chat(message)
            ...     metrics.prompt_tokens = 100
        """
        metrics = ExecutionMetrics(
            agent_name=self.agent_name,
            agent_type=self.agent_type,
        )
        
        # Start span if tracer is available
        span = None
        if self._tracer:
            try:
                span = self._tracer.start_span(
                    f"agent.{operation}",
                    attributes={
                        "agent.name": self.agent_name,
                        "agent.type": self.agent_type,
                        "agent.operation": operation,
                        **(attributes or {}),
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to start span: {e}")
        
        try:
            yield metrics
            metrics.success = True
            
        except Exception as e:
            metrics.success = False
            metrics.error = str(e)
            
            if span:
                try:
                    from opentelemetry.trace import StatusCode
                    span.set_status(StatusCode.ERROR, str(e))
                    span.record_exception(e)
                except Exception:
                    pass
            raise
            
        finally:
            metrics.finalize()
            
            # Record metrics
            self._record_metrics(metrics)
            
            # End span
            if span:
                try:
                    span.set_attribute("execution.duration_ms", metrics.duration_ms)
                    span.set_attribute("tokens.prompt", metrics.prompt_tokens)
                    span.set_attribute("tokens.completion", metrics.completion_tokens)
                    span.set_attribute("tokens.total", metrics.total_tokens)
                    span.set_attribute("execution.success", metrics.success)
                    span.end()
                except Exception as e:
                    logger.debug(f"Failed to finalize span: {e}")
    
    def _record_metrics(self, metrics: ExecutionMetrics) -> None:
        """Record execution metrics to OpenTelemetry."""
        labels = {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
        }
        
        try:
            if hasattr(self, "_duration_histogram"):
                self._duration_histogram.record(metrics.duration_ms, labels)
            
            if hasattr(self, "_token_counter") and metrics.total_tokens > 0:
                self._token_counter.add(metrics.total_tokens, labels)
            
            if hasattr(self, "_request_counter"):
                self._request_counter.add(1, labels)
            
            if hasattr(self, "_error_counter") and not metrics.success:
                self._error_counter.add(1, labels)
                
        except Exception as e:
            logger.debug(f"Failed to record metrics: {e}")
    
    def record_tokens(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record token usage.
        
        Args:
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
        """
        labels = {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
        }
        
        try:
            if hasattr(self, "_token_counter"):
                self._token_counter.add(prompt_tokens + completion_tokens, labels)
        except Exception as e:
            logger.debug(f"Failed to record tokens: {e}")
    
    def log_request(
        self,
        message: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an incoming request.
        
        Args:
            message: The user message.
            session_id: Optional session identifier.
            metadata: Additional request metadata.
        """
        if not self.config.enable_request_logging:
            return
            
        logger.info(
            f"[{self.agent_name}] Request: {message[:100]}...",
            extra={
                "agent_name": self.agent_name,
                "agent_type": self.agent_type,
                "session_id": session_id,
                "message_length": len(message),
                **(metadata or {}),
            },
        )
    
    def log_response(
        self,
        response: str,
        session_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an outgoing response.
        
        Args:
            response: The agent response.
            session_id: Optional session identifier.
            duration_ms: Request duration in milliseconds.
            metadata: Additional response metadata.
        """
        if not self.config.enable_request_logging:
            return
            
        logger.info(
            f"[{self.agent_name}] Response: {response[:100]}...",
            extra={
                "agent_name": self.agent_name,
                "agent_type": self.agent_type,
                "session_id": session_id,
                "response_length": len(response),
                "duration_ms": duration_ms,
                **(metadata or {}),
            },
        )


def trace_agent(
    operation: str = "invoke",
    track_tokens: bool = True,
):
    """Decorator for tracing agent methods.
    
    This decorator adds OpenTelemetry tracing to agent methods,
    automatically recording execution time and optional token usage.
    
    Args:
        operation: Name of the operation being traced.
        track_tokens: Whether to track token usage from response.
        
    Example:
        >>> class MyAgent:
        ...     @trace_agent("chat")
        ...     def chat(self, message: str) -> str:
        ...         return self.agent.invoke(message)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get or create telemetry
            telemetry = getattr(self, "_telemetry", None)
            if telemetry is None:
                agent_name = getattr(self, "name", "unknown")
                agent_type = getattr(self, "agent_type", "custom")
                if hasattr(agent_type, "value"):
                    agent_type = agent_type.value
                telemetry = AgentTelemetry(agent_name, agent_type)
            
            with telemetry.track_execution(operation) as metrics:
                result = func(self, *args, **kwargs)
                
                # Try to extract token usage from result
                if track_tokens and isinstance(result, dict):
                    usage = result.get("usage", {})
                    metrics.prompt_tokens = usage.get("prompt_tokens", 0)
                    metrics.completion_tokens = usage.get("completion_tokens", 0)
                
                return result
        
        return wrapper  # type: ignore
    return decorator


# Export public API
__all__ = [
    "TelemetryConfig",
    "ExecutionMetrics",
    "AgentTelemetry",
    "setup_azure_monitor",
    "get_tracer",
    "get_meter",
    "trace_agent",
]
