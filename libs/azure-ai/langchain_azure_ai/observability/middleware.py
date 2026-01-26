"""FastAPI middleware for observability and request tracking.

This module provides middleware components for:
- Request/response logging
- Execution time tracking
- OpenTelemetry context propagation
- Error tracking and alerting

Usage:
    from fastapi import FastAPI
    from langchain_azure_ai.observability.middleware import (
        TracingMiddleware,
        RequestLoggingMiddleware,
    )
    
    app = FastAPI()
    app.add_middleware(TracingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses.
    
    This middleware logs:
    - Request method, path, and client IP
    - Response status code and execution time
    - Request body (configurable)
    
    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> app.add_middleware(
        ...     RequestLoggingMiddleware,
        ...     log_request_body=True,
        ... )
    """
    
    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        exclude_paths: Optional[list] = None,
        max_body_log_size: int = 1000,
    ):
        """Initialize the middleware.
        
        Args:
            app: The ASGI application.
            log_request_body: Whether to log request bodies.
            exclude_paths: List of paths to exclude from logging.
            max_body_log_size: Maximum body size to log (truncates beyond this).
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
        self.max_body_log_size = max_body_log_size
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process the request and log details."""
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        
        # Log request
        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", "unknown"),
        }
        
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                body_str = body.decode("utf-8")[:self.max_body_log_size]
                log_data["request_body"] = body_str
                # Reset body for downstream handlers
                request._body = body
            except Exception as e:
                log_data["request_body_error"] = str(e)
        
        logger.info(f"[{request_id}] → {request.method} {request.url.path}", extra=log_data)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log response
            response_log = {
                **log_data,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            }
            
            log_level = logging.INFO if response.status_code < 400 else logging.WARNING
            logger.log(
                log_level,
                f"[{request_id}] ← {response.status_code} ({duration_ms:.2f}ms)",
                extra=response_log,
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            
            return response
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{request_id}] ✗ Error: {str(e)} ({duration_ms:.2f}ms)",
                extra={
                    **log_data,
                    "error": str(e),
                    "duration_ms": round(duration_ms, 2),
                },
                exc_info=True,
            )
            raise


class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware for OpenTelemetry distributed tracing.
    
    This middleware:
    - Creates spans for HTTP requests
    - Propagates trace context
    - Records request/response attributes
    - Captures errors as span events
    
    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> app.add_middleware(TracingMiddleware)
    """
    
    def __init__(
        self,
        app: ASGIApp,
        service_name: str = "azure-ai-agents",
        exclude_paths: Optional[list] = None,
    ):
        """Initialize the middleware.
        
        Args:
            app: The ASGI application.
            service_name: Name of the service for tracing.
            exclude_paths: List of paths to exclude from tracing.
        """
        super().__init__(app)
        self.service_name = service_name
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
        self._tracer = None
    
    def _get_tracer(self):
        """Lazy load the tracer."""
        if self._tracer is None:
            try:
                from opentelemetry import trace
                self._tracer = trace.get_tracer(self.service_name)
            except ImportError:
                return None
        return self._tracer
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process the request with tracing."""
        # Skip tracing for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        tracer = self._get_tracer()
        if tracer is None:
            return await call_next(request)
        
        # Extract trace context from headers
        try:
            from opentelemetry.propagate import extract
            from opentelemetry.trace import SpanKind
            
            context = extract(dict(request.headers))
            
            with tracer.start_as_current_span(
                f"{request.method} {request.url.path}",
                context=context,
                kind=SpanKind.SERVER,
                attributes={
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "http.route": request.url.path,
                    "http.scheme": request.url.scheme,
                    "http.host": request.url.hostname or "",
                    "http.client_ip": request.client.host if request.client else "",
                    "http.user_agent": request.headers.get("user-agent", ""),
                },
            ) as span:
                try:
                    response = await call_next(request)
                    
                    span.set_attribute("http.status_code", response.status_code)
                    
                    if response.status_code >= 400:
                        from opentelemetry.trace import StatusCode
                        span.set_status(
                            StatusCode.ERROR,
                            f"HTTP {response.status_code}",
                        )
                    
                    return response
                    
                except Exception as e:
                    from opentelemetry.trace import StatusCode
                    span.set_status(StatusCode.ERROR, str(e))
                    span.record_exception(e)
                    raise
                    
        except ImportError:
            return await call_next(request)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for recording HTTP metrics.
    
    This middleware records:
    - Request count by endpoint and status
    - Request duration histogram
    - Active request gauge
    
    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> app.add_middleware(MetricsMiddleware)
    """
    
    def __init__(
        self,
        app: ASGIApp,
        service_name: str = "azure-ai-agents",
        exclude_paths: Optional[list] = None,
    ):
        """Initialize the middleware.
        
        Args:
            app: The ASGI application.
            service_name: Name of the service for metrics.
            exclude_paths: List of paths to exclude from metrics.
        """
        super().__init__(app)
        self.service_name = service_name
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
        self._meter = None
        self._request_counter = None
        self._duration_histogram = None
        self._active_requests = None
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Set up metrics instruments."""
        try:
            from opentelemetry import metrics
            
            self._meter = metrics.get_meter(self.service_name)
            
            self._request_counter = self._meter.create_counter(
                name="http.server.requests",
                description="Total HTTP requests",
                unit="requests",
            )
            
            self._duration_histogram = self._meter.create_histogram(
                name="http.server.duration",
                description="HTTP request duration",
                unit="ms",
            )
            
            self._active_requests = self._meter.create_up_down_counter(
                name="http.server.active_requests",
                description="Number of active HTTP requests",
                unit="requests",
            )
            
        except ImportError:
            logger.debug("OpenTelemetry metrics not available")
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process the request and record metrics."""
        # Skip metrics for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        if self._meter is None:
            return await call_next(request)
        
        labels = {
            "method": request.method,
            "route": request.url.path,
        }
        
        # Track active requests
        if self._active_requests:
            self._active_requests.add(1, labels)
        
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            
            labels["status_code"] = str(response.status_code)
            
            return response
            
        except Exception:
            labels["status_code"] = "500"
            raise
            
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            if self._duration_histogram:
                self._duration_histogram.record(duration_ms, labels)
            
            if self._request_counter:
                self._request_counter.add(1, labels)
            
            if self._active_requests:
                self._active_requests.add(-1, {"method": request.method, "route": request.url.path})


# Export public API
__all__ = [
    "RequestLoggingMiddleware",
    "TracingMiddleware",
    "MetricsMiddleware",
]
