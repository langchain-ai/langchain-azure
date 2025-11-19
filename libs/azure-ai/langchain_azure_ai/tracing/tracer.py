"""Azure AI OpenTelemetry Tracer with static autolog API.

This module provides a static facade over the existing AzureAIOpenTelemetryTracer
to enable MLflow-like autolog functionality with automatic callback registration.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any, Dict, Optional

# Import the existing tracer implementation
from langchain_azure_ai.callbacks.tracers.inference_tracing import (
    AzureAIOpenTelemetryTracer as _BaseTracer,
)
from langchain_azure_ai.tracing.callback_manager import (
    apply_monkey_patches,
    is_monkey_patched,
    remove_monkey_patches,
    set_active_tracer,
)
from langchain_azure_ai.tracing.config import (
    TracingConfig,
    validate_config,
)

LOGGER = logging.getLogger(__name__)


class AzureAIOpenTelemetryTracer(_BaseTracer):
    """Azure AI OpenTelemetry tracer with static autolog API.
    
    This class extends the base tracer with static methods for global
    configuration and automatic callback registration.
    
    Usage:
        # Configure once at startup
        AzureAIOpenTelemetryTracer.set_app_insights(
            "InstrumentationKey=...;IngestionEndpoint=..."
        )
        AzureAIOpenTelemetryTracer.set_config({
            "provider_name": "azure.ai.openai",
            "redact_messages": False
        })
        
        # Enable automatic tracing
        AzureAIOpenTelemetryTracer.autolog()
        
        # All LangChain operations are now automatically traced
        result = chain.invoke(input)  # No callbacks needed!
    """
    
    # Class-level state for autolog
    _global_config: TracingConfig = TracingConfig()
    _app_insights_connection_string: Optional[str] = None
    _autolog_active: bool = False
    _global_tracer_instance: Optional[AzureAIOpenTelemetryTracer] = None
    _global_tags: Dict[str, Any] = {}
    _state_lock: Lock = Lock()
    
    @classmethod
    def set_app_insights(cls, connection_string: str) -> None:
        """Set the Application Insights connection string.
        
        Args:
            connection_string: Azure Application Insights connection string
                in the format "InstrumentationKey=...;IngestionEndpoint=..."
        
        Example:
            AzureAIOpenTelemetryTracer.set_app_insights(
                "InstrumentationKey=abc123;IngestionEndpoint=https://..."
            )
        """
        with cls._state_lock:
            cls._app_insights_connection_string = connection_string
            LOGGER.info("Application Insights connection string configured")
    
    @classmethod
    def set_config(cls, config: Dict[str, Any]) -> None:
        """Update global tracer configuration.
        
        Args:
            config: Configuration dictionary with options like:
                - provider_name: Provider name (e.g., "azure.ai.openai")
                - redact_messages: Whether to redact message content
                - redact_tool_arguments: Whether to redact tool arguments
                - redact_tool_results: Whether to redact tool results
                - sampling_rate: Sampling rate (0.0-1.0)
                - patch_mode: "callback", "hybrid", or "monkey"
                - And many more (see config.py for full list)
        
        Raises:
            ValueError: If configuration is invalid
        
        Example:
            AzureAIOpenTelemetryTracer.set_config({
                "provider_name": "azure.ai.openai",
                "redact_messages": False,
                "sampling_rate": 1.0
            })
        """
        validate_config(config)
        with cls._state_lock:
            cls._global_config.update(config)
            LOGGER.info(f"Configuration updated with {len(config)} options")
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get current global configuration.
        
        Returns:
            Copy of current configuration dictionary
        
        Example:
            config = AzureAIOpenTelemetryTracer.get_config()
            print(f"Provider: {config['provider_name']}")
        """
        with cls._state_lock:
            return cls._global_config.get_all()
    
    @classmethod
    def autolog(cls, **overrides: Any) -> None:
        """Enable automatic tracing of all LangChain operations.
        
        After calling this method, all LangChain/LangGraph executions will
        automatically emit OpenTelemetry spans without needing to pass
        callbacks explicitly.
        
        Args:
            **overrides: Configuration overrides to apply for this autolog session.
                These take precedence over values set via set_config().
        
        Raises:
            RuntimeError: If monkey patching fails
        
        Example:
            # Basic usage
            AzureAIOpenTelemetryTracer.autolog()
            
            # With overrides
            AzureAIOpenTelemetryTracer.autolog(
                sampling_rate=0.5,
                redact_messages=True
            )
        """
        with cls._state_lock:
            if cls._autolog_active:
                LOGGER.info("Autolog already active")
                return
            
            # Apply configuration overrides
            if overrides:
                validate_config(overrides)
                cls._global_config.update(overrides)
            
            # Get final configuration
            config = cls._global_config.get_all()
            
            # Configure Azure Monitor if connection string is set
            # The connection string will be passed to the tracer instance creation below
            if not cls._app_insights_connection_string:
                cls._app_insights_connection_string = os.getenv(
                    "APPLICATION_INSIGHTS_CONNECTION_STRING"
                )
                if not cls._app_insights_connection_string:
                    LOGGER.warning(
                        "No Application Insights connection string configured. "
                        "Call set_app_insights() or set "
                        "APPLICATION_INSIGHTS_CONNECTION_STRING environment variable."
                    )
            
            # Create global tracer instance (base class handles Azure Monitor configuration)
            enable_content_recording = not config.get("redact_messages", False)
            provider_name = config.get("provider_name")
            tracer_name = config.get("tracer_name", "azure_ai_genai_tracer")
            
            cls._global_tracer_instance = cls(
                connection_string=cls._app_insights_connection_string,
                enable_content_recording=enable_content_recording,
                provider_name=provider_name,
                name=tracer_name,
            )
            
            # Apply monkey patches for auto-registration
            patch_mode = config.get("patch_mode", "callback")
            
            if patch_mode in ("hybrid", "monkey"):
                try:
                    apply_monkey_patches()
                    set_active_tracer(cls._global_tracer_instance)
                    LOGGER.info(
                        f"Autolog enabled with patch_mode={patch_mode}. "
                        "All LangChain operations will be automatically traced."
                    )
                except Exception as e:
                    LOGGER.error(f"Failed to apply monkey patches: {e}")
                    raise RuntimeError(
                        "Failed to enable autolog with monkey patching. "
                        "Try patch_mode='callback' instead."
                    ) from e
            else:
                # Callback mode - user still needs to pass tracer
                # but we make it available via get_active_tracer()
                set_active_tracer(cls._global_tracer_instance)
                LOGGER.warning(
                    "Autolog enabled in callback mode. Tracer instance created but "
                    "you must still pass it via callbacks. Use patch_mode='monkey' "
                    "for true auto-registration."
                )
            
            cls._autolog_active = True
    
    @classmethod
    def is_active(cls) -> bool:
        """Check if autolog is currently active.
        
        Returns:
            True if autolog is enabled
        
        Example:
            if AzureAIOpenTelemetryTracer.is_active():
                print("Tracing is enabled")
        """
        with cls._state_lock:
            return cls._autolog_active
    
    @classmethod
    def add_tags(cls, tags: Dict[str, Any]) -> None:
        """Add global tags to be attached to all spans.
        
        Args:
            tags: Dictionary of tag key-value pairs
        
        Example:
            AzureAIOpenTelemetryTracer.add_tags({
                "environment": "production",
                "version": "1.0.0"
            })
        """
        with cls._state_lock:
            cls._global_tags.update(tags)
            LOGGER.info(f"Added {len(tags)} global tags")
            
            # If tracer is active, we would apply tags to it here
            # For now, tags will be used when creating new spans
    
    @classmethod
    def update_redaction_rules(cls, **flags: bool) -> None:
        """Update redaction configuration flags.
        
        Args:
            **flags: Redaction flags to update:
                - redact_messages: Redact message content
                - redact_tool_arguments: Redact tool arguments
                - redact_tool_results: Redact tool results
        
        Example:
            AzureAIOpenTelemetryTracer.update_redaction_rules(
                redact_messages=True,
                redact_tool_arguments=True
            )
        """
        with cls._state_lock:
            cls._global_config.update(flags)
            LOGGER.info(f"Updated {len(flags)} redaction rules")
    
    @classmethod
    def force_flush(cls) -> None:
        """Force flush all pending telemetry data.
        
        This ensures all spans are sent to Application Insights immediately.
        
        Example:
            # At end of critical operation
            AzureAIOpenTelemetryTracer.force_flush()
        """
        try:
            from opentelemetry import trace
            
            # Get the tracer provider and flush
            tracer_provider = trace.get_tracer_provider()
            if hasattr(tracer_provider, "force_flush"):
                tracer_provider.force_flush()
                LOGGER.info("Forced flush of telemetry data")
        except Exception as e:
            LOGGER.warning(f"Error during force flush: {e}")
    
    @classmethod
    def shutdown(cls) -> None:
        """Shutdown the tracer and disable autolog.
        
        This removes monkey patches, flushes pending data, and deactivates
        automatic tracing. Call this at application shutdown.
        
        Example:
            # At application exit
            AzureAIOpenTelemetryTracer.shutdown()
        """
        with cls._state_lock:
            if not cls._autolog_active:
                LOGGER.info("Autolog not active, nothing to shutdown")
                return
            
            # Force flush before shutdown
            cls.force_flush()
            
            # Remove monkey patches
            if is_monkey_patched():
                remove_monkey_patches()
            
            # Clear active tracer
            set_active_tracer(None)
            
            # Shutdown tracer provider
            try:
                from opentelemetry import trace
                
                tracer_provider = trace.get_tracer_provider()
                if hasattr(tracer_provider, "shutdown"):
                    tracer_provider.shutdown()
            except Exception as e:
                LOGGER.warning(f"Error during tracer provider shutdown: {e}")
            
            # Reset state
            cls._autolog_active = False
            cls._global_tracer_instance = None
            
            LOGGER.info("Autolog shutdown complete")
    
    @classmethod
    def get_tracer_instance(cls) -> Optional[AzureAIOpenTelemetryTracer]:
        """Get the global tracer instance if autolog is active.
        
        Returns:
            Global tracer instance or None if autolog is not active
        
        Example:
            tracer = AzureAIOpenTelemetryTracer.get_tracer_instance()
            if tracer:
                # Use tracer explicitly if needed
                result = chain.invoke(input, config={"callbacks": [tracer]})
        """
        with cls._state_lock:
            return cls._global_tracer_instance


# Export the enhanced tracer
__all__ = ["AzureAIOpenTelemetryTracer"]
