"""Configuration management for Azure AI OpenTelemetry tracing."""

from __future__ import annotations

import os
from typing import Any, Dict
from threading import Lock


# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    # Provider configuration
    "provider_name": None,  # e.g., "azure.ai.openai", "openai"
    "tracer_name": "azure_ai_genai_tracer",
    
    # Feature flags
    "emit_mlflow_compat": True,  # Emit MLflow-compatible attributes
    "aggregate_usage": True,  # Aggregate token usage at root
    "enable_post_processor": True,  # Enable RunTree post-processing
    "normalize_operation_names": True,  # Normalize operation names to spec
    
    # Redaction and privacy
    "redact_messages": False,  # Redact message content
    "redact_tool_arguments": False,  # Redact tool call arguments
    "redact_tool_results": False,  # Redact tool call results
    "max_message_characters": None,  # Truncate messages if > this length
    
    # Tool and metadata inclusion
    "include_tool_definitions": False,  # Include tool schemas in spans
    "log_stream_chunks": False,  # Log streaming token chunks
    
    # Sampling and performance
    "sampling_rate": 1.0,  # Sample rate (0.0-1.0)
    
    # Span hierarchy configuration
    "prefer_last_chat_parent": True,  # Prefer last chat as parent for tools
    "honor_external_parent": True,  # Use external spans as parents
    "enable_span_links": True,  # Enable span links for relationships
    
    # Patch mode for auto-registration
    "patch_mode": "callback",  # "callback" | "hybrid" | "monkey"
    
    # Thread/session tracking
    "thread_id_attribute": "gen_ai.conversation.id",
    
    # Post-processing configuration
    "post_process_root_tag": "langchain.azure.ai",
    "store_run_tree_snapshot": False,  # Store RunTree for debugging
}


# Environment variable mappings
ENV_VAR_MAPPINGS = {
    "AZURE_GENAI_REDACT_MESSAGES": ("redact_messages", bool),
    "AZURE_GENAI_REDACT_TOOL_ARGUMENTS": ("redact_tool_arguments", bool),
    "AZURE_GENAI_REDACT_TOOL_RESULTS": ("redact_tool_results", bool),
    "AZURE_GENAI_PROVIDER_NAME": ("provider_name", str),
    "AZURE_GENAI_SAMPLING_RATE": ("sampling_rate", float),
    "AZURE_GENAI_PATCH_MODE": ("patch_mode", str),
    "AZURE_GENAI_ENABLE_POST_PROCESSOR": ("enable_post_processor", bool),
}


class TracingConfig:
    """Thread-safe configuration manager for tracing."""
    
    def __init__(self):
        self._config: Dict[str, Any] = DEFAULT_CONFIG.copy()
        self._lock = Lock()
        self._load_from_environment()
    
    def _load_from_environment(self) -> None:
        """Load configuration overrides from environment variables."""
        for env_var, (config_key, value_type) in ENV_VAR_MAPPINGS.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    if value_type == bool:
                        # Handle boolean values
                        parsed_value = value.lower() in ("true", "1", "yes", "on")
                    elif value_type == float:
                        parsed_value = float(value)
                    else:
                        parsed_value = value
                    self._config[config_key] = parsed_value
                except (ValueError, TypeError):
                    # Skip invalid environment variable values
                    pass
    
    def update(self, config: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            config: Dictionary of configuration updates
        """
        with self._lock:
            self._config.update(config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        with self._lock:
            return self._config.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values.
        
        Returns:
            Copy of current configuration
        """
        with self._lock:
            return self._config.copy()
    
    def reset(self) -> None:
        """Reset configuration to defaults."""
        with self._lock:
            self._config = DEFAULT_CONFIG.copy()
            self._load_from_environment()


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration values.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate sampling_rate
    sampling_rate = config.get("sampling_rate", 1.0)
    if not 0.0 <= sampling_rate <= 1.0:
        raise ValueError(f"sampling_rate must be between 0.0 and 1.0, got {sampling_rate}")
    
    # Validate patch_mode
    patch_mode = config.get("patch_mode", "callback")
    valid_modes = {"callback", "hybrid", "monkey"}
    if patch_mode not in valid_modes:
        raise ValueError(
            f"patch_mode must be one of {valid_modes}, got {patch_mode}"
        )
    
    # Validate max_message_characters
    max_chars = config.get("max_message_characters")
    if max_chars is not None and (not isinstance(max_chars, int) or max_chars < 0):
        raise ValueError(
            f"max_message_characters must be a positive integer or None, got {max_chars}"
        )
