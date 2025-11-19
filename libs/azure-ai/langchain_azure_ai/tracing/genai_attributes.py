"""GenAI semantic conventions attribute mapping utilities.

This module provides utilities for mapping LangChain execution data to
OpenTelemetry GenAI semantic convention attributes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


class GenAIAttributes:
    """Attribute names from GenAI semantic conventions."""
    
    # Operation
    OPERATION_NAME = "gen_ai.operation.name"
    
    # Request attributes
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_TOP_P = "gen_ai.request.top_p"
    REQUEST_TOP_K = "gen_ai.request.top_k"
    REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    
    # Provider and infrastructure
    PROVIDER_NAME = "gen_ai.provider.name"
    SERVER_ADDRESS = "server.address"
    SERVER_PORT = "server.port"
    
    # Messages
    INPUT_MESSAGES = "gen_ai.input.messages"
    SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
    OUTPUT_MESSAGES = "gen_ai.output.messages"
    INPUT_MESSAGES_TRUNCATED = "gen_ai.input.messages.truncated"
    OUTPUT_MESSAGES_TRUNCATED = "gen_ai.output.messages.truncated"
    INPUT_MESSAGES_HASH = "gen_ai.input.messages.hash"
    OUTPUT_MESSAGES_HASH = "gen_ai.output.messages.hash"
    
    # Tool attributes
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_TYPE = "gen_ai.tool.type"
    TOOL_DESCRIPTION = "gen_ai.tool.description"
    TOOL_CALL_ID = "gen_ai.tool.call.id"
    TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
    TOOL_CALL_RESULT = "gen_ai.tool.call.result"
    
    # Usage/token tracking
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    USAGE_INPUT_TOKENS_TOTAL = "gen_ai.usage.input_tokens.total"
    USAGE_OUTPUT_TOKENS_TOTAL = "gen_ai.usage.output_tokens.total"
    USAGE_TOTAL_TOKENS_TOTAL = "gen_ai.usage.total_tokens.total"
    
    # Response metadata
    RESPONSE_ID = "gen_ai.response.id"
    RESPONSE_MODEL = "gen_ai.response.model"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    
    # Agent metadata
    AGENT_NAME = "gen_ai.agent.name"
    AGENT_ID = "gen_ai.agent.id"
    AGENT_DESCRIPTION = "gen_ai.agent.description"
    CONVERSATION_ID = "gen_ai.conversation.id"
    
    # Error
    ERROR_TYPE = "error.type"
    
    # MLflow compatibility
    MLFLOW_CHAT_TOKEN_USAGE = "mlflow.chat.tokenUsage"
    MLFLOW_TRACE_TOKEN_USAGE = "mlflow.trace.tokenUsage"


def format_messages(
    messages: List[Any],
    redact: bool = False,
    max_chars: Optional[int] = None,
    role_filter: Optional[List[str]] = None
) -> tuple[str, bool, Optional[str]]:
    """Format messages according to GenAI semantic conventions.
    
    Args:
        messages: List of LangChain messages
        redact: Whether to redact message content
        max_chars: Maximum characters before truncation
        role_filter: Only include messages with these roles
        
    Returns:
        Tuple of (formatted_json, truncated, hash_if_redacted)
    """
    formatted = []
    truncated = False
    
    for msg in messages:
        # Extract role and content
        role = getattr(msg, "type", "unknown")
        
        # Filter by role if specified
        if role_filter and role not in role_filter:
            continue
        
        # Build message object
        msg_obj = {"role": role}
        
        if redact:
            # Redact content but include hash
            msg_obj["parts"] = [{"type": "text", "text": "[REDACTED]"}]
        else:
            content = getattr(msg, "content", "")
            
            # Handle truncation
            if max_chars and len(content) > max_chars:
                content = content[:max_chars] + "..."
                truncated = True
            
            msg_obj["parts"] = [{"type": "text", "text": content}]
        
        # Add tool calls if present
        if hasattr(msg, "additional_kwargs"):
            tool_calls = msg.additional_kwargs.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    msg_obj.setdefault("parts", []).append({
                        "type": "tool_call",
                        "id": tc.get("id"),
                        "function": {
                            "name": tc.get("function", {}).get("name"),
                            "arguments": tc.get("function", {}).get("arguments")
                        }
                    })
        
        formatted.append(msg_obj)
    
    # Compute hash if redacted
    hash_value = None
    if redact:
        content_str = json.dumps(formatted, sort_keys=True)
        hash_value = hashlib.sha256(content_str.encode()).hexdigest()
    
    return json.dumps(formatted), truncated, hash_value


def format_system_instructions(messages: List[Any], redact: bool = False) -> Optional[str]:
    """Extract and format system instructions from messages.
    
    Args:
        messages: List of LangChain messages
        redact: Whether to redact content
        
    Returns:
        Formatted system instructions or None
    """
    system_messages = [
        msg for msg in messages
        if getattr(msg, "type", None) == "system"
    ]
    
    if not system_messages:
        return None
    
    if redact:
        return "[REDACTED]"
    
    # Combine all system messages
    instructions = "\n".join(
        getattr(msg, "content", "") for msg in system_messages
    )
    
    return instructions if instructions else None


def normalize_token_usage(usage_data: Dict[str, Any]) -> Dict[str, int]:
    """Normalize token usage from various formats.
    
    LLMs report token usage in different formats (prompt_tokens vs input_tokens,
    completion_tokens vs output_tokens, etc.). This normalizes to a standard format.
    
    Args:
        usage_data: Raw usage data from LLM response
        
    Returns:
        Normalized dict with input_tokens, output_tokens, total_tokens
    """
    normalized = {}
    
    # Input tokens (various names)
    input_tokens = (
        usage_data.get("input_tokens") or
        usage_data.get("prompt_tokens") or
        usage_data.get("prompt") or
        0
    )
    normalized["input_tokens"] = int(input_tokens)
    
    # Output tokens (various names)
    output_tokens = (
        usage_data.get("output_tokens") or
        usage_data.get("completion_tokens") or
        usage_data.get("completion") or
        0
    )
    normalized["output_tokens"] = int(output_tokens)
    
    # Total tokens
    total_tokens = usage_data.get("total_tokens")
    if total_tokens:
        normalized["total_tokens"] = int(total_tokens)
    else:
        # Compute if not provided
        normalized["total_tokens"] = normalized["input_tokens"] + normalized["output_tokens"]
    
    return normalized


def infer_provider(
    serialized: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    invocation_params: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Infer the provider name from available data.
    
    Args:
        serialized: Serialized LLM/Chain data
        metadata: Metadata dictionary
        invocation_params: Invocation parameters
        
    Returns:
        Provider name (e.g., "azure.ai.openai", "openai", "anthropic")
    """
    # Check explicit metadata
    if metadata and "provider" in metadata:
        return metadata["provider"]
    
    # Check serialized data
    if serialized:
        # Look for class name indicators
        id_list = serialized.get("id", [])
        if isinstance(id_list, list):
            class_name = id_list[-1] if id_list else ""
            
            if "AzureChatOpenAI" in class_name or "AzureOpenAI" in class_name:
                return "azure.ai.openai"
            elif "ChatOpenAI" in class_name or "OpenAI" in class_name:
                return "openai"
            elif "ChatAnthropic" in class_name or "Anthropic" in class_name:
                return "anthropic"
    
    # Check invocation params for endpoint hints
    if invocation_params:
        base_url = invocation_params.get("base_url", "")
        if "azure" in base_url.lower():
            return "azure.ai.openai"
        elif "api.openai.com" in base_url:
            return "openai"
    
    return None


def infer_server_address_port(
    serialized: Optional[Dict[str, Any]] = None,
    invocation_params: Optional[Dict[str, Any]] = None
) -> tuple[Optional[str], Optional[int]]:
    """Infer server address and port from available data.
    
    Args:
        serialized: Serialized LLM/Chain data
        invocation_params: Invocation parameters
        
    Returns:
        Tuple of (address, port)
    """
    address = None
    port = None
    
    if invocation_params:
        base_url = invocation_params.get("base_url", "")
        if base_url:
            # Parse URL
            try:
                from urllib.parse import urlparse
                parsed = urlparse(base_url)
                address = parsed.hostname
                port = parsed.port
            except Exception:
                # Ignore all exceptions during URL parsing; return None for address and port if parsing fails.
                pass
    
    return address, port


def truncate_and_redact(
    value: Any,
    redact: bool = False,
    max_chars: Optional[int] = None
) -> tuple[Any, bool]:
    """Truncate and/or redact a value.
    
    Args:
        value: Value to process
        redact: Whether to redact
        max_chars: Maximum characters
        
    Returns:
        Tuple of (processed_value, was_truncated)
    """
    if redact:
        return "[REDACTED]", False
    
    if not isinstance(value, str):
        value = str(value)
    
    if max_chars and len(value) > max_chars:
        return value[:max_chars] + "...", True
    
    return value, False


def create_mlflow_compat_usage(usage: Dict[str, int]) -> str:
    """Create MLflow-compatible token usage JSON.
    
    Args:
        usage: Normalized usage dict
        
    Returns:
        JSON string for MLflow compatibility
    """
    mlflow_usage = {
        "promptTokens": usage.get("input_tokens", 0),
        "completionTokens": usage.get("output_tokens", 0),
        "totalTokens": usage.get("total_tokens", 0)
    }
    return json.dumps(mlflow_usage)
